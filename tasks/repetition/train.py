"""Training script for CTM repetition counting.

Counts are treated as classification over n_count_buckets discrete buckets.
This keeps the existing CTM certainty mechanism (entropy-based) intact and
produces a full distribution over counts at each tick, enabling both argmax
(OBO accuracy) and soft expected-count (MAE) evaluation.

The frame-tick coupling is inherited from ContinuousThoughtMachineRepCount /
ContinuousThoughtMachineVideo: at internal tick t the model attends to frame
t // iterations_per_frame. Using n_frames=64 with iterations_per_frame=1
gives 64 ticks, yielding a Nyquist limit of 32 distinct repetitions.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm.auto import tqdm

sns.set_style("darkgrid")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from tasks.repetition.dataset import build_datasets, video_count_collate
from tasks.repetition.losses import count_loss, count_loss_survival
from tasks.repetition.model import ContinuousThoughtMachineRepCount
from tasks.repetition.utils import (
    count_from_buckets, count_from_hazards, extract_counts, mae, obo_accuracy,
)
from utils.housekeeping import set_seed
from utils.run import init_run, load_checkpoint, save_checkpoint
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "synthetic-v2", "countix", "repcount", "ucfrep"])
    parser.add_argument("--data_root", type=str, default="data/repetition")
    parser.add_argument("--kinetics_root", type=str, default=None,
                        help="For --dataset countix: root of an official Kinetics-400 "
                             "mirror, used to look up videos by youtube_id when the "
                             "Countix CSVs are not co-located with the videos.")
    parser.add_argument("--n_frames", type=int, default=64,
                        help="Frames sampled per clip in legacy TSN mode. "
                             "Used by real datasets only when target_fps<=0; "
                             "ignored for the synthetic backend (which always "
                             "uses fps-based sampling).")
    parser.add_argument("--target_fps", type=float, default=8.0,
                        help="Frames sampled per second. Required for synthetic. "
                             "For real datasets (countix/repcount/ucfrep): "
                             "any value > 0 enables FPS-based variable-length "
                             "sampling (each clip gets T = round(duration_s × "
                             "target_fps) frames); set to 0 or negative to "
                             "fall back to legacy fixed n_frames TSN sampling.")
    parser.add_argument("--clip_duration_s_min", type=float, default=4.0,
                        help="Synthetic-only — lower bound on per-clip duration.")
    parser.add_argument("--clip_duration_s_max", type=float, default=12.0,
                        help="Synthetic-only — upper bound on per-clip duration. "
                             "Set equal to _min for fixed-length clips.")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--max_count", type=int, default=16,
                        help="Max oscillation count for the synthetic dataset.")
    parser.add_argument("--n_count_buckets", type=int, default=32,
                        help="Number of discrete count classes. Labels >= n_count_buckets "
                             "are clamped to n_count_buckets-1.")
    parser.add_argument("--head_type", type=str, default="classification",
                        choices=["classification", "survival"],
                        help="Loss/decoder framing. 'classification' is vanilla "
                             "K-way categorical CE; 'survival' is CORN-style "
                             "discrete-hazard BCE (ordinal).")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_input", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--iterations_per_frame", type=int, default=1,
                        help="CTM internal ticks spent on each frame.")
    parser.add_argument("--synapse_depth", type=int, default=4)
    parser.add_argument("--n_synch_out", type=int, default=64)
    parser.add_argument("--n_synch_action", type=int, default=64)
    parser.add_argument("--neuron_select_type", type=str, default="random-pairing")
    parser.add_argument("--memory_length", type=int, default=32,
                        help="NLM history length. ~n_frames/2 covers one full period "
                             "at the median count.")
    parser.add_argument("--memory_hidden_dims", type=int, default=16)
    parser.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--backbone_type", type=str, default="resnet18-2")
    parser.add_argument("--pretrained_backbone",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Use ImageNet-pretrained backbone (default). "
                             "Pass --no-pretrained_backbone to train from scratch.")
    parser.add_argument("--freeze_backbone",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze backbone params and lock BN to eval mode "
                             "(only meaningful with --pretrained_backbone).")
    parser.add_argument("--positional_embedding_type", type=str, default="none",
                        choices=["none", "learnable-fourier", "multi-learnable-fourier",
                                 "custom-rotational"])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout_nlm", type=float, default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--batch_size_test", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--training_iterations", type=int, default=50001)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler_type", type=str, default="cosine",
                        choices=["cosine", "multistep"])
    parser.add_argument("--milestones", type=int, nargs="+", default=[20000, 35000])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clipping", type=float, default=-1)

    # Housekeeping
    parser.add_argument("--log_dir", type=str, default="logs/repetition/scratch")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Subdirectory of --log_dir for this run. "
                             "Defaults to ${SLURM_JOB_ID} on JZ, else a timestamp.")
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--track_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers_train", type=int, default=2)
    parser.add_argument("--num_workers_test", type=int, default=1)
    parser.add_argument("--n_test_batches", type=int, default=20)
    parser.add_argument("--device", type=int, nargs="+", default=[-1])
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


def pick_device(args):
    if args.device[0] != -1:
        return f"cuda:{args.device[0]}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log_split_stats(ds, name: str, n_count_buckets: int) -> None:
    """Print per-split count histogram and clamp summary to the stdout log."""
    from collections import Counter

    n = len(ds)
    print(f"  {name}: len={n}")
    if n == 0:
        return

    counts = extract_counts(ds).tolist()
    if hasattr(ds, "records") and ds.records and "count" in ds.records[0]:
        pass  # used the full record list
    elif len(counts) < n:
        print(f"    (count histogram from first {len(counts)}/{n} samples)")

    arr = np.asarray(counts)
    n_clamped = int((arr >= n_count_buckets).sum())
    print(f"    count: min={arr.min()} max={arr.max()} "
          f"mean={arr.mean():.2f} median={float(np.median(arr)):.1f} "
          f"std={arr.std():.2f}")
    print(f"    clamped (>= n_count_buckets={n_count_buckets}): "
          f"{n_clamped}/{len(arr)} ({100.0 * n_clamped / max(1, len(arr)):.1f}%)")

    # Coarse decade-style histogram so the txt log stays readable.
    edges = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    hist = Counter()
    for v in counts:
        placed = False
        for lo, hi in zip(edges, edges[1:]):
            if lo <= v < hi:
                hist[f"[{lo},{hi})"] += 1
                placed = True
                break
        if not placed:
            hist["other"] += 1
    print("    histogram (count buckets):")
    for lo, hi in zip(edges, edges[1:]):
        k = f"[{lo},{hi})"
        if hist[k]:
            print(f"      {k:>10s}: {hist[k]}")
    if hist["other"]:
        print(f"      {'other':>10s}: {hist['other']}")

    # Top 10 modes (handy for spotting "most clips have count=2" baselines).
    top = Counter(counts).most_common(10)
    pretty = ", ".join(f"{c}×{k}" for k, c in top)
    print(f"    top modes: {pretty}")


def _dispatch_head(head_type: str):
    """Pick (loss_fn, decode_fn) for the requested head."""
    if head_type == "survival":
        return count_loss_survival, count_from_hazards
    return count_loss, count_from_buckets


def main():
    args = parse_args()
    args.out_dims = args.n_count_buckets
    loss_fn, decode_fn = _dispatch_head(args.head_type)
    print(f"head_type={args.head_type}  loss={loss_fn.__name__}  decode={decode_fn.__name__}")
    set_seed(args.seed, deterministic=False)
    init_run(args)
    print(f"Output dir: {os.path.abspath(args.log_dir)}")

    # --- Data ---
    # FPS-based variable-length sampling is gated by ``--target_fps > 0``.
    # The synthetic backend always uses it (it has no fixed-T fallback).
    # Real backends opt in: when target_fps > 0, they switch from TSN
    # fixed-``n_frames`` sampling to ``_decode_clip_fps`` and produce
    # variable-T clips that share a batch via padding + frame mask.
    fps = args.target_fps if args.target_fps and args.target_fps > 0 else None
    args.is_variable_length = (
        args.dataset in ("synthetic", "synthetic-v2") or fps is not None
    )

    extra = {}
    if args.dataset in ("synthetic", "synthetic-v2"):
        extra = dict(
            target_fps=fps,
            duration_s_min=args.clip_duration_s_min,
            duration_s_max=args.clip_duration_s_max,
        )
    elif fps is not None:
        extra = dict(target_fps=fps)
    train_data, test_data = build_datasets(
        args.dataset, args.data_root, args.n_frames, args.image_size,
        max_count=args.max_count, kinetics_root=args.kinetics_root,
        **extra,
    )

    print(f"Dataset={args.dataset}  train={len(train_data)}  test={len(test_data)}  "
          f"buckets={args.n_count_buckets}  max_count={args.max_count}")
    if args.is_variable_length:
        if args.dataset in ("synthetic", "synthetic-v2"):
            print(f"  variable-length sampling: target_fps={fps} "
                  f"duration_s∈[{args.clip_duration_s_min}, {args.clip_duration_s_max}]")
        else:
            print(f"  variable-length sampling: target_fps={fps} "
                  f"(real-dataset clip durations come from the source videos)")
    else:
        print(f"  fixed-count TSN sampling: n_frames={args.n_frames}")
    print("Dataset stats:")
    _log_split_stats(train_data, "train", args.n_count_buckets)
    _log_split_stats(test_data,  "test ", args.n_count_buckets)
    print("(For chance-level reference, run "
          "`python -m tasks.repetition.baselines.modal_count` with the same "
          "--dataset / --data_root / --n_count_buckets.)")

    collate_fn = video_count_collate if args.is_variable_length else None
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers_train, drop_last=True, pin_memory=False,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=True,
        num_workers=args.num_workers_test, drop_last=False, pin_memory=False,
        collate_fn=collate_fn,
    )

    # --- Model ---
    device = pick_device(args)
    print(f"Using device: {device}")

    model = ContinuousThoughtMachineRepCount(
        n_frames=args.n_frames,
        iterations_per_frame=args.iterations_per_frame,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=args.deep_memory,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=args.do_normalisation,
        backbone_type=args.backbone_type,
        positional_embedding_type=args.positional_embedding_type,
        out_dims=args.out_dims,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    dummy_clip, _ = train_data[0]
    dummy_clip_b = dummy_clip.unsqueeze(0).to(device)
    if args.is_variable_length:
        dummy_mask = torch.ones(
            (1, dummy_clip_b.shape[1]), dtype=torch.bool, device=device,
        )
        model(dummy_clip_b, frame_mask=dummy_mask)
    else:
        model(dummy_clip_b)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Wrap in DataParallel when multiple GPU IDs are given (e.g. --device 0 1).
    # raw_model is used wherever we need the underlying module (state_dict, eval hooks).
    if len(args.device) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=args.device)
        print(f"DataParallel on GPUs: {args.device}")
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # --- Optimiser ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {n_trainable:,} / {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
        eps=1e-8 if not args.use_amp else 1e-6,
    )
    if args.use_scheduler:
        if args.scheduler_type == "cosine":
            scheduler = WarmupCosineAnnealingLR(
                optimizer, args.warmup_steps, args.training_iterations,
                warmup_start_lr=1e-20, eta_min=1e-7,
            )
        else:
            scheduler = WarmupMultiStepLR(
                optimizer, warmup_steps=args.warmup_steps,
                milestones=args.milestones, gamma=args.gamma,
            )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup(args.warmup_steps).step,
        )
    scaler = torch.amp.GradScaler(
        "cuda" if "cuda" in device else "cpu", enabled=args.use_amp,
    )

    # --- Metrics ---
    start_iter = 0
    iters = []
    train_losses, test_losses = [], []
    train_obo_per_tick, test_obo_per_tick = [], []
    train_mae_per_tick, test_mae_per_tick = [], []
    train_obo_mc, test_obo_mc = [], []
    train_mae_mc, test_mae_mc = [], []

    # --- Optional reload ---
    ckpt = load_checkpoint(args.log_dir, map_location=device) if args.reload else None
    if ckpt is not None:
        print(f"Reloading {args.log_dir}/checkpoint.pt")
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_iter = ckpt["iteration"]
        iters = ckpt["iters"]
        train_losses = ckpt["train_losses"]
        test_losses = ckpt["test_losses"]
        train_obo_per_tick = ckpt["train_obo_per_tick"]
        test_obo_per_tick = ckpt["test_obo_per_tick"]
        train_mae_per_tick = ckpt["train_mae_per_tick"]
        test_mae_per_tick = ckpt["test_mae_per_tick"]
        train_obo_mc = ckpt["train_obo_mc"]
        test_obo_mc = ckpt["test_obo_mc"]
        train_mae_mc = ckpt["train_mae_mc"]
        test_mae_mc = ckpt["test_mae_mc"]

    # --- Training loop ---
    data_iter = iter(train_loader)
    with tqdm(total=args.training_iterations, initial=start_iter,
              leave=False, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            clips, frame_mask, targets = _unpack_batch(batch, device)

            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16, enabled=args.use_amp,
            ):
                if frame_mask is not None:
                    predictions, certainties, _, tick_mask = model(
                        clips, frame_mask=frame_mask
                    )
                else:
                    out = model(clips)
                    # Backward compat: model now always returns 4 values, but
                    # if a future caller wraps it differently, accept 3 too.
                    if len(out) == 4:
                        predictions, certainties, _, tick_mask = out
                    else:
                        predictions, certainties, _ = out
                        tick_mask = None
                loss, where_certain = loss_fn(
                    predictions, certainties, targets,
                    use_most_certain=True, tick_mask=tick_mask,
                )

            # Quick train-batch metric for the progress bar.
            with torch.no_grad():
                B = predictions.size(0)
                batch_idx = torch.arange(B, device=device)
                mc_logits = predictions[batch_idx, :, where_certain]    # (B, buckets)
                _, argmax_counts = decode_fn(mc_logits)
                batch_obo = obo_accuracy(argmax_counts, targets)

            scaler.scale(loss).backward()
            if args.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            lr_now = optimizer.param_groups[-1]["lr"]
            pbar.set_description(
                f"[{args.dataset}] loss={loss.item():.3f} obo={batch_obo:.3f} "
                f"lr={lr_now:.2e} tick={where_certain.float().mean().item():.1f}"
            )
            pbar.update(1)

            # --- Eval & plots ---
            if (bi % args.track_every == 0 and bi > 0) or bi == args.training_iterations - 1:
                iters.append(bi)
                tr = _evaluate(model, train_loader, device, args, use_inference_mode=False)
                te = _evaluate(model, test_loader, device, args, use_inference_mode=True)
                train_losses.append(tr[0]); test_losses.append(te[0])
                train_obo_per_tick.append(tr[1]); test_obo_per_tick.append(te[1])
                train_mae_per_tick.append(tr[2]); test_mae_per_tick.append(te[2])
                train_obo_mc.append(tr[3]); test_obo_mc.append(te[3])
                train_mae_mc.append(tr[4]); test_mae_mc.append(te[4])
                print(
                    f"[eval @ iter {bi}]  "
                    f"train OBO={tr[3]:.3f} MAE={tr[4]:.3f}  |  "
                    f"test OBO={te[3]:.3f} MAE={te[4]:.3f}"
                )
                _plot_metrics(
                    args, iters,
                    train_losses, test_losses,
                    train_obo_per_tick, test_obo_per_tick,
                    train_mae_per_tick, test_mae_per_tick,
                    train_obo_mc, test_obo_mc,
                    train_mae_mc, test_mae_mc,
                )
                model.train()

            # --- Checkpoint ---
            if (bi % args.save_every == 0 and bi > start_iter) or bi == args.training_iterations - 1:
                save_checkpoint(args.log_dir, {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "iteration": bi,
                    "iters": iters,
                    "args": args,
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_obo_per_tick": train_obo_per_tick,
                    "test_obo_per_tick": test_obo_per_tick,
                    "train_mae_per_tick": train_mae_per_tick,
                    "test_mae_per_tick": test_mae_per_tick,
                    "train_obo_mc": train_obo_mc,
                    "test_obo_mc": test_obo_mc,
                    "train_mae_mc": train_mae_mc,
                    "test_mae_mc": test_mae_mc,
                }, bi)


def _unpack_batch(batch, device):
    """Return (clips, frame_mask_or_None, targets) on the given device."""
    if len(batch) == 3:
        clips, frame_mask, targets = batch
        clips = clips.to(device, non_blocking=True)
        frame_mask = frame_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return clips, frame_mask, targets
    clips, targets = batch
    clips = clips.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return clips, None, targets


def _evaluate(model, loader, device, args, use_inference_mode):
    """Evaluate on loader. Returns (loss, per_tick_obo, per_tick_mae, mc_obo, mc_mae).

    For variable-length batches (synthetic with FPS sampling) the per-tick
    arrays are empty — different batches have different T_max, so per-tick
    aggregation across the loader isn't well-defined and the most-certain
    metrics are what we track.
    """
    model.eval()
    loss_fn, decode_fn = _dispatch_head(args.head_type)

    all_targets, all_argmax_per_tick, all_expected_per_tick = [], [], []
    all_argmax_mc, all_expected_mc, all_losses = [], [], []

    context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with context:
        for i, batch in enumerate(loader):
            clips, frame_mask, targets = _unpack_batch(batch, device)
            if frame_mask is not None:
                predictions, certainties, _, tick_mask = model(clips, frame_mask=frame_mask)
            else:
                out = model(clips)
                if len(out) == 4:
                    predictions, certainties, _, tick_mask = out
                else:
                    predictions, certainties, _ = out
                    tick_mask = None

            loss, where_certain = loss_fn(
                predictions, certainties, targets,
                use_most_certain=True, tick_mask=tick_mask,
            )
            all_losses.append(loss.item())

            B = predictions.size(0)
            batch_idx = torch.arange(B, device=device)

            # Per-tick aggregation only makes sense when T is fixed across the
            # whole loader. Skip it for variable-length batches.
            if not args.is_variable_length:
                T = predictions.size(-1)
                preds_T = predictions.permute(0, 2, 1)              # (B, T, n_buckets)
                expected_T, argmax_T = decode_fn(
                    preds_T.reshape(B * T, args.n_count_buckets)
                )
                expected_T = expected_T.reshape(B, T)
                argmax_T = argmax_T.reshape(B, T)

            # Most-certain tick selection.
            mc_logits = predictions[batch_idx, :, where_certain]    # (B, n_buckets)
            expected_mc_batch, argmax_mc_batch = decode_fn(mc_logits)

            all_targets.append(targets.cpu())
            if not args.is_variable_length:
                all_argmax_per_tick.append(argmax_T.cpu())
                all_expected_per_tick.append(expected_T.cpu())
            all_argmax_mc.append(argmax_mc_batch.cpu())
            all_expected_mc.append(expected_mc_batch.cpu())

            if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                break

    all_targets = torch.cat(all_targets)                            # (N,)
    all_argmax_mc = torch.cat(all_argmax_mc)                        # (N,)
    all_expected_mc = torch.cat(all_expected_mc)                    # (N,)

    if args.is_variable_length:
        # Per-tick aggregation isn't well-defined across batches with
        # different T_max; report as empty.
        per_tick_obo = np.zeros(0, dtype=np.float32)
        per_tick_mae = np.zeros(0, dtype=np.float32)
    else:
        all_argmax_pt = torch.cat(all_argmax_per_tick)              # (N, T)
        all_expected_pt = torch.cat(all_expected_per_tick)          # (N, T)
        per_tick_obo = (
            (all_argmax_pt.float() - all_targets.float().unsqueeze(1)).abs() <= 1
        ).float().mean(dim=0).numpy()                               # (T,)
        per_tick_mae = (
            (all_expected_pt.float() - all_targets.float().unsqueeze(1)).abs()
        ).mean(dim=0).numpy()                                       # (T,)

    mc_obo = obo_accuracy(all_argmax_mc, all_targets)
    mc_mae = mae(all_expected_mc, all_targets)

    return float(np.mean(all_losses)), per_tick_obo, per_tick_mae, mc_obo, mc_mae


def _plot_metrics(
    args, iters,
    train_losses, test_losses,
    train_obo_per_tick, test_obo_per_tick,
    train_mae_per_tick, test_mae_per_tick,
    train_obo_mc, test_obo_mc,
    train_mae_mc, test_mae_mc,
):
    have_per_tick = bool(train_obo_per_tick) and len(train_obo_per_tick[-1]) > 0
    T = len(train_obo_per_tick[-1]) if have_per_tick else 1
    cm = plt.cm.viridis

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(iters, train_losses, "b-", label=f"train ({train_losses[-1]:.3f})")
    ax.plot(iters, test_losses, "r-", label=f"test ({test_losses[-1]:.3f})")
    ax.legend(loc="upper right"); ax.set_ylim(bottom=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/losses.png", dpi=140); plt.close(fig)

    # --- OBO accuracy ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    if have_per_tick:
        tr_obo = np.array(train_obo_per_tick)
        te_obo = np.array(test_obo_per_tick)
        for t in range(T):
            axes[0].plot(iters, tr_obo[:, t], color=cm(t / max(1, T - 1)), alpha=0.3)
            axes[1].plot(iters, te_obo[:, t], color=cm(t / max(1, T - 1)), alpha=0.3)
    axes[0].plot(iters, train_obo_mc, "k--", label="most-certain", lw=1.5)
    axes[1].plot(iters, test_obo_mc, "k--", label="most-certain", lw=1.5)
    axes[0].set_title("train OBO accuracy (per tick, black = most certain)")
    axes[1].set_title("test OBO accuracy (per tick, black = most certain)")
    for ax in axes:
        ax.set_ylim(0, 1); ax.legend(loc="lower right"); ax.set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/obo_accuracy.png", dpi=140); plt.close(fig)

    # --- MAE ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    if have_per_tick:
        tr_mae = np.array(train_mae_per_tick)
        te_mae = np.array(test_mae_per_tick)
        for t in range(T):
            axes[0].plot(iters, tr_mae[:, t], color=cm(t / max(1, T - 1)), alpha=0.3)
            axes[1].plot(iters, te_mae[:, t], color=cm(t / max(1, T - 1)), alpha=0.3)
    axes[0].plot(iters, train_mae_mc, "k--", label="most-certain", lw=1.5)
    axes[1].plot(iters, test_mae_mc, "k--", label="most-certain", lw=1.5)
    axes[0].set_title("train MAE (per tick, black = most certain)")
    axes[1].set_title("test MAE (per tick, black = most certain)")
    for ax in axes:
        ax.set_ylim(bottom=0); ax.legend(loc="upper right"); ax.set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/mae.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
