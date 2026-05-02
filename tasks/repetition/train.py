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

from tasks.repetition.dataset import build_datasets
from tasks.repetition.losses import count_loss
from tasks.repetition.model import ContinuousThoughtMachineRepCount
from tasks.repetition.utils import count_from_buckets, mae, obo_accuracy
from utils.housekeeping import set_seed
from utils.run import init_run, load_checkpoint, save_checkpoint
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "countix", "repcount", "ucfrep"])
    parser.add_argument("--data_root", type=str, default="data/repetition")
    parser.add_argument("--kinetics_root", type=str, default=None,
                        help="For --dataset countix: root of an official Kinetics-400 "
                             "mirror, used to look up videos by youtube_id when the "
                             "Countix CSVs are not co-located with the videos.")
    parser.add_argument("--n_frames", type=int, default=64,
                        help="Frames sampled per clip. Nyquist limit = n_frames / 2 reps.")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--max_count", type=int, default=16,
                        help="Max oscillation count for the synthetic dataset.")
    parser.add_argument("--n_count_buckets", type=int, default=32,
                        help="Number of discrete count classes. Labels >= n_count_buckets "
                             "are clamped to n_count_buckets-1.")

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
    parser.add_argument("--weight_decay", type=float, default=0.0)
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


def main():
    args = parse_args()
    args.out_dims = args.n_count_buckets
    set_seed(args.seed, deterministic=False)
    init_run(args)

    # --- Data ---
    train_data, test_data = build_datasets(
        args.dataset, args.data_root, args.n_frames, args.image_size,
        max_count=args.max_count, kinetics_root=args.kinetics_root,
    )
    print(f"Dataset={args.dataset}  train={len(train_data)}  test={len(test_data)}  "
          f"buckets={args.n_count_buckets}  max_count={args.max_count}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers_train, drop_last=True, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=True,
        num_workers=args.num_workers_test, drop_last=False, pin_memory=False,
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
    model(dummy_clip.unsqueeze(0).to(device))
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
                clips, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                clips, targets = next(data_iter)

            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16, enabled=args.use_amp,
            ):
                predictions, certainties, _ = model(clips)
                loss, where_certain = count_loss(
                    predictions, certainties, targets, use_most_certain=True
                )

            # Quick train-batch metric for the progress bar.
            with torch.no_grad():
                B = predictions.size(0)
                batch_idx = torch.arange(B, device=device)
                mc_logits = predictions[batch_idx, :, where_certain]    # (B, buckets)
                _, argmax_counts = count_from_buckets(mc_logits)
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


def _evaluate(model, loader, device, args, use_inference_mode):
    """Evaluate on loader. Returns (loss, per_tick_obo, per_tick_mae, mc_obo, mc_mae)."""
    model.eval()
    T = args.n_frames * args.iterations_per_frame

    all_targets, all_argmax_per_tick, all_expected_per_tick = [], [], []
    all_argmax_mc, all_expected_mc, all_losses = [], [], []

    context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with context:
        for i, (clips, targets) in enumerate(loader):
            clips = clips.to(device)
            targets = targets.to(device)
            predictions, certainties, _ = model(clips)

            loss, where_certain = count_loss(
                predictions, certainties, targets, use_most_certain=True
            )
            all_losses.append(loss.item())

            B = predictions.size(0)
            batch_idx = torch.arange(B, device=device)

            # Per-tick predictions: (B, T) argmax and expected counts.
            preds_T = predictions.permute(0, 2, 1)              # (B, T, n_buckets)
            expected_T, argmax_T = count_from_buckets(
                preds_T.reshape(B * T, args.n_count_buckets)
            )
            expected_T = expected_T.reshape(B, T)
            argmax_T = argmax_T.reshape(B, T)

            # Most-certain tick selection.
            mc_logits = predictions[batch_idx, :, where_certain]    # (B, n_buckets)
            expected_mc_batch, argmax_mc_batch = count_from_buckets(mc_logits)

            all_targets.append(targets.cpu())
            all_argmax_per_tick.append(argmax_T.cpu())
            all_expected_per_tick.append(expected_T.cpu())
            all_argmax_mc.append(argmax_mc_batch.cpu())
            all_expected_mc.append(expected_mc_batch.cpu())

            if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                break

    all_targets = torch.cat(all_targets)                            # (N,)
    all_argmax_pt = torch.cat(all_argmax_per_tick)                  # (N, T)
    all_expected_pt = torch.cat(all_expected_per_tick)              # (N, T)
    all_argmax_mc = torch.cat(all_argmax_mc)                        # (N,)
    all_expected_mc = torch.cat(all_expected_mc)                    # (N,)

    # Per-tick OBO and MAE: average over samples at each tick.
    per_tick_obo = (
        (all_argmax_pt.float() - all_targets.float().unsqueeze(1)).abs() <= 1
    ).float().mean(dim=0).numpy()                                   # (T,)
    per_tick_mae = (
        (all_expected_pt.float() - all_targets.float().unsqueeze(1)).abs()
    ).mean(dim=0).numpy()                                           # (T,)

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
    T = len(train_obo_per_tick[-1]) if train_obo_per_tick else 1
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
