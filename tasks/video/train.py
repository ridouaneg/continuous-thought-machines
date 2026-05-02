"""Training script for CTM action recognition on UCF-101 / HMDB-51.

Couples internal ticks to video frames. Loss is the existing
``image_classification_loss`` (min-CE + max-certainty, broadcast across
ticks) — for clip-level labels that is exactly right: the tick axis is
the frame axis, and the single clip label is broadcast.

Training uses full BPTT through every tick. For 32 frames with
``iterations_per_frame=1`` that is 32 ticks, comparable in cost to
mazes (75) or ImageNet (75).
"""

from __future__ import annotations

import argparse
import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm.auto import tqdm

sns.set_style("darkgrid")
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

from tasks.video.dataset import build_datasets
from tasks.video.model import ContinuousThoughtMachineVideo
from utils.housekeeping import set_seed
from utils.losses import image_classification_loss
from utils.run import init_run, load_checkpoint, save_checkpoint
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "ucf101", "hmdb51", "kinetics"])
    parser.add_argument("--data_root", type=str, default="data/video")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--n_frames", type=int, default=16,
                        help="Frames sampled per clip (== tick count if iterations_per_frame=1).")
    parser.add_argument("--image_size", type=int, default=112)

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
    parser.add_argument("--memory_length", type=int, default=16)
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
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clipping", type=float, default=-1)

    # Housekeeping
    parser.add_argument("--log_dir", type=str, default="logs/video/scratch")
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
    set_seed(args.seed, deterministic=False)
    init_run(args)

    # --- Data ---
    train_data, test_data, class_labels = build_datasets(
        args.dataset, args.data_root, args.n_frames, args.image_size, fold=args.fold
    )
    args.out_dims = len(class_labels)
    print(f"Dataset={args.dataset}  train={len(train_data)}  test={len(test_data)}  "
          f"classes={args.out_dims}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers_train, drop_last=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=True,
        num_workers=args.num_workers_test, drop_last=False, pin_memory=False
    )

    # --- Model ---
    device = pick_device(args)
    print(f"Using device: {device}")

    model = ContinuousThoughtMachineVideo(
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

    # Initialise lazy modules with a dummy forward.
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
            optimizer, lr_lambda=warmup(args.warmup_steps).step
        )
    scaler = torch.amp.GradScaler(
        "cuda" if "cuda" in device else "cpu", enabled=args.use_amp
    )

    # --- Metrics ---
    start_iter = 0
    iters, train_losses, test_losses = [], [], []
    train_acc_per_tick, test_acc_per_tick = [], []
    train_acc_mc, test_acc_mc = [], []

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
        train_acc_per_tick = ckpt["train_acc_per_tick"]
        test_acc_per_tick = ckpt["test_acc_per_tick"]
        train_acc_mc = ckpt["train_acc_mc"]
        test_acc_mc = ckpt["test_acc_mc"]

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
                loss, where_certain = image_classification_loss(
                    predictions, certainties, targets, use_most_certain=True,
                    label_smoothing=args.label_smoothing,
                )

            batch_idx = torch.arange(predictions.size(0), device=predictions.device)
            acc = (predictions.argmax(1)[batch_idx, where_certain] == targets).float().mean().item()

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
                f"[{args.dataset}] loss={loss.item():.3f} acc={acc:.3f} "
                f"lr={lr_now:.2e} tick_certain={where_certain.float().mean().item():.1f}"
            )
            pbar.update(1)

            # --- Eval & plots ---
            if (bi % args.track_every == 0 and bi > 0) or bi == args.training_iterations - 1:
                iters.append(bi)
                tr_loss, tr_per_tick, tr_mc = _evaluate(
                    model, train_loader, device, args, use_inference_mode=False
                )
                te_loss, te_per_tick, te_mc = _evaluate(
                    model, test_loader, device, args, use_inference_mode=True
                )
                train_losses.append(tr_loss); test_losses.append(te_loss)
                train_acc_per_tick.append(tr_per_tick); test_acc_per_tick.append(te_per_tick)
                train_acc_mc.append(tr_mc); test_acc_mc.append(te_mc)
                _plot_metrics(args, iters, train_losses, test_losses,
                              train_acc_per_tick, test_acc_per_tick,
                              train_acc_mc, test_acc_mc)
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
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_acc_per_tick": train_acc_per_tick,
                    "test_acc_per_tick": test_acc_per_tick,
                    "train_acc_mc": train_acc_mc,
                    "test_acc_mc": test_acc_mc,
                    "args": args,
                    "class_labels": class_labels,
                }, bi)


def _evaluate(model, loader, device, args, use_inference_mode):
    model.eval()
    all_targets, all_preds, all_preds_mc, all_losses = [], [], [], []
    context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with context:
        for i, (clips, targets) in enumerate(loader):
            clips = clips.to(device); targets = targets.to(device)
            predictions, certainties, _ = model(clips)
            loss, where_certain = image_classification_loss(
                predictions, certainties, targets, use_most_certain=True,
                label_smoothing=args.label_smoothing,
            )
            all_losses.append(loss.item())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(predictions.argmax(1).cpu().numpy())  # (B, T)
            batch_idx = torch.arange(predictions.size(0), device=device)
            all_preds_mc.append(
                predictions.argmax(1)[batch_idx, where_certain].cpu().numpy()
            )
            if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                break
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)  # (N, T)
    all_preds_mc = np.concatenate(all_preds_mc)
    per_tick = (all_preds == all_targets[:, None]).mean(axis=0)  # (T,)
    mc = (all_preds_mc == all_targets).mean()
    return float(np.mean(all_losses)), per_tick, float(mc)


def _plot_metrics(args, iters, train_losses, test_losses,
                  train_acc_per_tick, test_acc_per_tick, train_acc_mc, test_acc_mc):
    # Loss curve
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(iters, train_losses, "b-", label=f"train ({train_losses[-1]:.3f})")
    ax.plot(iters, test_losses, "r-", label=f"test ({test_losses[-1]:.3f})")
    ax.legend(loc="upper right"); ax.set_ylim(bottom=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/losses.png", dpi=140); plt.close(fig)

    # Per-tick accuracy heatmap
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    cm = sns.color_palette("viridis", as_cmap=True)
    tr = np.array(train_acc_per_tick)
    te = np.array(test_acc_per_tick)
    T = tr.shape[1]
    for t in range(T):
        axes[0].plot(iters, tr[:, t], color=cm(t / max(1, T - 1)), alpha=0.35)
        axes[1].plot(iters, te[:, t], color=cm(t / max(1, T - 1)), alpha=0.35)
    axes[0].plot(iters, train_acc_mc, "k--", label="most-certain", linewidth=1.5)
    axes[1].plot(iters, test_acc_mc, "k--", label="most-certain", linewidth=1.5)
    axes[0].set_title("train accuracy (per tick = per frame, black = most certain)")
    axes[1].set_title("test accuracy (per tick = per frame, black = most certain)")
    for ax in axes: ax.legend(loc="lower right"); ax.set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/accuracies.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
