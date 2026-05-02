"""Predictive-coding pre-training for the video CTM on Kinetics.

Loss: at the last internal tick of each frame f, the predictor head reads the
synch_out and predicts the mean-pooled (frozen) backbone features of frame
f+1. Cosine similarity, stop-gradient on the target.

Components trained (the "CTM core"):
    kv_proj, q_proj, attention, synapses, trace_processor, sync params,
    start states, predictor_head.

Components frozen:
    initial_rgb (Identity), backbone (ImageNet-pretrained ResNet).
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

from tasks.pretrain.dataset import build_pretrain_dataset
from tasks.pretrain.model import CTMVideoPredictiveCoding
from utils.housekeeping import set_seed
from utils.run import init_run, load_checkpoint, save_checkpoint
from utils.schedulers import WarmupCosineAnnealingLR, warmup


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "kinetics", "video_folder"])
    p.add_argument("--data_root", type=str, default="data/kinetics")
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--image_size", type=int, default=112)
    p.add_argument("--max_videos", type=int, default=None)

    # Model — must match downstream finetuning to be useful.
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_input", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--iterations_per_frame", type=int, default=1)
    p.add_argument("--synapse_depth", type=int, default=4)
    p.add_argument("--n_synch_out", type=int, default=64)
    p.add_argument("--n_synch_action", type=int, default=64)
    p.add_argument("--neuron_select_type", type=str, default="random-pairing")
    p.add_argument("--memory_length", type=int, default=16)
    p.add_argument("--memory_hidden_dims", type=int, default=16)
    p.add_argument("--deep_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do_normalisation", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--backbone_type", type=str, default="resnet18-2")
    p.add_argument("--positional_embedding_type", type=str, default="none")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dropout_nlm", type=float, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--training_iterations", type=int, default=30001)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_clipping", type=float, default=1.0)

    # Housekeeping
    p.add_argument("--log_dir", type=str, default="logs/pretrain/kinetics")
    p.add_argument("--run_name", type=str, default=None,
                   help="Subdirectory of --log_dir for this run. "
                        "Defaults to ${SLURM_JOB_ID} on JZ, else a timestamp.")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--track_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=int, nargs="+", default=[-1])
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reload", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def pick_device(args):
    if args.device[0] != -1:
        return f"cuda:{args.device[0]}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(args, device):
    # out_dims is irrelevant during pretraining (we don't use the output projector),
    # but the parent class requires it. Use a small placeholder.
    model = CTMVideoPredictiveCoding(
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
        out_dims=10,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
    ).to(device)

    # Initialise lazy modules with a dummy forward. The standard forward also
    # initialises output_projector, which forward_pretrain alone wouldn't touch.
    dummy = torch.randn(1, args.n_frames, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        _ = model(dummy)

    # Freeze the output_projector during pretraining — it's unused and the random
    # head is irrelevant. Saves a few params and avoids confusing optimiser state.
    for p in model.output_projector.parameters():
        p.requires_grad_(False)
    return model


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    init_run(args)

    train_data = build_pretrain_dataset(
        args.dataset, args.data_root, args.n_frames, args.image_size,
        max_videos=args.max_videos,
    )
    print(f"Dataset={args.dataset}  size={len(train_data)}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=False,
    )

    device = pick_device(args)
    print(f"Using device: {device}")

    model = build_model(args, device)
    raw_model = model
    if len(args.device) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=args.device)
        raw_model = model.module
        print(f"DataParallel on GPUs: {args.device}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,} (encoder frozen)")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        eps=1e-8 if not args.use_amp else 1e-6,
    )
    scheduler = WarmupCosineAnnealingLR(
        optimizer, args.warmup_steps, args.training_iterations,
        warmup_start_lr=1e-20, eta_min=1e-7,
    )
    scaler = torch.amp.GradScaler(
        "cuda" if "cuda" in device else "cpu", enabled=args.use_amp
    )

    start_iter = 0
    iters, losses, mean_cosines = [], [], []

    ckpt = load_checkpoint(args.log_dir, map_location=device) if args.reload else None
    if ckpt is not None:
        print(f"Reloading {args.log_dir}/checkpoint.pt")
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_iter = ckpt["iteration"]
        iters = ckpt["iters"]; losses = ckpt["losses"]; mean_cosines = ckpt["mean_cosines"]

    data_iter = iter(train_loader)
    with tqdm(total=args.training_iterations, initial=start_iter,
              leave=False, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            try:
                clips, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                clips, _ = next(data_iter)
            clips = clips.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16, enabled=args.use_amp,
            ):
                if isinstance(model, torch.nn.DataParallel):
                    # DataParallel only dispatches `forward`; call the underlying loss method.
                    loss, cos = raw_model.predictive_coding_loss(clips)
                else:
                    loss, cos = model.predictive_coding_loss(clips)

            scaler.scale(loss).backward()
            if args.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clipping)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            lr_now = optimizer.param_groups[-1]["lr"]
            mean_cos = cos.mean().item()
            pbar.set_description(
                f"loss={loss.item():.4f} cos={mean_cos:.4f} lr={lr_now:.2e}"
            )
            pbar.update(1)

            if (bi % args.track_every == 0 and bi > 0) or bi == args.training_iterations - 1:
                iters.append(bi)
                losses.append(loss.item())
                mean_cosines.append(mean_cos)
                _plot_curves(args, iters, losses, mean_cosines)

            if (bi % args.save_every == 0 and bi > start_iter) or bi == args.training_iterations - 1:
                save_checkpoint(args.log_dir, {
                    "model_state_dict": raw_model.state_dict(),
                    "core_state_dict": raw_model.core_state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "iteration": bi,
                    "iters": iters,
                    "losses": losses,
                    "mean_cosines": mean_cosines,
                    "args": args,
                }, bi)


def _plot_curves(args, iters, losses, mean_cosines):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(iters, losses, "b-")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss (1 - cos)")
    axes[0].set_title("predictive-coding loss"); axes[0].set_ylim(bottom=0)
    axes[1].plot(iters, mean_cosines, "g-")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("mean cosine sim")
    axes[1].set_title("predictor↔target alignment")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/curves.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
