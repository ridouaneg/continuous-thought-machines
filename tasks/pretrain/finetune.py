"""Action-recognition fine-tuning on top of a (optionally) pre-trained CTM.

The encoder is the ImageNet-pretrained ResNet, frozen throughout. The CTM
core is either:

  - randomly initialised (``--init_from`` omitted) — this is the no-pretrain
    BASELINE, and
  - loaded from a predictive-coding checkpoint produced by ``pretrain.py``
    (``--init_from <path>``) — the pre-trained variant.

In both cases the CTM core + output projector are trained with
``image_classification_loss`` exactly as in ``tasks/video/train.py``.
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

from tasks.pretrain.dataset import build_finetune_datasets
from tasks.pretrain.model import CTMVideoPredictiveCoding
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import image_classification_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "ucf101", "hmdb51", "kinetics"])
    p.add_argument("--data_root", type=str, default="data/video")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--image_size", type=int, default=112)

    # Model — must match pretraining args.
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

    # Pre-training source — omit to run the no-pretrain baseline.
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to a pretrain checkpoint (.pt). Omit for the baseline.")

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--batch_size_test", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--training_iterations", type=int, default=30001)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--use_scheduler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scheduler_type", type=str, default="cosine",
                   choices=["cosine", "multistep"])
    p.add_argument("--milestones", type=int, nargs="+", default=[15000, 25000])
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_clipping", type=float, default=-1)

    # Housekeeping
    p.add_argument("--log_dir", type=str, default="logs/pretrain/finetune")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--track_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers_train", type=int, default=2)
    p.add_argument("--num_workers_test", type=int, default=1)
    p.add_argument("--n_test_batches", type=int, default=20)
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


def build_model(args, n_classes, device):
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
        out_dims=n_classes,
        prediction_reshaper=[-1],
        dropout=args.dropout,
        dropout_nlm=args.dropout_nlm,
        neuron_select_type=args.neuron_select_type,
    ).to(device)

    # Initialise lazy modules with a dummy forward.
    dummy = torch.randn(1, args.n_frames, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        _ = model(dummy)

    # The predictor head is pretraining-only; freeze it so it doesn't waste optim state.
    for p in model.predictor_head.parameters():
        p.requires_grad_(False)
    return model


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=False)
    os.makedirs(args.log_dir, exist_ok=True)

    train_data, test_data, class_labels = build_finetune_datasets(
        args.dataset, args.data_root, args.n_frames, args.image_size, fold=args.fold,
    )
    args.out_dims = len(class_labels)
    print(f"Dataset={args.dataset}  train={len(train_data)}  test={len(test_data)}  "
          f"classes={args.out_dims}")

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers_train, drop_last=True, pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=True,
        num_workers=args.num_workers_test, drop_last=False, pin_memory=False,
    )

    zip_python_code(f"{args.log_dir}/repo_state.zip")
    with open(f"{args.log_dir}/args.txt", "w") as f:
        print(args, file=f)

    device = pick_device(args)
    print(f"Using device: {device}")

    model = build_model(args, args.out_dims, device)

    # --- Optional load of the pre-trained CTM core ---
    init_mode = "baseline (random init)"
    if args.init_from:
        if not os.path.isfile(args.init_from):
            raise FileNotFoundError(f"--init_from path does not exist: {args.init_from}")
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        # Prefer the dedicated `core_state_dict` if present; fall back to model_state_dict.
        sd = ckpt.get("core_state_dict") or ckpt["model_state_dict"]
        result = model.load_core_state_dict(
            sd, drop_predictor_head=True, drop_output_projector=True
        )
        # Split missing keys into expected categories for a clearer report.
        backbone_missing = [k for k in result.missing_keys
                            if k.startswith("backbone.") or k.startswith("initial_rgb.")]
        head_missing = [k for k in result.missing_keys if k.startswith("output_projector.")]
        predictor_missing = [k for k in result.missing_keys if k.startswith("predictor_head.")]
        accounted = set(backbone_missing) | set(head_missing) | set(predictor_missing)
        other_missing = [k for k in result.missing_keys if k not in accounted]
        print(f"Loaded pre-trained core from {args.init_from}")
        print(f"  backbone kept from ImageNet:   {len(backbone_missing)} keys")
        print(f"  output_projector re-init:      {len(head_missing)} keys (per-task head)")
        print(f"  predictor_head kept random:    {len(predictor_missing)} keys (frozen, unused)")
        if other_missing:
            print(f"  WARNING — other missing:       {len(other_missing)} keys: {other_missing[:5]}")
        print(f"  unexpected (ignored):          {len(result.unexpected_keys)} keys")
        init_mode = f"pretrained from {args.init_from}"
    print(f"Init mode: {init_mode}")

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

    start_iter = 0
    iters = []
    train_losses, test_losses = [], []
    train_acc_per_tick, test_acc_per_tick = [], []
    train_acc_mc, test_acc_mc = [], []

    ckpt_path = f"{args.log_dir}/checkpoint.pt"
    if args.reload and os.path.isfile(ckpt_path):
        print(f"Reloading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_iter = ckpt["iteration"]
        iters = ckpt["iters"]
        train_losses = ckpt["train_losses"]; test_losses = ckpt["test_losses"]
        train_acc_per_tick = ckpt["train_acc_per_tick"]; test_acc_per_tick = ckpt["test_acc_per_tick"]
        train_acc_mc = ckpt["train_acc_mc"]; test_acc_mc = ckpt["test_acc_mc"]

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
                    predictions, certainties, targets, use_most_certain=True
                )

            batch_idx = torch.arange(predictions.size(0), device=predictions.device)
            acc = (predictions.argmax(1)[batch_idx, where_certain] == targets).float().mean().item()

            scaler.scale(loss).backward()
            if args.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clipping)
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

            if (bi % args.track_every == 0 and bi > 0) or bi == args.training_iterations - 1:
                iters.append(bi)
                tr_loss, tr_per_tick, tr_mc = _evaluate(model, train_loader, device, args, False)
                te_loss, te_per_tick, te_mc = _evaluate(model, test_loader, device, args, True)
                train_losses.append(tr_loss); test_losses.append(te_loss)
                train_acc_per_tick.append(tr_per_tick); test_acc_per_tick.append(te_per_tick)
                train_acc_mc.append(tr_mc); test_acc_mc.append(te_mc)
                _plot_metrics(args, iters, train_losses, test_losses,
                              train_acc_per_tick, test_acc_per_tick,
                              train_acc_mc, test_acc_mc)
                model.train()

            if (bi % args.save_every == 0 and bi > start_iter) or bi == args.training_iterations - 1:
                torch.save({
                    "model_state_dict": raw_model.state_dict(),
                    "core_state_dict": raw_model.core_state_dict(),
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
                    "init_mode": init_mode,
                }, ckpt_path)


def _evaluate(model, loader, device, args, use_inference_mode):
    model.eval()
    all_targets, all_preds, all_preds_mc, all_losses = [], [], [], []
    context = torch.inference_mode() if use_inference_mode else torch.no_grad()
    with context:
        for i, (clips, targets) in enumerate(loader):
            clips = clips.to(device); targets = targets.to(device)
            predictions, certainties, _ = model(clips)
            loss, where_certain = image_classification_loss(
                predictions, certainties, targets, use_most_certain=True
            )
            all_losses.append(loss.item())
            all_targets.append(targets.cpu().numpy())
            all_preds.append(predictions.argmax(1).cpu().numpy())
            batch_idx = torch.arange(predictions.size(0), device=device)
            all_preds_mc.append(
                predictions.argmax(1)[batch_idx, where_certain].cpu().numpy()
            )
            if args.n_test_batches != -1 and i >= args.n_test_batches - 1:
                break
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_preds_mc = np.concatenate(all_preds_mc)
    per_tick = (all_preds == all_targets[:, None]).mean(axis=0)
    mc = (all_preds_mc == all_targets).mean()
    return float(np.mean(all_losses)), per_tick, float(mc)


def _plot_metrics(args, iters, train_losses, test_losses,
                  train_acc_per_tick, test_acc_per_tick, train_acc_mc, test_acc_mc):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(iters, train_losses, "b-", label=f"train ({train_losses[-1]:.3f})")
    ax.plot(iters, test_losses, "r-", label=f"test ({test_losses[-1]:.3f})")
    ax.legend(loc="upper right"); ax.set_ylim(bottom=0)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    fig.tight_layout(); fig.savefig(f"{args.log_dir}/losses.png", dpi=140); plt.close(fig)

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
