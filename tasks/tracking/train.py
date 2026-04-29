"""Train the CTM on multi-object tracking.

Supports three dataset backends:

  synthetic  : fast, fully synthetic Gaussian blobs (no download required)
  mot17      : MOT17 pedestrian tracking (~5 GB download, motchallenge.net)
  dancetrack : DanceTrack dancer tracking (github.com/DanceTrack/DanceTrack)

Key design
----------
- Positions are discretised into n_bins bins per axis; tracking becomes a
  classification task compatible with the CTM certainty mechanism.
- Targets use -1 as a sentinel for absent / occluded objects; the loss
  automatically masks those positions out.
- The frame encoder is swappable: 'tiny' (synthetic), 'medium' or 'resnet18'
  (real data with pretrained features).

Example usage
-------------
# Synthetic (quick test)
python -m tasks.tracking.train --dataset synthetic

# MOT17 (after downloading to /data/MOT17)
python -m tasks.tracking.train \\
    --dataset mot17 --data_root /data/MOT17 \\
    --encoder_type resnet18 --in_channels 3 \\
    --n_objects 8 --img_size 128 --n_bins 16 \\
    --d_model 512 --d_input 256 --iterations 30 \\
    --device 0 --log_dir logs/tracking/mot17

# DanceTrack
python -m tasks.tracking.train \\
    --dataset dancetrack --data_root /data/DanceTrack \\
    --encoder_type resnet18 --in_channels 3 \\
    --n_objects 8 --img_size 128 \\
    --device 0 --log_dir logs/tracking/dancetrack
"""

import argparse
import os
import random

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='warn')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from tqdm.auto import tqdm

from models.utils import get_latest_checkpoint
from tasks.tracking.losses import tracking_loss, position_mae
from tasks.tracking.utils import prepare_model, build_datasets
from utils.housekeeping import set_seed, zip_python_code
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on multi-object tracking")

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'mot17', 'dancetrack'],
                        help="Dataset backend.")
    parser.add_argument('--data_root', type=str, default='',
                        help="Root directory for real datasets (ignored for synthetic).")
    parser.add_argument('--n_objects', type=int, default=2,
                        help="Number of objects to track per window.")
    parser.add_argument('--n_frames', type=int, default=8,
                        help="Number of frames per sequence window.")
    parser.add_argument('--img_size', type=int, default=32,
                        help="Frames are resized to (img_size × img_size).")
    parser.add_argument('--n_bins', type=int, default=16,
                        help="Position discretisation bins per axis.")
    parser.add_argument('--stride', type=int, default=4,
                        help="Frame stride between consecutive windows (real datasets).")
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Fraction of frames reserved for validation (MOT17).")
    # Synthetic-only
    parser.add_argument('--blob_sigma_px', type=float, default=1.5)
    parser.add_argument('--velocity_scale', type=float, default=0.07)
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--n_test',  type=int, default=5000)

    # ── Frame encoder ─────────────────────────────────────────────────────
    parser.add_argument('--encoder_type', type=str, default='resnet18',
                        choices=['tiny', 'medium', 'resnet18'],
                        help="Frame encoder architecture. Default 'resnet18' uses "
                             "ImageNet-pretrained weights, fully frozen.")
    parser.add_argument('--in_channels', type=int, default=3,
                        help="Input image channels. Default 3 matches the pretrained "
                             "ResNet; the synthetic dataset replicates its grayscale "
                             "blob across RGB and ImageNet-normalises when this is 3. "
                             "Pass 1 with --encoder_type tiny/medium for the legacy "
                             "grayscale synthetic flow.")
    parser.add_argument('--d_feat', type=int, default=64,
                        help="Frame encoder output dimension.")

    # ── CTM architecture ──────────────────────────────────────────────────
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_input', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=32)
    parser.add_argument('--n_synch_action', type=int, default=32)
    parser.add_argument('--synapse_depth', type=int, default=1)
    parser.add_argument('--memory_length', type=int, default=10)
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_hidden_dims', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing',
                        choices=['first-last', 'random', 'random-pairing'])
    parser.add_argument('--n_random_pairing_self', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=20)

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=30001)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=int, nargs='+', default=[10000, 20000])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--use_most_certain', action=argparse.BooleanOptionalAction, default=True)

    # ── Housekeeping ──────────────────────────────────────────────────────
    parser.add_argument('--log_dir', type=str, default='logs/tracking')
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--track_every', type=int, default=1000)
    parser.add_argument('--n_test_batches', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', type=int, nargs='+', default=[-1])
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def reshape_preds(predictions, n_objects, n_frames, n_bins):
    """(B, N*T*2*K, iters) → (B, N*T*2, K, iters)"""
    B, _, iters = predictions.shape
    return predictions.reshape(B, n_objects * n_frames * 2, n_bins, iters)


def flatten_targets(targets, n_objects, n_frames):
    """(B, T, N, 2) → (B, N*T*2)  matching the model output ordering."""
    B = targets.shape[0]
    # (B, T, N, 2) → (B, N, T, 2) → (B, N*T*2)
    return targets.permute(0, 2, 1, 3).reshape(B, n_objects * n_frames * 2)


def eval_loop(model, loader, args, device, n_test_batches):
    """Run evaluation; returns (mean_loss, mean_mae, per_tick_acc)."""
    all_losses, all_maes = [], []
    per_tick_correct = None
    total = 0

    with torch.inference_mode():
        for bi, (frames, targets) in enumerate(loader):
            frames  = frames.to(device)
            targets = targets.to(device)

            preds, certs, _ = model(frames)
            preds = reshape_preds(preds, args.n_objects, args.n_frames, args.n_bins)

            B          = frames.shape[0]
            tgts_flat  = flatten_targets(targets, args.n_objects, args.n_frames)

            loss, where_cert = tracking_loss(preds, certs, tgts_flat, args.use_most_certain)
            all_losses.append(loss.item())

            batch_idx     = torch.arange(B, device=device)
            preds_at_tick = preds[batch_idx, :, :, where_cert]
            mae = position_mae(preds_at_tick, tgts_flat, args.n_bins)
            all_maes.append(mae)

            # Per-tick bin accuracy (only over valid positions)
            valid     = (tgts_flat >= 0)                           # (B, N*T*2)
            correct   = (preds.argmax(2) == tgts_flat.clamp(0).unsqueeze(-1))
            correct   = (correct * valid.unsqueeze(-1)).float().sum(1)
            n_valid   = valid.float().sum(1, keepdim=True).clamp(1)
            tick_acc  = (correct / n_valid).sum(0).cpu()           # (iters,)

            if per_tick_correct is None:
                per_tick_correct = tick_acc
            else:
                per_tick_correct += tick_acc
            total += B

            if bi + 1 >= n_test_batches:
                break

    per_tick_acc = (per_tick_correct / total).numpy()
    return np.mean(all_losses), np.mean(all_maes), per_tick_acc


def plot_metrics(iters, train_losses, test_losses, train_maes, test_maes,
                 train_tick_accs, test_tick_accs, log_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(iters, train_losses, label='train')
    axes[0].plot(iters, test_losses,  label='val')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(iters, train_maes, label='train MAE')
    axes[1].plot(iters, test_maes,  label='val MAE')
    axes[1].set_ylabel('Position MAE (normalised)')
    axes[1].legend()

    cm = plt.get_cmap('viridis')
    n_curves = len(train_tick_accs[0]) if train_tick_accs else 1
    for ti, (tr_acc, te_acc) in enumerate(zip(np.array(train_tick_accs).T,
                                               np.array(test_tick_accs).T)):
        c = cm(ti / max(n_curves - 1, 1))
        axes[2].plot(iters, tr_acc, color=c, alpha=0.4)
        axes[2].plot(iters, te_acc, color=c, alpha=0.8, linestyle='--')
    axes[2].set_ylabel('Per-tick bin accuracy')
    axes[2].set_xlabel('Iteration')

    plt.tight_layout()
    fig.savefig(f'{log_dir}/metrics.png', dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}  |  Dataset: {args.dataset}')

    # ── Datasets ──────────────────────────────────────────────────────────
    train_data, test_data = build_datasets(
        dataset=args.dataset,
        data_root=args.data_root,
        n_frames=args.n_frames,
        img_size=args.img_size,
        n_objects=args.n_objects,
        n_bins=args.n_bins,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        blob_sigma_px=args.blob_sigma_px,
        velocity_scale=args.velocity_scale,
        in_channels=args.in_channels,
    )
    print(f'Train: {len(train_data)} windows  |  Val: {len(test_data)} windows')

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = prepare_model(args, device)
    model.train()

    dummy_frames, _ = train_data[0]
    with torch.no_grad():
        model(dummy_frames.unsqueeze(0).to(device))

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {total_params:,}  (trainable: {trainable:,})')

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(
                optimizer, args.warmup_steps, args.training_iterations,
                warmup_start_lr=1e-8, eta_min=1e-6,
            )
        elif args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(
                optimizer, warmup_steps=args.warmup_steps,
                milestones=args.milestones, gamma=args.gamma,
            )

    scaler = torch.amp.GradScaler(
        "cuda" if "cuda" in device else "cpu", enabled=args.use_amp
    )

    # ── Metric storage ────────────────────────────────────────────────────
    start_iter = 0
    train_losses, test_losses = [], []
    train_maes,   test_maes   = [], []
    train_tick_accs, test_tick_accs = [], []
    logged_iters = []

    # ── Reload ────────────────────────────────────────────────────────────
    if args.reload and (ckpt_path := get_latest_checkpoint(args.log_dir)):
        print(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        if not args.reload_model_only:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_iter       = ckpt['iteration']
            train_losses     = ckpt.get('train_losses', [])
            test_losses      = ckpt.get('test_losses', [])
            train_maes       = ckpt.get('train_maes', [])
            test_maes        = ckpt.get('test_maes', [])
            train_tick_accs  = ckpt.get('train_tick_accs', [])
            test_tick_accs   = ckpt.get('test_tick_accs', [])
            logged_iters     = ckpt.get('iters', [])
            if 'torch_rng_state' in ckpt:
                torch.set_rng_state(ckpt['torch_rng_state'].cpu().byte())
                np.random.set_state(ckpt['numpy_rng_state'])
                random.setstate(ckpt['random_rng_state'])
        del ckpt

    # ── Training loop ─────────────────────────────────────────────────────
    iterator = iter(trainloader)

    with tqdm(total=args.training_iterations, initial=start_iter, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            try:
                frames, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                frames, targets = next(iterator)

            frames  = frames.to(device)
            targets = targets.to(device)
            B       = frames.shape[0]
            tgts_flat = flatten_targets(targets, args.n_objects, args.n_frames)

            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16, enabled=args.use_amp,
            ):
                preds, certs, _ = model(frames)
                preds = reshape_preds(preds, args.n_objects, args.n_frames, args.n_bins)
                loss, where_cert = tracking_loss(preds, certs, tgts_flat, args.use_most_certain)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            batch_idx     = torch.arange(B, device=device)
            preds_at_tick = preds[batch_idx, :, :, where_cert]
            mae_approx    = position_mae(preds_at_tick, tgts_flat, args.n_bins)

            pbar.set_description(
                f'Loss={loss.item():.3f}  MAE={mae_approx:.3f}'
                f'  tick={where_cert.float().mean():.1f}±{where_cert.float().std():.1f}'
                f'  lr={optimizer.param_groups[0]["lr"]:.2e}'
            )
            pbar.update(1)

            # ── Evaluation ────────────────────────────────────────────────
            if bi % args.track_every == 0:
                model.eval()

                tr_loss, tr_mae, tr_tick_acc = eval_loop(
                    model, trainloader, args, device, args.n_test_batches)
                te_loss, te_mae, te_tick_acc = eval_loop(
                    model, testloader,  args, device, args.n_test_batches)

                logged_iters.append(bi)
                train_losses.append(tr_loss);     test_losses.append(te_loss)
                train_maes.append(tr_mae);        test_maes.append(te_mae)
                train_tick_accs.append(tr_tick_acc.tolist())
                test_tick_accs.append(te_tick_acc.tolist())

                print(
                    f'[{bi}] Train loss={tr_loss:.4f} MAE={tr_mae:.4f}'
                    f'  |  Val loss={te_loss:.4f} MAE={te_mae:.4f}'
                )
                plot_metrics(logged_iters, train_losses, test_losses,
                             train_maes, test_maes,
                             train_tick_accs, test_tick_accs, args.log_dir)
                model.train()

            # ── Checkpoint ────────────────────────────────────────────────
            if bi % args.save_every == 0 and bi > 0:
                torch.save({
                    'iteration':           bi,
                    'model_state_dict':    model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict':   scaler.state_dict(),
                    'args':                args,
                    'train_losses':        train_losses,
                    'test_losses':         test_losses,
                    'train_maes':          train_maes,
                    'test_maes':           test_maes,
                    'train_tick_accs':     train_tick_accs,
                    'test_tick_accs':      test_tick_accs,
                    'iters':               logged_iters,
                    'torch_rng_state':     torch.get_rng_state(),
                    'numpy_rng_state':     np.random.get_state(),
                    'random_rng_state':    random.getstate(),
                }, f'{args.log_dir}/checkpoint_{bi:07d}.pt')
