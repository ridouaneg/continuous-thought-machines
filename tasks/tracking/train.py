"""Train the on-the-fly tracking CTM.

Each internal tick attends to one frame and emits the position prediction
for that same frame — there is no look-ahead. State carries across frames.

Supports three dataset backends:

  synthetic  : fast, fully synthetic Gaussian blobs (no download required)
  mot17      : MOT17 pedestrian tracking (~5 GB download, motchallenge.net)
  dancetrack : DanceTrack dancer tracking (github.com/DanceTrack/DanceTrack)

Example usage
-------------
# Synthetic (quick test)
python -m tasks.tracking.train --dataset synthetic

# MOT17 (after downloading to /data/MOT17)
python -m tasks.tracking.train \\
    --dataset mot17 --data_root /data/MOT17 \\
    --backbone_type resnet18-2 --in_channels 3 \\
    --n_objects 8 --img_size 128 --n_bins 16 \\
    --d_model 512 --d_input 256 \\
    --n_frames 8 --iterations_per_frame 4 \\
    --device 0 --log_dir logs/tracking/mot17
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

from tasks.tracking.losses import tracking_loss, position_mae
from tasks.tracking.utils import prepare_model, build_datasets
from utils.housekeeping import set_seed
from utils.run import init_run, load_checkpoint, save_checkpoint
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on multi-object tracking (on-the-fly)")

    # ── Dataset ───────────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'mot17', 'dancetrack'])
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--n_objects', type=int, default=2)
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--n_bins', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    # Synthetic-only
    parser.add_argument('--blob_sigma_px', type=float, default=1.5)
    parser.add_argument('--velocity_scale', type=float, default=0.07)
    parser.add_argument('--n_train', type=int, default=50000)
    parser.add_argument('--n_test',  type=int, default=5000)
    parser.add_argument('--in_channels', type=int, default=3,
                        help="Input image channels. Synthetic emits 1; with 3 it "
                             "replicates and ImageNet-normalises.")

    # ── CTM architecture ──────────────────────────────────────────────────
    parser.add_argument('--backbone_type', type=str, default='resnet18-2',
                        help="Standard CTM backbone (see models/constants.py).")
    parser.add_argument('--pretrained_backbone',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--freeze_backbone',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--positional_embedding_type', type=str, default='learnable-fourier',
                        choices=['none', 'learnable-fourier', 'multi-learnable-fourier',
                                 'custom-rotational'])
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
    parser.add_argument('--iterations_per_frame', type=int, default=4,
                        help="CTM internal ticks spent on each frame "
                             "(total iterations = n_frames * iterations_per_frame).")

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
    parser.add_argument('--run_name', type=str, default=None)
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


def reshape_preds(predictions, n_objects, n_bins):
    """(B, N*2*K, iters) → (B, N*2, K, iters)"""
    B, _, iters = predictions.shape
    return predictions.reshape(B, n_objects * 2, n_bins, iters)


def per_tick_bin_accuracy(predictions, targets, iterations_per_frame):
    """Per-tick bin accuracy aligned with the on-the-fly target schedule.

    predictions : (B, N*2, K, iters)
    targets     : (B, T, N, 2) bin indices (-1 = absent)

    Returns (iters,) accuracy averaged over valid positions and batch.
    """
    B, n_per_frame, _, iters = predictions.shape
    T = targets.shape[1]
    ipf = iterations_per_frame

    targets_pf = targets.reshape(B, T, n_per_frame)                   # (B, T, N*2)
    targets_per_tick = targets_pf.repeat_interleave(ipf, dim=1)        # (B, iters, N*2)
    targets_per_tick = targets_per_tick.transpose(1, 2)                # (B, N*2, iters)
    valid = (targets_per_tick >= 0)
    safe = targets_per_tick.clamp(min=0)

    correct = (predictions.argmax(2) == safe) & valid                 # (B, N*2, iters)
    n_valid = valid.float().sum(dim=1).clamp(min=1)                   # (B, iters)
    per_sample = correct.float().sum(dim=1) / n_valid                 # (B, iters)
    return per_sample.mean(dim=0).cpu().numpy()                       # (iters,)


def eval_loop(model, loader, args, device, n_test_batches):
    """Run evaluation; returns (mean_loss, mean_mae, per_tick_acc)."""
    all_losses, all_maes = [], []
    per_tick_sum = None
    n_batches = 0

    with torch.inference_mode():
        for bi, (frames, targets) in enumerate(loader):
            frames  = frames.to(device)
            targets = targets.to(device)

            preds, certs, _ = model(frames)
            preds = reshape_preds(preds, args.n_objects, args.n_bins)

            loss, where_cert = tracking_loss(
                preds, certs, targets,
                iterations_per_frame=args.iterations_per_frame,
                use_most_certain=args.use_most_certain,
            )
            all_losses.append(loss.item())

            mae = position_mae(preds, targets, args.n_bins, where_cert)
            all_maes.append(mae)

            tick_acc = per_tick_bin_accuracy(preds, targets, args.iterations_per_frame)
            per_tick_sum = tick_acc if per_tick_sum is None else per_tick_sum + tick_acc
            n_batches += 1

            if bi + 1 >= n_test_batches:
                break

    per_tick_acc = per_tick_sum / max(n_batches, 1)
    return float(np.mean(all_losses)), float(np.mean(all_maes)), per_tick_acc


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
    init_run(args)

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
    ckpt = load_checkpoint(args.log_dir, map_location=device) if args.reload else None
    if ckpt is not None:
        print(f'Reloading {args.log_dir}/checkpoint.pt')
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

            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16, enabled=args.use_amp,
            ):
                preds, certs, _ = model(frames)
                preds = reshape_preds(preds, args.n_objects, args.n_bins)
                loss, where_cert = tracking_loss(
                    preds, certs, targets,
                    iterations_per_frame=args.iterations_per_frame,
                    use_most_certain=args.use_most_certain,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            mae_approx = position_mae(preds, targets, args.n_bins, where_cert)

            mean_tick = where_cert.float().mean()
            std_tick  = where_cert.float().std()
            pbar.set_description(
                f'Loss={loss.item():.3f}  MAE={mae_approx:.3f}'
                f'  tick={mean_tick:.1f}±{std_tick:.1f}'
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
                save_checkpoint(args.log_dir, {
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
                }, bi, keep_history=True)
