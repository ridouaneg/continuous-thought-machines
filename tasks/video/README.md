# Video action recognition with the Continuous Thought Machine

Proof-of-concept exploration: use the CTM as a streaming video classifier
on UCF-101 / HMDB-51. This folder is self-contained — the only imports
from the rest of the repo are the base `ContinuousThoughtMachine`, the
`image_classification_loss` (which broadcasts a scalar label across the
tick axis), and the standard schedulers / housekeeping utilities.

## Design

The base CTM computes key/value features once from a static image and
then attends to that same tensor at every internal tick. For video we
reinterpret the tick axis as the **frame axis**:

- Sample `n_frames` frames per clip (e.g. 16 or 32).
- Run the backbone on all frames in parallel (batched along the time
  axis) to produce per-frame spatial features.
- Run the CTM for `n_frames * iterations_per_frame` internal ticks. At
  internal tick `t`, the attention query (computed from the **action
  synchronisation**) reads from the spatial features of frame
  `t // iterations_per_frame`. All recurrent state — pre-activation
  trace, NLM outputs, and the leaky (alpha, beta) synchronisation
  accumulators — carries over across frames.
- Emit a prediction per tick. Loss is the existing
  `image_classification_loss` (mean of min-CE tick + max-certainty
  tick). Full BPTT through every tick.

For a PoC with `iterations_per_frame=1`, a 32-frame clip corresponds to
32 CTM ticks, which is cheaper than the 75 ticks used for ImageNet.

## What runs out of the box

- **Synthetic dataset** (`--dataset synthetic`). Generates a tiny
  "moving shapes" dataset on the fly. No downloads, no video decoding.
  Trains to ~100% in a few thousand iterations on CPU. Useful for
  sanity checks and for generating the visualisations in the notebook.
- **UCF-101** (`--dataset ucf101`). Expects the standard UCF-101
  directory layout under `--data_root` (videos grouped by class folder)
  plus the canonical split files. Uses `torchvision.io.read_video` for
  decoding.
- **HMDB-51** (`--dataset hmdb51`). Same layout, different split format.

## Training

```
# Synthetic sanity check (CPU-friendly)
python -m tasks.video.train --dataset synthetic --n_frames 16 \
    --d_model 256 --d_input 128 --backbone_type resnet18-1 \
    --iterations_per_frame 1 --synapse_depth 2 --heads 4 \
    --n_synch_out 64 --n_synch_action 64 --memory_length 8 \
    --batch_size 16 --training_iterations 2001 --log_dir logs/video/synthetic

# UCF-101
bash tasks/video/scripts/train_ucf101.sh

# HMDB-51
bash tasks/video/scripts/train_hmdb51.sh
```

## Visualisations

After training, open `tasks/video/analysis/visualize.ipynb`. Point it
at a checkpoint directory. It generates:

1. Neuron raster over the clip (D rows × n_frames×K columns).
2. Per-frame attention heatmap overlaid on the video frames.
3. Certainty curve across ticks with a frame-boundary axis.
4. UMAP of the output-synchronisation vector across ticks, coloured by
   tick index — shows trajectories of the latent as the CTM watches.
5. FFT of each neuron's trace — shows that different neurons learn
   different temporal frequencies (a central prediction of the paper,
   now keyed to real video timescales).
6. Synchronisation-accumulator magnitudes over time, which reflect the
   "soft-infinite" memory channel.

## Files

```
tasks/video/
├── README.md          (this file)
├── model.py           (CTMVideo — subclass with per-frame attention)
├── dataset.py         (synthetic / UCF-101 / HMDB-51 loaders)
├── train.py           (training loop, mirrors image_classification/train.py)
├── scripts/
│   ├── train_synthetic.sh
│   ├── train_ucf101.sh
│   └── train_hmdb51.sh
└── analysis/
    └── visualize.ipynb
```
