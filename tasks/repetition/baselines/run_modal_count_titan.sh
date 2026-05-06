#!/bin/bash
# Tier-0 chance-level baselines for the repetition task — runs on Titan
# (CPU-fast: only label histograms, no video decoding past header).
#
# Outputs each result as JSON next to the matching log dir so the loss
# plots have a chance bar to compare against. Adjust DATA_BASE if your
# titan symlinks live elsewhere.
#
# Usage:
#   bash tasks/repetition/baselines/run_modal_count_titan.sh
#
# Note: synthetic-v2 is not in modal_count.py's choices list as of writing
# (see modal_count.py:43). Its label distribution differs from v1 (sums of
# per-segment counts, not uniform), so the v1 number below is only a loose
# upper bound. Patch the choices list if you want a true v2 baseline.

set -e

cd "$(dirname "$0")/../../.."   # -> repo root

DATA_BASE="/geovic/ghermi/data"
OUT_BASE="logs/repetition"

mkdir -p "${OUT_BASE}/countix" "${OUT_BASE}/ucfrep" "${OUT_BASE}/repcount" "${OUT_BASE}/synthetic"

echo "=== Countix ==="
python -m tasks.repetition.baselines.modal_count \
    --dataset countix \
    --data_root "${DATA_BASE}/countix/" \
    --kinetics_root "${DATA_BASE}/kinetics/" \
    --image_size 112 \
    --n_count_buckets 32 \
    --target_fps 8 \
    --save_json "${OUT_BASE}/countix/modal_baseline.json"

echo
echo "=== UCFRep ==="
python -m tasks.repetition.baselines.modal_count \
    --dataset ucfrep \
    --data_root "${DATA_BASE}/ucfrep/" \
    --image_size 112 \
    --n_count_buckets 32 \
    --target_fps 8 \
    --save_json "${OUT_BASE}/ucfrep/modal_baseline.json"

#echo
#echo "=== RepCount-A ==="
#python -m tasks.repetition.baselines.modal_count \
#    --dataset repcount \
#    --data_root "${DATA_BASE}/repcount/" \
#    --image_size 112 \
#    --n_count_buckets 32 \
#    --target_fps 8 \
#    --save_json "${OUT_BASE}/repcount/modal_baseline.json"

echo
echo "=== Synthetic v1 (uniform[1,8] labels — also a loose upper bound for v2) ==="
python -m tasks.repetition.baselines.modal_count \
    --dataset synthetic \
    --image_size 64 \
    --max_count 8 \
    --n_count_buckets 16 \
    --target_fps 16 \
    --clip_duration_s_min 4 \
    --clip_duration_s_max 12 \
    --save_json "${OUT_BASE}/synthetic/modal_baseline.json"

echo
echo "Done. JSONs saved under ${OUT_BASE}/*/modal_baseline.json"
