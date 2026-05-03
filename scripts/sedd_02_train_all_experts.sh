#!/usr/bin/env bash
# Phase-2 of Paper 2: train all 6 SEDD-small LoRA experts sequentially.
# Run inside a tmux session. ~10 min per expert × 6 = ~1h on A100.

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="${OUTDIR:-artifacts/sedd_checkpoints}"
NUM_STEPS="${NUM_STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-256}"
LR="${LR:-3e-4}"
BACKBONE="${BACKBONE:-louaaron/sedd-small}"

step() { printf "\n=== %s — %s ===\n" "$(date -Iseconds)" "$1"; }

mkdir -p "$OUTDIR"

for EXPERT in long formal positive positive2 concrete sports; do
    step "training $EXPERT"
    python scripts/sedd_01_train_lora.py \
        --expert "$EXPERT" \
        --output-dir "$OUTDIR" \
        --backbone "$BACKBONE" \
        --num-steps "$NUM_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --sequence-length "$SEQ_LEN" \
        --learning-rate "$LR"
done

step "all 6 experts trained"
ls -la "$OUTDIR"
