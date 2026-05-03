#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/auto_tune
step() { printf "\n=== %s — %s ===\n" "$(date -Iseconds)" "$1"; }

QW="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
QW_CKPT="artifacts/checkpoints_qwen3"
MD="kuleshov-group/mdlm-owt"
MD_CKPT="artifacts/checkpoints"

step "Setup 1: Qwen3 formal × positive × concrete (known μ*=-1, r*=0.61)"
python3 scripts/15_auto_tune_mu.py \
    --triplet formal --triplet positive --triplet concrete \
    --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --n-samples 50 --bo-iters 4 --candidates "-2,-1,-0.5,0" \
    --out-json artifacts/auto_tune/qwen3_fpc.json

step "Setup 2: Qwen3 positive2 × concrete × sports (known μ*=-2, r*=3.23)"
python3 scripts/15_auto_tune_mu.py \
    --triplet positive2 --triplet concrete --triplet sports \
    --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --n-samples 50 --bo-iters 4 --candidates "-2,-1,-0.5,0" \
    --out-json artifacts/auto_tune/qwen3_pcs.json

step "Setup 3: MDLM-OWT formal × positive × concrete (known μ*=0, r*=0.71)"
python3 scripts/15_auto_tune_mu.py \
    --triplet formal --triplet positive --triplet concrete \
    --backbone "$MD" --checkpoints-dir "$MD_CKPT" \
    --n-samples 50 --bo-iters 4 --candidates "-2,-1,-0.5,0" \
    --out-json artifacts/auto_tune/mdlm_fpc.json

step "All auto-tune validations done"
