#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/auto_tune_n200 artifacts/mu_extra
step() { printf "\n=== %s — %s ===\n" "$(date -Iseconds)" "$1"; }

QW="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
QW_CKPT="artifacts/checkpoints_qwen3"
MD="kuleshov-group/mdlm-owt"
MD_CKPT="artifacts/checkpoints"

# === Phase A (auto-tune at n=200) — 3 setups ===

step "12b/1: A+B at n=200 on Qwen3 fpc (GT μ*=-1)"
python3 scripts/15_auto_tune_mu.py \
    --triplet formal --triplet positive --triplet concrete \
    --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --n-samples 200 --bo-iters 6 \
    --candidates "-2,-1.5,-1,-0.5,0" \
    --out-json artifacts/auto_tune_n200/qwen3_fpc.json

step "12b/2: A+B at n=200 on Qwen3 pcs (GT μ*=-2 canonical)"
python3 scripts/15_auto_tune_mu.py \
    --triplet positive2 --triplet concrete --triplet sports \
    --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --n-samples 200 --bo-iters 6 \
    --candidates "-2,-1.5,-1,-0.5,0" \
    --out-json artifacts/auto_tune_n200/qwen3_pcs.json

step "12b/3: A+B at n=200 on MDLM-OWT fpc (GT μ*=0)"
python3 scripts/15_auto_tune_mu.py \
    --triplet formal --triplet positive --triplet concrete \
    --backbone "$MD" --checkpoints-dir "$MD_CKPT" \
    --n-samples 200 --bo-iters 6 \
    --candidates "-2,-1.5,-1,-0.5,0" \
    --out-json artifacts/auto_tune_n200/mdlm_fpc.json

# === Phase B (extra μ-sweeps for predictor training) — 8 N=2 pairs + 2 triplets ===
# N=2 pairs on Qwen3 not yet done: 8 pairs
# Reduced sweep: μ ∈ {-1.5, -1, -0.5, 0} (4 values) at n=200

step "12c/1: Qwen3 N=2 formal × positive2"
python3 scripts/14_mu_sweep.py --triplet formal --triplet positive2 \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_fp2.json

step "12c/2: Qwen3 N=2 formal × concrete"
python3 scripts/14_mu_sweep.py --triplet formal --triplet concrete \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_fc.json

step "12c/3: Qwen3 N=2 formal × sports"
python3 scripts/14_mu_sweep.py --triplet formal --triplet sports \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_fs.json

step "12c/4: Qwen3 N=2 positive × positive2"
python3 scripts/14_mu_sweep.py --triplet positive --triplet positive2 \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_pp2.json

step "12c/5: Qwen3 N=2 positive × concrete"
python3 scripts/14_mu_sweep.py --triplet positive --triplet concrete \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_pc.json

step "12c/6: Qwen3 N=2 positive × sports"
python3 scripts/14_mu_sweep.py --triplet positive --triplet sports \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_ps.json

step "12c/7: Qwen3 N=2 positive2 × concrete"
python3 scripts/14_mu_sweep.py --triplet positive2 --triplet concrete \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_p2c.json

step "12c/8: Qwen3 N=2 positive2 × sports"
python3 scripts/14_mu_sweep.py --triplet positive2 --triplet sports \
    --n-samples 200 --max-new-tokens 48 --backbone "$QW" --checkpoints-dir "$QW_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_qwen3_p2s.json

# Plus 2 N=2 on MDLM-OWT for cross-backbone in predictor
step "12c/9: MDLM-OWT N=2 formal × positive"
python3 scripts/14_mu_sweep.py --triplet formal --triplet positive \
    --n-samples 200 --max-new-tokens 48 --backbone "$MD" --checkpoints-dir "$MD_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_mdlm_fp.json

step "12c/10: MDLM-OWT N=2 concrete × sports"
python3 scripts/14_mu_sweep.py --triplet concrete --triplet sports \
    --n-samples 200 --max-new-tokens 48 --backbone "$MD" --checkpoints-dir "$MD_CKPT" \
    --mu-values "-1.5,-1,-0.5,0" --out-json artifacts/mu_extra/n2_mdlm_cs.json

step "Phase 12 full done"
