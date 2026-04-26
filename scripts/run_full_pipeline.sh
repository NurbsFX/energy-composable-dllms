#!/usr/bin/env bash
# Pod-side wrapper that runs every phase end-to-end and writes a marker at
# the end. Run it inside a tmux session on the pod so disconnections do
# not kill it:
#
#     tmux new -s poc
#     ./scripts/run_full_pipeline.sh
#
# The local-side scripts/auto_download_artifacts.sh polls for the marker
# and pulls everything down to ~/Documents/composable-dllms-artifacts.
#
# `set -e` aborts on the first failed phase, so a non-zero exit means the
# marker was *not* written — the auto-downloader will keep polling and
# eventually time out (configured locally), prompting you to investigate.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p artifacts

step() {
    printf '\n=== %s — %s ===\n' "$(date -Iseconds)" "$1"
}

step "Phase 2: build datasets (80k per vertical)"
python scripts/02_build_datasets.py --target-size 80000

step "Phase 3: train 6 LoRA experts"
python scripts/03_train_experts.py

step "Phase 3b: train 2 intersection experts (long∩formal, formal∩concrete)"
python scripts/03b_train_intersection_expert.py \
    --pairs long:formal --pairs formal:concrete

step "Phase 3.5: validate experts (cross-vertical scoring)"
python scripts/04_validate_experts.py

step "Phase 4: 15-pair composition sweep (Plan-B λ extended)"
python scripts/05_run_composition.py \
    --pairs long:formal --pairs long:positive --pairs long:positive2 \
    --pairs long:concrete --pairs long:sports \
    --pairs formal:positive --pairs formal:positive2 --pairs formal:concrete \
    --pairs formal:sports \
    --pairs positive:positive2 --pairs positive:concrete --pairs positive:sports \
    --pairs positive2:concrete --pairs positive2:sports \
    --pairs concrete:sports \
    --n-samples 500

step "Phase 4.5 Test 2: PoE formula check"
python scripts/06_poe_formula_check.py --pair long:formal --k-pairs 50

step "Phase 4.5 Test 1: long ∩ formal vs PoE(long, formal)"
python scripts/06b_test1_intersection_check.py --pair long:formal --n-samples 500

step "Phase 4.5 Test 1: formal ∩ concrete vs PoE(formal, concrete)"
python scripts/06b_test1_intersection_check.py --pair formal:concrete --n-samples 500

step "Phase 4.5 Test 3: N=3 extension"
python scripts/07_n3_extension.py \
    --triplet long --triplet formal --triplet positive \
    --n-samples 500

# Marker — only written if every phase above succeeded.
{
    echo "Pipeline finished at $(date -Iseconds)"
    echo "Host: $(hostname)"
} > artifacts/PIPELINE_DONE.txt

step "Pipeline complete"
echo "Marker written to artifacts/PIPELINE_DONE.txt — the local"
echo "auto-downloader will detect this and pull everything home."
