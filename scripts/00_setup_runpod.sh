#!/usr/bin/env bash
# Provision a fresh RunPod pod (template: pytorch:2.1.0-py3.10-cuda11.8).

set -euo pipefail

apt-get update -qq
apt-get install -y --no-install-recommends git tmux htop nvtop curl

pip install --upgrade pip
pip install -r requirements.txt

: "${WANDB_API_KEY:?Set WANDB_API_KEY in pod env}"
: "${HF_TOKEN:?Set HF_TOKEN in pod env}"
echo "${WANDB_API_KEY}" | wandb login --relogin
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential

# Pre-fetch model weights so the first scripted run isn't I/O-bound.
python - <<'PY'
from huggingface_hub import snapshot_download
for repo_id in [
    "kuleshov-group/mdlm-owt",
    "s-nlp/roberta-base-formality-ranker",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "unitary/toxic-bert",
    "gpt2",
]:
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "tokenizer*"],
    )
PY

python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))"
