#!/usr/bin/env bash
# Workaround for an upstream packaging bug in ZHZisZZ/dllm: its pyproject.toml
# declares `packages = ["dllm"]`, which tells setuptools to ship only the top-
# level package and silently drops every subpackage (`dllm.core`, `dllm.utils`,
# `dllm.data`, `dllm.pipelines`). The result is an install where
# `import dllm` fails with `cannot import name 'core'`.
#
# We additionally trim the eager imports in `dllm/__init__.py`,
# `dllm/core/__init__.py` and `dllm/pipelines/__init__.py` so the package
# does not try to load every pipeline (which pulls in lm-eval, trl, vllm,
# flash-attn, deepspeed RL extras) for our use case. We only need MDLM
# training and MDLM sampling, both in `dllm.core`.
#
# Run this once after `pip install -r requirements.txt` (or
# `pip install -e ".[dev]"`).

set -euo pipefail

# Honour an active virtualenv if there is one; this is the common case
# locally but not on a fresh GPU pod where `pip` and `python` are on PATH.
PYTHON="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON="${PYTHON:-python}"
PIP="${PIP:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/pip}}"
PIP="${PIP:-pip}"

DLLM_DIR="${DLLM_DIR:-./external/dllm}"
DLLM_URL="${DLLM_URL:-https://github.com/ZHZisZZ/dllm.git}"

if [ ! -d "$DLLM_DIR" ]; then
    git clone --depth 1 "$DLLM_URL" "$DLLM_DIR"
fi

"$PYTHON" - <<EOF
from pathlib import Path

dllm_dir = Path("$DLLM_DIR")

# 1. Patch pyproject.toml: include subpackages.
p = dllm_dir / "pyproject.toml"
content = p.read_text()
patched = content.replace(
    '[tool.setuptools]\npackages = ["dllm"]',
    '[tool.setuptools.packages.find]\nwhere = ["."]\ninclude = ["dllm*"]',
)
if patched != content:
    p.write_text(patched)
    print("Patched dllm pyproject.toml.")

# 2. Patch dllm/__init__.py: drop pipelines (pulls heavy deps we do not need).
init = dllm_dir / "dllm" / "__init__.py"
init.write_text(
    'from . import core, data, utils\n\n'
    '__all__ = ["core", "data", "utils"]\n'
)
print("Trimmed dllm/__init__.py to {core, data, utils}.")

# 3. Patch dllm/core/__init__.py: drop the eval submodule (needs lm-eval).
core_init = dllm_dir / "dllm" / "core" / "__init__.py"
core_init.write_text(
    'from . import samplers, schedulers, trainers\n\n'
    '__all__ = ["samplers", "schedulers", "trainers"]\n'
)
print("Trimmed dllm/core/__init__.py to {samplers, schedulers, trainers}.")

# 4. Patch dllm/pipelines/__init__.py: drop rl (needs trl) and editflow (heavy).
#    dllm.utils.get_tokenizer transitively imports dllm.pipelines.a2d, which
#    triggers this __init__ — and the upstream version eagerly imports rl.
pipelines_init = dllm_dir / "dllm" / "pipelines" / "__init__.py"
pipelines_init.write_text(
    'from . import a2d\n\n'
    '__all__ = ["a2d"]\n'
)
print("Trimmed dllm/pipelines/__init__.py to {a2d}.")

# 5. Patch dllm/utils/models.py: add trust_remote_code=True everywhere
#    (MDLM-OWT ships custom modeling code) and add a GPT-2 tokenizer fallback
#    (MDLM-OWT does not ship a tokenizer in its repo).
models_py = dllm_dir / "dllm" / "utils" / "models.py"
src = models_py.read_text()

# 5a. trust_remote_code in get_model params dict.
src = src.replace(
    'params = {\n        "dtype": dtype,',
    'params = {\n        "trust_remote_code": True,\n        "dtype": dtype,',
)
# 5b. trust_remote_code in get_tokenizer's primary AutoTokenizer call, plus
#     a GPT-2 fallback for repos without a tokenizer (MDLM-OWT).
old_tok = (
    '    # ---------------- Tokenizer loading ----------------\n'
    '    tokenizer = transformers.AutoTokenizer.from_pretrained(\n'
    '        model_name_or_path,\n'
    '        padding_side="right",\n'
    '    )'
)
new_tok = (
    '    # ---------------- Tokenizer loading ----------------\n'
    '    try:\n'
    '        tokenizer = transformers.AutoTokenizer.from_pretrained(\n'
    '            model_name_or_path,\n'
    '            padding_side="right",\n'
    '            trust_remote_code=True,\n'
    '        )\n'
    '    except (ValueError, OSError):\n'
    '        # MDLM-OWT (kuleshov-group/mdlm-owt) ships no tokenizer; falls back to GPT-2.\n'
    '        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", padding_side="right")\n'
    '        # MDLM expects vocab_size = GPT-2 (50257) + 1 mask token = 50258 — id 50257.\n'
    '        tokenizer.add_special_tokens({"mask_token": "<|mdlm_mask|>"})'
)
assert old_tok in src, "tokenizer patch anchor not found"
src = src.replace(old_tok, new_tok)
# 5c. trust_remote_code on AutoConfig + handle custom configs not in AutoModel
#     mapping (MDLMConfig is registered by trust_remote_code but absent from
#     the standard AutoModel._model_mapping).
old_cfg = (
    '    # If model is not provided, return as-is\n'
    '    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path)\n'
    '    model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]'
)
new_cfg = (
    '    # If model is not provided, return as-is\n'
    '    model_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)\n'
    '    if type(model_cfg) not in transformers.AutoModel._model_mapping:\n'
    '        # Custom configs (e.g. MDLMConfig) are not in the standard mapping;\n'
    '        # skip the model-specific tokenizer customization.\n'
    '        return tokenizer\n'
    '    model_cls = transformers.AutoModel._model_mapping[type(model_cfg)]'
)
assert old_cfg in src, "config patch anchor not found"
src = src.replace(old_cfg, new_cfg)
models_py.write_text(src)
print("Patched dllm/utils/models.py for MDLM-OWT compatibility.")
EOF

"$PIP" install --force-reinstall --no-deps -e "$DLLM_DIR"
"$PYTHON" -c "from dllm.core.trainers import MDLMTrainer; from dllm.core.samplers import MDLMSampler; print('dllm import OK')"
