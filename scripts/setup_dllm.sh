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
EOF

"$PIP" install --force-reinstall --no-deps -e "$DLLM_DIR"
"$PYTHON" -c "from dllm.core.trainers import MDLMTrainer; from dllm.core.samplers import MDLMSampler; print('dllm import OK')"
