#!/usr/bin/env bash
# Vendor the SEDD reference (Lou et al. 2024, ICML) under external/sedd/.
#
# SEDD = Score-Entropy Discrete Diffusion. We use it for the parallel
# Paper 2 track that tests score-based composition (vs the marginal-based
# composition of Paper 1, MDLM + PoE-logits).
#
# Lou's repo is pure PyTorch (no HF Trainer integration). We do not patch
# upstream code here — instead, we wrap it from src/sedd_composition/.
# That keeps the vendored snapshot pristine and re-cloneable.
#
# Run once after `pip install -r requirements.txt`.

set -euo pipefail

PYTHON="${PYTHON:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON="${PYTHON:-python}"
PIP="${PIP:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/pip}}"
PIP="${PIP:-pip}"

SEDD_DIR="${SEDD_DIR:-./external/sedd}"
SEDD_URL="${SEDD_URL:-https://github.com/louaaron/Score-Entropy-Discrete-Diffusion.git}"
# Pin a commit so the patch surface stays stable; bump explicitly when
# upgrading. Setting to "main" (default) is fine for early dev; pin before
# running real experiments.
SEDD_REV="${SEDD_REV:-main}"

if [ ! -d "$SEDD_DIR" ]; then
    git clone "$SEDD_URL" "$SEDD_DIR"
fi
(cd "$SEDD_DIR" && git fetch --all --quiet && git checkout --quiet "$SEDD_REV")

# We do NOT pip-install SEDD as a package — Lou's repo has no
# pyproject.toml. Instead, src/sedd_composition/ adds external/sedd to
# sys.path lazily (see src/sedd_composition/load.py).

# Verify the file layout we depend on hasn't drifted upstream, and patch
# transformer.py with an SDPA fallback so it loads on machines without
# flash_attn (Mac local dev, CUDA-13 pods, etc.). The patch is idempotent.
"$PYTHON" - <<EOF
from pathlib import Path
sedd = Path("$SEDD_DIR")
required = [
    "model/transformer.py",
    "sampling.py",
    "graph_lib.py",
    "noise_lib.py",
    "losses.py",
    "load_model.py",
    "configs",
]
missing = [r for r in required if not (sedd / r).exists()]
if missing:
    raise SystemExit(f"SEDD layout drift — missing: {missing}")

# 1. Make flash_attn import optional.
trf = sedd / "model" / "transformer.py"
src = trf.read_text()
old_imp = "from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func"
new_imp = (
    "try:\n"
    "    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func\n"
    "    _HAS_FLASH_ATTN = True\n"
    "except ImportError:\n"
    "    flash_attn_varlen_qkvpacked_func = None\n"
    "    _HAS_FLASH_ATTN = False"
)
if old_imp in src:
    src = src.replace(old_imp, new_imp)
    print("Patched flash_attn import to try/except.")

# 2. Replace the flash_attn call with an SDPA fallback. The original
#    call returns (b*s, n_heads, head_dim); we reshape the qkv pack into
#    standard (b, h, s, d) tensors, apply F.scaled_dot_product_attention
#    (non-causal — SEDD is bidirectional), and reshape back to match.
old_call = (
    "        x = flash_attn_varlen_qkvpacked_func(\n"
    "            qkv, cu_seqlens, seq_len, 0., causal=False)"
)
new_call = (
    "        if _HAS_FLASH_ATTN:\n"
    "            x = flash_attn_varlen_qkvpacked_func(\n"
    "                qkv, cu_seqlens, seq_len, 0., causal=False)\n"
    "        else:\n"
    "            # SDPA fallback for envs without flash_attn (Mac local, CUDA-13 pods).\n"
    "            # qkv at this point is (b*s, 3, n_heads, head_dim).\n"
    "            head_dim = qkv.shape[-1]\n"
    "            qkv_b = qkv.reshape(batch_size, seq_len, 3, self.n_heads, head_dim)\n"
    "            q, k, v = qkv_b.permute(2, 0, 3, 1, 4).unbind(0)  # each (b, h, s, d)\n"
    "            x = F.scaled_dot_product_attention(q, k, v, is_causal=False)\n"
    "            # back to (b*s, n_heads, head_dim) to match flash_attn output shape\n"
    "            x = x.permute(0, 2, 1, 3).reshape(batch_size * seq_len, self.n_heads, head_dim)"
)
if old_call in src:
    src = src.replace(old_call, new_call)
    print("Patched flash_attn call site to SDPA fallback.")

trf.write_text(src)

# 3. Disable @torch.jit.script decorators in fused_add_dropout_scale.py and
#    rotary.py — they crash on torch 2.11 (vector range check) even though
#    they worked under torch 2.4. Eager Python execution is fast enough for
#    our scale and is bit-identical.
for fname in ("model/fused_add_dropout_scale.py", "model/rotary.py"):
    p = sedd / fname
    s = p.read_text()
    if "@torch.jit.script" in s:
        s = s.replace("@torch.jit.script\n", "")
        p.write_text(s)
        print(f"Stripped @torch.jit.script in {fname}.")

print("SEDD layout OK at", sedd)
EOF
