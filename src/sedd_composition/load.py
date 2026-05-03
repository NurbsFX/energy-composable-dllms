"""Bridge to ``external/sedd``.

Lou's repo ships no ``pyproject.toml`` and is not pip-installable. We add
``external/sedd/`` to ``sys.path`` lazily, so the rest of our code can do
``from model import SEDD`` etc. without further plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SEDD_DIR = REPO_ROOT / "external" / "sedd"


def _ensure_sedd_on_path() -> None:
    """Idempotently add external/sedd to sys.path so its top-level modules
    (``model``, ``sampling``, ``graph_lib``, ``noise_lib``, ``utils``) become
    importable. We do not patch the upstream code; we live with its flat
    layout."""
    if not SEDD_DIR.exists():
        raise RuntimeError(
            f"SEDD reference not found at {SEDD_DIR}. Run `bash scripts/setup_sedd.sh` first."
        )
    p = str(SEDD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_sedd_from_hub(
    repo_id: str = "louaaron/sedd-small",
    device: str | None = None,
):
    """Fetch a published SEDD checkpoint and return ``(model, graph, noise)``.

    ``model`` is an ``nn.Module`` whose ``forward(x, sigma)`` returns log-scores
    of shape ``(batch, seq_len, vocab)`` where ``vocab = 50258`` (GPT-2 vocab
    + 1 absorbing/MASK token at index 50257).

    Parameters
    ----------
    repo_id : str
        HuggingFace repo. Available: ``louaaron/sedd-small`` (90M),
        ``louaaron/sedd-medium`` (320M).
    device : str | None
        Torch device (default: cuda if available, else mps, else cpu).

    Returns
    -------
    score_model : nn.Module
        The SEDD score model.
    graph : graph_lib.Graph
        The discrete absorbing graph (provides ``staggered_score``,
        ``transp_transition``, etc.). Tokenizer-aware; do not swap.
    noise : noise_lib.Noise
        The noise schedule. Provides ``noise(t) -> (sigma, dsigma)``.
    """
    _ensure_sedd_on_path()

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    from load_model import load_model_hf  # type: ignore

    score_model, graph, noise = load_model_hf(repo_id, device=device)
    return score_model, graph, noise


def get_gpt2_tokenizer_for_sedd():
    """Return a GPT-2 tokenizer with the SEDD MASK token registered at
    id 50257. Mirrors what `dllm.utils.get_tokenizer` does for MDLM-OWT.

    SEDD's effective vocab is 50258 = 50257 GPT-2 + 1 absorbing. We do not
    need the model side to know about a special token (it's just an
    extra column in the score tensor); but the tokenizer-decode path
    must skip it cleanly when we decode samples.
    """
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Register the absorbing token so .decode() doesn't choke.
    tokenizer.add_special_tokens({"mask_token": "<|sedd_mask|>"})
    return tokenizer


def repo_root() -> Path:
    """Convenience for callers that need the project root."""
    return REPO_ROOT


def sedd_dir() -> Path:
    """Convenience for callers that need the vendored SEDD directory."""
    return SEDD_DIR
