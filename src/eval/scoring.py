"""Per-sample scoring helpers used by Phase 3.5, 4 and 4.5.

A generated sample is scored on:
* the six proxy energies → ``score_<key>`` fields (raw signal in [0, 1] for
  classifier-backed proxies, [1, 5] for concreteness, token count for length).
* GPT-2 perplexity → ``ppl_gpt2`` (fluency proxy independent of the experts).
* distinct-2 → fraction of unique bigrams (mode-collapse signal).

The :class:`SampleScorer` caches the GPT-2 model so the cost is amortised
over a sweep, and uses the same lazy-loaded proxy energies as Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .joint_satisfaction import SampleRecord, compute_distinct_2

if TYPE_CHECKING:
    from ..energies.proxies import Energy


@dataclass
class SampleScorer:
    energies: dict[str, Energy]
    gpt2_model_name: str = "gpt2"
    _gpt2_model: object = None
    _gpt2_tokenizer: object = None

    def _ensure_gpt2(self) -> None:
        if self._gpt2_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._gpt2_tokenizer = AutoTokenizer.from_pretrained(self.gpt2_model_name)
        self._gpt2_model = AutoModelForCausalLM.from_pretrained(self.gpt2_model_name)
        self._gpt2_model.eval()
        if torch.cuda.is_available():
            self._gpt2_model = self._gpt2_model.cuda()

    def gpt2_ppl(self, text: str) -> float:
        """Sentence-level GPT-2 perplexity. Lower = more fluent under GPT-2."""
        import math

        import torch

        self._ensure_gpt2()
        tok = self._gpt2_tokenizer
        ids = tok.encode(text, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            ids = ids.cuda()
        if ids.shape[1] < 2:
            return float("nan")
        with torch.no_grad():
            out = self._gpt2_model(ids, labels=ids)
        return float(math.exp(out.loss.item()))

    def score(self, text: str) -> SampleRecord:
        """Single-sample scoring: proxies + GPT-2 PPL + distinct-2."""
        proxy_scores: dict[str, float] = {
            key: float(e.raw_signal(text)) for key, e in self.energies.items()
        }
        length = int(proxy_scores["len"]) if "len" in proxy_scores else len(text.split())
        return SampleRecord(
            text=text,
            length=length,
            proxy_scores=proxy_scores,
            ppl_gpt2=self.gpt2_ppl(text),
            distinct_2=compute_distinct_2(text.split()),
        )
