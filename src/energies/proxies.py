"""Proxy energies for the four verticals (length, formality, sentiment, toxicity).

Each :class:`Energy` maps a piece of text to a scalar; a *low* value means a
*strong* fit to the vertical. We deliberately use external pretrained
classifiers rather than the DLLMs we will fine-tune, so that the orthogonality
index κ derived from these energies characterises the vertical itself rather
than a particular model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Energy(ABC):
    name: str

    @abstractmethod
    def __call__(self, text: str) -> float: ...


class LengthEnergy(Energy):
    """E_len(x) = |log(L(x) / L_star)| with L(x) the GPT-2 token count."""

    name = "len"

    def __init__(self, L_star: int = 100, tokenizer_name: str = "gpt2"):
        from transformers import AutoTokenizer

        self.L_star = L_star
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, text: str) -> float:
        n = max(len(self._tok.encode(text, add_special_tokens=False)), 1)
        return float(abs(np.log(n / self.L_star)))


class _ClassifierEnergy(Energy):
    """E(x) = -logit(p) where p comes from a HuggingFace text classifier.

    Subclasses override :meth:`_prob` to pick which class probability to read.
    The classifier is loaded lazily so importing this module is cheap.
    """

    def __init__(self, model_name: str, name: str, *, device: str | None = None):
        self.model_name = model_name
        self.name = name
        self._device = device
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            device=0 if device == "cuda" else -1,
            truncation=True,
        )

    def _prob(self, scores: list[dict]) -> float:
        raise NotImplementedError

    @staticmethod
    def _neg_logit(p: float, eps: float = 1e-6) -> float:
        p = float(min(max(p, eps), 1.0 - eps))
        return float(-np.log(p / (1.0 - p)))

    def __call__(self, text: str) -> float:
        self._ensure()
        return self._neg_logit(self._prob(self._pipe(text)[0]))


class FormalityEnergy(_ClassifierEnergy):
    def __init__(self, model_name: str = "s-nlp/roberta-base-formality-ranker"):
        super().__init__(model_name=model_name, name="form")

    def _prob(self, scores: list[dict]) -> float:
        return _by_label(scores, "formal")


class SentimentEnergy(_ClassifierEnergy):
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__(model_name=model_name, name="sent")

    def _prob(self, scores: list[dict]) -> float:
        return _by_label(scores, "POSITIVE")


class ToxicityEnergy(_ClassifierEnergy):
    """E_tox(x) = -logit(P(non-toxic)). Low energy = non-toxic."""

    def __init__(self, model_name: str = "unitary/toxic-bert"):
        super().__init__(model_name=model_name, name="tox")

    def _prob(self, scores: list[dict]) -> float:
        return 1.0 - _by_label(scores, "toxic")


def _by_label(scores: list[dict], label: str) -> float:
    target = label.lower()
    for s in scores:
        if s["label"].lower() == target:
            return float(s["score"])
    raise ValueError(f"label {label!r} not in {[s['label'] for s in scores]}")


def build_default_energies() -> dict[str, Energy]:
    """The canonical four energies, in the order used by the Gram matrix."""
    return {
        "len": LengthEnergy(),
        "form": FormalityEnergy(),
        "sent": SentimentEnergy(),
        "tox": ToxicityEnergy(),
    }
