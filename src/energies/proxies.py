"""Proxy energies for the project's verticals.

Each :class:`Energy` maps a piece of text to a scalar; a *low* value means a
*strong* fit to the vertical. We deliberately use external pretrained
classifiers (or, for length and concreteness, no neural model at all) rather
than the DLLMs we will fine-tune, so that the orthogonality index κ derived
from these energies characterises the vertical itself rather than a
particular model.

The default set is six energies — ``len, form, sent, sent2, conc, topic`` —
where ``sent2`` is the §3.7 redundant proxy that anchors a high-κ pair into
the Gram matrix and ``conc`` / ``topic`` replace the original ``tox`` axis,
which had near-zero variance on OpenWebText.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

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
        # max_length is set explicitly because some HF tokenizer configs
        # (e.g. cardiffnlp/twitter-roberta-base-sentiment-latest) do not
        # declare model_max_length, in which case `truncation=True` alone
        # does not actually truncate and the model receives more tokens
        # than its position-embedding table can index.
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512,
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


class SentimentEnergyV2(_ClassifierEnergy):
    """Second sentiment classifier with a different decision boundary.

    Deliberately redundant with :class:`SentimentEnergy`. Introduces a
    high-κ pair (``sent × sent2``) into the Gram matrix so that the
    κ-vs-deficit experiment has a non-trivial gradient on its X axis when
    the four base verticals turn out near-orthogonal on the target corpus
    (the §3.7 fallback path).
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ):
        super().__init__(model_name=model_name, name="sent2")

    def _prob(self, scores: list[dict]) -> float:
        return _by_label(scores, "positive")


class TopicEnergy(_ClassifierEnergy):
    """E_topic(x) = -logit(P(target_class | x)) via an AG News classifier.

    Default target is ``Sports`` because it is strongly discriminative on
    OpenWebText (sports vocabulary is very distinct from the other AG News
    classes). The fabriceyhc model emits ``LABEL_0`` … ``LABEL_3`` rather
    than human-readable names, so the energy class also passes a numeric
    fallback index to :func:`_by_label`.
    """

    AG_NEWS_INDICES: dict[str, int] = {
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Sci/Tech": 3,
    }

    def __init__(
        self,
        model_name: str = "fabriceyhc/bert-base-uncased-ag_news",
        target_class: str = "Sports",
    ):
        if target_class not in self.AG_NEWS_INDICES:
            raise ValueError(
                f"unknown AG News class {target_class!r}; "
                f"expected one of {sorted(self.AG_NEWS_INDICES)}"
            )
        super().__init__(model_name=model_name, name="topic")
        self.target_class = target_class
        self._target_index = self.AG_NEWS_INDICES[target_class]

    def _prob(self, scores: list[dict]) -> float:
        return _by_label(scores, self.target_class, fallback_index=self._target_index)


class ConcretenessEnergy(Energy):
    """E_conc(x) = -mean concreteness via Brysbaert et al. 2014 ratings.

    Pure dictionary lookup over the Brysbaert/Warriner/Kuperman 2014 norms
    (~40k English lemmas, scored 1=abstract to 5=concrete by crowdsourcing).
    Low energy = highly concrete text. The ratings file is fetched on
    first use into ``~/.cache/composable-dllms/`` so the dependency is
    transparent and the package itself stays small.

    Methodologically this proxy has the appeal of being entirely
    non-neural: there is no classifier to debug or to suspect of bias —
    only crowdsourced lexical scores.
    """

    name: str = "conc"

    # Springer journal supplementary; stable URL associated with the DOI.
    BRYSBAERT_URL = (
        "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5"
        "/MediaObjects/13428_2013_403_MOESM1_ESM.xlsx"
    )
    BRYSBAERT_FILENAME = "brysbaert_concreteness_2014.xlsx"

    def __init__(self, ratings_path: Path | None = None):
        self._ratings_path = ratings_path
        self._scores: dict[str, float] | None = None

    def _ensure(self) -> None:
        if self._scores is not None:
            return
        path = self._ratings_path or _ensure_brysbaert_ratings(
            self.BRYSBAERT_URL, self.BRYSBAERT_FILENAME
        )
        self._scores = _load_brysbaert_ratings(path)

    def __call__(self, text: str) -> float:
        self._ensure()
        assert self._scores is not None
        words = re.findall(r"\b[a-z]+\b", text.lower())
        scores = [self._scores[w] for w in words if w in self._scores]
        if not scores:
            return 0.0
        return -float(np.mean(scores))


def _load_brysbaert_ratings(path: Path) -> dict[str, float]:
    """Parse the Brysbaert ratings file into ``{word: score}``.

    Supports both the original tab-separated text export and the official
    Springer xlsx supplementary; the format is dispatched on the file
    extension so tests can exercise the loader without an xlsx dependency
    on a tiny fixture.
    """
    import pandas as pd

    if str(path).lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep="\t")
    # Single-word entries only; the original file marks bigrams with Bigram=1.
    df = df[df["Bigram"] == 0]
    return dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"].astype(float), strict=True))


def _ensure_brysbaert_ratings(url: str, filename: str) -> Path:
    """Lazy download of the Brysbaert ratings into the user cache directory."""
    cache_dir = Path.home() / ".cache" / "composable-dllms"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve(url, path)  # noqa: S310 — Springer journal URL
    return path


def _by_label(
    scores: list[dict],
    label: str,
    *,
    fallback_index: int | None = None,
) -> float:
    """Find the score for ``label`` in a HF text-classification output.

    If ``fallback_index`` is given and the human-readable label is not in
    ``scores``, also try ``LABEL_<fallback_index>`` for models that ship
    without an ``id2label`` mapping (e.g. the AG News fine-tunes on the
    Hub).
    """
    target = label.lower()
    for s in scores:
        if s["label"].lower() == target:
            return float(s["score"])
    if fallback_index is not None:
        fallback = f"label_{fallback_index}"
        for s in scores:
            if s["label"].lower() == fallback:
                return float(s["score"])
    raise ValueError(f"label {label!r} not in {[s['label'] for s in scores]}")


def build_default_energies() -> dict[str, Energy]:
    """The six proxy energies, in the order used by the Gram matrix.

    * ``len, form, sent`` — three style verticals.
    * ``sent2`` — §3.7 redundant sentiment proxy that anchors a high-κ
      pair into the Gram matrix (``sent × sent2``).
    * ``conc`` — concreteness from Brysbaert 2014 ratings; non-neural,
      no classifier biases to worry about.
    * ``topic`` — Sports topic via an AG News classifier.

    The previous ``tox`` vertical was dropped because ``Var(E_tox)`` is
    near zero on OpenWebText and the resulting "non-toxic expert" would
    have been indistinguishable from the backbone.
    """
    return {
        "len": LengthEnergy(),
        "form": FormalityEnergy(),
        "sent": SentimentEnergy(),
        "sent2": SentimentEnergyV2(),
        "conc": ConcretenessEnergy(),
        "topic": TopicEnergy(),
    }
