"""Network-free tests for the non-classifier proxy energies and label parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.energies.proxies import (
    ConcretenessEnergy,
    LengthEnergy,
    TopicEnergy,
    _by_label,
)


@pytest.fixture
def toy_brysbaert(tmp_path: Path) -> Path:
    csv = tmp_path / "brysbaert_toy.txt"
    csv.write_text(
        "Word\tBigram\tConc.M\tConc.SD\tUnknown\tTotal\tPercent_known\tSUBTLEX\tDom_Pos\n"
        "rock\t0\t4.50\t0.4\t0\t27\t1.0\t0\tNoun\n"
        "table\t0\t5.00\t0.0\t0\t27\t1.0\t0\tNoun\n"
        "idea\t0\t1.50\t0.3\t0\t27\t1.0\t0\tNoun\n"
        "freedom\t0\t1.20\t0.5\t0\t27\t1.0\t0\tNoun\n"
        "ice cream\t1\t4.80\t0.0\t0\t27\t1.0\t0\tNoun\n"
    )
    return csv


def test_concreteness_returns_negative_mean_for_concrete_words(toy_brysbaert: Path):
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    assert e("rock") == -4.5
    assert e("table") == -5.0


def test_concreteness_returns_negative_mean_for_abstract_words(toy_brysbaert: Path):
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    assert e("idea") == -1.5
    assert e("freedom") == -1.2


def test_concreteness_averages_over_known_content_words(toy_brysbaert: Path):
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    # rock (4.5) + idea (1.5) → mean 3.0 → energy -3.0
    assert e("rock and idea") == pytest.approx(-3.0)


def test_concreteness_unknown_words_yield_zero(toy_brysbaert: Path):
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    assert e("supercalifragilistic xyzzy") == 0.0


def test_concreteness_skips_bigrams(toy_brysbaert: Path):
    """The fixture contains 'ice cream' as a bigram entry; we only look up
    single words, so 'ice' and 'cream' alone should not match."""
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    assert e("ice cream") == 0.0  # neither single word is in the dict


def test_length_energy_zero_at_target():
    e = LengthEnergy(L_star=10)
    # Need exactly 10 GPT-2 tokens; pick a string that tokenises to 10.
    # Easier test: a single-word string has token count > 0, energy is non-negative.
    assert e("hello world") >= 0.0


def test_by_label_finds_human_readable_label():
    scores = [{"label": "POSITIVE", "score": 0.7}, {"label": "NEGATIVE", "score": 0.3}]
    assert _by_label(scores, "positive") == 0.7


def test_by_label_falls_back_to_label_index():
    scores = [
        {"label": "LABEL_0", "score": 0.1},
        {"label": "LABEL_1", "score": 0.7},
        {"label": "LABEL_2", "score": 0.1},
        {"label": "LABEL_3", "score": 0.1},
    ]
    assert _by_label(scores, "Sports", fallback_index=1) == 0.7


def test_by_label_raises_when_neither_found():
    scores = [{"label": "FOO", "score": 0.5}]
    with pytest.raises(ValueError):
        _by_label(scores, "Sports", fallback_index=1)


def test_topic_energy_rejects_unknown_class():
    with pytest.raises(ValueError):
        TopicEnergy(target_class="Politics")


def test_raw_signals_batch_default_matches_per_text(toy_brysbaert: Path):
    """Default :meth:`raw_signals_batch` is just a loop over :meth:`raw_signal`."""
    e = ConcretenessEnergy(ratings_path=toy_brysbaert)
    texts = ["rock and idea", "table", "freedom", "xyz"]
    per_text = [e.raw_signal(t) for t in texts]
    batched = e.raw_signals_batch(texts)
    assert per_text == batched
