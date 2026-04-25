import numpy as np

from src.eval.joint_satisfaction import (
    SampleRecord,
    compute_distinct_2,
    compute_thresholds,
    summarize,
)


def test_distinct_2_unique_bigrams():
    assert compute_distinct_2(["a", "b", "c", "d"]) == 1.0


def test_distinct_2_repeated():
    assert compute_distinct_2(["a", "a", "a"]) < 1.0


def test_distinct_2_short_sequences():
    assert compute_distinct_2([]) == 0.0
    assert compute_distinct_2(["a"]) == 0.0


def test_compute_thresholds_quantile():
    th = compute_thresholds({"x": np.arange(100, dtype=float)}, q=0.75)
    assert abs(th["x"] - 74.25) < 1.0


def _record(form: float, sent: float) -> SampleRecord:
    return SampleRecord(
        text="",
        length=10,
        score_len=0.0,
        score_form=form,
        score_sent=sent,
        score_tox=0.0,
        ppl_gpt2=10.0,
        distinct_2=0.5,
    )


def test_summarize_joint_and_marginals():
    samples = [_record(0.9, 0.9), _record(0.9, 0.1), _record(0.1, 0.9), _record(0.1, 0.1)]
    s = summarize(
        samples,
        score_keys=("form", "sent"),
        thresholds={"form": 0.5, "sent": 0.5},
        baseline_distinct_2=0.5,
        baseline_ppl=10.0,
    )
    assert s.joint_satisfaction == 0.25
    assert s.marginal_a == 0.5
    assert s.marginal_b == 0.5
    assert s.mode_collapse_ratio == 1.0
    assert s.fluency_ratio == 1.0


def test_summarize_empty():
    s = summarize(
        [],
        score_keys=("form", "sent"),
        thresholds={"form": 0.5, "sent": 0.5},
        baseline_distinct_2=1.0,
        baseline_ppl=10.0,
    )
    assert s.n_samples == 0
    assert s.joint_satisfaction == 0.0
