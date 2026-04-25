"""Vertical-specific sub-corpora built from OpenWebText.

For each vertical we want a JSONL file containing OWT documents that score
*high* on the proxy of that vertical, so that fine-tuning a LoRA expert on
the file pushes the model in the desired direction.

The implementation streams OWT once. For each text we score with the
classifiers/dictionaries we still need (a vertical that has reached
``target_size`` is skipped on subsequent texts), so the wall-clock cost
falls as the run progresses.

A run also produces ``cross_table.json`` reporting the mean raw signal of
each proxy on each vertical's accepted documents — this should be
diagonal-dominant (each vertical scores high on its own proxy and not
much higher than baseline on the others). It is the §4.3 sanity check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ..energies import build_default_energies
from ..energies.proxies import Energy


@dataclass(frozen=True)
class VerticalSpec:
    """Filter spec for one expert's training corpus.

    The threshold is expressed in the natural unit of the proxy's
    ``raw_signal``: a token count for length, a probability in [0, 1] for
    classifier-backed proxies, a mean rating in [1, 5] for concreteness.
    """

    name: str
    energy_key: str
    keep_above: float
    target_size: int = 80_000


# Ordered for the §4.3 cross-table; one entry per LoRA expert we will train.
# Thresholds calibrated on a 5000-doc OWT smoke test:
# * long > 700 because OWT articles already average ~700 tokens; the
#   roadmap default of 150 was below the corpus mean and did not actually
#   filter anything.
# * concrete > 2.8 (instead of 3.0) so that the acceptance rate stays
#   above ~3% — at 3.0 the slowest-filling vertical would dominate the
#   wall-clock cost of the build.
DEFAULT_VERTICAL_SPECS: list[VerticalSpec] = [
    VerticalSpec(name="long", energy_key="len", keep_above=700.0),
    VerticalSpec(name="formal", energy_key="form", keep_above=0.75),
    VerticalSpec(name="positive", energy_key="sent", keep_above=0.85),
    VerticalSpec(name="positive2", energy_key="sent2", keep_above=0.70),
    VerticalSpec(name="concrete", energy_key="conc", keep_above=2.80),
    VerticalSpec(name="sports", energy_key="topic", keep_above=0.50),
]


def build_all(
    *,
    out_dir: str | Path = "artifacts/datasets",
    energies: dict[str, Energy] | None = None,
    specs: list[VerticalSpec] = DEFAULT_VERTICAL_SPECS,
    streaming: bool = True,
    max_examples_seen: int = 5_000_000,
    text_max_chars: int = 4096,
    batch_size: int = 32,
) -> dict[str, Path]:
    """Stream OWT, dispatch each document to the verticals it qualifies for.

    Documents are read into batches of ``batch_size`` and every batch is
    scored against all proxies in one pipeline call — the only design
    that gives a realistic GPU throughput, because the per-call overhead
    of HF text-classification pipelines dominates per-text scoring.

    Returns a mapping ``vertical_name → output JSONL path``. Also writes
    ``out_dir / cross_table.json`` with the per-(vertical, proxy) mean
    raw signal for the §4.3 diagonal-dominance check.
    """
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    energies = energies or build_default_energies()

    out_paths = {s.name: out_dir / f"{s.name}.jsonl" for s in specs}
    targets = {s.name: s.target_size for s in specs}
    energy_keys = {s.name: s.energy_key for s in specs}
    thresholds = {s.name: s.keep_above for s in specs}

    counts: dict[str, int] = dict.fromkeys(targets, 0)
    proxy_names = list(energies.keys())
    cross_sums: dict[str, dict[str, float]] = {
        s.name: dict.fromkeys(proxy_names, 0.0) for s in specs
    }

    files = {name: open(path, "w", encoding="utf-8") for name, path in out_paths.items()}
    try:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
        pbar = tqdm(total=sum(targets.values()), desc="documents written")

        batch: list[str] = []
        seen = 0

        def _flush_batch() -> None:
            if not batch:
                return
            # Score the whole batch on every proxy in one shot.
            signals: dict[str, list[float]] = {
                key: energies[key].raw_signals_batch(batch, batch_size=batch_size)
                for key in proxy_names
            }
            for i, text in enumerate(batch):
                for spec in specs:
                    if counts[spec.name] >= targets[spec.name]:
                        continue
                    if signals[spec.energy_key][i] <= spec.keep_above:
                        continue
                    files[spec.name].write(json.dumps({"text": text}) + "\n")
                    counts[spec.name] += 1
                    pbar.update(1)
                    for proxy in proxy_names:
                        cross_sums[spec.name][proxy] += signals[proxy][i]
            batch.clear()

        for ex in ds:
            if seen >= max_examples_seen:
                break
            if all(counts[n] >= targets[n] for n in counts):
                break
            text = ex.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            batch.append(text[:text_max_chars])
            seen += 1
            if len(batch) >= batch_size:
                _flush_batch()
        _flush_batch()  # tail
        pbar.close()
    finally:
        for f in files.values():
            f.close()

    # Drop empty files so callers can rely on the returned mapping.
    out_paths = {n: p for n, p in out_paths.items() if counts[n] > 0}

    cross_table = {
        n: {proxy: (s / counts[n] if counts[n] else 0.0) for proxy, s in sums.items()}
        for n, sums in cross_sums.items()
    }
    (out_dir / "cross_table.json").write_text(
        json.dumps(
            {
                "counts": counts,
                "thresholds": thresholds,
                "energy_keys": energy_keys,
                "cross_table_mean_raw_signal": cross_table,
            },
            indent=2,
        )
    )
    return out_paths
