"""Vertical-specific sub-corpora built from OpenWebText.

For each vertical, stream OWT, score every document with the relevant proxy
classifier, and keep documents that pass the filter until ``target_size``
matches have been written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..energies.proxies import Energy


@dataclass(frozen=True)
class VerticalSpec:
    name: str
    energy_key: str
    keep_below: float | None  # raw proxy probability threshold; low = match
    keep_above: float | None  # raw proxy probability threshold; high = match
    target_size: int = 80_000


# The "long" vertical applies a token-length filter directly rather than a
# probability threshold; build_all is responsible for handling that case.
DEFAULT_VERTICAL_SPECS: list[VerticalSpec] = [
    VerticalSpec(name="long", energy_key="len", keep_below=None, keep_above=None),
    VerticalSpec(name="formal", energy_key="form", keep_below=None, keep_above=0.75),
    VerticalSpec(name="positive", energy_key="sent", keep_below=None, keep_above=0.85),
    VerticalSpec(name="nontoxic", energy_key="tox", keep_below=0.10, keep_above=None),
]


def build_all(
    *,
    out_dir: str | Path = "artifacts/datasets",
    energies: dict[str, Energy] | None = None,
    specs: list[VerticalSpec] = DEFAULT_VERTICAL_SPECS,
    streaming: bool = True,
    max_examples_seen: int = 5_000_000,
) -> dict[str, Path]:
    """Score and dispatch OWT documents across the requested verticals."""
    raise NotImplementedError
