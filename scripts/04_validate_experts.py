#!/usr/bin/env python
"""Cross-vertical validation of the four fine-tuned experts.

For each expert, generate samples and score them with the four proxies; the
resulting matrix should be diagonal-dominant.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoints_dir: Path = Path("artifacts/checkpoints"),
    out_json: Path = Path("artifacts/expert_validation.json"),
    n_samples: int = 128,
    max_new_tokens: int = 128,
) -> None:
    raise NotImplementedError("run after expert checkpoints exist")


if __name__ == "__main__":
    app()
