#!/usr/bin/env python
"""Direct check of the PoE composition formula on K sequence pairs."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pair: str = typer.Option("len:form", help="Expert pair as 'name_a:name_b'."),
    k_pairs: int = 50,
    num_t_samples: int = 32,
    out_json: Path = Path("artifacts/poe_formula_check.json"),
    out_png: Path = Path("artifacts/plots/poe_formula_check.png"),
) -> None:
    raise NotImplementedError("run after the composition sweep")


if __name__ == "__main__":
    app()
