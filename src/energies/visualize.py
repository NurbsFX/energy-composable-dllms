"""Heatmap and pairwise-scatter plots for the proxy Gram matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_gram_heatmap(
    G: np.ndarray,
    names: list[str],
    out_path: str | Path,
    *,
    title: str = "Empirical Gram matrix",
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        G,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        center=0,
        cbar_kws={"label": r"Cov$(E_i, E_j)$"},
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pair_scatters(
    E_matrix: np.ndarray,
    names: list[str],
    out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(E_matrix, columns=names)
    grid = sns.pairplot(df, plot_kws={"alpha": 0.3, "s": 6}, corner=True)
    grid.fig.suptitle(r"Pairwise proxy energies on $p_{\mathrm{base}}$ samples", y=1.02)
    grid.fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(grid.fig)
