import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.providers.logger_provider import global_logger


COLORS = {
    "morpheus": "#1F3A5F",
    "berturk": "#B5651D",
}


def _color(name: str) -> str:
    return COLORS.get(str(name).lower(), "#8C8C8C")


def plot_tsne(coords_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(coords_csv)
    roots = sorted(df["root"].unique())
    cmap = plt.get_cmap("tab20")
    color_of = {r: cmap(i % 20) for i, r in enumerate(roots)}

    fig, ax = plt.subplots(figsize=(11, 9))
    for _, row in df.iterrows():
        ax.scatter(row["x"], row["y"], color=color_of[row["root"]], s=55, alpha=0.85)
        ax.annotate(str(row["word"]), (row["x"], row["y"]), fontsize=6, alpha=0.7)
    ax.set_title(f"t-SNE — root-family embedding clustering ({coords_csv.stem})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    global_logger.info(f"[embedding_report] Wrote {out_path}")


def _grouped_bar(df: pd.DataFrame, value_col: str, group_col: str, label_col: str,
                 title: str, ylabel: str, out_path: Path) -> None:
    groups = list(df[group_col].unique())
    labels = list(df[label_col].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.6), 5))
    for i, lab in enumerate(labels):
        sub = df[df[label_col] == lab].set_index(group_col)
        vals = [float(sub.loc[g, value_col]) if g in sub.index else 0.0 for g in groups]
        ax.bar(x + i * width, vals, width, label=str(lab), color=_color(lab))
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    global_logger.info(f"[embedding_report] Wrote {out_path}")


def _simple_bar(df: pd.DataFrame, value_col: str, label_col: str,
                title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = df[label_col].tolist()
    vals = df[value_col].astype(float).tolist()
    ax.bar(labels, vals, color=[_color(l) for l in labels])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    global_logger.info(f"[embedding_report] Wrote {out_path}")


def run(results_root: Optional[str] = None) -> None:
    base = Path(__file__).resolve().parents[3]
    root = Path(results_root) if results_root else (
        base / "src" / "benchmarker" / "results" / "paper_eval" / "embeddings"
    )
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    neighbors = root / "neighbors"
    if neighbors.exists():
        for coords in sorted(neighbors.glob("tsne_coords_*.csv")):
            enc = coords.stem.replace("tsne_coords_", "")
            plot_tsne(coords, fig_dir / f"tsne_{enc}.png")
        summ = neighbors / "neighbors_summary.csv"
        if summ.exists():
            df = pd.read_csv(summ)
            value_col = next((c for c in df.columns if c.startswith("purity_at_")), None)
            if value_col:
                _simple_bar(df, value_col, "encoder",
                            "Root-family neighbour purity", value_col, fig_dir / "neighbors_purity.png")

    probing = root / "probing" / "probing_summary.csv"
    if probing.exists():
        df = pd.read_csv(probing)
        _grouped_bar(df, "probe_acc", "task", "encoder",
                     "Morphological probing accuracy", "linear probe acc",
                     fig_dir / "probing_accuracy.png")

    ner = root / "ner" / "ner_summary.csv"
    if ner.exists():
        df = pd.read_csv(ner)
        melted = df.melt(id_vars=["encoder"], value_vars=["macro_f1", "entity_micro_f1"],
                         var_name="metric", value_name="score")
        _grouped_bar(melted, "score", "metric", "encoder",
                     "WikiANN-tr NER probe", "F1", fig_dir / "ner_f1.png")

    print()
    print("=" * 70)
    print("EMBEDDING REPORT FIGURES")
    print("=" * 70)
    for p in sorted(fig_dir.glob("*.png")):
        print(f"  {p}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()
    run(results_root=args.results_root)
