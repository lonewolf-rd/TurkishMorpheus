import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def collect_runs(runs_dir: str, glob_pattern: str = "*") -> List[Tuple[str, Path]]:
    runs: List[Tuple[str, Path]] = []
    base = Path(runs_dir)
    if not base.exists():
        print(f"[ScaleAblation] runs_dir not found: {runs_dir}")
        return runs

    for sub in sorted(base.glob(glob_pattern)):
        if not sub.is_dir():
            continue
        summary = sub / "summary_metrics.csv"
        if summary.exists():
            runs.append((sub.name, summary))
    return runs


def merge_runs(runs: List[Tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for run_name, summary_path in runs:
        df = pd.read_csv(summary_path)
        df["run"] = run_name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pivot_metric(merged: pd.DataFrame, category: str, metric: str, model: str = "morpheus_trained") -> pd.DataFrame:
    sub = merged[
        (merged["category"] == category)
        & (merged["metric"] == metric)
        & (merged["model"] == model)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index="run",
        columns="stratum",
        values="value",
        aggfunc="first",
    ).reset_index()
    return pivot


def plot_scale_curve(
        merged: pd.DataFrame,
        category: str,
        metric: str,
        output_path: str,
        title: str = "",
        model: str = "morpheus_trained",
        run_order: Optional[List[str]] = None,
):
    pivot = pivot_metric(merged, category, metric, model)
    if pivot.empty:
        print(f"[ScaleAblation] No data for {category}/{metric}, skipping plot")
        return

    if run_order is not None:
        pivot["__order"] = pivot["run"].apply(
            lambda r: run_order.index(r) if r in run_order else len(run_order)
        )
        pivot = pivot.sort_values("__order").drop(columns=["__order"])
    else:
        pivot = pivot.sort_values("run")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))
    stratum_cols = [c for c in pivot.columns if c != "run"]
    for col in stratum_cols:
        ax.plot(pivot["run"], pivot[col], marker="o", label=col, linewidth=2)

    ax.set_title(title or f"{category} — {metric}")
    ax.set_xlabel("Training run (data scale)")
    ax.set_ylabel(metric)
    ax.legend(title="stratum", loc="best")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ScaleAblation] Plot saved: {output_path}")


def model_comparison_at_run(merged: pd.DataFrame, run: str, category: str, metric: str) -> pd.DataFrame:
    sub = merged[
        (merged["run"] == run)
        & (merged["category"] == category)
        & (merged["metric"] == metric)
    ].copy()
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(
        index="stratum",
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()
    return pivot


def run_comparison(
        runs_dir: str,
        output_dir: str,
        run_order: Optional[List[str]] = None,
):
    runs = collect_runs(runs_dir)
    if not runs:
        print(f"[ScaleAblation] No runs found in {runs_dir}")
        return

    print(f"[ScaleAblation] Found {len(runs)} runs: {[r[0] for r in runs]}")

    merged = merge_runs(runs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    merged_path = out / "merged_metrics.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[ScaleAblation] Merged metrics: {merged_path}")

    coh_pivot = pivot_metric(merged, "coherence", "delta")
    if not coh_pivot.empty:
        coh_pivot.to_csv(out / "scale_root_coherence_delta.csv", index=False)
        plot_scale_curve(
            merged, "coherence", "delta",
            str(out / "scale_root_coherence_delta.png"),
            title="Root cluster coherence (intra - inter) vs data scale",
            run_order=run_order,
        )

    f1_pivot = pivot_metric(merged, "boundary_vs_morfessor", "f1")
    if not f1_pivot.empty:
        f1_pivot.to_csv(out / "scale_boundary_f1.csv", index=False)
        plot_scale_curve(
            merged, "boundary_vs_morfessor", "f1",
            str(out / "scale_boundary_f1.png"),
            title="Morpheus boundary F1 (vs Morfessor) vs data scale",
            run_order=run_order,
        )

    fert_pivot = pivot_metric(merged, "fertility", "tokens_per_word", model="morpheus")
    if not fert_pivot.empty:
        fert_pivot.to_csv(out / "scale_morpheus_fertility.csv", index=False)
        plot_scale_curve(
            merged, "fertility", "tokens_per_word",
            str(out / "scale_morpheus_fertility.png"),
            title="Morpheus fertility (tokens/word) vs data scale",
            model="morpheus",
            run_order=run_order,
        )

    mas_pivot = pivot_metric(merged, "mas_vs_morfessor", "boundary_agreement_pct", model="morpheus")
    if not mas_pivot.empty:
        mas_pivot.to_csv(out / "scale_morpheus_mas.csv", index=False)
        plot_scale_curve(
            merged, "mas_vs_morfessor", "boundary_agreement_pct",
            str(out / "scale_morpheus_mas.png"),
            title="Morpheus MAS (% agreement with Morfessor) vs data scale",
            model="morpheus",
            run_order=run_order,
        )

    print(f"\n[ScaleAblation] All artifacts written to: {out.absolute()}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.parent
    runs_dir = str(base / "src/model_development/paper_eval_results_ablation")
    output_dir = str(base / "src/model_development/paper_eval_results_ablation/_comparison")

    run_order = [
        "scale_500MB",
        "scale_1GB",
        "scale_2_5GB",
        "scale_5GB",
    ]

    run_comparison(runs_dir, output_dir, run_order=run_order)
