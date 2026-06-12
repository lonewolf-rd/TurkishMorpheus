import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.providers.logger_provider import global_logger


CANON_ORDER = ["Morpheus", "Morfessor", "BPE", "ByteBPE", "Unigram", "WordPiece", "TurkishTokenizer"]

COLORS = {
    "Morpheus": "#1F3A5F",
    "TurkishTokenizer": "#B5651D",
    "Morfessor": "#555555",
    "BPE": "#8C8C8C",
    "ByteBPE": "#AAAAAA",
    "Unigram": "#C2C2C2",
    "WordPiece": "#D9D9D9",
}


def _canon(name: str) -> str:
    n = str(name).lower().replace("_", "-")
    if "morpheus" in n:
        return "Morpheus"
    if "morfessor" in n:
        return "Morfessor"
    if "byte" in n:
        return "ByteBPE"
    if n.startswith("bpe"):
        return "BPE"
    if "unigram" in n:
        return "Unigram"
    if "wordpiece" in n:
        return "WordPiece"
    if "turkish-tokenizer" in n or "turkishtokenizer" in n:
        return "TurkishTokenizer"
    return str(name)


def _color(name: str) -> str:
    return COLORS.get(_canon(name), "#999999")


def _order_index(name: str) -> int:
    c = _canon(name)
    return CANON_ORDER.index(c) if c in CANON_ORDER else len(CANON_ORDER)


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 11,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def _hbar(ax, names: Sequence[str], values: Sequence[float], fmt: str,
          lossy: Optional[set] = None) -> None:
    y = np.arange(len(names))
    colors = [_color(n) for n in names]
    bars = ax.barh(y, values, color=colors, edgecolor="#333333", linewidth=0.7)
    if lossy:
        for b, n in zip(bars, names):
            if _canon(n) in lossy:
                b.set_hatch("////")
    ax.set_yticks(y)
    ax.set_yticklabels([_canon(n) for n in names])
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for b, v in zip(bars, values):
        ax.text(b.get_width() + vmax * 0.01, b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", ha="left", fontsize=9, color="#222222")
    ax.set_xlim(0, vmax * 1.15)


def _grouped_bars(ax, names: Sequence[str], series: Dict[str, Sequence[float]],
                  group_labels: Sequence[str]) -> None:
    n_groups = len(group_labels)
    n_series = len(names)
    x = np.arange(n_groups)
    width = 0.8 / max(n_series, 1)
    for i, name in enumerate(names):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, series[name], width=width, label=_canon(name),
               color=_color(name), edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(n_series, 4), loc="upper center", bbox_to_anchor=(0.5, -0.10))


def _sorted_names(names: Sequence[str], values: Sequence[float], ascending: bool):
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=not ascending)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_bpc(lm_summary: Path, out: Path, lossy: Optional[set]) -> None:
    df = pd.read_csv(lm_summary)
    col = "best_bpc" if "best_bpc" in df.columns else "final_bpc"
    names, vals = _sorted_names(df["tokenizer"].tolist(), df[col].tolist(), ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.4f}", lossy=lossy)
    ax.set_title("Bits per character (BPC) — lower is better")
    sub = "hatched = lossy reconstruction (information-destroying)" if lossy else ""
    ax.set_xlabel("BPC" + (f"     {sub}" if sub else ""))
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_roundtrip(summary: Path, out: Path) -> None:
    df = pd.read_csv(summary)
    df["acc_pct"] = df["roundtrip_acc"] * 100
    names, vals = _sorted_names(df["tokenizer"].tolist(), df["acc_pct"].tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.1f}%")
    ax.axvline(100, color="#1F3A5F", linestyle=":", linewidth=1.0)
    ax.set_title("Roundtrip reconstruction accuracy — decode(encode(w)) == w")
    ax.set_xlabel("% of inflected wordforms reconstructed exactly")
    ax.set_xlim(0, 108)
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_morphscore(summary: Path, out: Path) -> None:
    df = pd.read_csv(summary)
    df = df.sort_values("morphscore_recall", ascending=False)
    names = df["model"].tolist()
    metrics = [("morphscore_recall", "Recall"),
               ("morphscore_precision", "Precision"),
               ("macro_f1", "Macro F1")]
    series = {n: [df[df["model"] == n][m].iloc[0] for m, _ in metrics] for n in names}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _grouped_bars(ax, names, series, [lbl for _, lbl in metrics])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.set_title("MorphScore — gold morpheme boundaries (UD_Turkish-Kenet)")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_sigmorphon(summary: Path, out: Path) -> None:
    df = pd.read_csv(summary)
    df = df.sort_values("lemma_prefix_rate", ascending=False)
    names = df["model"].tolist()
    metrics = [("lemma_prefix_rate", "Lemma prefix"),
               ("root_in_segments_rate", "Root in segs"),
               ("first_segment_match_rate", "First seg"),
               ("exact_suffix_count_rate", "Suffix count")]
    series = {n: [df[df["model"] == n][m].iloc[0] for m, _ in metrics] for n in names}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _grouped_bars(ax, names, series, [lbl for _, lbl in metrics])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("SIGMORPHON 2022 Turkish — gold inflection")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_mas_strata(summary: Path, out: Path) -> None:
    df = pd.read_csv(summary)
    strata = df["stratum"].tolist()
    tok_cols = [c for c in df.columns if c.endswith("_mas")]
    names = [c[:-4] for c in tok_cols]
    names = sorted(names, key=_order_index)
    series = {n: df[f"{n}_mas"].tolist() for n in names}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    _grouped_bars(ax, names, series, strata)
    ax.set_ylabel("boundary agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Morphological alignment by stratum (vs Morfessor) — generalization")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_trmmlu(summary: Path, out: Path) -> None:
    df = pd.read_csv(summary)
    df = df.sort_values("pure_pct", ascending=False)
    names = df["tokenizer"].tolist()
    series = {n: [df[df["tokenizer"] == n]["tr_pct"].iloc[0],
                  df[df["tokenizer"] == n]["pure_pct"].iloc[0]] for n in names}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    _grouped_bars(ax, names, series, ["%TR", "%Pure"])
    ax.set_ylabel("% of unique tokens")
    ax.set_ylim(0, 100)
    ax.set_title("TR-MMLU token validity (Kalbur validator)")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_gpu_memory(inference_summary: Path, out: Path) -> None:
    df = pd.read_csv(inference_summary)
    col = "peak_gpu_memory_mb_B32"
    if col not in df.columns:
        return
    names, vals = _sorted_names(df["tokenizer"].tolist(), df[col].tolist(), ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.0f} MB")
    ax.set_title("Peak GPU memory during generation (batch 32) — lower is better")
    ax.set_xlabel("MB")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_gen_throughput(inference_summary: Path, out: Path) -> None:
    df = pd.read_csv(inference_summary)
    col = "chars_per_sec_gen_eff_B1"
    if col not in df.columns:
        return
    names, vals = _sorted_names(df["tokenizer"].tolist(), df[col].tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.0f}")
    ax.set_title("Generation throughput, same param-equalized LM (batch 1) — higher is better")
    ax.set_xlabel("effective chars/sec")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_encode_speed(inference_summary: Path, out: Path) -> None:
    df = pd.read_csv(inference_summary)
    col = "chars_per_sec_encode"
    if col not in df.columns:
        return
    df = df.copy()
    df["k"] = df[col] / 1000.0
    names, vals = _sorted_names(df["tokenizer"].tolist(), df["k"].tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.0f}k")
    ax.set_title("Tokenization (encode) speed — higher is better")
    ax.set_xlabel("kchars/sec")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_decode_speed(roundtrip_summary: Path, out: Path) -> None:
    df = pd.read_csv(roundtrip_summary)
    if "decode_words_per_sec" not in df.columns:
        return
    names, vals = _sorted_names(df["tokenizer"].tolist(),
                                df["decode_words_per_sec"].tolist(), ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(names) + 1.6))
    _hbar(ax, names, vals, "{:.0f}")
    ax.set_title("Detokenization (decode) throughput — higher is better")
    ax.set_xlabel("words/sec")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_qualitative_table(seg_wide: Path, out: Path) -> None:
    df = pd.read_csv(seg_wide)
    want = ["evlerimizdekiler", "kitabımdaki", "gelebilseydik", "gidiyor",
            "muvaffakiyetsizleştiriciler", "görevlendirilemeyeceklerinden",
            "çalışmalarımızdan", "gloplaştırıcılık", "şarpınsızca"]
    cols = [c for c in ["morpheus", "morfessor", "turkish-tokenizer", "bpe-64k", "wordpiece-64k"]
            if c in df.columns]
    body: List[List[str]] = []
    for w in want:
        sub = df[df["word"] == w]
        if sub.empty:
            continue
        r = sub.iloc[0]
        body.append([w] + [str(r[c]) for c in cols])
    if not body:
        return

    headers = ["word"] + [_canon(c) for c in cols]
    fig, ax = plt.subplots(figsize=(2.4 * len(headers), 0.55 * len(body) + 1.4))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=headers, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)
    morph_col = headers.index("Morpheus") if "Morpheus" in headers else -1
    for (rr, cc), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.get_text().set_fontfamily("monospace")
        if rr == 0:
            cell.set_facecolor("#2C2C2C")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif cc == morph_col:
            cell.set_facecolor("#E8EEF5")
    ax.set_title("Qualitative segmentation comparison", fontweight="bold", loc="left")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def plot_pareto(inference_summary: Path, out: Path) -> None:
    df = pd.read_csv(inference_summary)
    gen_col = "chars_per_sec_gen_eff_B1"
    if gen_col not in df.columns or "best_bpc" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    for _, r in df.iterrows():
        name = r["tokenizer"]
        c = _color(name)
        focal = _canon(name) in ("Morpheus", "TurkishTokenizer")
        ax.scatter(r[gen_col], r["best_bpc"], s=130 if focal else 90,
                   color=c, edgecolor="#222222", linewidth=1.0 if focal else 0.5, zorder=3)
        ax.annotate(_canon(name), (r[gen_col], r["best_bpc"]),
                    textcoords="offset points", xytext=(7, 3), fontsize=9,
                    fontweight="bold" if focal else "normal", color="#222222")
    ax.grid(True)
    ax.set_xlabel("generation throughput (chars/sec, B=1) — higher is better →")
    ax.set_ylabel("BPC — lower is better ↓")
    ax.set_title("Quality vs speed (Pareto)")
    fig.savefig(out)
    plt.close(fig)
    global_logger.info(f"[eval_report] {out.name}")


def run(base: Optional[Path] = None) -> Path:
    _setup_style()
    base = base or Path(__file__).resolve().parents[3]
    results = base / "src" / "benchmarker" / "results"
    paper = results / "paper_eval"
    figdir = paper / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    lossy: set = set()
    rt = paper / "roundtrip" / "roundtrip_summary.csv"
    if rt.exists():
        rdf = pd.read_csv(rt)
        for _, r in rdf.iterrows():
            if r["roundtrip_acc"] < 0.999:
                lossy.add(_canon(r["tokenizer"]))

    jobs = [
        (results / "lm_eval" / "full" / "summary.csv",
         lambda p, o: plot_bpc(p, o, lossy), "fig_bpc.png"),
        (rt, plot_roundtrip, "fig_roundtrip.png"),
        (paper / "morphscore" / "morphscore_summary.csv", plot_morphscore, "fig_morphscore.png"),
        (paper / "sigmorphon" / "sigmorphon_summary.csv", plot_sigmorphon, "fig_sigmorphon.png"),
        (paper / "mas_vs_morfessor.csv", plot_mas_strata, "fig_mas_strata.png"),
        (paper / "trmmlu" / "trmmlu_summary.csv", plot_trmmlu, "fig_trmmlu.png"),
        (results / "lm_eval" / "inference" / "inference_summary.csv", plot_pareto, "fig_pareto_bpc_gen.png"),
        (results / "lm_eval" / "inference" / "inference_summary.csv", plot_gpu_memory, "fig_gpu_memory.png"),
        (results / "lm_eval" / "inference" / "inference_summary.csv", plot_gen_throughput, "fig_gen_throughput.png"),
        (results / "lm_eval" / "inference" / "inference_summary.csv", plot_encode_speed, "fig_encode_speed.png"),
        (rt, plot_decode_speed, "fig_decode_speed.png"),
        (paper / "segmentation_comparison_wide.csv", plot_qualitative_table, "fig_qualitative_table.png"),
    ]

    made = 0
    for src, fn, name in jobs:
        if not Path(src).exists():
            global_logger.warning(f"[eval_report] missing, skipped: {src}")
            continue
        try:
            fn(Path(src), figdir / name)
            made += 1
        except Exception as e:
            global_logger.error(f"[eval_report] {name} failed: {e}")
            import traceback
            traceback.print_exc()

    global_logger.info(f"[eval_report] {made} figures written to {figdir}")
    print(f"\nFigures: {figdir.absolute()} ({made} generated)")
    return figdir


def main():
    parser = argparse.ArgumentParser(prog="src.benchmarker.visualization.eval_report")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
