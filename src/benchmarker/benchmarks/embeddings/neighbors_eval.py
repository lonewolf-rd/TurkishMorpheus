import sys
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.embeddings.encoders import build_encoders, find_checkpoint
from src.benchmarker.benchmarks.embeddings.gold_data import load_root_families, default_gold_path


FALLBACK_FAMILIES: Dict[str, List[str]] = {
    "kitap": ["kitap", "kitaplar", "kitabı", "kitabımız", "kitaplıkta", "kitapçı"],
    "ev": ["ev", "evler", "evimiz", "evlerde", "evcil", "evsiz"],
    "göz": ["göz", "gözler", "gözlük", "gözcü", "gözlemci", "gözünde"],
    "su": ["su", "sular", "suluk", "susuz", "suda", "sucu"],
    "yol": ["yol", "yollar", "yolcu", "yolculuk", "yolda", "yolsuz"],
    "demir": ["demir", "demirci", "demircilik", "demirden", "demirler", "demirli"],
    "balık": ["balık", "balıkçı", "balıkçılık", "balıklar", "balığı", "balıksız"],
    "orman": ["orman", "ormancı", "ormancılık", "ormanlar", "ormanda", "ormanlık"],
}


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return norm @ norm.T


def cluster_purity_at_k(vecs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    sim = cosine_matrix(vecs)
    np.fill_diagonal(sim, -np.inf)
    hits = 0
    total = 0
    for i in range(len(vecs)):
        topk = np.argsort(-sim[i])[:k]
        hits += int((labels[topk] == labels[i]).sum())
        total += k
    return hits / max(total, 1)


def _resolve_families() -> Dict[str, List[str]]:
    gold = default_gold_path()
    if gold.exists():
        fams = load_root_families(str(gold))
        if fams:
            return fams
    global_logger.warning("[neighbors_eval] SIGMORPHON gold unavailable; using fallback families.")
    return FALLBACK_FAMILIES


def run(include_berturk: bool = True, k: int = 5, make_umap: bool = True) -> None:
    base = Path(__file__).resolve().parents[4]
    artifacts = base / "src" / "model_development" / "artifacts"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "embeddings" / "neighbors"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[neighbors_eval] No Morpheus checkpoint found.")
        sys.exit(1)

    families = _resolve_families()
    roots = sorted(families)
    words: List[str] = []
    label_names: List[str] = []
    for root in roots:
        for w in families[root]:
            words.append(w)
            label_names.append(root)
    label_ids = np.array([roots.index(n) for n in label_names])

    encoders = build_encoders(str(checkpoint), include_berturk=include_berturk)

    summary_rows = []
    for enc in encoders:
        vecs = enc.encode_words(words)
        purity = cluster_purity_at_k(vecs, label_ids, k=k)
        summary_rows.append({"encoder": enc.name, "dim": enc.dim, f"purity_at_{k}": round(purity, 4)})
        global_logger.info(f"[neighbors_eval] {enc.name}: root purity@{k} = {purity:.4f}")
        _save_coords(vecs, label_names, words, enc.name, output_dir)
        if make_umap:
            _plot_umap(vecs, label_names, words, enc.name, output_dir)

    summary_path = output_dir / "neighbors_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 70)
    print(f"NEAREST-NEIGHBOUR ROOT COHERENCE  (k={k}, {len(roots)} root families)")
    print("=" * 70)
    for r in summary_rows:
        print(f"  {r['encoder']:<12s} dim={r['dim']:<4d} purity@{k}={r[f'purity_at_{k}']:.4f}")
    print("=" * 70)
    print(f"Summary: {summary_path}")
    print(f"Artifacts: {output_dir}")


def _save_coords(vecs, label_names, words, enc_name, output_dir) -> None:
    try:
        from sklearn.manifold import TSNE
    except Exception as e:
        global_logger.warning(f"[neighbors_eval] sklearn TSNE unavailable: {e}")
        return
    perplexity = min(30, max(5, len(vecs) // 4))
    coords = TSNE(n_components=2, metric="cosine", init="pca",
                  perplexity=perplexity, random_state=42).fit_transform(vecs)
    path = output_dir / f"tsne_coords_{enc_name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "root", "x", "y"])
        for (x, y), name, word in zip(coords, label_names, words):
            writer.writerow([word, name, f"{x:.6f}", f"{y:.6f}"])


def _plot_umap(vecs, label_names, words, enc_name, output_dir) -> None:
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        global_logger.warning(f"[neighbors_eval] UMAP/matplotlib unavailable: {e}")
        return

    n_neighbors = min(8, max(2, len(vecs) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42)
    emb2d = reducer.fit_transform(vecs)

    roots = sorted(set(label_names))
    cmap = plt.get_cmap("tab20")
    color_of = {r: cmap(i % 20) for i, r in enumerate(roots)}

    fig, ax = plt.subplots(figsize=(11, 9))
    for (x, y), name, word in zip(emb2d, label_names, words):
        ax.scatter(x, y, color=color_of[name], s=55, alpha=0.85)
        ax.annotate(word, (x, y), fontsize=6, alpha=0.7)
    ax.set_title(f"{enc_name} — root-family clustering (UMAP, cosine)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / f"umap_{enc_name}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
