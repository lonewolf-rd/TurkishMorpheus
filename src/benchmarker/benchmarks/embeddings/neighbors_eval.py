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


def retrieval_metrics(vecs: np.ndarray, labels: np.ndarray, k: int = 5) -> Dict[str, float]:
    sim = cosine_matrix(vecs)
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)

    purity_hits = purity_total = 0
    precR_sum = ap_sum = mrr_sum = r1_sum = r5_sum = 0.0
    n = len(vecs)
    for i in range(n):
        ranked = order[i]
        rel = (labels[ranked] == labels[i])
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue

        topk = ranked[:k]
        purity_hits += int((labels[topk] == labels[i]).sum())
        purity_total += k

        precR_sum += float(rel[:n_rel].sum()) / n_rel
        r1_sum += float(rel[0])
        r5_sum += float(rel[:5].sum()) / min(5, n_rel)

        first_hit = int(np.argmax(rel)) if rel.any() else -1
        mrr_sum += 1.0 / (first_hit + 1) if first_hit >= 0 else 0.0

        cum = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1)
        precisions = cum / ranks
        ap_sum += float((precisions * rel).sum()) / n_rel

    return {
        "MAP": ap_sum / max(n, 1),
        "MRR": mrr_sum / max(n, 1),
        "recall_at_1": r1_sum / max(n, 1),
        "recall_at_5": r5_sum / max(n, 1),
        "precision_at_R": precR_sum / max(n, 1),
        f"purity_at_{k}": purity_hits / max(purity_total, 1),
    }


def _resolve_families() -> Dict[str, List[str]]:
    gold = default_gold_path()
    if gold.exists():
        fams = load_root_families(str(gold))
        if fams:
            return fams
    global_logger.warning("[neighbors_eval] SIGMORPHON gold unavailable; using fallback families.")
    return FALLBACK_FAMILIES


def run(include_berturk: bool = True, k: int = 5) -> None:
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

    sizes = [len(families[r]) for r in roots]
    summary_rows = []
    for enc in encoders:
        vecs = enc.encode_words(words)
        m = retrieval_metrics(vecs, label_ids, k=k)
        row = {"encoder": enc.name, "dim": enc.dim}
        row.update({key: round(val, 4) for key, val in m.items()})
        summary_rows.append(row)
        global_logger.info(
            f"[neighbors_eval] {enc.name}: MAP={m['MAP']:.4f} MRR={m['MRR']:.4f} "
            f"R@1={m['recall_at_1']:.4f} R@5={m['recall_at_5']:.4f}"
        )
        _save_coords(vecs, label_names, words, enc.name, output_dir)

    summary_path = output_dir / "neighbors_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 78)
    print(f"ROOT-FAMILY RETRIEVAL  (lexical / keyword-style)")
    print(f"  {len(roots)} root families, {len(words)} words, "
          f"family size min/mean/max = {min(sizes)}/{np.mean(sizes):.1f}/{max(sizes)}")
    print("=" * 78)
    print(f"  {'encoder':<12s} {'MAP':>8s} {'MRR':>8s} {'R@1':>8s} {'R@5':>8s} {'P@R':>8s}")
    for r in summary_rows:
        print(f"  {r['encoder']:<12s} {r['MAP']:>8.4f} {r['MRR']:>8.4f} "
              f"{r['recall_at_1']:>8.4f} {r['recall_at_5']:>8.4f} {r['precision_at_R']:>8.4f}")
    print("=" * 78)
    print("Family-size invariant metrics. purity@k (in CSV) is capped by family size.")
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


if __name__ == "__main__":
    run()
