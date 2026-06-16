import sys
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.embeddings.encoders import build_encoders, find_checkpoint
from src.benchmarker.benchmarks.embeddings.neighbors_eval import _resolve_families


def build_pairs(
        families: Dict[str, List[str]],
        n_neg_per_pos: int = 1,
        seed: int = 42,
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    rng = random.Random(seed)
    roots = list(families)
    pos: List[Tuple[str, str]] = []
    for forms in families.values():
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                pos.append((forms[i], forms[j]))

    neg: List[Tuple[str, str]] = []
    target_neg = len(pos) * n_neg_per_pos
    attempts = 0
    while len(neg) < target_neg and attempts < target_neg * 20:
        attempts += 1
        ra, rb = rng.sample(roots, 2)
        wa = rng.choice(families[ra])
        wb = rng.choice(families[rb])
        neg.append((wa, wb))

    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return pairs, labels


def verification_metrics(sims: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    auc = roc_auc_score(labels, sims)
    ap = average_precision_score(labels, sims)

    thresholds = np.unique(sims)
    best_f1, best_t = 0.0, 0.0
    for t in thresholds:
        pred = (sims >= t).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return {"roc_auc": float(auc), "avg_precision": float(ap),
            "best_f1": best_f1, "best_threshold": best_t}


def run(include_berturk: bool = True, include_bge: bool = True) -> None:
    base = Path(__file__).resolve().parents[4]
    artifacts = base / "src" / "model_development" / "artifacts"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "embeddings" / "dedup"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[dedup_eval] No Morpheus checkpoint found.")
        sys.exit(1)

    families = _resolve_families()
    pairs, labels = build_pairs(families)
    word_index: Dict[str, int] = {}
    for a, b in pairs:
        word_index.setdefault(a, len(word_index))
        word_index.setdefault(b, len(word_index))
    vocab = list(word_index)
    ai = np.array([word_index[a] for a, _ in pairs])
    bi = np.array([word_index[b] for _, b in pairs])

    global_logger.info(
        f"[dedup_eval] {labels.sum()} positive, {len(labels) - labels.sum()} negative pairs, "
        f"{len(vocab)} unique words"
    )

    encoders = build_encoders(str(checkpoint), include_berturk=include_berturk,
                              include_bge=include_bge)

    rows = []
    for enc in encoders:
        vecs = enc.encode_words(vocab)
        norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sims = (norm[ai] * norm[bi]).sum(axis=1)
        m = verification_metrics(sims, labels)
        rows.append({"encoder": enc.name, "dim": enc.dim,
                     "roc_auc": round(m["roc_auc"], 4),
                     "avg_precision": round(m["avg_precision"], 4),
                     "best_f1": round(m["best_f1"], 4),
                     "best_threshold": round(m["best_threshold"], 4)})
        global_logger.info(
            f"[dedup_eval] {enc.name}: AUC={m['roc_auc']:.4f} AP={m['avg_precision']:.4f} "
            f"bestF1={m['best_f1']:.4f}"
        )

    summary_path = output_dir / "dedup_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 78)
    print("SAME-ROOT VERIFICATION  (lexical dedup: are two words the same root?)")
    print("=" * 78)
    print(f"  {'encoder':<12s} {'ROC-AUC':>9s} {'AvgPrec':>9s} {'bestF1':>9s} {'thr':>8s}")
    for r in rows:
        print(f"  {r['encoder']:<12s} {r['roc_auc']:>9.4f} {r['avg_precision']:>9.4f} "
              f"{r['best_f1']:>9.4f} {r['best_threshold']:>8.4f}")
    print("=" * 78)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run()
