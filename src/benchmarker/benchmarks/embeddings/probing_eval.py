import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.embeddings.encoders import build_encoders, find_checkpoint
from src.benchmarker.benchmarks.embeddings.gold_data import load_probe_sets, default_gold_path


FALLBACK_TASKS: Dict[str, List[Tuple[str, str]]] = {
    "number": [
        ("kitap", "sg"), ("kitaplar", "pl"), ("ev", "sg"), ("evler", "pl"),
        ("göz", "sg"), ("gözler", "pl"), ("yol", "sg"), ("yollar", "pl"),
        ("araba", "sg"), ("arabalar", "pl"), ("çocuk", "sg"), ("çocuklar", "pl"),
        ("masa", "sg"), ("masalar", "pl"), ("kuş", "sg"), ("kuşlar", "pl"),
        ("kapı", "sg"), ("kapılar", "pl"), ("şehir", "sg"), ("şehirler", "pl"),
    ],
    "case": [
        ("evde", "loc"), ("evden", "abl"), ("eve", "dat"), ("evi", "acc"),
        ("yolda", "loc"), ("yoldan", "abl"), ("yola", "dat"), ("yolu", "acc"),
        ("okulda", "loc"), ("okuldan", "abl"), ("okula", "dat"), ("okulu", "acc"),
        ("şehirde", "loc"), ("şehirden", "abl"), ("şehire", "dat"), ("şehiri", "acc"),
        ("masada", "loc"), ("masadan", "abl"), ("masaya", "dat"), ("masayı", "acc"),
    ],
}


def linear_probe_cv(vecs: np.ndarray, labels: np.ndarray, n_splits: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs: List[float] = []
    for train_idx, test_idx in skf.split(vecs, labels):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(vecs[train_idx], labels[train_idx])
        accs.append(clf.score(vecs[test_idx], labels[test_idx]))
    return float(np.mean(accs))


def _resolve_tasks() -> Dict[str, List[Tuple[str, str]]]:
    gold = default_gold_path()
    if gold.exists():
        tasks = load_probe_sets(str(gold))
        if tasks:
            return tasks
    global_logger.warning("[probing_eval] SIGMORPHON gold unavailable; using fallback sets.")
    return FALLBACK_TASKS


def run(include_berturk: bool = True, n_splits: int = 5) -> None:
    base = Path(__file__).resolve().parents[4]
    artifacts = base / "src" / "model_development" / "artifacts"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "embeddings" / "probing"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[probing_eval] No Morpheus checkpoint found.")
        sys.exit(1)

    probe_tasks = _resolve_tasks()
    encoders = build_encoders(str(checkpoint), include_berturk=include_berturk)

    rows = []
    for task, pairs in probe_tasks.items():
        words = [w for w, _ in pairs]
        names = sorted({lab for _, lab in pairs})
        labels = np.array([names.index(lab) for _, lab in pairs])
        for enc in encoders:
            vecs = enc.encode_words(words)
            acc = linear_probe_cv(vecs, labels, n_splits=n_splits)
            rows.append({"task": task, "encoder": enc.name, "n": len(words),
                         "classes": len(names), "probe_acc": round(acc, 4)})
            global_logger.info(f"[probing_eval] {task}/{enc.name}: acc={acc:.4f}")

    summary_path = output_dir / "probing_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("MORPHOLOGICAL PROBING  (frozen embeddings, linear probe, stratified CV)")
    print("=" * 70)
    print(f"  {'task':<10s} {'encoder':<12s} {'classes':>8s} {'probe_acc':>10s}")
    for r in rows:
        print(f"  {r['task']:<10s} {r['encoder']:<12s} {r['classes']:>8d} {r['probe_acc']:>10.4f}")
    print("=" * 70)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run()
