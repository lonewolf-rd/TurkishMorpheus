import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.embeddings.encoders import build_encoders, find_checkpoint


WIKIANN_CONFIG = "tr"
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def load_wikiann(split: str, cap: int = None) -> List[Tuple[List[str], List[int]]]:
    from datasets import load_dataset

    ds = load_dataset("wikiann", WIKIANN_CONFIG, split=split)
    out: List[Tuple[List[str], List[int]]] = []
    for ex in ds:
        out.append((list(ex["tokens"]), list(ex["ner_tags"])))
        if cap is not None and len(out) >= cap:
            break
    global_logger.info(f"[ner_eval] Loaded {len(out)} WikiANN-{WIKIANN_CONFIG} {split} sentences")
    return out


def build_features(encoder, data: List[Tuple[List[str], List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    tags: List[int] = []
    for tokens, ner_tags in data:
        if not tokens:
            continue
        vecs = encoder.encode_tokens_in_context(tokens)
        feats.append(vecs)
        tags.extend(ner_tags)
    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, encoder.dim), dtype=np.float32)
    y = np.array(tags, dtype=np.int64)
    return X, y


def evaluate_probe(encoder, train, test) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, classification_report

    Xtr, ytr = build_features(encoder, train)
    Xte, yte = build_features(encoder, test)

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    entity_mask = yte != 0
    macro = f1_score(yte, pred, average="macro")
    micro_ent = f1_score(yte[entity_mask], pred[entity_mask], average="micro") if entity_mask.any() else 0.0
    report = classification_report(yte, pred, zero_division=0)
    return {"macro_f1": round(float(macro), 4), "entity_micro_f1": round(float(micro_ent), 4),
            "report": report}


def run(include_berturk: bool = True, train_cap: int = 3000, test_cap: int = 1000) -> None:
    base = Path(__file__).resolve().parents[4]
    artifacts = base / "src" / "model_development" / "artifacts"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "embeddings" / "ner"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[ner_eval] No Morpheus checkpoint found.")
        sys.exit(1)

    train = load_wikiann("train", cap=train_cap)
    test = load_wikiann("test", cap=test_cap)

    encoders = build_encoders(str(checkpoint), include_berturk=include_berturk)

    rows = []
    for enc in encoders:
        res = evaluate_probe(enc, train, test)
        rows.append({"encoder": enc.name, "dim": enc.dim,
                     "macro_f1": res["macro_f1"], "entity_micro_f1": res["entity_micro_f1"]})
        global_logger.info(
            f"[ner_eval] {enc.name}: macro_f1={res['macro_f1']} entity_micro_f1={res['entity_micro_f1']}"
        )
        (output_dir / f"report_{enc.name}.txt").write_text(res["report"], encoding="utf-8")

    summary_path = output_dir / "ner_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("NER PROBE  (WikiANN-tr, frozen embeddings + linear classifier)")
    print("=" * 70)
    print(f"  {'encoder':<12s} {'dim':>5s} {'macro_f1':>10s} {'entity_micro_f1':>16s}")
    for r in rows:
        print(f"  {r['encoder']:<12s} {r['dim']:>5d} {r['macro_f1']:>10.4f} {r['entity_micro_f1']:>16.4f}")
    print("=" * 70)
    print("Note: NER is contextual; Morpheus produces static word vectors, BERTurk contextual ones.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run()
