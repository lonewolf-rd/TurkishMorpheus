import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from src.common.text_utils import turkish_lower
from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.sigmorphon import build_segmenters
from src.benchmarker.benchmarks.trmmlu_eval import find_morpheus_checkpoint


GOLD: List[Dict] = [
    {"w": "fonksiyonlar", "m": ["fonksiyon", "lar"], "s": "rare_root"},
    {"w": "moleküllerin", "m": ["molekül", "ler", "in"], "s": "rare_root"},
    {"w": "parametreler", "m": ["parametre", "ler"], "s": "rare_root"},
    {"w": "enzimlerde", "m": ["enzim", "ler", "de"], "s": "rare_root"},
    {"w": "izotoplar", "m": ["izotop", "lar"], "s": "rare_root"},
    {"w": "katalizörler", "m": ["katalizör", "ler"], "s": "rare_root"},
    {"w": "polimerler", "m": ["polimer", "ler"], "s": "rare_root"},
    {"w": "vektörler", "m": ["vektör", "ler"], "s": "rare_root"},
    {"w": "matrisler", "m": ["matris", "ler"], "s": "rare_root"},
    {"w": "türevler", "m": ["türev", "ler"], "s": "rare_root"},
    {"w": "integraller", "m": ["integral", "ler"], "s": "rare_root"},
    {"w": "hipotezler", "m": ["hipotez", "ler"], "s": "rare_root"},
    {"w": "nöronların", "m": ["nöron", "lar", "ın"], "s": "rare_root"},
    {"w": "proteinler", "m": ["protein", "ler"], "s": "rare_root"},
    {"w": "bakterilerin", "m": ["bakteri", "ler", "in"], "s": "rare_root"},
    {"w": "virüsler", "m": ["virüs", "ler"], "s": "rare_root"},
    {"w": "algoritmalar", "m": ["algoritma", "lar"], "s": "rare_root"},
    {"w": "sentaksta", "m": ["sentaks", "ta"], "s": "rare_root"},
    {"w": "gradyanlar", "m": ["gradyan", "lar"], "s": "rare_root"},
    {"w": "kromozomlar", "m": ["kromozom", "lar"], "s": "rare_root"},

    {"w": "demircilik", "m": ["demir", "ci", "lik"], "s": "derivation"},
    {"w": "balıkçılık", "m": ["balık", "çı", "lık"], "s": "derivation"},
    {"w": "ormancılık", "m": ["orman", "cı", "lık"], "s": "derivation"},
    {"w": "gözlemciler", "m": ["gözlem", "ci", "ler"], "s": "derivation"},
    {"w": "renksizlik", "m": ["renk", "siz", "lik"], "s": "derivation"},
    {"w": "susuzluk", "m": ["su", "suz", "luk"], "s": "derivation"},
    {"w": "bulutsuz", "m": ["bulut", "suz"], "s": "derivation"},
    {"w": "yağmurlardan", "m": ["yağmur", "lar", "dan"], "s": "derivation"},
    {"w": "menziller", "m": ["menzil", "ler"], "s": "derivation"},
    {"w": "tuzsuzdu", "m": ["tuz", "suz", "du"], "s": "derivation"},

    {"w": "yaprağın", "m": ["yaprağ", "ın"], "s": "softening"},
    {"w": "bardağı", "m": ["bardağ", "ı"], "s": "softening"},
    {"w": "çiçeğin", "m": ["çiçeğ", "in"], "s": "softening"},
    {"w": "köpeğim", "m": ["köpeğ", "im"], "s": "softening"},
    {"w": "ekmeği", "m": ["ekmeğ", "i"], "s": "softening"},
    {"w": "sokağa", "m": ["sokağ", "a"], "s": "softening"},
    {"w": "tarağı", "m": ["tarağ", "ı"], "s": "softening"},

    {"w": "aklın", "m": ["akl", "ın"], "s": "vowel_drop"},
    {"w": "burnu", "m": ["burn", "u"], "s": "vowel_drop"},
    {"w": "şehrin", "m": ["şehr", "in"], "s": "vowel_drop"},
    {"w": "ağzı", "m": ["ağz", "ı"], "s": "vowel_drop"},
    {"w": "boynu", "m": ["boyn", "u"], "s": "vowel_drop"},
    {"w": "karnı", "m": ["karn", "ı"], "s": "vowel_drop"},

    {"w": "saatlerde", "m": ["saat", "ler", "de"], "s": "loanword_exc"},
    {"w": "rollerde", "m": ["rol", "ler", "de"], "s": "loanword_exc"},
    {"w": "gollerin", "m": ["gol", "ler", "in"], "s": "loanword_exc"},
    {"w": "hallerden", "m": ["hal", "ler", "den"], "s": "loanword_exc"},
    {"w": "harflerle", "m": ["harf", "ler", "le"], "s": "loanword_exc"},
    {"w": "usullerin", "m": ["usul", "ler", "in"], "s": "loanword_exc"},
    {"w": "kalplerin", "m": ["kalp", "ler", "in"], "s": "loanword_exc"},
]


def score(pred: List[str], gold: List[str]) -> Dict[str, bool]:
    pred_l = [turkish_lower(p) for p in pred]
    gold_l = [turkish_lower(g) for g in gold]
    return {
        "root_ok": bool(pred_l) and pred_l[0] == gold_l[0],
        "count_ok": len(pred) == len(gold),
        "len_ok": [len(p) for p in pred] == [len(g) for g in gold],
        "exact_ok": pred_l == gold_l,
    }


def run(classical_vocab: int = 64000) -> None:
    base = Path(__file__).resolve().parents[3]
    artifacts = base / "src" / "model_development" / "artifacts"
    output_dir = base / "src" / "benchmarker" / "results" / "paper_eval" / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = find_morpheus_checkpoint(artifacts / "checkpoints")
    if checkpoint is None:
        global_logger.error("[qualitative_eval] No Morpheus checkpoint found.")
        sys.exit(1)

    segmenters = build_segmenters(
        checkpoint_path=str(checkpoint),
        morfessor_path=str(artifacts / "tokenizers" / "classical" / "morfessor_model.bin"),
        benchmarker_results_dir=str(artifacts / "tokenizers" / "classical"),
        preferred_classical_vocab=classical_vocab,
    )

    names = list(segmenters.keys())
    metrics = ["root_ok", "count_ok", "len_ok", "exact_ok"]
    agg: Dict[str, Dict[str, int]] = {n: {m: 0 for m in metrics} for n in names}
    detail_rows: List[Dict] = []

    for entry in GOLD:
        word, gold, stratum = entry["w"], entry["m"], entry["s"]
        row = {"word": word, "stratum": stratum, "gold": " | ".join(gold)}
        for name, seg_fn in segmenters.items():
            try:
                segs = seg_fn(word)
            except Exception:
                segs = []
            sc = score(segs, gold)
            row[f"{name}_seg"] = " | ".join(segs)
            for m in metrics:
                row[f"{name}_{m[:-3]}"] = int(sc[m])
                agg[name][m] += int(sc[m])
        detail_rows.append(row)

    details_path = output_dir / "qualitative_details.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    n = len(GOLD)
    summary_rows = []
    for name in names:
        summary_rows.append({
            "tokenizer": name,
            "root_pct": round(agg[name]["root_ok"] / n * 100, 1),
            "count_pct": round(agg[name]["count_ok"] / n * 100, 1),
            "len_pct": round(agg[name]["len_ok"] / n * 100, 1),
            "exact_pct": round(agg[name]["exact_ok"] / n * 100, 1),
        })
    summary_rows.sort(key=lambda r: -r["exact_pct"])
    summary_path = output_dir / "qualitative_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 78)
    print(f"QUALITATIVE MORPHOLOGY TEST  ({n} curated words, gold surface segmentation)")
    print("=" * 78)
    print(f"  {'tokenizer':<20s} {'root%':>7s} {'count%':>7s} {'len%':>7s} {'exact%':>7s}")
    print(f"  {'':<20s} {'(kök)':>7s} {'(ek say)':>7s} {'(uzunluk)':>7s} {'(birebir)':>7s}")
    for r in summary_rows:
        print(f"  {r['tokenizer']:<20s} {r['root_pct']:>7.1f} {r['count_pct']:>7.1f} "
              f"{r['len_pct']:>7.1f} {r['exact_pct']:>7.1f}")
    print("=" * 78)
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run()
