from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.common.providers.logger_provider import global_logger
from src.benchmarker.benchmarks.sigmorphon import load_sigmorphon_inflection_gold


NUMBER_TAGS = {"SG", "PL"}
CASE_TAGS = {"NOM", "ACC", "GEN", "DAT", "ABL", "LOC", "INS", "ESS", "EQU", "PRIV"}


def default_gold_path() -> Path:
    base = Path(__file__).resolve().parents[4]
    return base / "data" / "sigmorphon_tr" / "tur.gold"


def load_root_families(
        gold_path: str,
        min_members: int = 8,
        max_families: int = 120,
        max_per_family: int = 20,
) -> Dict[str, List[str]]:
    entries = load_sigmorphon_inflection_gold(gold_path)
    by_root: Dict[str, List[str]] = defaultdict(list)
    seen: set = set()
    for e in entries:
        root = e["lemma_root"]
        form = e["inflected"]
        key = (root, form)
        if key in seen:
            continue
        seen.add(key)
        by_root[root].append(form)

    families = {
        root: forms[:max_per_family]
        for root, forms in by_root.items()
        if len(forms) >= min_members
    }
    families = dict(sorted(families.items(), key=lambda kv: -len(kv[1]))[:max_families])
    global_logger.info(
        f"[gold_data] Built {len(families)} root families from SIGMORPHON gold "
        f"(min_members={min_members})"
    )
    return families


def load_probe_sets(
        gold_path: str,
        min_per_class: int = 20,
) -> Dict[str, List[Tuple[str, str]]]:
    entries = load_sigmorphon_inflection_gold(gold_path)
    tasks: Dict[str, List[Tuple[str, str]]] = {"number": [], "case": []}

    for e in entries:
        feats = set(e["features"])
        form = e["inflected"]
        num = feats & NUMBER_TAGS
        if len(num) == 1:
            tasks["number"].append((form, next(iter(num))))
        case = feats & CASE_TAGS
        if len(case) == 1:
            tasks["case"].append((form, next(iter(case))))

    cleaned: Dict[str, List[Tuple[str, str]]] = {}
    for task, pairs in tasks.items():
        counts: Dict[str, int] = defaultdict(int)
        for _, lab in pairs:
            counts[lab] += 1
        keep = {lab for lab, c in counts.items() if c >= min_per_class}
        kept = [(w, lab) for w, lab in pairs if lab in keep]
        if len({lab for _, lab in kept}) >= 2:
            cleaned[task] = kept
        global_logger.info(
            f"[gold_data] probe task '{task}': {len(kept)} items, "
            f"{len(keep)} classes (>= {min_per_class} each)"
        )
    return cleaned
