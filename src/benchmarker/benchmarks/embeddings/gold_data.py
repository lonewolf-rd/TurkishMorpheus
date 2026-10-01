import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def default_morphscore_path() -> Path:
    base = Path(__file__).resolve().parents[4]
    return base / "data" / "morphscore" / "turkish_data.csv"


def load_root_families_ud(
        data_path: Optional[Path] = None,
        min_members: int = 8,
        max_families: int = 120,
        max_per_family: int = 20,
        min_root_len: int = 2,
        seed: int = 0,
) -> Dict[str, List[str]]:
    """Root families from UD_Turkish-Kenet gold stems (the MorphScore data, ~30K forms).

    SIGMORPHON has too few single-word forms per lemma for retrieval (3 lemmas with
    >= 4 forms), so families come from the UD gold stems instead. Families are sampled
    at random (fixed seed) rather than by size, so frequent verbs do not dominate.
    """
    from src.benchmarker.benchmarks.morphscore_eval import load_morphscore_turkish

    df = load_morphscore_turkish(data_path or default_morphscore_path())
    by_root: Dict[str, List[str]] = defaultdict(list)
    for form, stem in zip(df["wordform"], df["stem"]):
        if form.isalpha() and len(stem) >= min_root_len:
            by_root[stem].append(form)

    rng = random.Random(seed)
    eligible = sorted(r for r, forms in by_root.items() if len(set(forms)) >= min_members)
    chosen = sorted(rng.sample(eligible, min(max_families, len(eligible))))
    families = {}
    for root in chosen:
        forms = sorted(set(by_root[root]))
        families[root] = sorted(rng.sample(forms, min(max_per_family, len(forms))))
    global_logger.info(
        f"[gold_data] Built {len(families)} root families ({sum(map(len, families.values()))} words) "
        f"from UD_Turkish-Kenet gold stems ({len(eligible)} eligible, min_members={min_members})"
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
