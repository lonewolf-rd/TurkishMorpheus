import math
from collections import Counter
from typing import Dict, Optional


def shannon_entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def renyi_entropy_bits(counts: Counter, alpha: float = 2.5) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    if abs(alpha - 1.0) < 1e-9:
        return shannon_entropy_bits(counts)
    s = sum((c / total) ** alpha for c in counts.values() if c > 0)
    if s <= 0:
        return 0.0
    return math.log2(s) / (1.0 - alpha)


def renyi_efficiency(
        counts: Counter,
        alpha: float = 2.5,
        vocab_size: Optional[int] = None,
) -> Dict[str, float]:
    h_alpha = renyi_entropy_bits(counts, alpha)
    n_observed = len(counts)
    result = {
        "renyi_alpha": alpha,
        "renyi_entropy_bits": round(h_alpha, 4),
        "shannon_entropy_bits": round(shannon_entropy_bits(counts), 4),
        "renyi_efficiency_observed": round(
            h_alpha / math.log2(max(n_observed, 2)), 4
        ),
    }
    if vocab_size:
        result["renyi_efficiency_vocab"] = round(
            h_alpha / math.log2(max(vocab_size, 2)), 4
        )
    return result
