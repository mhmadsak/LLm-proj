# src/spans.py
import re
from typing import List, Tuple, Literal, Dict

_TOKEN_RE = re.compile(r"\S+")

def token_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]

def _overlap(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def probs_to_hard_spans(
    probs: List[float],
    hard_threshold: float = 0.6,
    dead_zone: float = 0.08,
) -> List[Tuple[int, int]]:
    """
    Maximal contiguous spans [start, end) of token indices where token is 'hard':
      p_i >= hard_threshold AND |p_i - hard_threshold| >= dead_zone
    """
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(probs)
    while i < n:
        p = float(probs[i])
        if (p >= hard_threshold) and (abs(p - hard_threshold) >= dead_zone):
            start = i
            i += 1
            while i < n:
                q = float(probs[i])
                if not ((q >= hard_threshold) and (abs(q - hard_threshold) >= dead_zone)):
                    break
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans

def merge_to_soft_spans(
    probs: List[float],
    hard_spans: List[Tuple[int, int]],   # kept for signature compatibility; unused  # kept; unused
) -> List[Dict]:
    """
    Return one probability per token (word).
    Each token i is represented as a unit span [i, i+1) with its own 'prob'.
    Example: [{"start": 0, "end": 1, "prob": 0.12}, ..., {"start": n-1, "end": n, "prob": 0.87}]
    """
    out: List[Dict] = []
    for i, p in enumerate(probs):
        out.append({"start": i, "end": i + 1, "prob": float(p)})
    return out
