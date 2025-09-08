# src/spans.py
import re
from typing import List, Dict, Tuple

_TOKEN_RE = re.compile(r"\S+")

def _tokenize_with_spans(text: str):
    return [{"i": i, "start": m.start(), "end": m.end()} for i, m in enumerate(_TOKEN_RE.finditer(text or ""))]

def _merge_adjacent(spans: List[List[int]]) -> List[List[int]]:
    if not spans: return []
    spans = sorted(spans)
    out = [spans[0]]
    for s, e in spans[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1][1] = max(le, e)
        else:
            out.append([s, e])
    return out

def _combine(vals: List[float], mode: str) -> float:
    if not vals: return 0.5
    if mode == "mean":
        return sum(vals) / len(vals)
    if mode == "min":
        return min(vals)
    # default: "max" → conservative for hallucinations
    return max(vals)

def compute_spans(answer_text: str,
                  originals: List[str],
                  verifs: List[Dict],
                  threshold: float = 0.5,
                  combine: str = "max") -> Tuple[List[List[int]], List[Dict]]:
    """
    Aggregate per-token probabilities across multiple facts that point to the same substring.
    Returns:
      hard: [[start, end], ...]
      soft: [{"start": s, "end": e, "prob": p}, ...]
    """
    acc: Dict[Tuple[int,int], List[float]] = {}

    for orig, v in zip(originals, verifs):
        if not isinstance(v, dict) or "token_probs" not in v:
            continue
        probs = v["token_probs"]
        base = (answer_text or "").find(orig or "")
        if base < 0:
            continue
        toks = _tokenize_with_spans(orig or "")
        for t in toks:
            i, s, e = t["i"], t["start"], t["end"]
            p = float(probs[i]) if i < len(probs) else 0.5
            key = (base + s, base + e)
            acc.setdefault(key, []).append(p)

    # combine and build soft/hard
    soft = []
    for (s, e), vals in acc.items():
        p = _combine(vals, combine)
        soft.append({"start": s, "end": e, "prob": float(p)})

    hard = [[d["start"], d["end"]] for d in soft if d["prob"] >= threshold]
    hard = _merge_adjacent(hard)
    return hard, soft

# Backwards-compatible wrappers (if other code still imports these names)
def extract_predicted_spans_hard(answer_text, originals, verifs, threshold=0.5, combine="max"):
    hard, _ = compute_spans(answer_text, originals, verifs, threshold=threshold, combine=combine)
    return hard

def extract_predicted_spans_soft(answer_text, originals, verifs, combine="max"):
    _, soft = compute_spans(answer_text, originals, verifs, threshold=0.5, combine=combine)
    return soft
