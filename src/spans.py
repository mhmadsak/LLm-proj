# src/spans.py
import re
from typing import List, Tuple, Literal, Dict

_TOKEN_RE = re.compile(r"\S+")

def token_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]

def _overlap(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def aggregate_claims_to_answer_probs(
    answer_text: str,
    claim_originals: List[str],
    claim_token_probs: List[List[float]],
    per_token_agg: Literal["max", "mean", "min"] = "max",
    default_prob: float = 0.6
) -> List[float]:
    """
    Map per-claim token probs onto answer tokens and aggregate per answer token.
    - claim_originals[i] must be the exact substring from the answer for claim i.
    - claim_token_probs[i] corresponds to tokenization of claim_originals[i] w/ \S+ regex.
    Returns list of probs aligned to answer tokens.
    """
    ans_spans = token_spans(answer_text)
    n_ans = len(ans_spans)
    buckets: List[List[float]] = [[] for _ in range(n_ans)]

    for orig, probs in zip(claim_originals or [], claim_token_probs or []):
        if not orig:
            continue
        base = (answer_text or "").find(orig)
        if base < 0:
            continue  # can't map if substring not found
        claim_spans_rel = token_spans(orig)
        claim_spans_abs = [(base + s, base + e) for (s, e) in claim_spans_rel]
        # map each claim token prob to overlapping answer tokens
        for p, cspan in zip(probs, claim_spans_abs):
            try:
                pv = float(p)
            except Exception:
                pv = default_prob
            pv = 0.0 if pv < 0.0 else 1.0 if pv > 1.0 else pv
            for i, aspan in enumerate(ans_spans):
                if _overlap(cspan, aspan) > 0:
                    buckets[i].append(pv)

    def _reduce(vals: List[float]) -> float:
        if not vals: return default_prob
        if per_token_agg == "mean": return sum(vals)/len(vals)
        if per_token_agg == "min":  return min(vals)
        return max(vals)  # default "max" conservative for hallucination

    return [_reduce(vs) for vs in buckets]

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

def _combine(vals: List[float], how: Literal["mean", "max", "min"]) -> float:
    if not vals: return 0.0
    if how == "max": return max(vals)
    if how == "min": return min(vals)
    return sum(vals) / len(vals)

def merge_to_soft_spans(
    probs: List[float],
    hard_spans: List[Tuple[int, int]],
    combine: Literal["mean", "max", "min"] = "mean",
) -> List[Dict]:
    """
    Convert hard spans to soft spans with an aggregate probability per span.
    Returns: [{"start": s, "end": e, "prob": p}, ...]
    """
    out: List[Dict] = []
    for s, e in hard_spans:
        p = _combine([float(x) for x in probs[s:e]], combine)
        out.append({"start": s, "end": e, "prob": float(p)})
    return out
