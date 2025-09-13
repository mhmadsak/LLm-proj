# src/splitting_en.py
# Atomic English-only statement splitter with exact substring preservation.

import json
import re
from typing import List, Dict, Tuple

# --- English short claim words ----------------------------------------------
SHORT_CLAIM_WORDS = {"yes", "no", "certainly", "indeed"}

SHORT_CLAIM_RE = re.compile(
    r"^\s*(?P<claim>(?:yes|no|certainly|indeed))(?:\W+|$)",
    re.IGNORECASE
)

# Sentence segmentation: split on ., !, ?, ;, or newline.
SENTENCE_RE = re.compile(
    r"""
    (?P<seg>
        .+?                                 # minimal chars
        (?:
            (?<=[.!?])(?:["')\]]+)?         # keep closing punctuation/brackets
            (?:\s+|$)
          | \n+
          | ;\s*
          | $                               # end of string
        )
    )
    """,
    re.VERBOSE | re.DOTALL
)

# ---------------------------------------------------------------------------

def _split_leading_short_claims(text: str, seg_start: int, seg_end: int) -> List[Tuple[int, int]]:
    """If a segment starts with a short claim word, split it off."""
    seg = text[seg_start:seg_end]
    m = SHORT_CLAIM_RE.match(seg)
    if not m:
        return [(seg_start, seg_end)]

    claim_end = seg_start + m.end()
    spans = []
    if text[seg_start:claim_end].strip():
        spans.append((seg_start, claim_end))
    if text[claim_end:seg_end].strip():
        spans.append((claim_end, seg_end))
    return spans

def extract_statements(model_output_text: str) -> List[Dict[str, str]]:
    """
    Split English text into atomic claims/statements.
    - Handles short claim words ("Yes", "No", "Certainly", "Indeed").
    - Preserves exact substrings.
    """
    if model_output_text is None:
        model_output_text = ""
    text = model_output_text

    spans: List[Tuple[int, int]] = []
    for m in SENTENCE_RE.finditer(text):
        s, e = m.span("seg")
        if text[s:e].strip():
            spans.extend(_split_leading_short_claims(text, s, e))

    # Build JSON objects with indexed keys
    result: List[Dict[str, str]] = []
    for i, (s, e) in enumerate(spans, start=1):
        original = text[s:e]
        factual = original.strip()
        result.append({
            f"factual_statement_{i}": factual,
            f"original_substring_{i}": original
        })
    return result

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data = sys.stdin.read()
    out = extract_statements(data)
    print(json.dumps(out, ensure_ascii=False))