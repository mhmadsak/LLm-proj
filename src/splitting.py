# src/splitting_en.py
import re
from typing import List, Dict, Tuple

import spacy
from spacy.language import Language

# --- config ----------------------------------------------------------------

YES_WORDS = {"yes","yeah","yep","certainly","indeed","affirmative","correct","true","right"}
NO_WORDS  = {"no","nope","nah","false","incorrect","wrong"}

# bullets like: "- foo", "* bar", "• baz", "1. item", "a) item"
BULLET_RE = re.compile(
    r"^\s*(?:[-*•]\s+|(?:\d+|[a-zA-Z])[\.\)]\s+)(?P<txt>.+?)\s*$"
)

# --- spaCy loader -----------------------------------------------------------

_EN_NLP: Language = None

def _load_en() -> Language:
    """Load English pipeline; fall back to blank('en') + sentencizer."""
    global _EN_NLP
    if _EN_NLP is not None:
        return _EN_NLP
    try:
        _EN_NLP = spacy.load("en_core_web_sm", disable=["ner","tagger","lemmatizer"])
    except Exception:
        _EN_NLP = spacy.blank("en")
    if "sentencizer" not in _EN_NLP.pipe_names:
        _EN_NLP.add_pipe("sentencizer")
    return _EN_NLP

# --- utils -----------------------------------------------------------------

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("…", " ").strip())

def _is_yesno_question(q: str) -> bool:
    return bool(re.match(
        r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|may|might|must)\b",
        q, flags=re.I
    ))

def _is_wh_question(q: str) -> bool:
    return bool(re.match(r"^\s*(who|what|where|when|which|how|why)\b", q, flags=re.I))

# --- QA → declarative heuristics (English) ---------------------------------

def _qa_to_statement_yesno(q: str, a: str) -> Tuple[str, str]:
    """Make a rough declarative statement from a yes/no Q + answer.
    Returns (statement, answer_anchor_for_original_substring)."""
    q = _clean(q)
    a = _clean(a)
    if not q: return "", ""
    first = (a.split()[:1] or [""])[0].lower()
    polarity = True if first in YES_WORDS else False if first in NO_WORDS else True
    stem = q[:-1] if q.endswith("?") else q

    # BE
    m = re.match(r"^\s*(is|are|was|were)\s+(.*)$", stem, flags=re.I)
    if m:
        be, rest = m.group(1).lower(), m.group(2)
        if polarity:
            # "Is X Y?" → "X Y."
            return f"{rest}.".strip(), (a.split()[:1] or [""])[0]
        else:
            parts = rest.split()
            if len(parts) >= 2:
                return f"{parts[0]} {be} not {' '.join(parts[1:])}.".strip(), (a.split()[:1] or [""])[0]
            return f"{rest} not.".strip(), (a.split()[:1] or [""])[0]

    # DO/DOES/DID
    m = re.match(r"^\s*(do|does|did)\s+(.*)$", stem, flags=re.I)
    if m:
        rest = m.group(2)
        if polarity:
            return f"{rest}.".strip(), (a.split()[:1] or [""])[0]
        parts = rest.split()
        if len(parts) >= 2:
            return f"{parts[0]} does not {' '.join(parts[1:])}.".strip(), (a.split()[:1] or [""])[0]
        return f"not {rest}.".strip(), (a.split()[:1] or [""])[0]

    # MODALS
    m = re.match(r"^\s*(can|could|will|would|should|may|might|must)\s+(.*)$", stem, flags=re.I)
    if m:
        modal, rest = m.group(1).lower(), m.group(2)
        if polarity:
            return f"{rest}.".strip(), (a.split()[:1] or [""])[0]
        neg = "cannot" if modal == "can" else f"{modal} not"
        return f"{rest} {neg}.".strip(), (a.split()[:1] or [""])[0]

    # HAVE/HAS/HAD
    m = re.match(r"^\s*(has|have|had)\s+(.*)$", stem, flags=re.I)
    if m:
        aux, rest = m.group(1).lower(), m.group(2)
        if polarity:
            return f"{rest}.".strip(), (a.split()[:1] or [""])[0]
        return f"{aux} not {rest}.".strip(), (a.split()[:1] or [""])[0]

    # Fallback
    return f"{stem}.".strip(), (a.split()[:1] or [""])[0]

def _qa_to_statement_wh(q: str, a: str) -> Tuple[str, str]:
    """Simple WH patterns."""
    q = _clean(q); a = _clean(a)
    a_short = a if len(a) <= 80 else a.split(".")[0]
    if not a_short: return "", ""
    # Who VERB ... ?
    m = re.match(r"^\s*who\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"{a_short} {m.group(1)}.".strip(), a
    # What is/are X?
    m = re.match(r"^\s*what\s+(is|are)\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"{m.group(2)} {m.group(1).lower()} {a_short}.".strip(), a
    # Where is/was/are/were X?
    m = re.match(r"^\s*where\s+(is|was|are|were)\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"{m.group(2)} {m.group(1).lower()} located in {a_short}.".strip(), a
    # When did/was/were/is X ...?
    m = re.match(r"^\s*when\s+(did|was|were|is)\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"{m.group(2)} in {a_short}.".strip(), a
    # How many X ...?
    m = re.match(r"^\s*how\s+many\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"There are {a_short} {m.group(1)}.".strip(), a
    # How much ...?
    m = re.match(r"^\s*how\s+much\s+(.*)\?\s*$", q, flags=re.I)
    if m: return f"The amount is {a_short}.".strip(), a
    # Fallback
    return (f"{q} {a_short}".strip() + ".", a)

# --- list splitting (bullets / numbered) -----------------------------------

def _extract_bullets(text: str) -> List[Tuple[int, int]]:
    """Return absolute (lo,hi) spans for bullet/numbered lines in the text."""
    spans = []
    if not text: return spans
    pos = 0
    for line in text.splitlines(keepends=True):
        m = BULLET_RE.match(line)
        if m:
            lo = pos + (m.start("txt"))
            hi = pos + (m.end("txt"))
            spans.append((lo, hi))
        pos += len(line)
    return spans

# --- public API -------------------------------------------------------------

def split_en_general(model_input: str, model_output: str, max_items: int = 30) -> List[Dict[str, str]]:
    """
    English splitter for mixed data:
      - If input is a question and output exists → derive a QA-based statement.
      - Split ANSWER sentences (spaCy).
      - Split bullet/numbered lines from INPUT and ANSWER.
      - Fallbacks if one side is empty.
    Returns: [{"factual_statement": "...", "original_substring": "...", "origin": "..."}]
    origin ∈ {"qa_derived","answer","input_line"}
    """
    q = _clean(model_input or "")
    a = _clean(model_output or "")

    out: List[Dict[str, str]] = []

    # (1) QA-derived statement (only if Q & A present)
    if q and a:
        if _is_yesno_question(q):
            stmt, anchor = _qa_to_statement_yesno(q, a)
            if stmt:
                out.append({"factual_statement": stmt, "original_substring": anchor or a, "origin": "qa_derived"})
        elif _is_wh_question(q):
            stmt, anchor = _qa_to_statement_wh(q, a)
            if stmt:
                out.append({"factual_statement": stmt, "original_substring": anchor or a, "origin": "qa_derived"})

    # (2) Answer sentence splitting
    if a:
        nlp = _load_en()
        doc = nlp(model_output)  # use ORIGINAL answer (to keep exact substrings)
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            out.append({"factual_statement": s, "original_substring": sent.text, "origin": "answer"})

    # (3) Bullet / numbered lines from INPUT and ANSWER (verbatim spans)
    for source_text, origin in ((model_input or "", "input_line"), (model_output or "", "answer")):
        for lo, hi in _extract_bullets(source_text):
            orig = source_text[lo:hi]
            fact = orig.strip()
            if fact:
                out.append({"factual_statement": fact, "original_substring": orig, "origin": origin})

    # (4) If no answer but input is a statement (not a command), split input sentences
    if not a and q and not q.endswith("?"):
        nlp = _load_en()
        doc = nlp(model_input)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                out.append({"factual_statement": s, "original_substring": sent.text, "origin": "input_line"})

    # Deduplicate (preserve order) and cap
    seen = set()
    dedup = []
    for item in out:
        key = (item["factual_statement"].strip().lower(), item["original_substring"])
        if key not in seen:
            seen.add(key)
            dedup.append(item)
        if len(dedup) >= max_items:
            break
    return dedup
