
import os
import re
import time
from typing import List, Tuple, Dict, Optional

import requests

# --------------------------- Config -----------------------------------------

CTX_MIN_LEN = int(os.getenv("CTX_MIN_LEN", "200"))

def _is_weak(ctx: str, min_len: int = CTX_MIN_LEN) -> bool:
    return not ctx or len(ctx) < min_len

# ----------------------- Google CSE Search ----------------------------------

def google_cse_search(query: str, k: int = 5, timeout_s: float = 8.0, retries: int = 2) -> str:
    """
    Query Google Custom Search Engine and return a newline-joined blob of
    (title + snippet + url) lines. Returns "" on failure or OFFLINE mode.
    """
    if not query or not query.strip():
        return ""
    if os.getenv("OFFLINE", "").lower() == "true":
        return ""

    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE")
    if not api_key or not engine_id:
        return ""

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": engine_id, "q": query, "num": min(k, 10)}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", []) or []
                lines = []
                for it in items[:k]:
                    title = it.get("title", "") or ""
                    snippet = it.get("snippet", "") or ""
                    link = it.get("link", "") or ""
                    lines.append(" ".join([title, snippet, link]).strip())
                return "\n".join(lines).strip()
            else:
                last_err = Exception(f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = e
        time.sleep(0.5 * (attempt + 1))  # simple backoff
    return ""

# -------------------- Keyword fallback (question-only) ----------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")

def extract_keywords(text: str, min_len: int = 3, max_keywords: int = 8) -> List[str]:
    """
    Minimal keyword extractor:
      - alphanumeric (allows -/_),
      - keep tokens with len >= min_len (digits may be shorter),
      - deduplicate preserving order.
    """
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        low = tok.lower()
        if len(low) < min_len and not low.isdigit():
            continue
        if low not in seen:
            seen.add(low)
            out.append(tok)
        if len(out) >= max_keywords:
            break
    return out

def build_keyword_query(user_question: str, max_keywords: int = 8) -> str:
    """
    Construct a keyword-only query from the user question.
    """
    kws = extract_keywords(user_question, max_keywords=max_keywords)
    return " ".join(kws)

# --------------------- Public: retrieve with fallback -----------------------

def retrieve_context(user_question: str, k: int = 5) -> Tuple[str, Dict]:
    """
    Try primary Google CSE with the raw question.
    If weak/empty, retry with keyword-only query.
    Returns (context, meta).
    meta includes:
      - ctx_source: 'cse' | 'cse_keywords' | 'none'
      - raw_query / keyword_query (when applicable)
      - ctx_len
    """
    meta: Dict = {
        "ctx_source": "none",
        "ctx_len": 0,
        "raw_query": (user_question or "").strip(),
    }

    # Primary: raw question
    ctx = google_cse_search(meta["raw_query"], k=k)
    if not _is_weak(ctx):
        meta["ctx_source"] = "cse"
        meta["ctx_len"] = len(ctx)
        return ctx, meta

    # Fallback #1: keyword query (question only)
    kw_query = build_keyword_query(user_question, max_keywords=8)
    meta["keyword_query"] = kw_query
    ctx2 = google_cse_search(kw_query, k=max(k, 6)) if kw_query else ""
    if not _is_weak(ctx2):
        meta["ctx_source"] = "cse_keywords"
        meta["ctx_len"] = len(ctx2)
        return ctx2, meta

    # Still weak → let pipeline decide LLM fallback
    meta["ctx_len"] = 0
    return "", meta