import os, re, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# robust .env load
load_dotenv(find_dotenv(usecwd=True), override=True)
root_env = Path(__file__).resolve().parents[1] / ".env"
if root_env.exists():
    load_dotenv(root_env, override=True)

GOOGLE_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CX  = os.getenv("GOOGLE_SEARCH_ENGINE")
OFFLINE    = os.getenv("OFFLINE", "false").lower() == "true"
DEBUG      = os.getenv("DEBUG", "1") == "1"

# Prefer authoritative domains; skip noisy ones
WHITELIST = (
    "wikipedia.org", "britannica.com", ".edu", ".gov", "un.org",
    "who.int", "nature.com", "sciencedirect.com", "nih.gov",
    "olympics.com", "olympedia.org"
)
BLOCKLIST = (
    "reddit.com", "facebook.com", "twitter.com", "x.com",
    "quora.com", "stackexchange.com", "flightaware.com",
    "aclanthology.org", "huggingface.co"
)

def _log(m): 
    if DEBUG: print(f"[retrieval] {m}")

def _clean_fact(f: str) -> str:
    f = (f or "").replace("…", " ").strip()
    f = re.sub(r"\s+", " ", f)
    f = re.sub(r"^(yes|no|however|but|and|so|thus),?\s+", "", f, flags=re.I)
    return f

def _ok_domain(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    if any(b in host for b in BLOCKLIST): return False
    return any(host == w or host.endswith(w) for w in WHITELIST)

def _require_google():
    if not GOOGLE_KEY or not GOOGLE_CX:
        raise RuntimeError("Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE in .env")

def google_search(query, num=8):
    if OFFLINE:
        _log("OFFLINE → skip CSE")
        return []
    _require_google()
    url = "https://www.googleapis.com/customsearch/v1"
    r = requests.get(url, params={"key": GOOGLE_KEY, "cx": GOOGLE_CX, "q": query, "num": min(10, num)}, timeout=25)
    if r.status_code != 200:
        _log(f"CSE {r.status_code}: {r.text[:160]}")
        return []
    items = r.json().get("items", []) or []
    links = [it.get("link") for it in items if it.get("link")]
    _log(f"q='{query[:70]}…' hits={len(links)}")
    return links[:num]

def get_page_text(url):
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "noscript"]): t.decompose()
        txt = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", txt)[:8000]
    except Exception as e:
        _log(f"fetch fail {url}: {e}")
        return ""

def _best_window(text: str, query: str, win=1200):
    """Cheap keyword windowing so verify sees the most relevant chunk."""
    if not text: return ""
    q_terms = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']+", query) if len(w) > 2][:12]
    if not q_terms: return text[:win]
    best, bi = 0, 0
    step = max(1, win // 2)
    L = len(text)
    for i in range(0, max(1, L - win), step):
        chunk = text[i:i+win].lower()
        score = sum(chunk.count(t) for t in q_terms)
        if score > best: best, bi = score, i
    return text[bi:bi+win]

def _person_hint(fact: str) -> str:
    """
    Very small heuristic: take words before ' won| is | was ' as a name candidate.
    Handles particles like 'van', 'de', 'von' in lowercase.
    """
    m = re.split(r"\b(won|is|was)\b", fact, maxsplit=1, flags=re.I)
    head = m[0].strip() if m else fact.strip()
    # Keep up to ~5 tokens, allow lowercase particles inside
    tokens = head.split()
    if 1 <= len(tokens) <= 7 and any(t[0:1].isupper() for t in tokens):
        return " ".join(tokens)
    return ""

def _topic_queries(fact: str):
    """Generate multiple queries; bias to authoritative sites for Olympics-like claims."""
    q = _clean_fact(fact)
    queries = [q]
    # quoted head (first ~10 tokens)
    head = " ".join(q.split()[:10])
    if head:
        queries.append(f"\"{head}\"")
    # Olympics / medal heuristics
    if re.search(r"\b(olympic|olympics|medal|summer games|winter games)\b", q, flags=re.I):
        name = _person_hint(q)
        if name:
            queries += [
                f"\"{name}\" site:olympics.com",
                f"\"{name}\" site:olympedia.org",
                f"\"{name}\" site:wikipedia.org",
            ]
        queries += [
            f"{q} site:olympics.com",
            f"{q} site:olympedia.org",
            f"{q} site:wikipedia.org",
        ]
    return queries

def retrieve_context(fact: str) -> str:
    # Try a few targeted queries, dedupe links, prefer whitelist
    seen = set()
    cand = []
    for qq in _topic_queries(fact):
        for link in google_search(qq, num=8):
            if link in seen: continue
            seen.add(link)
            cand.append(link)
    # prioritize authoritative domains first
    pri = [u for u in cand if _ok_domain(u)] + [u for u in cand if not _ok_domain(u)]
    for link in pri[:8]:
        txt = get_page_text(link)
        if len(txt) >= 400:
            snip = _best_window(txt, _clean_fact(fact), win=1200)
            _log(f"context_ok len={len(snip)} from={link}")
            return snip
    _log("no context found")
    return ""
