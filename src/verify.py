# src/verify.py
import os, json, re, requests
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# --- .env loading (robust) -------------------------------------------------
load_dotenv(find_dotenv(usecwd=True), override=True)
root_env = Path(__file__).resolve().parents[1] / ".env"
if root_env.exists():
    load_dotenv(root_env, override=True)

API_KEY  = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
OFFLINE  = os.getenv("OFFLINE", "false").lower() == "true"
DEBUG    = os.getenv("DEBUG", "0") == "1"

def _log(msg: str):
    if DEBUG:
        print(f"[verify] {msg}")

# --- tokenization with character spans -------------------------------------
_TOKEN_RE = re.compile(r"\S+")

def tokenize_with_spans(text: str):
    """Return list of tokens with (i, t, start, end)."""
    out = []
    for i, m in enumerate(_TOKEN_RE.finditer(text or "")):
        out.append({"i": i, "t": m.group(0), "start": m.start(), "end": m.end()})
    return out

# --- DeepSeek call ----------------------------------------------------------
def _deepseek_json(prompt: str, max_tokens: int = 1000):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    r = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    return content

# --- public API -------------------------------------------------------------
def verify_answer_probs(answer_text: str, context: str):
    """
    Compare full answer text against CONTEXT.
    Return dict: {"verdict": "SUPPORTED|NOT SUPPORTED|PARTIAL|UNKNOWN", "token_probs": [floats len=N]}
    """
    answer_text = answer_text or ""
    toks = tokenize_with_spans(answer_text)
    n = len(toks)

    # offline / empty context fallback
    if OFFLINE or not API_KEY:
        _log("offline or missing API key → UNKNOWN @ 0.6")
        return {"verdict": "UNKNOWN", "token_probs": [0.6] * n}
    if not context.strip():
        _log("empty context → UNKNOWN @ 0.6")
        return {"verdict": "UNKNOWN", "token_probs": [0.6] * n}

    tokens_json = [{"i": t["i"], "t": t["t"]} for t in toks]
    n_tokens = len(tokens_json)

    prompt = f"""
You are a professional Fact-Checking AI. Your primary task is to analyze text at the token level to identify hallucinations and contradictions. You must reason step-by-step using only the provided evidence.  
You will compare an ANSWER against the given CONTEXT.
Your job is to decide, for each token in the ANSWER (as tokenized below), the probability that this token reflects a hallucination or contradiction with the CONTEXT.

CONTEXT (the ONLY evidence you may use):
\"\"\"{context[:6000]}\"\"\"


ANSWER (hypothesis to evaluate; NOT evidence):
{answer_text}

TOKENS (index + token; NOT evidence):
{json.dumps(tokens_json, ensure_ascii=False)}

Output ONLY valid JSON with exactly these keys:
{{
  "verdict": "SUPPORTED" | "NOT SUPPORTED" | "PARTIAL",
  "token_probs": [p0, p1, ..., p{n_tokens-1}]
}}

Scoring rules:
- Length: "token_probs" MUST have exactly {n_tokens} floats in [0,1], one per token index.
0.00 — Explicitly supported (exact match in context)

0.10 — Strongly supported (clear paraphrase/alias/inflection)

0.20 — Supported (minor stylistic variation)

0.30 — Similar (close variant; likely same entity/meaning)

0.40 — Related (partial overlap; hints but not explicit)

0.50 — Unclear (no direct evidence either way)

0.60 — Probably unrelated (context weak/silent; leans negative)

0.70 — Mismatch (details don’t line up)

0.80 — Contradicted (clear conflict with context)

0.90 — Strongly contradicted (key facts wrong; multiple conflicts)

1.00 — Fabricated/Impossible (invented or directly refuted)

Verdict rule (overall):
- "SUPPORTED" if most key tokens (entities/dates/numbers/places) are clearly supported (low probs).
- "NOT SUPPORTED" if key tokens contradict (high probs).
- "PARTIAL" if mixed.
Example (for understanding ONLY — do NOT copy in your output; your output MUST use exactly {n_tokens} probabilities for the CURRENT tokens):
- Example CONTEXT: "… Petra van Staveren … won the women’s 100 m breaststroke **gold** medal at the **1984** Summer Olympics in **Los Angeles** …"
- Example FACT: "Petra van **Stoveren** won a **silver** medal in the **2008** Summer Olympics in **Beijing, China**."
- Example reasoning:
  • Name "Stoveren" is a close variant of "Staveren" → similar → probability < 0.5 (e.g., 0.3).  
  • "silver" vs CONTEXT "gold" → contradiction → high (e.g., 0.9).  
  • "2008" vs "1984" → contradiction → ~1.0.  
  • "Beijing/China" vs "Los Angeles/USA" → contradiction → ~0.9.  
- Example OUTPUT:
{{
  "verdict": "NOT SUPPORTED",
  "token_probs": [0.2, 0.2, 0.3, 0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 1.0, 0.9, 0.9, 0.2, 0.9, 0.9]
}}
Another Example:Example (for understanding ONLY — do NOT copy in your output; your output MUST use exactly {n_tokens} probabilities for the CURRENT tokens):

Example CONTEXT: "… The order Erysiphales contains 19 genera …"

Example FACT: "The Elysiphale order contains 5 genera."

Example reasoning:
• "Elysiphale" vs CONTEXT "Erysiphales" → spelling variant / close but not exact → similar → probability ≈0.3–0.4.
• "5" vs CONTEXT "19" → contradicted number → high → ≈0.9–1.0.
• Other tokens like "The", "order", "contains", ".", etc. → directly supported or neutral → very low (≈0.0–0.2).

Example OUTPUT:

{{
  "verdict": "NOT SUPPORTED",
  "token_probs": [0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.0, 0.95, 0.1, 0.0, 0.0]
}}

Now, for the CURRENT task, return ONLY the final JSON object with exactly {n_tokens} probabilities (no explanations, no backticks).

"""

    try:
        content = _deepseek_json(prompt)
        data = json.loads(content)
    except Exception as e:
        _log(f"DeepSeek parse error: {e}")
        return {"verdict": "UNKNOWN", "token_probs": [0.6] * n}

    # Validate and coerce
    probs = data.get("token_probs", [])
    if not isinstance(probs, list):
        probs = []
    coerced = []
    for x in probs[:n]:
        try:
            v = float(x)
        except Exception:
            v = 0.6
        v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
        coerced.append(v)
    if len(coerced) < n:
        coerced += [0.6] * (n - len(coerced))

    verdict = data.get("verdict", "UNKNOWN")
    if verdict not in ("SUPPORTED", "NOT SUPPORTED", "PARTIAL"):
        verdict = "UNKNOWN"

    return {"verdict": verdict, "token_probs": coerced}

# --- Backward-compat shims --------------------------------------------------
def verify_word_probs(fact: str, _unused: str, context: str):
    """Deprecated shim: routes to verify_answer_probs."""
    return verify_answer_probs(fact, context)

def verify_facts_with_context(fact: str, context: str):
    """Deprecated shim: routes to verify_answer_probs."""
    return verify_answer_probs(fact, context)
