# src/verify.py
import os, json, re, requests
from pathlib import Path
from dotenv import load_dotenv, find_dotenv, dotenv_values

# --- .env loading (robust) -------------------------------------------------
load_dotenv(find_dotenv(usecwd=True), override=True)
root_env = Path(__file__).resolve().parents[1] / ".env"
if root_env.exists():
    load_dotenv(root_env, override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")   # set to thinker model if you have it, e.g. "deepseek-reasoner"
OFFLINE  = os.getenv("OFFLINE", "false").lower() == "true"
DEBUG    = os.getenv("DEBUG", "0") == "1"

def _log(msg): 
    if DEBUG: print(f"[verify] {msg}")

# --- tokenization with character spans -------------------------------------
_TOKEN_RE = re.compile(r"\S+")

def tokenize_with_spans(text: str):
    """Return list of tokens with (i, t, start, end) for ORIGINAL_SUBSTRING."""
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
def verify_word_probs(fact: str, original_substring: str, context: str):
    """
    Compare FACT + ORIGINAL_SUBSTRING against CONTEXT.
    Return dict: {"verdict": "SUPPORTED|NOT SUPPORTED|PARTIAL|UNKNOWN", "token_probs": [floats len=N]}
      where token_probs[i] is P(token_i is hallucinated), in [0,1].
    """
    original_substring = original_substring or ""
    toks = tokenize_with_spans(original_substring)
    n = len(toks)

    # offline / empty context fallback
    if OFFLINE or not API_KEY:
        _log("offline or missing API key → zeros")
        return {"verdict": "UNKNOWN", "token_probs": [0.0] * n}
    if not context.strip():
        _log("empty context → conservative probs (0.5)")
        return {"verdict": "UNKNOWN", "token_probs": [0.5] * n}

    # Prepare stable token list for the model
    tokens_json = [{"i": t["i"], "t": t["t"]} for t in toks]

    prompt = f"""
You will compare a FACT and its ORIGINAL_SUBSTRING against the given CONTEXT.
Your job is to decide, for each token in the ORIGINAL_SUBSTRING (as tokenized below), the probability that this token reflects a hallucination or contradiction with the CONTEXT.

CONTEXT:
\"\"\"{context[:6000]}\"\"\"

FACT:
{fact}

ORIGINAL_SUBSTRING TOKENS (index + token):
{json.dumps(tokens_json, ensure_ascii=False)}

Output ONLY valid JSON with exactly these keys:
{{
  "verdict": "SUPPORTED" | "NOT SUPPORTED" | "PARTIAL",
  "token_probs": [p0, p1, ..., p{n-1}]
}}

Rules:
- "token_probs" MUST have length {n}, one float in [0,1] per token index (0 = safe/consistent, 1 = hallucinated/contradicted).
- Use the CONTEXT ONLY. If the context clearly supports a token (name/date/place/number/etc.), give it a low probability (near 0).
- If the context contradicts a token, give it a high probability (near 1).
- If the context is unrelated or unclear, use an intermediate value (e.g., ~0.5).
- The "verdict" is overall: SUPPORTED if most key tokens are consistent, NOT SUPPORTED if key tokens contradict, PARTIAL if mixed.
- Do NOT include explanations or any text besides the JSON object.
"""

    try:
        content = _deepseek_json(prompt)
        data = json.loads(content)
    except Exception as e:
        _log(f"DeepSeek parse error: {e}")
        return {"verdict": "UNKNOWN", "token_probs": [0.5] * n}

    # Validate and coerce lengths
    probs = data.get("token_probs", [])
    if not isinstance(probs, list):
        probs = []
    # coerce to floats in [0,1], pad/trim to n
    probs = [float(x) if isinstance(x, (int, float, str)) and str(x).strip() != "" else 0.5 for x in probs]
    probs = [max(0.0, min(1.0, p)) for p in probs][:n]
    if len(probs) < n:
        probs += [0.5] * (n - len(probs))

    verdict = data.get("verdict", "UNKNOWN")
    if verdict not in ("SUPPORTED", "NOT SUPPORTED", "PARTIAL"):
        verdict = "UNKNOWN"

    return {"verdict": verdict, "token_probs": probs}

# Backward-compat shim (if older code calls this name):
def verify_facts_with_context(fact: str, context: str):
    """Deprecated shim: returns a minimal object (no tokens)."""
    res = verify_word_probs(fact, fact, context)
    return {"supported": res["verdict"] == "SUPPORTED"}
