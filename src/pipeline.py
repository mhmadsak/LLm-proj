# src/pipeline.py
import os
from typing import Dict, Any, List, Tuple
import requests

from .retrieval import retrieve_context
from .verify import verify_answer_probs
from .spans import (
    token_spans,
    probs_to_hard_spans,
    merge_to_soft_spans,
)

# ----------------------- env & config -----------------------
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ----------------------- helpers ----------------------------
CTX_MIN_LEN = 250

def _is_weak(ctx: str, min_len: int = CTX_MIN_LEN) -> bool:
    return not ctx or len(ctx) < min_len

def _answer_token_count(ans: str) -> int:
    return len(token_spans(ans or ""))

def _llm_construct_reference_context(user_question: str, answer_text: str, min_chars: int = 400) -> str:
    """
    Fallback: Ask DeepSeek to draft a neutral, reference-like context
    when web retrieval fails. Returns "" if missing key/too short.
    """
    if not DEEPSEEK_API_KEY:
        return ""
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    prompt = (
        "You are a careful reference writer. Based ONLY on general encyclopedic knowledge, "
        "draft a neutral, citation-style context that could be used to check the answer. "
        "No opinions or reasoning steps. Write 1–2 concise paragraphs covering names, dates, "
        "definitions, lists, and distinctions likely relevant to verifying the answer.\n\n"
        f"QUESTION:\n{user_question}\n\nANSWER:\n{answer_text}"
    )
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 600,
    }
    try:
        r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        text = text.strip()
        if len(text) < min_chars:
            return ""
        return text
    except Exception:
        return ""

def _map_token_spans_to_char_spans(
    ans: str,
    token_spans_list: List[Tuple[int, int]],
    hard_spans_tok: List[Tuple[int, int]],
    soft_spans_tok: List[Dict[str, Any]],
):
    """Convert token-index spans to character-index spans using the provided token offsets."""
    def tokspan_to_charspan(s_tok: int, e_tok: int):
        if e_tok <= s_tok or s_tok < 0 or e_tok > len(token_spans_list):
            return None
        return [token_spans_list[s_tok][0], token_spans_list[e_tok - 1][1]]

    hard_labels_char: List[List[int]] = []
    for s_tok, e_tok in hard_spans_tok:
        ce = tokspan_to_charspan(s_tok, e_tok)
        if ce is not None:
            hard_labels_char.append(ce)

    soft_labels_char: List[Dict[str, Any]] = []
    for d in soft_spans_tok:
        ce = tokspan_to_charspan(d["start"], d["end"])
        if ce is not None:
            soft_labels_char.append({"start": ce[0], "end": ce[1], "prob": d["prob"]})

    return hard_labels_char, soft_labels_char

# ----------------------- main API ---------------------------
def process_sample(
    sample: Dict[str, Any],
    hard_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Flow:
      1) Retrieve with model_input only.
      2) If weak context → try LLM-constructed context.
      3) If still weak → return UNKNOWN with neutral per-token probs (0.5).
      4) Verify whole answer (no claim splitting).
      5) Produce hard/soft spans and map to character intervals.
    """
    sid  = sample.get("id")
    q_in = (sample.get("model_input") or "").strip()
    ans  = (sample.get("model_output_text") or "").strip()

    meta = {
        "ctx_source": "none",           # cse | cse_keywords | llm_constructed | none
        "ctx_len": 0,
        "verify_model": DEEPSEEK_MODEL,
        "hard_threshold": hard_threshold,
    }

    # 1) Retrieval
    ctx, rmeta = retrieve_context(q_in, k=5)
    meta.update({k: v for k, v in (rmeta or {}).items() if k in ("raw_query", "keyword_query", "ctx_source")})
    meta["ctx_source"] = (rmeta or {}).get("ctx_source", "none")
    meta["ctx_len"] = len(ctx or "")

    # 2) Fallback: LLM-constructed reference context
    if _is_weak(ctx):
        ctx_llm = _llm_construct_reference_context(q_in, ans)
        if not _is_weak(ctx_llm):
            ctx = ctx_llm
            meta["ctx_source"] = "llm_constructed"
            meta["ctx_len"] = len(ctx)

    # 3) If still weak → UNKNOWN
    if _is_weak(ctx):
        n_tok = _answer_token_count(ans)
        token_probs = [0.5] * n_tok
        return {
            "id": sid,
            "model_input": q_in,
            "model_output_text": ans,
            "retrieved_context": "",
            "verdict": "UNKNOWN",
            "token_probs": token_probs,
            "hard_labels": [],
            "soft_labels": [],
            "meta": meta,
        }

    # 4) Verification (whole-answer only)
    vres = verify_answer_probs(ans, ctx)
    token_probs = [float(x) for x in (vres.get("token_probs") or [])]
    verdict     = vres.get("verdict", "UNKNOWN")

    # 5) Spans
    hard_spans_tok = probs_to_hard_spans(token_probs, hard_threshold=hard_threshold, dead_zone=0.08)
    soft_spans_tok = merge_to_soft_spans(token_probs, hard_spans_tok)

    tok_offsets = token_spans(ans)
    hard_labels_char, soft_labels_char = _map_token_spans_to_char_spans(ans, tok_offsets, hard_spans_tok, soft_spans_tok)

    return {
        "id": sid,
        "model_input": q_in,
        "model_output_text": ans,
        "retrieved_context": ctx,
        "verdict": verdict,
        "token_probs": token_probs,
        "hard_labels": hard_labels_char,
        "soft_labels": soft_labels_char,
        "meta": meta,
    }

# ----------------------- legacy wrapper ---------------------
def HalluSearch_inference(input_path: str, output_file: str,
                          hard_threshold: float = 0.6,
                          max_items_per_sample: int = None):
    """
    Legacy wrapper. Ignores 'max_items_per_sample' (kept for compatibility).
    """
    from .io_utils import read_jsonl, write_jsonl
    rows_out = []
    for sample in read_jsonl(input_path):
        rows_out.append(process_sample(sample, hard_threshold=hard_threshold))
    write_jsonl(output_file, rows_out)
