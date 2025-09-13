# src/pipeline.py
import os
from typing import Dict, Any, List, Tuple
import requests

from .retrieval import retrieve_context
from .verify import verify_answer_probs
from .splitting import extract_statements
from .spans import (
    token_spans,
    aggregate_claims_to_answer_probs,
    probs_to_hard_spans,
    merge_to_soft_spans,
)

# ----------------------- env & config -----------------------
CTX_MIN_LEN = int(os.getenv("CTX_MIN_LEN", "200"))
OFFLINE = os.getenv("OFFLINE", "false").lower() == "true"
DEBUG = os.getenv("DEBUG", "0") == "1"

USE_CLAIM_SPLITTING = os.getenv("USE_CLAIM_SPLITTING", "0") == "1"   # toggle splitting
PER_TOKEN_AGG = os.getenv("PER_TOKEN_AGG", "max")                     # max | mean | min

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")      # or deepseek-reasoner

def _log(msg: str) -> None:
    if DEBUG:
        print(f"[pipeline] {msg}")

# ----------------------- helpers ----------------------------
def _is_weak(ctx: str, min_len: int = CTX_MIN_LEN) -> bool:
    return not ctx or len(ctx) < min_len

def _answer_token_count(ans: str) -> int:
    return len(token_spans(ans or ""))

def _llm_construct_reference_context(user_question: str, answer_text: str, min_chars: int = 400) -> str:
    """
    Fallback #2: Ask DeepSeek to draft a neutral, reference-like context
    when web retrieval fails. Returns "" if offline/missing key/too short.
    """
    if OFFLINE or not DEEPSEEK_API_KEY:
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
    except Exception as e:
        _log(f"LLM context error: {e}")
        return ""

def _map_token_spans_to_char_spans(ans: str, token_spans_list: List[Tuple[int, int]],
                                   hard_spans_tok: List[Tuple[int, int]],
                                   soft_spans_tok: List[Dict[str, Any]]):
    """
    Convert token-index spans to character-index spans using the provided token offsets.
    """
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
    for d in soft_spans_tok:  # {"start": s_tok, "end": e_tok, "prob": p}
        ce = tokspan_to_charspan(d["start"], d["end"])
        if ce is not None:
            soft_labels_char.append({"start": ce[0], "end": ce[1], "prob": d["prob"]})

    return hard_labels_char, soft_labels_char

# ----------------------- main API ---------------------------
def process_sample(
    sample: Dict[str, Any],
    hard_threshold: float = 0.6,
    combine: str = "mean",
) -> Dict[str, Any]:
    """
    Orchestrates:
      1) Retrieve with model_input only (primary + keyword retry).
      2) If still weak → Fallback #2 (LLM reference context).
      3) Verify:
         - whole-answer mode (default), or
         - claim-splitting mode (USE_CLAIM_SPLITTING=1) with aggregation.
      4) Build hard/soft spans and map to character intervals.
    """
    sid   = sample.get("id")
    q_in  = (sample.get("model_input") or "").strip()
    ans   = (sample.get("model_output_text") or "").strip()

    meta = {
        "ctx_source": "none",           # cse | cse_keywords | llm_constructed | none
        "ctx_len": 0,
        "offline": OFFLINE,
        "verify_model": DEEPSEEK_MODEL,
        "hard_threshold": hard_threshold,
        "combine": combine,
        "use_claim_splitting": USE_CLAIM_SPLITTING,
        "per_token_agg": PER_TOKEN_AGG,
    }

    # 1) Retrieval (model_input only) with keyword fallback
    ctx, rmeta = retrieve_context(q_in, k=5)
    meta.update({k: v for k, v in rmeta.items() if k in ("raw_query", "keyword_query")})
    meta["ctx_source"] = rmeta.get("ctx_source", "none")
    meta["ctx_len"]    = len(ctx)
    _log(f"retrieval source={meta['ctx_source']} len={meta['ctx_len']}")

    # 2) Fallback #2: LLM-constructed reference context
    if _is_weak(ctx):
        _log("primary retrieval weak → trying llm_constructed context")
        ctx_llm = _llm_construct_reference_context(q_in, ans)
        if not _is_weak(ctx_llm):
            ctx = ctx_llm
            meta["ctx_source"] = "llm_constructed"
            meta["ctx_len"] = len(ctx)
            _log(f"llm_constructed context len={meta['ctx_len']}")

    # 3) If still weak → UNKNOWN with neutral 0.6 per token
    if _is_weak(ctx):
        n_tok = _answer_token_count(ans)
        token_probs = [0.6] * n_tok
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

    # 4) Verification
    if USE_CLAIM_SPLITTING:
        # Split into claims (as exact substrings when available)
        parts = extract_statements(ans) or []
        claim_texts: List[str] = []
        for i, obj in enumerate(parts, start=1):
            orig_key = f"original_substring_{i}"
            fact_key = f"factual_statement_{i}"
            claim_texts.append(obj.get(orig_key) or obj.get(fact_key) or "")

        claim_probs_list: List[List[float]] = []
        claim_verdicts: List[str] = []
        for ct in claim_texts:
            v = verify_answer_probs(ct, ctx)
            claim_verdicts.append(v.get("verdict", "UNKNOWN"))
            claim_probs_list.append([float(x) for x in (v.get("token_probs") or [])])

        # Aggregate claim-level probs back to the full answer tokens
        token_probs = aggregate_claims_to_answer_probs(
            answer_text=ans,
            claim_originals=claim_texts,
            claim_token_probs=claim_probs_list,
            per_token_agg=PER_TOKEN_AGG,
            default_prob=0.6,
        )

        # Verdict policy from claims
        if any(v == "NOT SUPPORTED" for v in claim_verdicts):
            verdict = "NOT SUPPORTED"
        elif all(v == "SUPPORTED" for v in claim_verdicts if v != "UNKNOWN") and token_probs and max(token_probs) < hard_threshold:
            verdict = "SUPPORTED"
        else:
            verdict = "PARTIAL"
        meta["claims_count"] = len(claim_texts)
    else:
        vres = verify_answer_probs(ans, ctx)
        token_probs = [float(x) for x in (vres.get("token_probs") or [])]
        verdict     = vres.get("verdict", "UNKNOWN")

    # 5) Spans (token-index) → map to character intervals
    hard_spans_tok = probs_to_hard_spans(token_probs, hard_threshold=hard_threshold, dead_zone=0.08)
    soft_spans_tok = merge_to_soft_spans(token_probs, hard_spans_tok, combine=combine)

    tok_offsets = token_spans(ans)  # [(start_char, end_char) for each token]
    hard_labels_char, soft_labels_char = _map_token_spans_to_char_spans(
        ans, tok_offsets, hard_spans_tok, soft_spans_tok
    )

    # 6) Return record
    return {
        "id": sid,
        "model_input": q_in,
        "model_output_text": ans,
        "retrieved_context": ctx,
        "verdict": verdict,              # "SUPPORTED" | "NOT SUPPORTED" | "PARTIAL" | "UNKNOWN"
        "token_probs": token_probs,      # per-answer-token probs (raw floats)
        "hard_labels": hard_labels_char, # [[start_char, end_char], ...]
        "soft_labels": soft_labels_char, # [{"start":..,"end":..,"prob":..}, ...]
        "meta": meta,
    }

# ------------- optional back-compat wrapper (legacy code) -------------------
def HalluSearch_inference(input_path: str, output_file: str,
                          hard_threshold: float = 0.6,
                          combine: str = "mean",
                          max_items_per_sample: int = None):
    """
    Legacy wrapper to mimic older entrypoint.
    Reads JSONL, processes samples with process_sample, writes JSONL.
    Ignores 'max_items_per_sample' (kept for compatibility).
    """
    from .io_utils import read_jsonl, write_jsonl
    rows_out = []
    for sample in read_jsonl(input_path):
        rows_out.append(process_sample(sample, hard_threshold=hard_threshold, combine=combine))
    write_jsonl(output_file, rows_out)
