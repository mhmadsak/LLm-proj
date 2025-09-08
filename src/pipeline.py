# src/pipeline.py
import os
import sys
import json
import glob

# Make sibling imports work whether run as a module or a script
try:
    from .splitting import split_en_general            # Q/A-aware English splitter
    from .retrieval import retrieve_context               # Google CSE retriever (your version)
    from .verify import verify_word_probs                 # DeepSeek per-word probabilities
    from .spans import compute_spans                      # aggregate to soft + hard
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from splitting import split_en_general
    from retrieval import retrieve_context
    from verify import verify_word_probs
    from spans import compute_spans


def _iter_jsonl(path: str):
    """
    Yield dicts from a .jsonl file (one JSON object per line),
    or from all .jsonl files inside a directory.
    """
    enc = "utf-8-sig"

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if not files:
            raise ValueError(f"No .jsonl files found in directory: {path}")
        for fp in files:
            yield from _iter_jsonl(fp)
        return

    if not path.lower().endswith(".jsonl"):
        raise ValueError(f"Expected a .jsonl file (or directory). Got: {path}")

    with open(path, "r", encoding=enc) as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"[JSONL parse error] {path} line {i}: {e}") from e


def _enrich_context(primary: str, qa_contexts: list[str], max_chars: int = 1500, qa_take: int = 2) -> str:
    """
    Build the context for an ANSWER fact by concatenating:
      - its own primary context (if any)
      - up to `qa_take` QA-derived contexts
    Truncated to `max_chars`.
    """
    chunks = []
    if primary and primary.strip():
        chunks.append(primary)
    added = 0
    for cx in qa_contexts:
        if not cx or not cx.strip():
            continue
        chunks.append(cx)
        added += 1
        if added >= qa_take:
            break
    joined = "\n---\n".join(chunks) if chunks else ""
    return joined[:max_chars]


def HalluSearch_inference(
    input_path: str,
    output_file: str,
    max_items_per_sample: int = 30,
    hard_threshold: float = 0.5,
    combine: str = "max",
    qa_take: int = 2,
    max_context_chars: int = 1500,
):
    """
    Pipeline:
      1) Read .jsonl (file or directory)
      2) Split (EN) using both model_input (question) and model_output_text (answer)
      3) Retrieve contexts for:
           - answer-derived facts (for labeling)
           - QA-derived facts (to enrich evidence)
      4) For EACH answer fact, enrich its context with a few QA contexts (web text only)
      5) Verify ONLY answer sentences → per-word probabilities
      6) Aggregate to soft labels; threshold to hard labels
      7) Write one JSONL line per input item
    """
    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for item in _iter_jsonl(input_path):
            question = (item.get("model_input") or "").strip()
            answer   = (item.get("model_output_text") or "").strip()

            # (1) Split to atomic pieces (keeps verbatim substrings + origin tags)
            pieces = split_en_general(question, answer, max_items=max_items_per_sample)

            # Separate ANSWER vs QA-DERIVED
            ans_idxs = [i for i, p in enumerate(pieces) if p.get("origin") == "answer"]
            qa_idxs  = [i for i, p in enumerate(pieces) if p.get("origin") == "qa_derived"]

            ans_facts     = [pieces[i]["factual_statement"]  for i in ans_idxs]
            ans_originals = [pieces[i]["original_substring"] for i in ans_idxs]
            qa_facts      = [pieces[i]["factual_statement"]  for i in qa_idxs]

            # (2) Retrieve contexts for BOTH groups
            ctx_ans = [retrieve_context(f) for f in ans_facts]
            ctx_qa  = [retrieve_context(f) for f in qa_facts]

            # (3) Enrich each answer context with a few QA contexts (still web-only evidence)
            enriched_ctx = [
                _enrich_context(c_primary, ctx_qa, max_chars=max_context_chars, qa_take=qa_take)
                for c_primary in ctx_ans
            ]

            # (4) Verify ONLY answer pieces → per-word probabilities
            verifs_ans = [
                verify_word_probs(fact, orig, ctx)
                for fact, orig, ctx in zip(ans_facts, ans_originals, enriched_ctx)
            ]

            # (5) Aggregate to soft; threshold to hard (answer-only)
            hard_spans, soft_spans = compute_spans(
                answer_text=answer,
                originals=ans_originals,
                verifs=verifs_ans,
                threshold=hard_threshold,
                combine=combine,  # "max" | "mean" | "min"
            )

            # (6) Emit one JSONL line
            pred = {
                "id": item.get("id"),
                "model_input": question,
                "model_output_text": answer,
                "hard_labels": hard_spans,   # [[start, end], ...]
                "soft_labels": soft_spans,   # [{start,end,prob}, ...]
                "verifications": verifs_ans, # ONLY answer-based verifications
            }
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".jsonl file or directory")
    ap.add_argument("--output", required=True, help="output .jsonl path")
    ap.add_argument("--max-items", dest="max_items", type=int, default=30, help="max facts per sample")
    ap.add_argument("--hard-threshold", dest="hard_threshold", type=float, default=0.5,
                    help="prob >= threshold → hard hallucination token")
    ap.add_argument("--combine", choices=["max", "mean", "min"], default="max",
                    help="how to combine duplicate-token probabilities across answer sentences")
    ap.add_argument("--qa-take", dest="qa_take", type=int, default=2,
                    help="how many QA contexts to append to each answer context")
    ap.add_argument("--max-context-chars", dest="max_context_chars", type=int, default=1500,
                    help="truncate enriched context to this many characters")
    args = ap.parse_args()

    HalluSearch_inference(
        input_path=args.input,
        output_file=args.output,
        max_items_per_sample=args.max_items,
        hard_threshold=args.hard_threshold,
        combine=args.combine,
        qa_take=args.qa_take,
        max_context_chars=args.max_context_chars,
    )
