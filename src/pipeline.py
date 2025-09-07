# src/pipeline.py
import os
import sys
import glob
import json


# make sibling imports work whether run as a package or plain script
try:
    from .retrieval import retrieve_context
    from .verify import verify_facts_with_context
    from .spans import extract_predicted_spans_hard, extract_predicted_spans_soft
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from retrieval import retrieve_context
    from verify import verify_facts_with_context
    from spans import extract_predicted_spans_hard, extract_predicted_spans_soft


def _iter_jsonl(path: str):
    """
    Yield dicts from:
      - a .jsonl file (one JSON object per line), or
      - a directory containing .jsonl files.
    Uses utf-8-sig to tolerate BOM on Windows.
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


def _split_facts(text: str):
    """ultra-minimal fact splitter: split on periods."""
    return [t.strip() for t in (text or "").split(".") if t.strip()]


def HalluSearch_inference(input_path: str, output_file: str):
    """
    JSONL → JSONL:
      - read items from .jsonl or a dir of .jsonl
      - split model_output_text into facts
      - retrieve context per fact
      - verify facts
      - emit hard/soft spans per item (one line per input)
    """
    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out:
        for item in _iter_jsonl(input_path):
            model_output = item.get("model_output_text", "") or ""
            facts = _split_facts(model_output)

            contexts = [retrieve_context(f) for f in facts]
            verifs   = [verify_facts_with_context(f, c) for f, c in zip(facts, contexts)]

            hard_spans = extract_predicted_spans_hard(facts, verifs)
            soft_spans = extract_predicted_spans_soft(facts, verifs)

            pred = {
                "id": item.get("id"),
                "model_output_text": model_output,
                "hard_labels": hard_spans,
                "soft_labels": soft_spans,
            }
            out.write(json.dumps(pred, ensure_ascii=False) + "\n")
