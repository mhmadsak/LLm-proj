import json
from .retrieval import retrieve_context
from .verify import verify_facts_with_context
from .spans import extract_predicted_spans_hard, extract_predicted_spans_soft

def HalluSearch_inference(input_file, output_file):
    """Run the hallucination search pipeline on a JSON input file."""

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in data:
        model_input = item.get("model_input", "")
        model_output = item.get("model_output_text", "")

        facts = model_output.split(".")  # very minimal split
        facts = [f.strip() for f in facts if f.strip()]

        contexts = [retrieve_context(f) for f in facts]
        verifications = [verify_facts_with_context(f, c) for f, c in zip(facts, contexts)]

        hard_spans = extract_predicted_spans_hard(facts, verifications)
        soft_spans = extract_predicted_spans_soft(facts, verifications)

        results.append({
            "id": item.get("id"),
            "model_output_text": model_output,
            "hard_labels": hard_spans,
            "soft_labels": soft_spans,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved results to {output_file}")