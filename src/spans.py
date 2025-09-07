def extract_predicted_spans_hard(facts, verifications):
    """Return spans as [start, end] indices for unsupported facts."""
    spans = []
    start = 0
    for fact, v in zip(facts, verifications):
        end = start + len(fact)
        if not v["supported"]:
            spans.append([start, end])
        start = end + 1
    return spans

def extract_predicted_spans_soft(facts, verifications):
    """Return spans with probabilities."""
    spans = []
    start = 0
    for fact, v in zip(facts, verifications):
        end = start + len(fact)
        prob = 0.0 if v["supported"] else 1.0
        spans.append({"start": start, "end": end, "prob": prob})
        start = end + 1
    return spans