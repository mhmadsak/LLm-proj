# inference.py
import argparse
from src.pipeline import HalluSearch_inference

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hard-threshold", dest="hard_threshold", type=float, default=0.5)
    p.add_argument("--max-items", dest="max_items", type=int, default=30)
    args = p.parse_args()

    HalluSearch_inference(
        input_path=args.input,
        output_file=args.output,
        max_items_per_sample=args.max_items,
        hard_threshold=args.hard_threshold,
    )
