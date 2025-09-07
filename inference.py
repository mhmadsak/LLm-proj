import argparse
from src.pipeline import HalluSearch_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON")
    parser.add_argument("--output", required=True, help="Path to save predictions JSONL")
    args = parser.parse_args()

    HalluSearch_inference(args.input, args.output)
