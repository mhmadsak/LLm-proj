# inference.py
import argparse
import os
from src.pipeline import process_sample  # new per-sample orchestrator
from src.io_utils import read_jsonl, write_jsonl

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hard-threshold", dest="hard_threshold", type=float, default=0.6)

    p.add_argument("--limit", type=int, default=0, help="process only the first N samples (0=all)")
    
    args = p.parse_args()

    if args.offline:
        os.environ["OFFLINE"] = "true"

    rows_out = []
    for i, sample in enumerate(read_jsonl(args.input), start=1):
        if args.limit and i > args.limit:
            break
        rows_out.append(process_sample(sample,
                                       hard_threshold=args.hard_threshold))

    write_jsonl(args.output, rows_out)

if __name__ == "__main__":
    main()
