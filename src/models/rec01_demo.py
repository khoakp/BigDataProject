"""Print Top-20 recommendations from a verified REC-01 Kaggle artifact.

Load the selected checkpoint and train graph once, then rank each requested user.
Cold users receive the train-only Popularity fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.models.rec01_inference import load_rec01_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path,
                        help="Directory containing report.json and the selected checkpoint")
    parser.add_argument("--movies", required=True, type=Path,
                        help="MovieLens movies.csv for title/genres after ranking")
    parser.add_argument("--user", required=True, type=int, action="append",
                        help="Raw user ID; repeat to query multiple users")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    service = load_rec01_inference(args.run, args.movies, device_name=args.device)
    results = [service.recommend(user) for user in args.user]
    json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
