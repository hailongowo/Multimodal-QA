"""Generate the text-preprocessed corpus used by the official baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.baselines.official.preprocess import build_processed_law_file


DEFAULT_INPUT = ROOT / "data" / "law_db" / "vlsp2025_law.json"
DEFAULT_OUTPUT = ROOT / "data" / "baselines" / "official" / "vlsp2025_law.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert HTML law tables to Markdown for the official baseline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    processed = build_processed_law_file(args.input, args.output)
    article_count = sum(len(law["articles"]) for law in processed)
    print(f"Wrote {len(processed)} laws and {article_count} articles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
