"""Prepare hierarchical Visualized-BGE fine-tuning data."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.finetune_visual_bge.prepare import prepare_finetune_data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/train data/vlsp_2025_train.json")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/train data/train_images")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/structured_corpus/structured_corpus.jsonl")
    parser.add_argument("--aliases", type=Path, default=ROOT / "data/finetune_visual_bge/label_aliases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/finetune_visual_bge/manifest.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = prepare_finetune_data(args.train, args.images_dir, args.corpus, args.aliases, args.output,
                                     args.validation_fraction, args.seed)
    print(f"prepared {len(manifest['questions'])} questions, {len(manifest['candidates'])} candidates")
    print(f"coverage: {manifest['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
