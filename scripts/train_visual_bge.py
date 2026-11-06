"""Fine-tune Visualized-BGE with mined hard negatives."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.finetune_visual_bge.trainer import TrainingConfig, train


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/finetune_visual_bge/manifest.json")
    parser.add_argument("--negatives", type=Path, default=ROOT / "data/finetune_visual_bge/hard_negatives.json")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/train data/train_images")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/models/Visualized_m3.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/finetune_visual_bge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=.02)
    parser.add_argument("--hard-negative-count", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=100,
        help="Maximum train queries to use (default: 100; 0 uses the full train split).",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=10,
        help="Write progress to console and train.log every N batches (default: 10).",
    )
    parser.add_argument("--freeze-policy", choices=("none", "vision", "all_but_projection"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.max_train_samples < 0:
        parser.error("--max-train-samples must be 0 or a positive integer")
    if args.log_every_steps < 1:
        parser.error("--log-every-steps must be a positive integer")
    if args.max_train_samples == 0:
        args.max_train_samples = None
    result = train(TrainingConfig(**vars(args)))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
