"""Evaluate a fine-tuned Visualized-BGE checkpoint."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.evaluation.results import save_results
from legalqa.finetune_visual_bge.evaluation import build_predictions, score_queries, select_questions
from legalqa.finetune_visual_bge.model import load_visualized_bge


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/finetune_visual_bge/manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/finetune_visual_bge/best.pt")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/train data/train_images")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/models/Visualized_m3.pth")
    parser.add_argument("--output-dir", type=Path, help="Result root containing predictions/ and metrics/; defaults to outputs for external questions, runs for validation")
    parser.add_argument("--questions", type=Path, help="Optional labeled JSON; omit for manifest validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    external = json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else None
    questions, positives = select_questions(manifest, external)
    adapter = load_visualized_bge("BAAI/bge-m3", args.weights, args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    adapter.load_state_dict(state["model"])
    scores = score_queries(adapter, questions, manifest["candidates"], args.images_dir)
    predictions = build_predictions(questions, scores, manifest["candidates"], max(10, args.top_k))
    if positives is not None:
        truth = [{"id": q["id"], "relevant_articles": [
            {"law_id": law_id, "article_id": label_id} for law_id, label_id in expected]}
                 for q, expected in zip(questions, positives)]
        output_dir = args.output_dir or (ROOT / "outputs" if args.questions
                                         else ROOT / "runs/finetune_visual_bge/validation")
        metrics = save_results(predictions, truth, predictions, output_dir,
                               "finetune_visual_bge", args.top_k)
        print(metrics)
    else:
        raise ValueError("Evaluation requires labeled questions to produce both result files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


