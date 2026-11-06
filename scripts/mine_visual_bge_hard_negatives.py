"""Mine hard negatives with the frozen baseline Visualized-BGE."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.finetune_visual_bge.artifacts import fingerprint_json, write_json_atomic
from legalqa.finetune_visual_bge.mining import mine_hard_negatives
from legalqa.finetune_visual_bge.model import load_visualized_bge


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/finetune_visual_bge/manifest.json")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data/train data/train_images")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/models/Visualized_m3.pth")
    parser.add_argument("--output", type=Path, default=ROOT / "data/finetune_visual_bge/hard_negatives.json")
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_ids = set(manifest["splits"]["train"])
    questions = [q for q in manifest["questions"] if q["id"] in train_ids]
    model = load_visualized_bge(args.model_name, args.weights, args.device, freeze_policy="none")
    candidate_embeddings = model.encode_documents([c["text"] for c in manifest["candidates"]])
    query_embeddings = model.encode_queries(
        [args.images_dir / f"{q['image_id']}.jpg" for q in questions], [q["question"] for q in questions])
    result = mine_hard_negatives(query_embeddings, candidate_embeddings, questions, manifest["candidates"], args.top_n)
    result.update({"manifest_fingerprint": fingerprint_json(manifest), "model_name": args.model_name,
                   "model_weight": str(args.weights), "visual_bge_revision": "fd1a2bdf69488ffebe0327999d4400d8c8058a0b"})
    write_json_atomic(args.output, result)
    print(f"mined negatives for {len(questions)} training queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
