"""Validate local inputs and prepare reproducible fine-tuning label aliases."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.finetune_visual_bge.artifacts import write_json_atomic
from legalqa.finetune_visual_bge.corpus import build_candidate_units, read_structured_corpus
from legalqa.finetune_visual_bge.labels import apply_aliases, resolve_label
from legalqa.finetune_visual_bge.model import ensure_visual_bge_available
from legalqa.finetune_visual_bge.prepare import filter_training_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aliases", type=Path, default=ROOT / "data/finetune_visual_bge/label_aliases.json")
    args = parser.parse_args()
    required = [
        "data/train data/vlsp_2025_train.json",
        "data/public_test data/vlsp_2025_public_test.json",
        "data/structured_corpus/structured_corpus.jsonl",
        "artifacts/models/Visualized_m3.pth",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Required local inputs are missing: " + ", ".join(missing))
    ensure_visual_bge_available()
    dependencies = ("numpy", "torch", "PIL", "transformers")
    missing_dependencies = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing_dependencies:
        raise RuntimeError("Missing modules: " + ", ".join(missing_dependencies)
                           + ". Install requirements.txt using this Python interpreter first.")

    candidates = build_candidate_units(read_structured_corpus(ROOT / required[2]))
    if args.aliases.exists():
        aliases = json.loads(args.aliases.read_text(encoding="utf-8"))
    else:
        law = "QCVN 41:2024/BGTVT"
        # Keep the original output IDs; only the candidate text comes from these sources.
        # B names the whole appendix, so retain its full scope instead of guessing one sign.
        aliases = {law: {
            "22 B.15": ["22", "B.15"],
            "B.4_x0000_": ["B.4"],
            "B": sorted(c.label_id for c in candidates
                        if c.law_id == law and c.unit_type == "article" and c.label_id.startswith("B.")),
            # train_432 / train_7_35 shows the R.420 built-up-area sign (D.17).
            "D.1777": ["D.17"],
            # I.414 is the sign number documented in appendix article E.14.
            "I.414": ["E.14"],
        }}
        # A future corpus may already contain one of these IDs.
        keys = {candidate.key for candidate in candidates}
        aliases = {law: {key: sources for key, sources in aliases[law].items() if (law, key) not in keys}}

    expanded = apply_aliases(candidates, aliases)
    errors = []
    training = json.loads((ROOT / required[0]).read_text(encoding="utf-8"))
    valid, rejected = filter_training_questions(
        training, ROOT / "data/train data/train_images", expanded, aliases)
    report = ROOT / "data/finetune_visual_bge/input_rejections.json"
    write_json_atomic(report, {"count": len(rejected), "questions": rejected})
    print(f"Training inputs: kept={len(valid)}, rejected={len(rejected)}; report={report}")
    if not valid:
        raise ValueError("No valid training questions remain; see rejection report")
    for questions_path, images_dir in (
        (required[1], "data/public_test data/public_test_images"),
    ):
        questions = json.loads((ROOT / questions_path).read_text(encoding="utf-8"))
        for question in questions:
            image = ROOT / images_dir / f"{question['image_id']}.jpg"
            if not image.is_file():
                errors.append(f"{question['id']}: missing image {image}")
    if errors:
        raise ValueError("Prerequisite validation failed:\n" + "\n".join(errors))
    if not args.aliases.exists():
        write_json_atomic(args.aliases, aliases)
        print(f"Created label aliases: {args.aliases}")
    else:
        print(f"Validated existing label aliases without overwriting: {args.aliases}")
    print("Validated prerequisites; invalid training questions will be excluded from the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
