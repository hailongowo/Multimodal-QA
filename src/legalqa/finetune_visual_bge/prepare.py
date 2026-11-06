"""Prepare deterministic fine-tuning manifests."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .corpus import build_candidate_units, read_structured_corpus
from .labels import apply_aliases, resolve_label


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def grouped_split(questions: list[dict], validation_fraction: float = 0.2, seed: int = 42) -> tuple[list[str], list[str]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    groups: dict[str, list[str]] = {}
    for question in questions:
        groups.setdefault(str(question["image_id"]), []).append(str(question["id"]))
    items = sorted(groups.items())
    random.Random(seed).shuffle(items)
    target = round(len(questions) * validation_fraction)
    validation_groups = []
    count = 0
    for image_id, query_ids in sorted(items, key=lambda item: (-len(item[1]), item[0])):
        if count < target:
            validation_groups.append((image_id, query_ids))
            count += len(query_ids)
    validation = {query_id for _, query_ids in validation_groups for query_id in query_ids}
    return ([str(q["id"]) for q in questions if str(q["id"]) not in validation],
            [str(q["id"]) for q in questions if str(q["id"]) in validation])


def prepare_finetune_data(train_path: Path, images_dir: Path, corpus_path: Path, aliases_path: Path,
                          output_path: Path, validation_fraction: float = 0.2, seed: int = 42) -> dict:
    questions = json.loads(train_path.read_text(encoding="utf-8"))
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
    candidates = apply_aliases(build_candidate_units(read_structured_corpus(corpus_path)), aliases)
    candidate_index = {candidate.key: index for index, candidate in enumerate(candidates)}
    coverage = Counter()
    prepared = []
    for question in questions:
        image_path = images_dir / f"{question['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for query {question['id']}: {image_path}")
        positives = []
        for label in question.get("relevant_articles", []):
            resolved = resolve_label(label["law_id"], label["article_id"], candidates, aliases)
            coverage[resolved.resolution] += 1
            positives.append({
                "law_id": resolved.law_id,
                "raw_id": resolved.raw_id,
                "label_id": resolved.canonical_id,
                "unit_type": resolved.unit_type,
                "resolution": resolved.resolution,
                "candidate_index": candidate_index[resolved.candidate_key],
            })
        if not positives:
            raise ValueError(f"query {question['id']} has no positive labels")
        prepared.append({
            "id": question["id"], "image_id": question["image_id"], "question": question["question"],
            "question_type": question.get("question_type"), "positives": positives,
        })
    train_ids, validation_ids = grouped_split(questions, validation_fraction, seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "sources": {"train_sha256": _sha256(train_path), "corpus_sha256": _sha256(corpus_path),
                    "aliases_sha256": _sha256(aliases_path)},
        "candidates": [{"law_id": c.law_id, "label_id": c.label_id, "unit_type": c.unit_type,
                        "text": c.text, "provenance": list(c.provenance)} for c in candidates],
        "questions": prepared,
        "splits": {"train": train_ids, "validation": validation_ids},
        "coverage": dict(sorted(coverage.items())),
    }
    _write_json_atomic(output_path, manifest)
    return manifest
