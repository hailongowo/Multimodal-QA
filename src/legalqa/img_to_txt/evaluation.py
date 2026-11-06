"""Description loading and legacy evaluation helpers for image-to-text queries."""

from __future__ import annotations

import json
from pathlib import Path

from legalqa.evaluation.metrics import eval_retrieval
from legalqa.img_to_txt.documents import flatten_description


def build_query(
    question: dict[str, object], description: dict[str, object] | None
) -> str:
    parts = [str(question.get("question", ""))]
    if description is not None:
        parts.append(flatten_description(description))
    return "\n".join(part for part in parts if part)


def build_dense_query(question: dict, descriptions: dict[str, dict[str, object]]) -> str:
    image_id = question.get("image_id")
    text = question.get("question")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("question image_id must be a non-empty string")
    if not isinstance(text, str) or not text:
        raise ValueError("question text must be a non-empty string")
    return build_query(question, descriptions.get(image_id))


def load_test_descriptions(path: Path) -> dict[str, dict[str, object]]:
    """Load and validate precomputed descriptions for evaluation images."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        actual = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(
            f"test description schema_version must be '1.0', got {actual!r}"
        )
    descriptions = payload.get("descriptions")
    if not isinstance(descriptions, dict):
        raise ValueError("test description descriptions must be an object")

    validated: dict[str, dict[str, object]] = {}
    for image_id, description in descriptions.items():
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("test description image_id must be a non-empty string")
        try:
            flatten_description(description)
        except ValueError as error:
            raise ValueError(
                f"invalid test description for image_id {image_id!r}: {error}"
            ) from error
        validated[image_id] = description
    return validated


def _save_predictions(path: Path, predictions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def evaluate_questions(
    questions: list[dict[str, object]],
    index,
    descriptions: dict[str, dict[str, object]],
    output_path: Path,
    top_k: int = 5,
) -> dict[str, float] | None:
    seen_question_ids: set[str] = set()
    for question in questions:
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question id must be a non-empty string")
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate question id: {question_id}")
        seen_question_ids.add(question_id)

    label_presence = ["relevant_articles" in question for question in questions]
    if any(label_presence) and not all(label_presence):
        raise ValueError("relevant_articles must be present for all questions or none")

    predictions: list[dict[str, object]] = []
    for current, question in enumerate(questions, start=1):
        question_id = question.get("id")
        image_id = question.get("image_id")
        text = question.get("question")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(f"question {question_id} has invalid image_id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"question {question_id} has invalid question text")
        print(
            f"[{current}/{len(questions)}] {question_id} ({image_id})",
            flush=True,
        )

        ranked = index.search(
            build_query(question, descriptions.get(image_id)), top_k=top_k
        )
        predictions.append(
            {
                "id": question_id,
                "image_id": image_id,
                "question": text,
                "relevant_articles": [
                    {"law_id": document.law_id, "article_id": document.article_id}
                    for document in ranked
                ],
            }
        )

    _save_predictions(output_path, predictions)
    if all(label_presence) and questions:
        return eval_retrieval(predictions, questions)
    return None
