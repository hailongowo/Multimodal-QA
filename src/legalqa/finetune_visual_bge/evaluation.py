"""Hierarchical retrieval metrics and prediction generation."""

import numpy as np


def select_questions(manifest: dict, external_questions: list[dict] | None = None):
    if external_questions is not None:
        questions = [{key: question[key] for key in ("id", "image_id", "question")} for question in external_questions]
        positives = None
        if all("relevant_articles" in question for question in external_questions):
            positives = [{(p["law_id"], p["article_id"]) for p in question["relevant_articles"]}
                         for question in external_questions]
        return questions, positives
    ids = set(manifest["splits"]["validation"])
    questions = [question for question in manifest["questions"] if question["id"] in ids]
    positives = [{(p["law_id"], p["label_id"]) for p in question["positives"]} for question in questions]
    return questions, positives


def retrieval_metrics(rankings: list[list[tuple[str, str]]], positives: list[set[tuple[str, str]]],
                      ks=(1, 5, 10)) -> dict:
    if len(rankings) != len(positives):
        raise ValueError("ranking and positive counts differ")
    result = {}
    for k in ks:
        values = [len(set(ranking[:k]) & expected) / len(expected) for ranking, expected in zip(rankings, positives)]
        result[f"recall_at_{k}"] = sum(values) / len(values) if values else 0.0
    reciprocal = []
    for ranking, expected in zip(rankings, positives):
        reciprocal.append(next((1 / rank for rank, key in enumerate(ranking[:10], 1) if key in expected), 0.0))
    result["mrr_at_10"] = sum(reciprocal) / len(reciprocal) if reciprocal else 0.0
    return result


def build_predictions(questions: list[dict], scores: np.ndarray, candidates: list[dict], top_k=5) -> list[dict]:
    predictions = []
    for row, question in enumerate(questions):
        references, seen = [], set()
        for index in np.argsort(-scores[row], kind="stable"):
            candidate = candidates[int(index)]
            key = (candidate["law_id"], candidate["label_id"])
            if key in seen:
                continue
            seen.add(key)
            references.append({"law_id": key[0], "article_id": key[1]})
            if len(references) == top_k:
                break
        predictions.append({"id": question["id"], "image_id": question["image_id"],
                            "question": question["question"], "relevant_articles": references})
    return predictions


def score_queries(adapter, questions: list[dict], candidates: list[dict], images_dir) -> np.ndarray:
    document_embeddings = adapter.encode_documents([candidate["text"] for candidate in candidates])
    query_embeddings = adapter.encode_queries(
        [images_dir / f"{question['image_id']}.jpg" for question in questions],
        [question["question"] for question in questions],
    )
    return query_embeddings @ document_embeddings.T


def validation_recall_at_5(adapter, manifest: dict, images_dir) -> float:
    validation_ids = set(manifest["splits"]["validation"])
    questions = [question for question in manifest["questions"] if question["id"] in validation_ids]
    if not questions:
        return 0.0
    scores = score_queries(adapter, questions, manifest["candidates"], images_dir)
    rankings = []
    for row in scores:
        rankings.append([(manifest["candidates"][i]["law_id"], manifest["candidates"][i]["label_id"])
                         for i in np.argsort(-row, kind="stable")])
    positives = [{(item["law_id"], item["label_id"]) for item in question["positives"]} for question in questions]
    return retrieval_metrics(rankings, positives, ks=(5,))["recall_at_5"]
