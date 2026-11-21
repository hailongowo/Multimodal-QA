"""Write the two final result files for an approach."""

import json
from pathlib import Path

from .metrics import eval_retrieval


def save_results(predictions, questions, rankings, output_dir, approach, top_k=5):
    """F2 uses exported predictions; recall uses unique ranked articles up to 10."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not questions or not all("relevant_articles" in q for q in questions):
        raise ValueError("Evaluation requires non-empty labeled questions")
    expected_ids = [q["id"] for q in questions]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Duplicate question IDs")
    for rows in (predictions, rankings):
        if [row["id"] for row in rows] != expected_ids:
            raise ValueError("Prediction/ranking IDs must match question order")
    final = [{**row, "relevant_articles": row["relevant_articles"][:top_k]}
             for row in predictions]
    metrics = {key: float(value) for key, value in eval_retrieval(final, questions).items()}
    for k in (1, 5, 10):
        values = []
        for question, row in zip(questions, rankings):
            truth = {(p["law_id"], p["article_id"]) for p in question["relevant_articles"]}
            ranked = list(dict.fromkeys((p["law_id"], p["article_id"])
                                        for p in row["relevant_articles"]))
            values.append(len(set(ranked[:k]) & truth) / len(truth) if truth else 0.0)
        metrics[f"recall_at_{k}"] = sum(values) / len(values)
    output_dir = Path(output_dir)
    for name, value in (("predictions", final), ("metrics", metrics)):
        directory = output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{approach}_{name}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    return metrics
