"""Run and evaluate any Subtask 1 retrieve(query) function."""

import json

from tqdm import tqdm

from .evaluation.metrics import eval_retrieval


def run_experiment(questions, retrieve, prediction_path):
    """Pass question metadata to a method; save its references and print F2."""
    predictions = []
    for question in tqdm(questions):
        query = {key: question[key] for key in ("id", "image_id", "question")}
        references = retrieve(query)
        predictions.append({**query, "relevant_articles": references})

    with open(prediction_path, "w", encoding="utf-8") as stream:
        json.dump(predictions, stream, ensure_ascii=False, indent=4)

    metrics = eval_retrieval(predictions, questions)
    print(metrics)
    return metrics
