"""Embedding files and model setup specific to the official baseline."""

import numpy as np

from ...data.loaders import read_json
from ...experiment import run_experiment
from .embedding import embedding_articles
from .model import load_model
from .retrieval import retrieve_articles


def make_embedding(law_path, image_path, model_weight, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_weight)
    law_db = read_json(law_path)
    law_id, law_embedding = embedding_articles(law_db, image_path, model)
    print(law_id.shape)
    print(law_embedding.shape)
    np.save(output_dir / "law_id.npy", law_id)
    np.save(output_dir / "law_embedding.npy", law_embedding)


def retrieval(question_path, image_path, model_weight, output_dir):
    model = load_model(model_weight)
    law_id = np.load(output_dir / "law_id.npy")
    law_embedding = np.load(output_dir / "law_embedding.npy")
    print(law_id.shape)
    print(law_embedding.shape)
    questions = read_json(question_path)
    return run_experiment(
        questions,
        lambda query: retrieve_articles(model, law_embedding, law_id, image_path, query),
        output_dir / "public_test_task_1.json",
    )
