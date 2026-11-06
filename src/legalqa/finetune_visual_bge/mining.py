"""Cosine retrieval and hard-negative selection."""

import numpy as np


def _normalize(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} embeddings must be a finite 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} embeddings contain a zero vector")
    return values / norms


def rank_candidates(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    scores = _normalize(query_embeddings, "query") @ _normalize(candidate_embeddings, "candidate").T
    return np.argsort(-scores, axis=1, kind="stable")


def mine_hard_negatives(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray,
                        questions: list[dict], candidates: list[dict], top_n: int = 20) -> dict:
    if len(query_embeddings) != len(questions):
        raise ValueError("query embedding count does not match question count")
    if len(candidate_embeddings) != len(candidates):
        raise ValueError("candidate embedding count does not match candidate count")
    normalized_queries = _normalize(query_embeddings, "query")
    normalized_candidates = _normalize(candidate_embeddings, "candidate")
    scores = normalized_queries @ normalized_candidates.T
    rankings = np.argsort(-scores, axis=1, kind="stable")
    rows = []
    for query_index, question in enumerate(questions):
        positives = {(item["law_id"], item["label_id"]) for item in question["positives"]}
        negatives = []
        for candidate_index in rankings[query_index]:
            candidate = candidates[int(candidate_index)]
            if (candidate["law_id"], candidate["label_id"]) in positives:
                continue
            negatives.append({**{key: candidate[key] for key in ("law_id", "label_id", "unit_type")},
                              "rank": len(negatives) + 1, "score": float(scores[query_index, candidate_index])})
            if len(negatives) == top_n:
                break
        rows.append({"id": question["id"], "negatives": negatives})
    return {"schema_version": 1, "top_n": top_n, "queries": rows}
