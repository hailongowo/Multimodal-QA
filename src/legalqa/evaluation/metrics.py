"""The official sample-wise F2 metric for Subtask 1."""

import numpy as np


def eval_retrieval(y_pred, y_true):
    y_pred = {e["id"]: set((p["law_id"], p["article_id"]) for p in e["relevant_articles"]) for e in y_pred}
    y_true = {e["id"]: set((p["law_id"], p["article_id"]) for p in e["relevant_articles"]) for e in y_true}
    ids = list(y_true)
    precision = np.array([len(y_true[k] & y_pred[k]) / len(y_pred[k]) if y_pred.get(k) else 0 for k in ids])
    recall = np.array([len(y_true[k] & y_pred.get(k, set())) / len(y_true[k]) for k in ids])
    f2 = (5 * precision * recall / (4 * precision + recall + 1e-9)).mean()
    return {"f2": f2}
