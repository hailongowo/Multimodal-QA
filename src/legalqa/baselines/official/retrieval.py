"""Top-5 retrieval adapted from VLSP2025-MLQA-TSR/retrieval.py."""

import heapq

import numpy as np


def get_top_k(predict_output, top_k):
    lst_out_one_hot = np.zeros(len(predict_output))
    idx_top_k = heapq.nlargest(top_k, range(len(predict_output)), predict_output.take)
    i = 1
    for idx in idx_top_k:
        lst_out_one_hot[idx] = i
        i = i + 1
    return lst_out_one_hot


def retrieve_articles(model, law_embeding, law_ids, image_path, question, top_k=5, return_rankings=False):
    def make_retrival(candidate):
        article_law = candidate.split("_")[0].split("-")[0]
        id_law = candidate.split("_")[1]
        return {"law_id": id_law, "article_id": article_law}

    relevant_ids = []
    img_path = str(image_path) + "/" + question["image_id"] + ".jpg"
    query_emb = model.encode(image=img_path, text=question["question"])
    similarities = query_emb.cpu().detach() @ law_embeding.T
    lst_similars = similarities.T.detach().numpy()
    # Keep the official top-5-vector prediction policy for F2.
    selected = get_top_k(lst_similars.reshape(-1), 5)
    official = []
    for i in range(len(selected)):
        reference = make_retrival(law_ids[i])
        if selected[i] > 0 and reference not in official:
            official.append(reference)
    # Recall uses score-ordered unique articles, independent of prediction cutoff.
    for i in np.argsort(-lst_similars.reshape(-1), kind="stable"):
        reference = make_retrival(law_ids[i])
        if reference not in relevant_ids:
            relevant_ids.append(reference)
            if len(relevant_ids) == top_k:
                break
    return (official, relevant_ids) if return_rankings else official
