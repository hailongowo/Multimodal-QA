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


def retrieve_articles(model, law_embeding, law_ids, image_path, question):
    def make_retrival(candidate):
        article_law = candidate.split("_")[0].split("-")[0]
        id_law = candidate.split("_")[1]
        return {"law_id": id_law, "article_id": article_law}

    relevant_ids = []
    img_path = str(image_path) + "/" + question["image_id"] + ".jpg"
    query_emb = model.encode(image=img_path, text=question["question"])
    similarities = query_emb.cpu().detach() @ law_embeding.T
    lst_similars = similarities.T.detach().numpy()
    lst_oh = get_top_k(lst_similars, 5)
    assert len(lst_oh) == len(law_ids)
    # The official baseline scans corpus order after selecting the top-5 vectors.
    for i in range(0, len(lst_oh)):
        if lst_oh[i] > 0:
            if make_retrival(law_ids[i]) not in relevant_ids:
                relevant_ids.append(make_retrival(law_ids[i]))
    return relevant_ids
