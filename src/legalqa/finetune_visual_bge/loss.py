"""Multi-positive contrastive retrieval objective."""


def multi_positive_info_nce(query_embeddings, document_embeddings, positive_mask, valid_mask, temperature=0.02):
    import torch
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if positive_mask.shape != valid_mask.shape or positive_mask.shape != (query_embeddings.shape[0], document_embeddings.shape[0]):
        raise ValueError("embedding and mask shapes are incompatible")
    if (~positive_mask.any(dim=1)).any():
        raise ValueError("query has no positive document")
    queries = torch.nn.functional.normalize(query_embeddings, dim=-1)
    documents = torch.nn.functional.normalize(document_embeddings, dim=-1)
    scores = queries @ documents.transpose(0, 1) / temperature
    negative_infinity = torch.finfo(scores.dtype).min
    numerator = torch.logsumexp(scores.masked_fill(~positive_mask, negative_infinity), dim=1)
    denominator = torch.logsumexp(scores.masked_fill(~valid_mask, negative_infinity), dim=1)
    return (denominator - numerator).mean()
