"""Structured-corpus dense retrieval followed by cross-encoder reranking."""

from .chunks import StructuredChunk, build_structured_chunks

from .pipeline import (
    CrossEncoderReranker,
    VisualizedBGERetriever,
    build_predictions,
    load_index,
    load_structured_candidates,
    load_structured_chunks,
    rerank_shortlist,
    rerank_articles,
    run_pipeline,
    save_index,
    shortlist_indices,
    select_article_evidence,
)

__all__ = [
    "StructuredChunk",
    "CrossEncoderReranker",
    "VisualizedBGERetriever",
    "build_structured_chunks",
    "build_predictions",
    "load_index",
    "load_structured_candidates",
    "load_structured_chunks",
    "rerank_shortlist",
    "rerank_articles",
    "run_pipeline",
    "save_index",
    "shortlist_indices",
    "select_article_evidence",
]
