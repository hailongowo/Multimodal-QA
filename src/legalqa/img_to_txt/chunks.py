"""Description-aware text chunks for dense retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .documents import load_search_documents


def load_description_chunks(
    corpus_path: Path,
    descriptions_path: Path,
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
    overlap_elements: int = 1,
) -> list[dict]:
    """Build deterministic chunks from legal text plus validated descriptions."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_elements < 0:
        raise ValueError("overlap_elements must not be negative")
    chunks: list[dict] = []
    for document in load_search_documents(corpus_path, descriptions_path):
        elements = [part.strip() for part in document.text.splitlines() if part.strip()]
        groups: list[list[str]] = []
        current: list[str] = []
        for element in elements:
            proposed = current + [element]
            if current and count_tokens("\n".join(proposed)) > max_tokens:
                groups.append(current)
                overlap = current[-overlap_elements:] if overlap_elements else []
                current = overlap if count_tokens("\n".join(overlap + [element])) <= max_tokens else []
            current.append(element)
        groups.append(current)
        for index, group in enumerate(groups, 1):
            chunks.append({
                "chunk_id": f"{document.article_key}/c{index:04d}",
                "article_key": document.article_key,
                "law_id": document.law_id,
                "label_id": document.article_id,
                "text": "\n".join(group),
                "block_ids": [],
            })
    return chunks
