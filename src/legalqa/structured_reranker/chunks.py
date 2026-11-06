"""Build deterministic multi-vector chunks from structured legal records."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


@dataclass(frozen=True)
class StructuredChunk:
    chunk_id: str
    article_key: str
    law_id: str
    label_id: str
    text: str
    block_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Element:
    text: str
    block_id: str


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _nested_text(block: dict) -> list[str]:
    kind = block.get("type")
    if kind == "figure":
        caption = _clean(block.get("caption_original"))
        return [caption] if caption else []
    if kind == "table":
        return _table_parts(block)
    text = _clean(block.get("text"))
    return [text] if text else []


def _table_parts(block: dict) -> list[str]:
    parts = [
        value
        for value in (
            _clean(block.get("caption_original")),
            _clean(block.get("unit_original")),
        )
        if value
    ]
    rows: dict[int, list[tuple[int, str]]] = {}
    for cell in block.get("cells", []):
        value = " ".join(
            text
            for nested in cell.get("blocks", [])
            for text in _nested_text(nested)
        )
        rows.setdefault(int(cell.get("row", 0)), []).append(
            (int(cell.get("column", 0)), value)
        )
    for row_index in sorted(rows):
        values = [value for _, value in sorted(rows[row_index])]
        parts.append(f"row {row_index + 1}: " + " | ".join(values))
    return parts


def _record_elements(record: dict) -> list[_Element]:
    elements = []
    for block in record.get("blocks", []):
        block_id = str(block.get("block_id") or "")
        kind = block.get("type")
        if kind == "figure":
            caption = _clean(block.get("caption_original"))
            if caption:
                elements.append(_Element(caption, block_id))
        elif kind == "table":
            elements.extend(_Element(text, block_id) for text in _table_parts(block) if text)
        else:
            text = _clean(block.get("text"))
            if text:
                elements.append(_Element(text, block_id))
    return elements


def _header(record: dict) -> str:
    article = f"Điều {record['id']}: {_clean(record.get('title'))}".rstrip(": ")
    return "\n".join(
        value for value in (_clean(record.get("document_title")), article) if value
    )


def _unique_block_ids(elements: list[_Element]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(element.block_id for element in elements if element.block_id))


def build_structured_chunks(
    records: list[dict],
    count_tokens: Callable[[str], int],
    max_tokens: int = 512,
    overlap_elements: int = 1,
) -> list[StructuredChunk]:
    """Greedily pack structural elements under a soft token budget."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if overlap_elements < 0:
        raise ValueError("overlap_elements must not be negative")

    chunks = []
    for record in sorted(records, key=lambda item: item.get("article_key", "")):
        header = _header(record)
        elements = _record_elements(record)
        groups: list[list[_Element]] = []
        current: list[_Element] = []
        for element in elements:
            proposed = current + [element]
            proposed_text = "\n".join([header] + [item.text for item in proposed])
            if current and count_tokens(proposed_text) > max_tokens:
                groups.append(current)
                overlap = current[-overlap_elements:] if overlap_elements else []
                overlap_text = "\n".join([header] + [item.text for item in overlap + [element]])
                current = overlap if not overlap or count_tokens(overlap_text) <= max_tokens else []
            current.append(element)
        if current or not groups:
            groups.append(current)

        article_key = str(record["article_key"])
        for index, group in enumerate(groups, 1):
            text = "\n".join([header] + [element.text for element in group])
            chunks.append(
                StructuredChunk(
                    chunk_id=f"{article_key}/c{index:04d}",
                    article_key=article_key,
                    law_id=str(record["document_id"]).strip(),
                    label_id=str(record["id"]).strip(),
                    text=text,
                    block_ids=_unique_block_ids(group),
                )
            )
    return chunks
