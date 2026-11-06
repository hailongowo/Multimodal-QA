"""Build searchable law-article documents from structured text and image descriptions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


DESCRIPTION_FIELDS = {"image_category", "summary", "items"}
ITEM_FIELDS = {
    "identifier",
    "visual_or_geometric_features",
    "meaning_or_specification",
}
CATEGORIES = {"SIGN_ILLUSTRATION", "TECHNICAL_DIAGRAM", "MARKING_AND_SIGNAL"}


@dataclass(frozen=True)
class SearchDocument:
    law_id: str
    article_id: str
    article_key: str
    text: str


def _required_string(value: object, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty string {field}")
    return value


def _paragraph_text(blocks: object) -> Iterator[str]:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("article blocks must contain objects")
        block_type = block.get("type")
        if block_type == "paragraph":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                yield text
        elif block_type == "table":
            cells = block.get("cells", [])
            if not isinstance(cells, list):
                raise ValueError("table cells must be a list")
            for cell in cells:
                if not isinstance(cell, dict):
                    raise ValueError("table cells must contain objects")
                yield from _paragraph_text(cell.get("blocks", []))


def flatten_description(description: dict[str, object]) -> str:
    """Flatten the schema-2.0 visual description into retrieval text."""
    if not isinstance(description, dict):
        raise ValueError("description must be an object")
    if set(description) != DESCRIPTION_FIELDS:
        raise ValueError(
            f"description fields must be exactly {sorted(DESCRIPTION_FIELDS)}"
        )
    if description["image_category"] not in CATEGORIES:
        raise ValueError("description image_category is not supported")
    if not isinstance(description["summary"], str) or not description["summary"].strip():
        raise ValueError("description summary must be a non-empty string")

    parts = [description["image_category"], description["summary"]]

    items = description["items"]
    if not isinstance(items, list):
        raise ValueError("description items must be a list")
    for item in items:
        if not isinstance(item, dict) or set(item) != ITEM_FIELDS:
            raise ValueError(f"description item fields must be exactly {sorted(ITEM_FIELDS)}")
        for field in (
            "identifier",
            "visual_or_geometric_features",
            "meaning_or_specification",
        ):
            value = item[field]
            if not isinstance(value, str):
                raise ValueError(f"description item {field} must be a string")
            if value.strip():
                parts.append(value)
    return "\n".join(parts)


def _load_articles(path: Path) -> list[dict[str, object]]:
    articles: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                article = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(article, dict):
                raise ValueError(f"Structured corpus line {line_number} must be an object")
            key = _required_string(article.get("article_key"), "article_key", f"line {line_number}")
            if key in seen_keys:
                raise ValueError(f"duplicate article_key: {key}")
            seen_keys.add(key)
            articles.append(article)
    return articles


def load_search_documents(
    corpus_path: Path, descriptions_path: Path
) -> list[SearchDocument]:
    """Load one searchable document per structured-corpus article."""
    articles = _load_articles(corpus_path)
    known_keys = {str(article["article_key"]) for article in articles}
    identity_by_key = {
        str(article["article_key"]): (
            _required_string(article.get("document_id"), "document_id", str(article["article_key"])),
            _required_string(article.get("id"), "id", str(article["article_key"])),
        )
        for article in articles
    }

    payload = json.loads(descriptions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "2.0":
        actual = payload.get("schema_version") if isinstance(payload, dict) else None
        raise ValueError(f"description schema_version must be '2.0', got {actual!r}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("description records must be a list")

    descriptions_by_article: dict[str, list[str]] = {key: [] for key in known_keys}
    seen_record_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"description record {index} must be an object")
        context = f"description record {index}"
        record_id = _required_string(record.get("record_id"), "record_id", context)
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate description record_id: {record_id}")
        seen_record_ids.add(record_id)
        article_key = _required_string(record.get("article_key"), "article_key", context)
        if article_key not in known_keys:
            raise ValueError(f"unknown article_key in descriptions: {article_key}")
        law_id = _required_string(record.get("document_id"), "document_id", context)
        article_id = _required_string(record.get("article_id"), "article_id", context)
        expected_law_id, expected_article_id = identity_by_key[article_key]
        if law_id != expected_law_id:
            raise ValueError(
                f"description document_id {law_id!r}; expected {expected_law_id!r}"
            )
        if article_id != expected_article_id:
            raise ValueError(
                f"description article_id {article_id!r}; expected {expected_article_id!r}"
            )
        status = record.get("status")
        if status not in {"success", "error"}:
            raise ValueError(f"description status must be 'success' or 'error', got {status!r}")
        if status == "success":
            descriptions_by_article[article_key].append(
                flatten_description(record.get("description"))
            )

    documents: list[SearchDocument] = []
    document_indexes: dict[tuple[str, str], int] = {}
    for article in articles:
        article_key = str(article["article_key"])
        law_id = _required_string(article.get("document_id"), "document_id", article_key)
        article_id = _required_string(article.get("id"), "id", article_key)
        text_parts = [
            value
            for value in (
                article.get("document_title"),
                article.get("title"),
            )
            if isinstance(value, str) and value.strip()
        ]
        text_parts.extend(_paragraph_text(article.get("blocks", [])))
        text_parts.extend(filter(None, descriptions_by_article[article_key]))
        segment_text = "\n".join(text_parts)
        identity = (law_id, article_id)
        if identity in document_indexes:
            index = document_indexes[identity]
            current = documents[index]
            documents[index] = SearchDocument(
                law_id=current.law_id,
                article_id=current.article_id,
                article_key=current.article_key,
                text="\n".join(filter(None, (current.text, segment_text))),
            )
            continue
        document_indexes[identity] = len(documents)
        documents.append(
            SearchDocument(law_id, article_id, article_key, segment_text)
        )
    return documents
