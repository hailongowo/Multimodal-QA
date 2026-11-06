"""Build the text-preprocessed law corpus for the official baseline."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import re

import pandas as pd


TABLE_MARKER_RE = re.compile(
    r"<<TABLE:\s*(.*?)\s*/TABLE>>",
    flags=re.DOTALL,
)


def _table_to_markdown(match: re.Match[str]) -> str:
    table_html = match.group(1).strip()
    tables = pd.read_html(StringIO(table_html), header=0)
    if not tables:
        raise ValueError("A table marker did not contain a readable HTML table")
    return f"<<TABLE: {tables[0].to_markdown()} /TABLE>>"


def process_law_corpus(laws: list[dict]) -> list[dict]:
    """Return a deep-copied corpus with HTML tables converted to Markdown."""
    if not isinstance(laws, list):
        raise ValueError("The source law corpus must be a JSON array")

    processed = deepcopy(laws)
    for law in processed:
        articles = law.get("articles")
        if not isinstance(articles, list):
            raise ValueError("Each law must contain an articles array")
        for article in articles:
            text = article.get("text")
            if not isinstance(text, str):
                raise ValueError("Each article must contain a text string")
            article["text"] = TABLE_MARKER_RE.sub(_table_to_markdown, text)
    return processed


def build_processed_law_file(source_path: Path, output_path: Path) -> list[dict]:
    """Read, process, and atomically write the official baseline law JSON."""
    laws = json.loads(source_path.read_text(encoding="utf-8"))
    processed = process_law_corpus(laws)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return processed
