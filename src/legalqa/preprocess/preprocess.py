"""Build the structured corpus; run via scripts/build_structured_corpus.py."""

import json
from pathlib import Path

from .blocks import ArticleParser
from .references import add_captions, add_references

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = PROJECT_ROOT / "data" / "law_db" / "vlsp2025_law.json"
OUTPUT = PROJECT_ROOT / "data" / "structured_corpus" / "structured_corpus.jsonl"


def main():
    documents = json.loads(INPUT.read_text(encoding="utf-8"))
    articles = []

    for doc_number, doc in enumerate(documents, start=1):
        for article_number, article in enumerate(doc["articles"], start=1):
            article_key = f"d{doc_number:02d}/a{article_number:04d}"
            parser = ArticleParser(article_key)
            blocks, block_ids = parser.parse(article["text"])
            articles.append({
                "article_key": article_key,
                "document_id": doc["id"],
                "document_title": doc["title"],
                "id": article["id"],
                "title": article["title"],
                "blocks": blocks,
                "all_block_ids": block_ids,
            })

    add_captions(articles)
    add_references(articles)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(article, ensure_ascii=False) + "\n")

    print(f"Created {len(articles)} articles -> {OUTPUT}")


if __name__ == "__main__":
    main()
