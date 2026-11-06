"""Tách text của mỗi điều thành paragraph / figure / table."""

import hashlib
import re

from .tables import parse_table

MARKER = re.compile(r"<<(?P<kind>IMAGE|TABLE):\s*(?P<value>.*?)\s*/(?P=kind)>>", re.DOTALL)
CLAUSE = re.compile(r"^((?:\d+(?:\.\d+)*\.?|[a-zđ]\)))\s+(?=\S)", re.IGNORECASE)


def clean_text(text):
    """Chuẩn hóa khoảng trắng; không dịch hoặc sửa nội dung."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


class ArticleParser:
    def __init__(self, article_key):
        self.article_key = article_key
        self.counter = 0
        self.all_ids = []

    def next_id(self):
        self.counter += 1
        block_id = f"{self.article_key}/b{self.counter:04d}"
        self.all_ids.append(block_id)
        return block_id

    def paragraph(self, text, span):
        text = clean_text(text)
        if not text:
            return None
        block = {
            "block_id": self.next_id(),
            "type": "paragraph",
            "source_span": list(span),
            "text": text,
            "references": [],
        }
        match = CLAUSE.match(text)
        if match:
            block["clause_label"] = match.group(1)
        return block

    def figure(self, source, span):
        filename = source.rsplit("/", 1)[-1]
        # ID ổn định từ tên nguồn; KHÔNG phải hash của nội dung ảnh.
        image_id = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
        return {
            "block_id": self.next_id(),
            "type": "figure",
            "source_span": list(span),
            "source_reference": source,
            "image_id": image_id,
            "description_vlm": None,
            "filename": filename,
        }

    def text_section(self, text, span):
        # Một dòng không rỗng thành một paragraph.
        blocks = []
        for line in text.splitlines():
            block = self.paragraph(line, span)
            if block:
                blocks.append(block)
        return blocks

    def parse(self, raw_text):
        blocks = []
        start = 0
        for match in MARKER.finditer(raw_text):
            blocks.extend(self.text_section(raw_text[start:match.start()], (start, match.start())))
            span = (match.start(), match.end())
            if match.group("kind") == "IMAGE":
                blocks.append(self.figure(match.group("value").strip(), span))
            else:
                table_id = self.next_id()
                table = parse_table(
                    match.group("value"), table_id, span,
                    self.paragraph, self.figure,
                )
                blocks.append(table)
            start = match.end()
        blocks.extend(self.text_section(raw_text[start:], (start, len(raw_text))))
        return blocks, self.all_ids
