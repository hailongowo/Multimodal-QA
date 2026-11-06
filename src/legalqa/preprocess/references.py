"""Gắn caption cho bảng/hình và nối các câu 'xem Hình/Bảng ...'."""

import re

CAPTION = re.compile(r"^((?:Hình|Bảng)\s+\S+)\s*[-–:]\s*", re.IGNORECASE)
MENTION = re.compile(r"\b(Hình|Bảng)\s+((?:[A-Z]\.)?\d+(?:\.\d+)*[a-z]?[’']?)", re.IGNORECASE)


def add_captions(articles):
    for article in articles:
        blocks = article["blocks"]
        for i, block in enumerate(blocks):
            if block["type"] not in ("figure", "table"):
                continue

            # Hình: chú thích thường ở ngay SAU; bảng: ở ngay TRƯỚC.
            expected = "Hình" if block["type"] == "figure" else "Bảng"
            if block["type"] == "figure":
                candidates = blocks[i + 1:i + 2]
            else:
                # Có thể có dòng "Đơn vị tính: ..." giữa caption và table.
                candidates = list(reversed(blocks[max(0, i - 3):i]))
            for caption in candidates:
                if caption["type"] != "paragraph":
                    break
                match = CAPTION.match(caption["text"])
                if match and match.group(1).startswith(expected):
                    block["label"] = match.group(1)
                    block["caption_original"] = caption["text"]
                    block["caption_block_id"] = caption["block_id"]
                    break

            # Nếu dòng đơn vị đứng ngay trên bảng, ghi lại làm metadata.
        for i, block in enumerate(blocks):
            if block["type"] != "table":
                continue
            for before in reversed(blocks[max(0, i - 3):i]):
                if before["type"] != "paragraph":
                    break
                if before["text"].startswith("Đơn vị"):
                    block["unit_original"] = before["text"]
                    break


def add_references(articles):
    # Tra cứu label trong cùng văn bản luật/quy chuẩn (không trộn tài liệu).
    indexes = {}
    for article in articles:
        labels = indexes.setdefault(article["document_id"], {})
        for block in article["blocks"]:
            if block["type"] in ("figure", "table") and block.get("label"):
                labels.setdefault(block["label"].casefold(), []).append(block["block_id"])

    for article in articles:
        labels = indexes[article["document_id"]]
        paragraphs = [b for b in article["blocks"] if b["type"] == "paragraph"]
        for table in article["blocks"]:
            if table["type"] == "table":
                paragraphs.extend(
                    b for cell in table["cells"] for b in cell["blocks"]
                    if b["type"] == "paragraph"
                )

        for block in paragraphs:
            # Không coi dòng chú thích 'Hình 1 - ...' là tham chiếu.
            if CAPTION.match(block["text"]):
                continue
            for match in MENTION.finditer(block["text"]):
                mention = match.group(0)
                targets = labels.get(mention.casefold(), [])
                status = "resolved" if len(targets) == 1 else "ambiguous" if targets else "unresolved"
                block["references"].append({
                    "mention": mention,
                    "target_block_id": targets[0] if len(targets) == 1 else None,
                    "status": status,
                })
