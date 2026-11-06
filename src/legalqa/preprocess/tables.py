"""Biến HTML bên trong <<TABLE: ... /TABLE>> thành cells + grid."""

from bs4 import BeautifulSoup


def html_text(node):
    """Giữ subscript/superscript dưới dạng V_(85), V^(2)."""
    for tag in node.find_all(["sub", "sup"]):
        content = tag.get_text("", strip=False)
        tag.replace_with(("_(" if tag.name == "sub" else "^(") + content + ")")
    return node.get_text("", strip=False)


def parse_table(html, table_id, span, make_paragraph, make_figure):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    cells, grid = [], []

    if table is None:
        return {"block_id": table_id, "type": "table", "source_span": list(span),
                "cells": cells, "grid": grid}

    for row_index, tr in enumerate(table.find_all("tr")):
        while len(grid) <= row_index:
            grid.append([])

        col_index = 0
        for tag in tr.find_all(["td", "th"], recursive=False):
            # Bỏ qua các vị trí đã bị chiếm bởi rowspan phía trên.
            while col_index < len(grid[row_index]) and grid[row_index][col_index]:
                col_index += 1

            rowspan = int(tag.get("rowspan", 1))
            colspan = int(tag.get("colspan", 1))
            cell_id = f"{table_id}/r{row_index + 1}c{col_index + 1}"

            content = []
            paragraphs = tag.find_all("p")
            if paragraphs:
                for p in paragraphs:
                    block = make_paragraph(html_text(p), span)
                    if block:
                        content.append(block)
                    for img in p.find_all("img"):
                        if img.get("src"):
                            content.append(make_figure(img["src"], span))
            else:
                block = make_paragraph(html_text(tag), span)
                if block:
                    content.append(block)
                for img in tag.find_all("img"):
                    if img.get("src"):
                        content.append(make_figure(img["src"], span))

            cells.append({
                "cell_id": cell_id,
                "row": row_index,
                "column": col_index,
                "rowspan": rowspan,
                "colspan": colspan,
                "is_header": tag.name == "th",
                "blocks": content,
            })

            for r in range(row_index, row_index + rowspan):
                while len(grid) <= r:
                    grid.append([])
                while len(grid[r]) < col_index + colspan:
                    grid[r].append("")
                for c in range(col_index, col_index + colspan):
                    grid[r][c] = cell_id
            col_index += colspan

    return {
        "block_id": table_id,
        "type": "table",
        "source_span": list(span),
        "cells": cells,
        "grid": grid,
    }
