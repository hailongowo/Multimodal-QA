"""Create deterministic article and clause retrieval units."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class CandidateUnit:
    law_id: str
    label_id: str
    unit_type: str
    text: str
    provenance: tuple[dict, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.law_id, self.label_id


def read_structured_corpus(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line {line_number} in {path}")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} in {path}: {error}") from error
    return records


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _serialize_block(block: dict, include_image_descriptions: bool = True) -> list[str]:
    kind = block.get("type")
    if kind == "paragraph" or (kind is None and "text" in block):
        return [_clean(block.get("text"))] if _clean(block.get("text")) else []
    if kind == "figure":
        values = [_clean(block.get("caption_original"))]
        if include_image_descriptions:
            values.append(_clean(block.get("description_vlm")))
        return [text for text in values if text]
    if kind == "table":
        parts = [text for text in (_clean(block.get("caption_original")), _clean(block.get("unit_original"))) if text]
        rows: dict[int, list[tuple[int, str]]] = {}
        for cell in block.get("cells", []):
            cell_text = " ".join(
                text for nested in cell.get("blocks", [])
                for text in _serialize_block(nested, include_image_descriptions)
            )
            rows.setdefault(int(cell.get("row", 0)), []).append((int(cell.get("column", 0)), cell_text))
        for row_index in sorted(rows):
            values = [value for _, value in sorted(rows[row_index])]
            parts.append(f"row {row_index + 1}: " + " | ".join(values))
        return parts
    return []


def _header(record: dict, label_id: str | None = None) -> list[str]:
    article = f"Điều {record['id']}: {_clean(record.get('title'))}".rstrip(": ")
    values = [_clean(record.get("document_title")), article]
    if label_id is not None:
        values.append(f"Khoản {label_id}")
    return [value for value in values if value]


def _merge_units(units: list[CandidateUnit]) -> list[CandidateUnit]:
    grouped: dict[tuple[str, str], list[CandidateUnit]] = {}
    for unit in units:
        grouped.setdefault(unit.key, []).append(unit)
    merged = []
    for key, members in grouped.items():
        members.sort(key=lambda unit: tuple(((p.get("article_key") or ""), (p.get("block_id") or ""))
                                            for p in unit.provenance))
        types = {member.unit_type for member in members}
        unit_type = members[0].unit_type if len(types) == 1 else "article+clause"
        merged.append(CandidateUnit(key[0], key[1], unit_type, "\n".join(member.text for member in members),
                                    tuple(p for member in members for p in member.provenance)))
    return merged


def build_candidate_units(
    records: list[dict],
    include_image_descriptions: bool = True,
    include_clause_candidates: bool = True,
) -> list[CandidateUnit]:
    units = []
    for record in sorted(records, key=lambda item: item.get("article_key", "")):
        law_id = str(record["document_id"]).strip()
        article_id = str(record["id"]).strip()
        blocks = record.get("blocks", [])
        article_parts = _header(record)
        for block in blocks:
            article_parts.extend(_serialize_block(block, include_image_descriptions))
        provenance = ({"article_key": record.get("article_key"), "block_id": None, "unit_type": "article"},)
        units.append(CandidateUnit(law_id, article_id, "article", "\n".join(article_parts), provenance))

        if not include_clause_candidates:
            continue
        starts = [index for index, block in enumerate(blocks) if _clean(block.get("clause_label"))]
        for position, start in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(blocks)
            label_id = _clean(blocks[start]["clause_label"]).rstrip(".")
            parts = _header(record, label_id)
            for block in blocks[start:end]:
                parts.extend(_serialize_block(block, include_image_descriptions))
            clause_provenance = ({
                "article_key": record.get("article_key"),
                "block_id": blocks[start].get("block_id"),
                "unit_type": "clause",
            },)
            units.append(CandidateUnit(law_id, label_id, "clause", "\n".join(parts), clause_provenance))
    return _merge_units(units)
