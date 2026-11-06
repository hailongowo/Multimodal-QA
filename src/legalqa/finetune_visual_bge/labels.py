"""Conservative training-label resolution and reviewed virtual aliases."""

from dataclasses import dataclass
import re

from .corpus import CandidateUnit


class LabelResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedLabel:
    law_id: str
    raw_id: str
    canonical_id: str
    unit_type: str
    resolution: str

    @property
    def candidate_key(self) -> tuple[str, str]:
        return self.law_id, self.canonical_id


def apply_aliases(candidates: list[CandidateUnit], aliases: dict) -> list[CandidateUnit]:
    expanded = list(candidates)
    index = {candidate.key: candidate for candidate in candidates}
    for law_id, law_aliases in aliases.items():
        for output_id, source_ids in law_aliases.items():
            if not isinstance(source_ids, list) or not source_ids:
                raise LabelResolutionError(f"alias {law_id}/{output_id} must list at least one source")
            sources = []
            for source_id in source_ids:
                source = index.get((law_id, str(source_id)))
                if source is None:
                    raise LabelResolutionError(f"alias {law_id}/{output_id} has missing source {source_id}")
                sources.append(source)
            key = (law_id, output_id)
            if key in index:
                raise LabelResolutionError(f"alias {law_id}/{output_id} collides with a corpus candidate")
            alias = CandidateUnit(
                law_id=law_id,
                label_id=output_id,
                unit_type="alias",
                text="\n".join(source.text for source in sources),
                provenance=tuple({**item, "alias_source": source.label_id}
                                 for source in sources for item in source.provenance),
            )
            expanded.append(alias)
            index[key] = alias
    return expanded


def resolve_label(law_id: str, raw_id: str, candidates: list[CandidateUnit], aliases: dict) -> ResolvedLabel:
    raw = str(raw_id).strip()
    index = {candidate.key: candidate for candidate in candidates}

    def found(label_id: str, resolution: str):
        candidate = index.get((law_id, label_id))
        if candidate is None:
            return None
        if candidate.unit_type == "alias":
            resolution = "alias"
        return ResolvedLabel(law_id, raw, label_id, candidate.unit_type, resolution)

    direct = found(raw, "clause" if index.get((law_id, raw), None) and index[(law_id, raw)].unit_type == "clause" else "article")
    if direct:
        return direct
    if re.fullmatch(r"\d+\.0", raw):
        normalized = found(raw[:-2], "normalized")
        if normalized:
            return normalized
    if re.fullmatch(r"[A-Z]\d+(?:\.\d+)?", raw):
        normalized_id = raw[0] + "." + raw[1:]
        normalized = found(normalized_id, "normalized")
        if normalized:
            return normalized
    raise LabelResolutionError(f"unresolved label {law_id}/{raw}")
