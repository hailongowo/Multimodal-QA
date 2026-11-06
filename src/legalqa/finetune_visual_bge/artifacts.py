"""Fingerprinted, atomic artifacts shared by pipeline stages."""

import hashlib
import json
from pathlib import Path


def fingerprint_json(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def require_compatible(actual: dict, expected: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise ValueError(f"incompatible {field}: {actual.get(field)!r} != {expected.get(field)!r}")
