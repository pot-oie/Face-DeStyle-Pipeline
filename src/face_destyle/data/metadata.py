"""JSONL metadata validation and persistence."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

RecordT = TypeVar("RecordT", bound=BaseModel)


def read_jsonl(
    path: str | Path,
    model: type[RecordT],
    *,
    check_paths: bool = False,
    path_fields: tuple[str, ...] = (),
) -> list[RecordT]:
    records: list[RecordT] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = model.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}: {exc}") from exc
            record_id = str(record.id)
            if record_id in seen:
                raise ValueError(f"Duplicate id {record_id!r} at {path}:{line_number}")
            seen.add(record_id)
            if check_paths:
                for field in path_fields:
                    candidate = Path(getattr(record, field))
                    if not candidate.exists():
                        raise FileNotFoundError(f"Missing {field} for {record_id}: {candidate}")
            records.append(record)
    return records


def write_jsonl(records: list[BaseModel], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
