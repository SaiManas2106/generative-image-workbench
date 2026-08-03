from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DatasetRecord:
    image_path: str
    caption: str
    split: str
    identity_id: str | None = None
    reference_image_path: str | None = None


@dataclass(frozen=True)
class DatasetReport:
    total_records: int
    split_counts: dict[str, int]
    identity_coverage: float
    duplicate_paths: list[str]
    errors: list[str]


VALID_SPLITS = {"train", "validation", "test"}


def validate_manifest(records: Iterable[DatasetRecord]) -> DatasetReport:
    records = list(records)
    split_counts: dict[str, int] = {}
    seen_paths: set[str] = set()
    duplicate_paths: list[str] = []
    errors: list[str] = []
    identity_count = 0

    for index, record in enumerate(records):
        if not record.image_path.strip():
            errors.append(f"record {index} has no image path")
        if not record.caption.strip():
            errors.append(f"record {index} has no caption")
        if record.split not in VALID_SPLITS:
            errors.append(f"record {index} has invalid split '{record.split}'")
        if record.image_path in seen_paths:
            duplicate_paths.append(record.image_path)
        seen_paths.add(record.image_path)
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        identity_count += int(bool(record.identity_id))

    coverage = identity_count / len(records) if records else 0.0
    if records and split_counts.get("validation", 0) == 0:
        errors.append("manifest requires at least one validation record")

    return DatasetReport(
        total_records=len(records),
        split_counts=split_counts,
        identity_coverage=round(coverage, 3),
        duplicate_paths=sorted(set(duplicate_paths)),
        errors=errors,
    )


def record_to_dict(record: DatasetRecord) -> dict[str, str | None]:
    return asdict(record)


def is_supported_image(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
