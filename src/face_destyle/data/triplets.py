"""Deterministic construction of accepted destylization training triplets."""

import hashlib
import random
import warnings

from face_destyle.schemas import EvaluationRecord, TripletRecord


def build_triplets(
    records: list[EvaluationRecord], *, references_per_target: int = 1, seed: int = 42
) -> list[TripletRecord]:
    if references_per_target < 1:
        raise ValueError("references_per_target must be at least 1")
    accepted = [record for record in records if record.accepted is True]
    rng = random.Random(seed)
    triplets: list[TripletRecord] = []
    for target in sorted(accepted, key=lambda item: item.id):
        candidates = [
            item
            for item in accepted
            if item.style_category == target.style_category and item.source_id != target.source_id
        ]
        if len(candidates) < references_per_target:
            warnings.warn(
                f"Target {target.id} has {len(candidates)} eligible references; "
                f"requested {references_per_target}",
                stacklevel=2,
            )
        chosen = rng.sample(candidates, k=min(references_per_target, len(candidates)))
        for reference in chosen:
            digest = hashlib.sha256(
                f"{seed}:{target.id}:{reference.id}".encode()
            ).hexdigest()[:16]
            triplets.append(
                TripletRecord(
                    id=f"triplet-{digest}",
                    destylized_content_path=target.output_path,
                    style_reference_path=reference.input_path,
                    original_style_target_path=target.input_path,
                    style_category=target.style_category,
                    target_source_id=target.source_id,
                    reference_source_id=reference.source_id,
                )
            )
    return triplets
