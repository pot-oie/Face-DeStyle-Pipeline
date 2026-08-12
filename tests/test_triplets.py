from face_destyle.data.triplets import build_triplets
from face_destyle.schemas import EvaluationRecord


def evaluation(source_id, style="comic", accepted=True):
    return EvaluationRecord(
        id=source_id,
        source_id=source_id,
        input_path=f"{source_id}-input.png",
        output_path=f"{source_id}-output.png",
        style_category=style,
        content_score=0.9,
        style_removal_score=0.9,
        smoke_test_similarity=1.0,
        accepted=accepted,
    )


def test_triplets_are_reproducible_and_respect_constraints():
    records = [
        evaluation("a"),
        evaluation("b"),
        evaluation("c"),
        evaluation("ignored", accepted=False),
    ]
    first = build_triplets(records, references_per_target=1, seed=7)
    second = build_triplets(records, references_per_target=1, seed=7)
    assert first == second
    assert len(first) == 3
    assert all(item.target_source_id != item.reference_source_id for item in first)
    assert all(item.style_category == "comic" for item in first)


def test_insufficient_references_warns():
    with __import__("pytest").warns(UserWarning, match="eligible references"):
        assert build_triplets([evaluation("solo")], seed=3) == []
