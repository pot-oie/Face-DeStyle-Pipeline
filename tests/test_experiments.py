import pytest

from face_destyle.experiments import (
    ExperimentSpec,
    expand_runs,
    load_evaluation_assets,
    load_experiment_specs,
    validate_asset_references,
)
from face_destyle.models import ModelRegistry


def test_repository_experiment_matrix_has_primary_and_extensions():
    specs = load_experiment_specs("configs/experiments.yaml")
    by_name = {spec.name: spec for spec in specs}

    assert by_name["prompt_generic"].settings["prompt_mode"] == "generic"
    assert by_name["global_canny"].settings["control_model"] == "canny_controlnet"
    assert by_name["canny_plus_pose"].settings["pose_extractor"] == "dwpose"
    assert by_name["depth_control"].extension
    assert by_name["generator_realvisxl"].settings["generator"] == "realvisxl_v5"


def test_repository_experiments_reference_registered_assets():
    specs = load_experiment_specs("configs/experiments.yaml")
    assets = set(ModelRegistry.from_yaml("configs/models.yaml").assets)

    validate_asset_references(specs, assets)
    assert load_evaluation_assets("configs/experiments.yaml") <= assets


def test_expanded_run_ids_are_stable_and_seed_specific():
    spec = ExperimentSpec("demo", {"generator": "sdxl_base"})

    first = expand_runs([spec], [42, 43])
    second = expand_runs([spec], [42, 43])

    assert first == second
    assert first[0].run_id != first[1].run_id
    assert first[0].settings["seed"] == 42


def test_unknown_asset_reference_is_rejected():
    spec = ExperimentSpec("bad", {"control_model": "does_not_exist"})

    with pytest.raises(ValueError, match="does_not_exist"):
        validate_asset_references([spec], {"sdxl_base"})
