import copy
from unittest.mock import PropertyMock, patch

import pytest
from pydantic import ValidationError

from dynopsi.data.featurization.inference_featurization_pipeline import (
    get_featurization_pipeline_from_config,
)
from dynopsi.data.utils.sequence_utils import ChainResidueRepr
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.simulation_pipeline.registry import get_simulation_pipeline_from_config


def _valid_sde_config() -> dict:
    """Build a fully valid ``sde_simulation`` config."""
    return {
        "type": "sde_simulation",
        "name": "strict_config_test",
        "inference_model_config": {
            "loader": {"type": "huggingface", "repo_id": "fake/repo", "filename": "fake.ckpt", "revision": "0"}
        },
        "time_sampler": {"type": "linear_time_sampler", "num_steps": 3},
        "diffusion_coefficient_fn": {"type": "inverse_parameter", "eps": 0.02, "clamp_max": 10.0},
        "noise_scale_fn": {"type": "constant", "constant": 0.0},
        "ode_warmup_score_scale_fn": {"type": "constant", "constant": 0.0},
        "ode_cooldown_score_scale_fn": {"type": "constant", "constant": 0.0},
    }


def _valid_featurization_config() -> dict:
    """Build a valid binder-design featurization config; validation does not read ``structure_path``."""
    return {
        "name": "strict_config_test",
        "type": "dynopsi_binder_design_featurization_pipeline",
        "index_type": "auth_residue_index",
        "structure_path": "missing.cif",
        "hotspots": {"hotspot_B58V_B80L_B139Y": "B58,B80,B139"},
        "binder_atoms_to_include": "ca",
        "new_chains_lengths": [60],
        "new_chains_centers_of_mass": {
            "hitcenter": {"type": "predefined_center_of_mass", "centers_of_mass": [[28.1, 41.4, 47.9]]}
        },
    }


@pytest.fixture(autouse=True)
def _no_checkpoint_load():
    with patch.object(DynoPsiModelConfig, "prior_stds", new_callable=PropertyMock, return_value={"coords": 1.0}):
        yield


def _assert_only_extra_forbidden(exc_info: pytest.ExceptionInfo[ValidationError]) -> None:
    """Assert the bogus key is the *sole* failure, so a base-config regression can't mask it."""
    errors = exc_info.value.errors()
    assert len(errors) == 1, f"expected exactly one error, got {errors}"
    assert errors[0]["type"] == "extra_forbidden"


def test_simulation_top_level_unknown_key_rejected() -> None:
    config = _valid_sde_config()
    config["bogus"] = 1
    with pytest.raises(ValidationError) as exc_info:
        get_simulation_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


def test_simulation_nested_time_sampler_unknown_key_rejected() -> None:
    config = _valid_sde_config()
    config["time_sampler"]["bogus"] = 1
    with pytest.raises(ValidationError) as exc_info:
        get_simulation_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


@pytest.mark.parametrize(
    "field",
    [
        "diffusion_coefficient_fn",
        "noise_scale_fn",
        "ode_warmup_score_scale_fn",
        "ode_cooldown_score_scale_fn",
    ],
)
def test_simulation_nested_single_parameter_fn_unknown_key_rejected(field: str) -> None:
    config = _valid_sde_config()
    config[field]["bogus"] = 1
    with pytest.raises(ValidationError) as exc_info:
        get_simulation_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


def test_simulation_nested_cross_union_member_key_rejected() -> None:
    # ``p`` is a valid field on LogTimeSampler but not LinearTimeSampler; forbid must reject
    # the misrouted key instead of silently dropping it.
    config = _valid_sde_config()
    config["time_sampler"] = {"type": "linear_time_sampler", "num_steps": 3, "p": 0.5}
    with pytest.raises(ValidationError) as exc_info:
        get_simulation_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


def test_featurization_unknown_key_rejected() -> None:
    config = _valid_featurization_config()
    config["bogus_key"] = 1
    with pytest.raises(ValidationError) as exc_info:
        get_featurization_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


def test_featurization_nested_center_spec_unknown_key_rejected() -> None:
    config = _valid_featurization_config()
    next(iter(config["new_chains_centers_of_mass"].values()))["bogus"] = 1
    with pytest.raises(ValidationError) as exc_info:
        get_featurization_pipeline_from_config(config)
    _assert_only_extra_forbidden(exc_info)


def test_chain_residue_repr_unknown_key_rejected() -> None:
    # Dict-authored form; the common string form ("A7-147") bypasses field validation via _coerce.
    with pytest.raises(ValidationError) as exc_info:
        ChainResidueRepr.model_validate({"chain_id_to_residue_index_map": {"A": [1, 2]}, "bogus": 1})
    _assert_only_extra_forbidden(exc_info)


def test_valid_configs_still_parse() -> None:
    featurization = get_featurization_pipeline_from_config(copy.deepcopy(_valid_featurization_config()))
    assert featurization.type == "dynopsi_binder_design_featurization_pipeline"
    simulation = get_simulation_pipeline_from_config(copy.deepcopy(_valid_sde_config()))
    assert simulation.type == "sde_simulation"
    chain_res = ChainResidueRepr.model_validate("A7-147")
    assert "A" in chain_res.chain_id_to_residue_index_map
