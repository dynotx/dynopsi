from typing import Any

import pytest
from pydantic import ValidationError

from dynopsi.data.constants import AtomFilter, DesignTask, InpaintFeature
from dynopsi.data.featurization.inference_featurization_pipeline.binder_design import (
    DynoPsiBinderDesignFeaturizationPipeline,
)
from dynopsi.data.featurization.inference_featurization_pipeline.registry import (
    get_featurization_pipeline_from_config,
)
from dynopsi.data.featurization.primitives import PredefinedCenterOfMassEstimator


def _pipeline_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(
        name="feat_test",
        index_type="original_residue_index",
        structure_path="path/to/structure.pdb",
        hotspots={"hs": "A5,A8"},
        new_chains_lengths=[10],
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=((25.0, 40.0, 30.0),))},
        binder_atoms_to_include=AtomFilter.CA,
    )
    kwargs.update(overrides)
    return kwargs


def _pipeline(**overrides: Any) -> DynoPsiBinderDesignFeaturizationPipeline:
    return DynoPsiBinderDesignFeaturizationPipeline(**_pipeline_kwargs(**overrides))


def _registry_config(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "type": "dynopsi_binder_design_featurization_pipeline",
        "name": "feat_test",
        "index_type": "original_residue_index",
        "structure_path": "path/to/structure.pdb",
        "hotspots": {"hs": "A5,A8"},
        "new_chains_lengths": [10],
        "new_chains_centers_of_mass": {
            "center": {
                "type": "predefined_center_of_mass",
                "centers_of_mass": [[25.0, 40.0, 30.0]],
            }
        },
        "binder_atoms_to_include": "ca",
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize(
    ("design_task", "expected_task", "binder_inpaint_noised", "target_context_noised"),
    [
        (None, DesignTask.MULTIMER_BINDER_DESIGN, True, False),
        ("MULTIMER_BINDER_DESIGN", DesignTask.MULTIMER_BINDER_DESIGN, True, False),
        ("MULTIMER_BINDER_DESIGN_NOISE_ALL", DesignTask.MULTIMER_BINDER_DESIGN_NOISE_ALL, True, True),
    ],
)
def test_design_task_sets_noise_mask(
    design_task: str | None,
    expected_task: DesignTask,
    binder_inpaint_noised: bool,
    target_context_noised: bool,
) -> None:
    pipeline = _pipeline() if design_task is None else _pipeline(design_task=design_task)
    assert pipeline.design_task == expected_task.name
    noise = pipeline._noise_mask_featurizer
    assert noise is not None
    assert noise.design_task is expected_task
    assert noise.design_task.get_noise_bool(InpaintFeature.BINDER_INPAINT) is binder_inpaint_noised
    assert noise.design_task.get_noise_bool(InpaintFeature.TARGET_CONTEXT) is target_context_noised
    dumped = pipeline.model_dump(mode="json")
    assert dumped["design_task"] == expected_task.name


def test_invalid_design_task_raises() -> None:
    with pytest.raises(ValidationError):
        _pipeline(design_task="MULTIMER_DOCKING_NOISE_ALL")


@pytest.mark.parametrize(
    ("overrides", "expected_task"),
    [
        ({}, DesignTask.MULTIMER_BINDER_DESIGN),
        ({"design_task": "MULTIMER_BINDER_DESIGN_NOISE_ALL"}, DesignTask.MULTIMER_BINDER_DESIGN_NOISE_ALL),
    ],
)
def test_registry_deserializes_design_task(overrides: dict[str, Any], expected_task: DesignTask) -> None:
    pipeline = get_featurization_pipeline_from_config(_registry_config(**overrides))
    assert isinstance(pipeline, DynoPsiBinderDesignFeaturizationPipeline)
    assert pipeline.design_task == expected_task.name
    assert pipeline._noise_mask_featurizer.design_task is expected_task
