from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def minimal_design_dict() -> dict[str, Any]:
    """Minimal valid design dict with one of each component of the design pipeline."""
    return {
        "featurization_pipelines": {
            "feat_a": {"ref_structure": "path/to/structure.pdb"},
        },
        "output_specifications": {
            "out_a": {"format": "pdb"},
        },
        "simulation_pipelines": {
            "sim_a": {"type": "sde_simulation"},
        },
        "design_pipelines": {
            "design_a": {
                "featurization_pipeline": "feat_a",
                "simulation_pipeline": "sim_a",
                "output_specification": "out_a",
            }
        },
    }


@pytest.fixture
def multi_pipeline_design_dict() -> dict[str, Any]:
    """Design dict with multiple featurization/simulation pipelines and combinations."""
    return {
        "featurization_pipelines": {
            "feat_a": {"ref_structure": "path/to/structure.pdb"},
            "feat_b": {"ref_rcsb_id": "1abc"},
        },
        "output_specifications": {
            "out_a": {"format": "pdb"},
            "out_b": {"format": "npz"},
        },
        "simulation_pipelines": {
            "sim_a": {"type": "sde_simulation"},
            "sim_b": {"type": "ode_simulation"},
        },
        "design_pipelines": {
            "design_a": {
                "featurization_pipeline": "feat_a",
                "simulation_pipeline": "sim_a",
                "output_specification": "out_a",
            },
            "design_b": {
                "featurization_pipeline": "feat_b",
                "simulation_pipeline": "sim_b",
                "output_specification": "out_b",
            },
        },
    }


@pytest.fixture
def mock_featurization_pipeline():
    return MagicMock(name="MockFeaturizationPipeline")


@pytest.fixture
def mock_simulation_pipeline():
    return MagicMock(name="MockSimulationPipeline")


@pytest.fixture
def mock_output_specification():
    return MagicMock(name="MockOutputSpecification")


@pytest.fixture
def mock_design_pipeline():
    return MagicMock(name="MockDesignPipeline")
