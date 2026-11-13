import pytest

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.inference_featurization_pipeline import (
    inference_featurization_pipeline_base as pipeline_base,
)


class TestValidateSupportedAtomsToInclude:
    def test_valid_atom_filter(self) -> None:
        atom_filter = pipeline_base.InferenceFeaturizationPipeline._validate_supported_atoms_to_include(
            atoms_to_include=AtomFilter.CA,
            supported_atoms_to_include=(AtomFilter.BB, AtomFilter.CA),
            pipeline_name="DummyPipeline",
        )
        assert atom_filter == AtomFilter.CA

    def test_invalid_atom_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="supports only atoms_to_include"):
            pipeline_base.InferenceFeaturizationPipeline._validate_supported_atoms_to_include(
                atoms_to_include=AtomFilter.FULL,
                supported_atoms_to_include=(AtomFilter.BB, AtomFilter.CA),
                pipeline_name="DummyPipeline",
            )
