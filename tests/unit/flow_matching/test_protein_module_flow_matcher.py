# type: ignore
from typing import Any

import pytest
import torch
import torch.nn as nn

from dynopsi.flow_matching.flow_matcher import FlowMatcherOutput
from dynopsi.flow_matching.interpolant import Interpolant
from dynopsi.flow_matching.priors import Sampleable
from dynopsi.flow_matching.protein_module_flow_matcher import ProteinModuleFlowMatcher


class MockSampleable(Sampleable):
    """Mock implementation of Sampleable for testing."""

    def sample(
        self, batch_size: int, unmasked: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return torch.randn(batch_size, 10, 3)


class MockInterpolant(Interpolant):
    """Mock implementation of Interpolant for testing."""

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x0 * (1 - t) + x1 * t

    def conditional_vector_field(
        self,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        if x0 is not None:
            return x1 - x0
        return x1 - xt

    def apply_conditional_vector_field(self, xt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        return xt + ut * dt


class MockDenoiser(nn.Module):
    """Mock denoiser for testing."""

    def __init__(self) -> None:
        super().__init__()
        # Add output_head attribute to prevent AttributeError in _postprocess_predictions
        self.output_head = type("MockOutputHead", (), {"distogram_head": None})()
        self.calls: list[dict[str, Any]] = []

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(inputs)
        return {
            "out_coords": torch.randn_like(inputs["coords"]),
            "distogram_prediction": torch.randn_like(inputs["coords"]),
        }


@pytest.fixture
def protein_flow_matcher_setup() -> ProteinModuleFlowMatcher:
    """Set up fixture for ProteinModuleFlowMatcher tests."""
    priors = {"coords": MockSampleable()}
    interpolants = {"coords": MockInterpolant()}
    denoiser = MockDenoiser()

    return ProteinModuleFlowMatcher(
        priors=priors,
        interpolants=interpolants,
        denoiser=denoiser,
        is_cvf_prediction=dict(coords=False),
        explicit_mask_fixed_positions=False,
    )


@pytest.fixture
def processed_batch() -> dict[str, Any]:
    """Sample batch for testing."""
    batch_size = 1
    seq_len = 10

    # Create coords tensor: (1, bs, max_atoms, 3)
    data_samples = {"coords": torch.randn(batch_size, seq_len, 3)}

    resolved_mask = torch.ones(batch_size, seq_len)  # (bs, L)
    noised_mask = torch.ones(batch_size, seq_len)  # (bs, L)

    other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}  # (bs, L)

    return batch_size, data_samples, resolved_mask, noised_mask, other_inputs


class TestProteinModuleFlowMatcher:
    """Test cases for ProteinModuleFlowMatcher class."""

    def test_init_success(self) -> None:
        """Test successful initialization with single modality."""
        priors = {"coords": MockSampleable()}
        interpolants = {"coords": MockInterpolant()}
        denoiser = MockDenoiser()

        flow_matcher = ProteinModuleFlowMatcher(
            priors=priors,
            interpolants=interpolants,
            denoiser=denoiser,
            is_cvf_prediction=dict(coords=False),
            explicit_mask_fixed_positions=False,
        )

        assert flow_matcher.modality == "coords"
        assert flow_matcher.modalities == ["coords"]
        assert flow_matcher.is_cvf_prediction[flow_matcher.modality] is False

    def test_init_multiple_modalities_fails(self) -> None:
        """Test initialization fails with multiple modalities."""
        priors = {"coords": MockSampleable(), "other": MockSampleable()}
        interpolants = {"coords": MockInterpolant(), "other": MockInterpolant()}
        denoiser = MockDenoiser()

        with pytest.raises(ValueError, match="Unsupported modality: other"):
            ProteinModuleFlowMatcher(
                priors=priors,
                interpolants=interpolants,
                denoiser=denoiser,
                is_cvf_prediction=dict(coords=False, other=False),
                explicit_mask_fixed_positions=False,
            )

    def test_construct_model_inputs(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        """Test _construct_model_inputs method."""
        batch_size = 1
        seq_len = 10
        n_atoms = 1

        current_states = {"coords": torch.randn(batch_size, seq_len, n_atoms * 3)}  # (bs, L, d)
        times = {"coords": torch.rand(batch_size, 1, 1)}  # (bs, 1, 1)
        self_conditioning_inputs = {"coords": torch.randn(batch_size, seq_len, n_atoms, 3)}  # (bs, L, n_atoms, 3)
        resolved_mask = torch.ones(batch_size, seq_len)  # (bs, L)
        noised_mask = torch.ones(batch_size, seq_len)  # (bs, L)
        other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}  # (bs, L)

        model_inputs = protein_flow_matcher_setup._construct_model_inputs(
            current_states, times, self_conditioning_inputs, resolved_mask, noised_mask, other_inputs
        )

        expected_keys = {"coords", "token_index", "t", "mask", "sc_coords", "noised_mask"}
        assert set(model_inputs.keys()) == expected_keys

        assert model_inputs["coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)
        assert model_inputs["token_index"].shape == (batch_size, seq_len)  # (bs, L)
        assert model_inputs["t"].shape == (batch_size,)  # (bs,)
        assert model_inputs["mask"].shape == (batch_size, seq_len)  # (bs, L)
        assert model_inputs["noised_mask"].shape == (batch_size, seq_len)  # (bs, L)
        assert model_inputs["sc_coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)

        expected_coords = current_states["coords"].reshape(batch_size, seq_len, n_atoms, 3)
        assert torch.allclose(model_inputs["coords"], expected_coords)

        expected_times = times["coords"].reshape(-1)
        assert torch.allclose(model_inputs["t"], expected_times)

        assert torch.allclose(model_inputs["mask"], resolved_mask)
        assert torch.allclose(model_inputs["noised_mask"], noised_mask)
        assert torch.allclose(model_inputs["token_index"], other_inputs["token_index"])
        assert torch.allclose(model_inputs["sc_coords"], self_conditioning_inputs["coords"])

    def test_construct_model_inputs_multiple_atoms(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        """Test _construct_model_inputs method with multiple atoms per residue."""
        batch_size = 1
        seq_len = 10
        n_atoms = 4  # Multiple atoms per residue

        current_states = {"coords": torch.randn(batch_size, seq_len, n_atoms * 3)}  # (bs, L, d)
        times = {"coords": torch.rand(batch_size, 1, 1)}  # (bs, 1, 1)
        self_conditioning_inputs = {"coords": torch.randn(batch_size, seq_len, n_atoms, 3)}  # (bs, L, n_atoms, 3)
        resolved_mask = torch.ones(batch_size, seq_len)  # (bs, L)
        noised_mask = torch.ones(batch_size, seq_len)  # (bs, L)
        other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}  # (bs, L)

        model_inputs = protein_flow_matcher_setup._construct_model_inputs(
            current_states, times, self_conditioning_inputs, resolved_mask, noised_mask, other_inputs
        )

        assert model_inputs["coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)
        assert model_inputs["sc_coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)

        expected_coords = current_states["coords"].reshape(batch_size, seq_len, n_atoms, 3)
        assert torch.allclose(model_inputs["coords"], expected_coords)

    def test_convert_raw_predictions_to_sc_inputs(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        """Test _convert_raw_predictions_to_sc_inputs method."""
        batch_size = 1
        seq_len = 10
        n_atoms = 1

        raw_predictions = {
            "out_coords": torch.randn(batch_size, seq_len, n_atoms, 3),  # (bs, L, n_atoms, 3)
            "distogram_prediction": torch.randn(
                batch_size, seq_len, n_atoms, n_atoms, 10
            ),  # (bs, L, n_atoms, n_atoms, 10)
        }

        current_states = {"coords": torch.randn(batch_size, seq_len, n_atoms * 3)}  # (bs, L, d)
        times = {"coords": torch.rand(batch_size, 1, 1)}  # (bs, 1, 1)

        sc_inputs = protein_flow_matcher_setup._convert_raw_predictions_to_sc_inputs(
            raw_predictions, current_states, times
        )

        assert "coords" in sc_inputs
        assert sc_inputs["coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)

        assert torch.allclose(sc_inputs["coords"], raw_predictions["out_coords"])

    def test_sc_inputs_from_flow_output_reshapes_coords(
        self, protein_flow_matcher_setup: ProteinModuleFlowMatcher
    ) -> None:
        batch_size = 2
        seq_len = 10
        n_atoms = 3
        clean = torch.randn(batch_size, seq_len, n_atoms * 3)
        sc_inputs = protein_flow_matcher_setup.sc_inputs_from_flow_output(
            {
                "coords": FlowMatcherOutput(
                    conditional_vector_field=torch.zeros_like(clean),
                    clean_data_point=clean,
                )
            }
        )
        assert sc_inputs["coords"].shape == (batch_size, seq_len, n_atoms, 3)
        assert torch.equal(sc_inputs["coords"], clean.reshape(batch_size, seq_len, n_atoms, 3))

    def test_denoise_feeds_provided_tensor_as_sc_coords(
        self, protein_flow_matcher_setup: ProteinModuleFlowMatcher
    ) -> None:
        batch_size = 1
        seq_len = 10
        current_states = {"coords": torch.randn(batch_size, seq_len, 3)}
        times = {"coords": torch.rand(batch_size, 1, 1)}
        resolved_mask = torch.ones(batch_size, seq_len)
        noised_mask = torch.ones(batch_size, seq_len)
        other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}
        provided = {"coords": torch.ones(batch_size, seq_len, 1, 3)}
        denoiser = protein_flow_matcher_setup.denoiser
        assert isinstance(denoiser, MockDenoiser)

        protein_flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
            self_conditioning_inputs=provided,
        )
        assert len(denoiser.calls) == 1
        assert denoiser.calls[0]["sc_coords"] is provided["coords"]

    def test_denoise_none_dict_sets_sc_coords_none(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        batch_size = 1
        seq_len = 10
        current_states = {"coords": torch.randn(batch_size, seq_len, 3)}
        times = {"coords": torch.rand(batch_size, 1, 1)}
        resolved_mask = torch.ones(batch_size, seq_len)
        noised_mask = torch.ones(batch_size, seq_len)
        other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}
        denoiser = protein_flow_matcher_setup.denoiser
        assert isinstance(denoiser, MockDenoiser)

        protein_flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
            self_conditioning_inputs={"coords": None},
        )
        assert len(denoiser.calls) == 1
        assert denoiser.calls[0]["sc_coords"] is None

    def test_denoise_omitted_self_conditioning_runs_dummy_pass(
        self, protein_flow_matcher_setup: ProteinModuleFlowMatcher
    ) -> None:
        batch_size = 1
        seq_len = 10
        current_states = {"coords": torch.randn(batch_size, seq_len, 3)}
        times = {"coords": torch.rand(batch_size, 1, 1)}
        resolved_mask = torch.ones(batch_size, seq_len)
        noised_mask = torch.ones(batch_size, seq_len)
        other_inputs = {"token_index": torch.randint(0, 20, (batch_size, seq_len))}
        denoiser = protein_flow_matcher_setup.denoiser
        assert isinstance(denoiser, MockDenoiser)

        protein_flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
        )
        assert len(denoiser.calls) == 2
        assert denoiser.calls[0]["sc_coords"] is None
        assert denoiser.calls[1]["sc_coords"] is not None
        assert denoiser.calls[1]["sc_coords"].shape == (batch_size, seq_len, 1, 3)

    def test_convert_raw_predictions_to_sc_inputs_multiple_atoms(
        self, protein_flow_matcher_setup: ProteinModuleFlowMatcher
    ) -> None:
        """Test _convert_raw_predictions_to_sc_inputs method with multiple atoms."""
        batch_size = 1
        seq_len = 10
        n_atoms = 4

        raw_predictions = {
            "out_coords": torch.randn(batch_size, seq_len, n_atoms, 3),  # (bs, L, n_atoms, 3)
            "distogram_prediction": torch.randn(
                batch_size, seq_len, n_atoms, n_atoms, 10
            ),  # (bs, L, n_atoms, n_atoms, 10)
        }
        current_states = {"coords": torch.randn(batch_size, seq_len, n_atoms * 3)}  # (bs, L, d)
        times = {"coords": torch.rand(batch_size, 1, 1)}  # (bs, 1, 1)

        sc_inputs = protein_flow_matcher_setup._convert_raw_predictions_to_sc_inputs(
            raw_predictions, current_states, times
        )

        assert sc_inputs["coords"].shape == (batch_size, seq_len, n_atoms, 3)  # (bs, L, n_atoms, 3)
        assert torch.allclose(sc_inputs["coords"], raw_predictions["out_coords"])

    def test_postprocess_predictions(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        """Test _postprocess_predictions method."""
        batch_size = 1
        seq_len = 10
        n_atoms = 1

        raw_predictions = {
            "out_coords": torch.randn(batch_size, seq_len, n_atoms, 3),  # (bs, L, n_atoms, 3)
            "distogram_prediction": torch.randn(
                batch_size, seq_len, n_atoms, n_atoms, 10
            ),  # (bs, L, n_atoms, n_atoms, 10)
        }

        flow_predictions, aux_predictions = protein_flow_matcher_setup._postprocess_predictions(raw_predictions)

        # Check shape transformation: (bs, L, n_atoms, 3) -> (bs, L, n_atoms * 3)
        assert flow_predictions["out_coords"].shape == (batch_size, seq_len, 3)
        assert "distogram_prediction" in aux_predictions

    def test_postprocess_predictions_multiple_atoms(self, protein_flow_matcher_setup: ProteinModuleFlowMatcher) -> None:
        """Test _postprocess_predictions method with multiple atoms."""
        batch_size = 1
        seq_len = 10
        n_atoms = 4

        raw_predictions = {
            "out_coords": torch.randn(batch_size, seq_len, n_atoms, 3),  # (bs, L, n_atoms, 3)
            "distogram_prediction": torch.randn(
                batch_size, seq_len, n_atoms, n_atoms, 10
            ),  # (bs, L, n_atoms, n_atoms, 10)
        }

        flow_predictions, aux_predictions = protein_flow_matcher_setup._postprocess_predictions(raw_predictions)

        # Check shape transformation: (bs, L, n_atoms, 3) -> (bs, L, n_atoms * 3)
        assert flow_predictions["out_coords"].shape == (batch_size, seq_len, n_atoms * 3)
        assert "distogram_prediction" in aux_predictions

    def test_end_to_end_processing(
        self, protein_flow_matcher_setup: ProteinModuleFlowMatcher, processed_batch: dict[str, Any]
    ) -> None:
        """Test end-to-end processing through all methods."""
        batch_size, data_samples, resolved_mask, noised_mask, other_inputs = processed_batch

        # Create inputs for model construction
        current_states = {"coords": torch.randn(batch_size, 10, 3)}  # (bs, L, d)
        times = {"coords": torch.rand(batch_size, 1, 1)}  # (bs, 1, 1)
        self_conditioning_inputs = {"coords": torch.randn(batch_size, 10, 1, 3)}  # (bs, L, n_atoms, 3)

        # Construct model inputs
        model_inputs = protein_flow_matcher_setup._construct_model_inputs(
            current_states, times, self_conditioning_inputs, resolved_mask, noised_mask, other_inputs
        )

        # Simulate denoiser output
        raw_predictions = {
            "out_coords": torch.randn(batch_size, 10, 1, 3),  # (bs, L, n_atoms, 3)
            "distogram_prediction": torch.randn(batch_size, 10, 1, 1, 10),  # (bs, L, n_atoms, n_atoms, 10)
        }

        # Convert to self-conditioning inputs
        sc_inputs = protein_flow_matcher_setup._convert_raw_predictions_to_sc_inputs(
            raw_predictions, current_states, times
        )

        # Postprocess predictions
        processed_predictions, aux_predictions = protein_flow_matcher_setup._postprocess_predictions(raw_predictions)

        # Verify shapes are consistent throughout the pipeline
        assert model_inputs["coords"].shape == (batch_size, 10, 1, 3)  # (bs, L, n_atoms, 3)
        assert sc_inputs["coords"].shape == (batch_size, 10, 1, 3)  # (bs, L, n_atoms, 3)
        assert processed_predictions["out_coords"].shape == (batch_size, 10, 3)  # (bs, L, d)
        assert "distogram_prediction" in aux_predictions

    def test_non_coords_modality_rejected(self) -> None:
        """Only the 'coords' modality is supported; constructing with another must raise."""
        priors = {"backbone": MockSampleable()}
        interpolants = {"backbone": MockInterpolant()}
        denoiser = MockDenoiser()

        with pytest.raises(ValueError, match="Unsupported modality"):
            ProteinModuleFlowMatcher(
                priors=priors,
                interpolants=interpolants,
                denoiser=denoiser,
                is_cvf_prediction=dict(backbone=False),
                explicit_mask_fixed_positions=False,
            )
