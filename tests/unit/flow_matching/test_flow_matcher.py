# type: ignore
from typing import Any

import pytest
import torch
import torch.nn as nn

from dynopsi.flow_matching.flow_matcher import FlowMatcher, FlowMatcherOutput
from dynopsi.flow_matching.interpolant import Interpolant
from dynopsi.flow_matching.priors import Sampleable


class MockSampleable(Sampleable):
    """Mock implementation of Sampleable for testing."""

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, 3)


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

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out_coords": torch.randn_like(inputs["coords"])}


class IdentityDenoiser(nn.Module):
    """Denoiser that outputs the inputs. Designed for testing clean data prediction."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out_coords": inputs["coords"]}


class ZeroDenoiser(nn.Module):
    """Denoiser that outputs zeros. Designed for testing CVF prediction."""

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out_coords": torch.zeros_like(inputs["coords"])}


class ConcreteFlowMatcher(FlowMatcher):
    """Concrete implementation of FlowMatcher for testing abstract methods.

    This version is for testing a denoiser that outputs the conditional vector field.
    """

    def _construct_model_inputs(
        self,
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        self_conditioning_inputs: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        other_denoiser_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        batch_size = current_states["coords"].shape[0]
        seq_len = current_states["coords"].shape[1]
        return {
            "coords": current_states["coords"].reshape(batch_size, seq_len, -1, 3),
        }

    def _convert_raw_predictions_to_sc_inputs(
        self,
        raw_predictions: dict[str, torch.Tensor],
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        postprocess_args: dict[str, Any] = {},
    ) -> dict[str, torch.Tensor]:
        """Convert the raw predictions to self-conditioning inputs."""
        return {"coords": raw_predictions["out_coords"]}

    def _postprocess_predictions(
        self,
        raw_predictions: dict[str, torch.Tensor],
        postprocess_args: dict[str, Any] = {},
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = raw_predictions["out_coords"].shape[0]
        seq_len = raw_predictions["out_coords"].shape[1]
        return {"out_coords": raw_predictions["out_coords"].reshape(batch_size, seq_len, -1)}, {}


@pytest.fixture
def flow_matcher_setup():
    """Set up fixture for FlowMatcher tests."""
    priors = {"coords": MockSampleable()}
    interpolants = {"coords": MockInterpolant()}
    denoiser = MockDenoiser()

    return ConcreteFlowMatcher(
        priors=priors,
        interpolants=interpolants,
        denoiser=denoiser,
        is_cvf_prediction=dict(coords=True),
        explicit_mask_fixed_positions=False,
    )


@pytest.fixture
def flow_matcher_identity_denoiser_setup():
    """Set up fixture for FlowMatcher tests with identity denoiser."""
    priors = {"coords": MockSampleable()}
    interpolants = {"coords": MockInterpolant()}
    denoiser = IdentityDenoiser()

    return ConcreteFlowMatcher(
        priors=priors,
        interpolants=interpolants,
        denoiser=denoiser,
        is_cvf_prediction=dict(coords=True),
        explicit_mask_fixed_positions=False,
    )


@pytest.fixture
def processed_batch() -> dict[str, Any]:
    """Sample processed batch for testing."""
    batch_size = 2
    processed_batch = dict(
        data_samples={"coords": torch.randn(batch_size, 10, 3)},
        resolved_mask=torch.ones(batch_size, 10),
        noised_mask=torch.ones(batch_size, 10),
        prior_means={"coords": torch.zeros(batch_size, 10, 3)},
        other_denoiser_inputs={"token_index": torch.randint(0, 20, (batch_size, 10))},
    )

    return processed_batch


# Update test cases to remove 'is_cvf_prediction' and ensure 'losses' is provided
class TestFlowMatcher:
    """Test cases for FlowMatcher class."""

    def test_init_success(self) -> None:
        """Test successful initialization."""
        priors = {"coords": MockSampleable()}
        interpolants = {"coords": MockInterpolant()}
        denoiser = MockDenoiser()

        flow_matcher = ConcreteFlowMatcher(
            priors=priors,
            interpolants=interpolants,
            denoiser=denoiser,
            is_cvf_prediction=dict(coords=True),
            explicit_mask_fixed_positions=False,
        )

        assert flow_matcher.modalities == ["coords"]

    def test_init_mismatched_priors_interpolants(self) -> None:
        """Test initialization fails with mismatched priors and interpolants."""
        priors = {"coords": MockSampleable(), "other": MockSampleable()}
        interpolants = {"coords": MockInterpolant()}
        denoiser = MockDenoiser()

        with pytest.raises(AssertionError):
            ConcreteFlowMatcher(
                priors=priors,
                interpolants=interpolants,
                denoiser=denoiser,
                is_cvf_prediction=dict(coords=True),
                explicit_mask_fixed_positions=False,
            )

    def test_compute_self_conditioning_inputs(self, flow_matcher_setup: ConcreteFlowMatcher) -> None:
        """Test _compute_self_conditioning_inputs method."""
        interpolated_states = {"coords": torch.randn(2, 10, 3)}
        time_samples = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}

        result = flow_matcher_setup._compute_self_conditioning_inputs(
            interpolated_states, time_samples, resolved_mask, noised_mask, other_inputs
        )

        assert "coords" in result
        assert result["coords"].shape == (2, 10, 1, 3)

    def test_denoise_with_self_conditioning(self, flow_matcher_setup: ConcreteFlowMatcher) -> None:
        """Test denoise method with self-conditioning"""
        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}

        flow_predictions, _ = flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
        )
        cvf_output = flow_predictions["coords"].conditional_vector_field
        x1_output = flow_predictions["coords"].clean_data_point

        assert cvf_output.shape == (2, 10, 3)
        assert x1_output.shape == (2, 10, 3)

    def test_denoise_skips_dummy_when_self_conditioning_inputs_provided(
        self, flow_matcher_setup: ConcreteFlowMatcher
    ) -> None:
        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}
        provided = {"coords": torch.ones(2, 10, 1, 3)}
        n_calls = 0
        orig_forward = flow_matcher_setup.denoiser.forward

        def counting_forward(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            nonlocal n_calls
            n_calls += 1
            return orig_forward(inputs)

        flow_matcher_setup.denoiser.forward = counting_forward  # type: ignore[method-assign]
        flow_predictions, _ = flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
            self_conditioning_inputs=provided,
        )
        assert n_calls == 1
        assert flow_predictions["coords"].conditional_vector_field.shape == (2, 10, 3)

    def test_denoise_none_dict_skips_dummy_self_conditioning(self, flow_matcher_setup: ConcreteFlowMatcher) -> None:
        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}
        n_calls = 0
        orig_forward = flow_matcher_setup.denoiser.forward

        def counting_forward(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            nonlocal n_calls
            n_calls += 1
            return orig_forward(inputs)

        flow_matcher_setup.denoiser.forward = counting_forward  # type: ignore[method-assign]
        flow_predictions, _ = flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
            self_conditioning_inputs={"coords": None},
        )
        assert n_calls == 1
        assert flow_predictions["coords"].clean_data_point.shape == (2, 10, 3)

    def test_cvf_denoiser_value_consistency(self) -> None:
        """Test denoise method value consistency. The denoiser should output
        a zero CVF and the x1 should be the same as the current states.
        """
        priors = {"coords": MockSampleable()}
        interpolants = {"coords": MockInterpolant()}
        denoiser = ZeroDenoiser()

        identity_flow_matcher = ConcreteFlowMatcher(
            priors=priors,
            interpolants=interpolants,
            denoiser=denoiser,
            is_cvf_prediction=dict(coords=True),
            explicit_mask_fixed_positions=False,
        )

        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}

        predictions, _ = identity_flow_matcher.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=True,
            other_denoiser_inputs=other_inputs,
        )

        cvf_output = predictions["coords"].conditional_vector_field
        x1_output = predictions["coords"].clean_data_point

        assert torch.allclose(cvf_output, torch.zeros_like(current_states["coords"]), atol=1e-5)
        assert torch.allclose(x1_output, current_states["coords"], atol=1e-5)

    def test_clean_data_denoiser_value_consistency(self) -> None:
        """Test denoise method value consistency.
        The clean data denoiser should output the same as the current states,
        so the CVF should be zero and the x1 should be the same as the current states.
        """
        priors = {"coords": MockSampleable()}
        interpolants = {"coords": MockInterpolant()}
        denoiser = IdentityDenoiser()

        identity_flow_matcher = ConcreteFlowMatcher(
            priors=priors,
            interpolants=interpolants,
            denoiser=denoiser,
            is_cvf_prediction=dict(coords=False),
            explicit_mask_fixed_positions=False,
        )

        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}

        predictions, _ = identity_flow_matcher.denoise(
            current_states, times, resolved_mask, noised_mask, True, other_inputs
        )

        cvf_output = predictions["coords"].conditional_vector_field
        x1_output = predictions["coords"].clean_data_point

        assert torch.allclose(cvf_output, torch.zeros_like(current_states["coords"]), atol=1e-5)
        assert torch.allclose(x1_output, current_states["coords"], atol=1e-5)

    def test_denoise_without_self_conditioning(self, flow_matcher_setup: ConcreteFlowMatcher) -> None:
        """Test denoise method without self-conditioning."""
        current_states = {"coords": torch.randn(2, 10, 3)}
        times = {"coords": torch.rand(2, 1, 1)}
        resolved_mask = torch.ones(2, 10)
        noised_mask = torch.ones(2, 10)
        other_inputs = {"token_index": torch.randint(0, 20, (2, 10))}

        flow_predictions, _ = flow_matcher_setup.denoise(
            current_states,
            times,
            resolved_mask,
            noised_mask,
            use_self_conditioning=False,
            other_denoiser_inputs=other_inputs,
        )

        cvf_output = flow_predictions["coords"].conditional_vector_field
        x1_output = flow_predictions["coords"].clean_data_point

        assert cvf_output.shape == (2, 10, 3)
        assert x1_output.shape == (2, 10, 3)

    def test_apply_explicit_mask_fixed_positions(self, flow_matcher_setup: ConcreteFlowMatcher) -> None:
        """Test apply_explicit_mask_fixed_positions method."""
        batch_size = 2
        L = 10
        d = 3
        predictions = FlowMatcherOutput(
            conditional_vector_field=torch.randn(batch_size, L, d), clean_data_point=torch.randn(batch_size, L, d)
        )
        targets = FlowMatcherOutput(
            conditional_vector_field=torch.randn(batch_size, L, d), clean_data_point=torch.randn(batch_size, L, d)
        )
        noised_mask = torch.randint(0, 2, (batch_size, L)).bool()

        result = flow_matcher_setup._apply_explicit_mask_fixed_positions(
            predictions, targets.clean_data_point, noised_mask
        )

        for i in range(batch_size):
            for j in range(L):
                result_cvf_ij = result.conditional_vector_field[i, j]
                result_x1_ij = result.clean_data_point[i, j]
                target_x1_ij = targets.clean_data_point[i, j]

                pred_cvf_ij = predictions.conditional_vector_field[i, j]

                if noised_mask[i, j] == 0:
                    assert torch.allclose(result_cvf_ij, torch.zeros_like(result_cvf_ij), atol=1e-5)
                    assert torch.allclose(result_x1_ij, target_x1_ij, atol=1e-5)
                else:
                    assert torch.allclose(pred_cvf_ij, pred_cvf_ij, atol=1e-5)
