from unittest import mock

import numpy as np
import pytest
import torch

from dynopsi.data.featurization.base import Features
from dynopsi.simulation.ops.sde_update import GaussianSDEStepUpdate
from dynopsi.utils import ConstantFunction
from tests.unit.simulation.conftest import make_store

# Captured before any test patches torch.randn — used by mock_randn_fn to avoid
# re-entering the mock (mock.patch on a module attribute patches the module singleton globally).
_real_torch_randn = torch.randn


def make_sde_op(
    save_intermediate: bool = False,
    use_legacy: bool = False,
    diffusion: float = 0.1,
    noise_scale: float = 1.0,
) -> GaussianSDEStepUpdate:
    return GaussianSDEStepUpdate(
        name="sde",
        valid_steps={"coords": list(range(5))},
        in_feat_loc_store="feat",
        in_x_loc_store="x",
        in_cvf_loc_store="cvf",
        diffusion_coefficient_fn=ConstantFunction(constant=diffusion),
        noise_scale_fn=ConstantFunction(constant=noise_scale),
        out_x_loc_store="x_out",
        prior_stds={"coords": 1.0},
        use_legacy_cvf_score_conversion=use_legacy,
        save_intermediate_terms=save_intermediate,
    )


# ---------------------------------------------------------------------------
# GaussianSDEStepUpdate (lifecycle + run)
# ---------------------------------------------------------------------------


class TestGaussianSDEStepUpdate:
    feat_key: str = "feat"
    in_x_key: str = "x"
    in_cvf_key: str = "cvf"
    out_x_key: str = "x_out"
    prior_stdev: dict[str, float] = {"coords": 1.0}
    diffusion_coef: float = 0.5
    noise_scale: float = 1.0

    def mock_randn_fn(self, *args, **kwargs):
        torch.manual_seed(0)
        return _real_torch_randn(*args, **kwargs)

    def make_sde_op(self, save_intermediate_terms: bool, **kwargs) -> GaussianSDEStepUpdate:
        return GaussianSDEStepUpdate(
            name="sde",
            valid_steps={"coords": [1]},
            in_feat_loc_store=self.feat_key,
            in_x_loc_store=self.in_x_key,
            in_cvf_loc_store=self.in_cvf_key,
            diffusion_coefficient_fn=ConstantFunction(constant=self.diffusion_coef),
            noise_scale_fn=ConstantFunction(constant=self.noise_scale),
            out_x_loc_store=self.out_x_key,
            prior_stds=self.prior_stdev,
            use_legacy_cvf_score_conversion=False,
            save_intermediate_terms=save_intermediate_terms,
            **kwargs,
        )

    @pytest.mark.parametrize("step_num, is_active", [(1, True), (3, False)])
    def test_input_state_keys(self, step_num: int, is_active: bool) -> None:
        sde_op = self.make_sde_op(save_intermediate_terms=False)
        keys = sde_op.input_state_keys(step_num)
        if is_active:
            assert keys == {self.in_x_key, self.in_cvf_key}
        else:
            assert keys == set()

    def test_input_feature_keys(self) -> None:
        sde_op = self.make_sde_op(save_intermediate_terms=False)
        assert sde_op.input_feature_keys() == set()

    @pytest.mark.parametrize("step_num, is_active", [(2, True), (3, False)])
    def test_output_state_keys(self, step_num: int, is_active: bool) -> None:
        sde_op = self.make_sde_op(save_intermediate_terms=False)
        keys = sde_op.output_state_keys(step_num)
        if is_active:
            assert keys == {self.out_x_key}
        else:
            assert keys == set()

    def test_model_configs(self) -> None:
        sde_op = self.make_sde_op(save_intermediate_terms=False)
        assert sde_op.model_configs() == []

    @pytest.mark.parametrize("save_intermediate_keys", [True, False])
    @pytest.mark.parametrize("custom_intermediate_keys", [True, False])
    def test_intermediate_observable_keys(self, save_intermediate_keys: bool, custom_intermediate_keys: bool) -> None:
        if custom_intermediate_keys:
            intermediate_keys = {
                "drift_terms_loc_store": "custom_drift",
                "score_terms_loc_store": "custom_score",
                "noise_terms_loc_store": "custom_noise",
            }
            sde_op = self.make_sde_op(save_intermediate_terms=save_intermediate_keys, **intermediate_keys)
        else:
            intermediate_keys = {
                "drift_terms_loc_store": "drift_terms",
                "score_terms_loc_store": "score_terms",
                "noise_terms_loc_store": "noise_terms",
            }
            sde_op = self.make_sde_op(save_intermediate_terms=save_intermediate_keys)
        keys = sde_op.intermediate_observable_keys()
        if save_intermediate_keys:
            assert keys == set(intermediate_keys.values())
        else:
            assert keys == set()

    @pytest.mark.parametrize(
        "step_num, is_active, save_intermediate_terms",
        [(1, True, False), (3, False, False), (1, True, True)],
        ids=["step1_active_no_intermediates", "step3_inactive_no_intermediates", "step1_active_save_intermediates"],
    )
    def test_run(
        self,
        step_num: int,
        is_active: bool,
        save_intermediate_terms: bool,
        num_timesteps: int,
        x_tensor: torch.Tensor,
        cvf_tensor: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> None:
        sde_op = self.make_sde_op(save_intermediate_terms=save_intermediate_terms)

        n_noised = seq_len // 2

        noised_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        noised_mask[:, :n_noised] = True
        features = {self.feat_key: Features(feature_dict={"coords": x_tensor.clone(), "noised_mask": noised_mask})}
        store = make_store(features=features, num_timesteps=num_timesteps, batch_size=batch_size)
        store.put(self.in_x_key, {"coords": x_tensor}, step_num)
        store.put(self.in_cvf_key, {"coords": cvf_tensor}, step_num)

        with mock.patch("dynopsi.simulation.ops.sde_update.torch.randn") as mock_randn:
            mock_randn.side_effect = self.mock_randn_fn
            sde_op.run(store, step_num)

        if is_active:
            result = store.get_state(self.out_x_key, step_num + 1)["coords"]
            dt = 1 / num_timesteps
            t = step_num / num_timesteps

            score = (t * cvf_tensor - x_tensor) / (self.prior_stdev["coords"] ** 2 * (1 - t))

            drift_term = cvf_tensor * dt
            score_term = self.diffusion_coef * score * dt
            noise_term = (
                np.sqrt(2 * self.diffusion_coef * self.noise_scale)
                * self.mock_randn_fn(x_tensor.shape, dtype=x_tensor.dtype, device=x_tensor.device)
                * np.sqrt(dt)
            )

            # diffusion=0 → score_term=0, noise_term=0 → x_out = x_in + cvf * dt
            expected = x_tensor + drift_term
            expected[noised_mask] = x_tensor[noised_mask] + (
                drift_term[noised_mask] + score_term[noised_mask] + noise_term[noised_mask]
            )
            assert torch.allclose(result, expected)

            if save_intermediate_terms:
                # check for terms
                assert torch.allclose(store.get_state(sde_op.drift_terms_loc_store, step_num + 1)["coords"], drift_term)
                score_term_actual = store.get_state(sde_op.score_terms_loc_store, step_num + 1)["coords"]
                assert torch.allclose(score_term_actual[noised_mask], score_term[noised_mask])
                assert score_term_actual[~noised_mask].isnan().all()

                noise_term_actual = store.get_state(sde_op.noise_terms_loc_store, step_num + 1)["coords"]
                assert torch.allclose(noise_term_actual[noised_mask], noise_term[noised_mask])
                assert noise_term_actual[~noised_mask].isnan().all()
        else:
            with pytest.raises(KeyError):
                store.get_state(self.out_x_key, step_num + 1)


# ---------------------------------------------------------------------------
# brownian_increment
# ---------------------------------------------------------------------------


class TestBrownianIncrement:
    def test_shape_matches_dt(self, batch_size: int, seq_len: int) -> None:
        op = make_sde_op()
        dt = torch.ones(batch_size, seq_len, 9) * 0.01
        assert op.brownian_increment(dt).shape == dt.shape

    def test_std_scales_with_sqrt_dt(self) -> None:
        # E[dW^2] = dt, so std(dW) = sqrt(dt).  With large N the ratio should be ~sqrt(100).
        torch.manual_seed(0)
        op = make_sde_op()
        n = 10_000
        dt_large = torch.full((n, 1, 1), 1.0)
        dt_small = torch.full((n, 1, 1), 0.01)
        ratio = op.brownian_increment(dt_large).std() / op.brownian_increment(dt_small).std()
        assert 8.0 < ratio.item() < 12.0


# ---------------------------------------------------------------------------
# vector_field_to_score
# ---------------------------------------------------------------------------


class TestVectorFieldToScore:
    def test_non_legacy_formula_zero_score(
        self, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor, batch_size: int
    ) -> None:
        # When t*cvf == x the score should be zero.
        # x_tensor=ones, cvf_tensor=2.0, t_tensor=0.5 → 0.5*2 = 1 = x ✓
        op = make_sde_op(use_legacy=False)
        prior_std = torch.ones(batch_size, 1, 1)
        score = op.vector_field_to_score(x_tensor, cvf_tensor, t_tensor, prior_std)
        assert torch.allclose(score, torch.zeros_like(score))

    def test_non_legacy_scales_by_prior_std(self, x_tensor: torch.Tensor, batch_size: int) -> None:
        # Doubling prior_std should halve the score magnitude.
        op = make_sde_op(use_legacy=False)
        x = torch.zeros_like(x_tensor)
        cvf = torch.ones_like(x_tensor)
        t = torch.full((batch_size, 1, 1), 0.5)
        std1 = torch.ones(batch_size, 1, 1)
        std2 = torch.full((batch_size, 1, 1), 2.0)
        s1 = op.vector_field_to_score(x, cvf, t, std1)
        s2 = op.vector_field_to_score(x, cvf, t, std2)
        assert torch.allclose(s1, s2 * 4)  # (2^2 factor from prior_std^2)

    def test_legacy_formula(self, cvf_tensor: torch.Tensor, x_tensor: torch.Tensor, batch_size: int) -> None:
        # legacy: score = (t*cvf - x) / (1 - t)
        op = make_sde_op(use_legacy=True)
        x = torch.zeros_like(x_tensor)
        t = torch.full((batch_size, 1, 1), 0.5)
        prior_std = torch.ones(batch_size, 1, 1)
        # (0.5*2 - 0) / (1 - 0.5) = 1 / 0.5 = 2
        score = op.vector_field_to_score(x, cvf_tensor, t, prior_std)
        assert torch.allclose(score, torch.full_like(score, 2.0))


# ---------------------------------------------------------------------------
# get_sde_update_terms
# ---------------------------------------------------------------------------


class TestGetSdeUpdateTerms:
    def test_output_shapes_match_input(
        self, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor, batch_size: int
    ) -> None:
        op = make_sde_op()
        dt = torch.full((batch_size, 1, 1), 0.01)
        prior_std = torch.ones(batch_size, 1, 1)
        drift, score, noise = op.get_sde_update_terms(x_tensor, cvf_tensor, t_tensor, dt, prior_std)
        assert drift.shape == x_tensor.shape
        assert score.shape == x_tensor.shape
        assert noise.shape == x_tensor.shape

    def test_drift_equals_cvf_times_dt(
        self, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor, batch_size: int
    ) -> None:
        op = make_sde_op()
        dt = torch.full((batch_size, 1, 1), 0.01)
        prior_std = torch.ones(batch_size, 1, 1)
        drift, _, _ = op.get_sde_update_terms(x_tensor, cvf_tensor, t_tensor, dt, prior_std)
        assert torch.allclose(drift, cvf_tensor * dt)

    def test_zero_diffusion_produces_zero_score_and_noise_terms(
        self, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor, batch_size: int
    ) -> None:
        op = make_sde_op(diffusion=0.0)
        dt = torch.full((batch_size, 1, 1), 0.01)
        prior_std = torch.ones(batch_size, 1, 1)
        _, score, noise = op.get_sde_update_terms(x_tensor, cvf_tensor, t_tensor, dt, prior_std)
        assert torch.allclose(score, torch.zeros_like(score))
        assert torch.allclose(noise, torch.zeros_like(noise))
