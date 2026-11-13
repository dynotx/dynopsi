import pytest
import torch

from dynopsi.data.featurization.base import Features
from dynopsi.simulation.ops.ode_update import GaussianScoreWeightedODEStepUpdate, ODEStepUpdate
from dynopsi.utils import ConstantFunction, InverseParameterFunction, SingleParameterFunctionType
from tests.unit.simulation.conftest import make_store

# ---------------------------------------------------------------------------
# ODEStepUpdate
# ---------------------------------------------------------------------------


class TestODEStepUpdate:
    in_x_key: str = "x"
    in_cvf_key: str = "cvf"
    out_x_key: str = "x"

    @pytest.fixture
    def ode_op(self) -> ODEStepUpdate:
        return ODEStepUpdate(
            name="ode_update",
            valid_steps={"coords": [1]},
            in_x_loc_store=self.in_x_key,
            in_cvf_loc_store=self.in_cvf_key,
            out_x_loc_store=self.out_x_key,
        )

    @pytest.mark.parametrize("step_num, is_active", [(1, True), (3, False)])
    def test_input_state_keys(self, step_num: int, is_active: bool, ode_op: ODEStepUpdate) -> None:
        keys = ode_op.input_state_keys(step_num)
        if is_active:
            assert keys == {self.in_x_key, self.in_cvf_key}
        else:
            assert keys == set()

    def test_input_feature_keys(self, ode_op: ODEStepUpdate) -> None:
        assert ode_op.input_feature_keys() == set()

    @pytest.mark.parametrize("step_num, is_active", [(2, True), (3, False)])
    def test_output_state_keys(self, step_num: int, is_active: bool, ode_op: ODEStepUpdate) -> None:
        # Op active at step 1 → output_state_keys checks step_num - 1,
        # so output_state_keys(2) is active, output_state_keys(3) is not.
        keys = ode_op.output_state_keys(step_num)
        if is_active:
            assert keys == {self.out_x_key}
        else:
            assert keys == set()

    def test_model_configs(self, ode_op: ODEStepUpdate) -> None:
        assert ode_op.model_configs() == []

    @pytest.mark.parametrize("step_num, is_active", [(1, True), (3, False)])
    def test_run(
        self,
        step_num: int,
        is_active: bool,
        num_timesteps: int,
        ode_op: ODEStepUpdate,
        x_tensor: torch.Tensor,
        cvf_tensor: torch.Tensor,
        t_tensor: torch.Tensor,
        batch_size: int,
    ) -> None:
        features = {"feat": Features(feature_dict={"coords": x_tensor.clone()})}
        store = make_store(features=features, num_timesteps=num_timesteps, batch_size=batch_size)
        store.put(self.in_x_key, {"coords": x_tensor}, step_num)
        store.put(self.in_cvf_key, {"coords": cvf_tensor}, step_num)

        ode_op.run(store, step_num)

        if is_active:
            result = store.get_state(self.out_x_key, step_num + 1)["coords"]
            dt = store.get_timestep_size(step_num, "coords", batch_size)
            expected = x_tensor + cvf_tensor * dt
            assert torch.allclose(result, expected)
        else:
            with pytest.raises(KeyError):
                store.get_state(self.out_x_key, step_num + 1)


# ---------------------------------------------------------------------------
# GaussianScoreWeightedODEStepUpdate
# ---------------------------------------------------------------------------


class TestGaussianScoreWeightedODEStepUpdate:
    feat_key: str = "feat"
    in_x_key: str = "x"
    in_cvf_key: str = "cvf"
    out_x_key: str = "x_out"

    def make_op(
        self, score_scale_fn: SingleParameterFunctionType = ConstantFunction(constant=1.0)
    ) -> GaussianScoreWeightedODEStepUpdate:
        return GaussianScoreWeightedODEStepUpdate(
            name="gs_ode",
            valid_steps={"coords": [1]},
            in_feat_loc_store=self.feat_key,
            in_x_loc_store=self.in_x_key,
            in_cvf_loc_store=self.in_cvf_key,
            out_x_loc_store=self.out_x_key,
            score_scale_fn=score_scale_fn,
        )

    @pytest.mark.parametrize("step_num, is_active", [(1, True), (3, False)])
    def test_input_state_keys(self, step_num: int, is_active: bool) -> None:
        op = self.make_op()
        keys = op.input_state_keys(step_num)
        if is_active:
            assert keys == {self.in_x_key, self.in_cvf_key}
        else:
            assert keys == set()

    @pytest.mark.parametrize("step_num, is_active", [(2, True), (3, False)])
    def test_output_state_keys(self, step_num: int, is_active: bool) -> None:
        op = self.make_op()
        keys = op.output_state_keys(step_num)
        if is_active:
            assert keys == {self.out_x_key}
        else:
            assert keys == set()

    @pytest.mark.parametrize("constant_scale", [1.0, 2.0])
    def test_scale_applies_formula(
        self, constant_scale: float, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor
    ) -> None:
        op = self.make_op(score_scale_fn=ConstantFunction(constant=constant_scale))
        expected = constant_scale * cvf_tensor + (1 - constant_scale) * x_tensor / t_tensor
        result = op.scale_cvf_by_score(x_tensor, cvf_tensor, t_tensor)
        assert torch.allclose(result, expected)

    def test_scale_not_constant_function(
        self, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, t_tensor: torch.Tensor
    ):
        op = self.make_op(score_scale_fn=InverseParameterFunction(eps=1))
        score_scale = 1 / 1.5
        expected = score_scale * cvf_tensor + (1 - score_scale) * x_tensor / t_tensor
        result = op.scale_cvf_by_score(x_tensor, cvf_tensor, t_tensor)
        assert torch.allclose(result, expected)

    def test_run_skips_when_inactive(
        self, num_timesteps: int, x_tensor: torch.Tensor, cvf_tensor: torch.Tensor, seq_len: int, batch_size: int
    ) -> None:
        noised_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        features = {"feat": Features(feature_dict={"coords": x_tensor.clone(), "noised_mask": noised_mask})}
        store = make_store(features=features, num_timesteps=num_timesteps, batch_size=batch_size)
        store.put(self.in_x_key, {"coords": x_tensor}, 0)
        store.put(self.in_cvf_key, {"coords": cvf_tensor}, 0)

        self.make_op().run(store, 0)  # op only active at step 1, not 0

        with pytest.raises(KeyError):
            store.get_state(self.out_x_key, 1)

    @pytest.mark.parametrize("step_num, is_active", [(1, True), (0, False)])
    def test_run(
        self,
        step_num: int,
        is_active: bool,
        num_timesteps: int,
        x_tensor: torch.Tensor,
        cvf_tensor: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> None:
        n_noised_residues = seq_len // 2
        noised_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        noised_mask[:, :n_noised_residues] = True

        features = {"feat": Features(feature_dict={"coords": x_tensor.clone(), "noised_mask": noised_mask})}
        store = make_store(features=features, num_timesteps=num_timesteps, batch_size=batch_size)
        store.put(self.in_x_key, {"coords": x_tensor}, step_num)
        store.put(self.in_cvf_key, {"coords": cvf_tensor}, step_num)

        self.make_op(score_scale_fn=ConstantFunction(constant=2.0)).run(store, step_num)

        if is_active:
            result = store.get_state(self.out_x_key, step_num + 1)["coords"]
            expected_result = x_tensor.clone() + cvf_tensor * (1 / num_timesteps)  # dt is 1/n_timesteps
            # 2 * cvf + (1-2)*x / t
            scaled_cvf = 2 * cvf_tensor[noised_mask] - x_tensor[noised_mask] / (step_num / num_timesteps)
            expected_result[noised_mask] = (
                x_tensor.clone()[noised_mask] + scaled_cvf * 1 / num_timesteps
            )  # dt is 1/n_timesteps

            assert torch.allclose(result, expected_result)
        else:
            with pytest.raises(KeyError):
                store.get_state(self.out_x_key, step_num + 1)
