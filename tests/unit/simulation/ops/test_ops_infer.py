import pytest
import torch

from dynopsi.data.featurization.base import Features
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.ops.infer import Infer
from tests.unit.simulation.conftest import make_store


class TestInfer:
    seq_len: int = 10
    batch_size: int = 5
    in_state_key: str = "x"
    in_feat_key: str = "feat"
    out_state_key: str = "cvf"

    @pytest.fixture
    def infer_op(self, inference_model_config: DynoPsiModelConfig) -> Infer:
        return Infer(
            name="infer",
            valid_steps={"coords": [1]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
        )

    @pytest.mark.parametrize("state_num, is_active", [(1, True), (3, False)])
    def test_input_state_keys(self, state_num: int, is_active: bool, infer_op: Infer) -> None:
        state_keys = infer_op.input_state_keys(state_num)
        if is_active:
            assert state_keys == {self.in_state_key}
        else:
            assert state_keys == set()

    def test_input_feature_keys(self, infer_op: Infer) -> None:
        assert infer_op.input_feature_keys() == {self.in_feat_key}

    @pytest.mark.parametrize("state_num, is_active", [(1, True), (3, False)])
    def test_output_state_keys(self, state_num: int, is_active: bool, infer_op: Infer) -> None:
        state_keys = infer_op.output_state_keys(state_num)
        if is_active:
            assert state_keys == {self.out_state_key}
        else:
            assert state_keys == set()

    def test_model_configs_returns_inference_model(
        self, infer_op: Infer, inference_model_config: DynoPsiModelConfig
    ) -> None:
        assert infer_op.model_configs() == [inference_model_config]

    @pytest.mark.parametrize("state_num, is_active", [(1, True), (3, False)])
    def test_run(self, state_num: int, is_active: bool, num_timesteps: int, infer_op: Infer) -> None:

        input_coords = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        features = {self.in_feat_key: Features(feature_dict={} | input_coords)}
        # get coord information from store
        store = make_store(features=features, num_timesteps=num_timesteps)
        # put the x coords at the anticipated state_num
        store.put(self.in_state_key, input_coords, state_num)
        # run the operation
        infer_op.run(store, state_num)
        model = infer_op.inference_model_config.dynopsi_model
        if is_active:
            assert self.out_state_key in store.states
            assert (
                store.get_state(self.out_state_key, state_num)["coords"]
                == torch.zeros(self.batch_size, 1, self.seq_len, 9) + 1 / num_timesteps * state_num
            ).all()
            assert model.infer_cvf.call_count == 1
            assert model.infer_cvf.call_args.kwargs["self_conditioning_inputs"] is None
        else:
            assert self.out_state_key not in store.states
            assert model.infer_cvf.call_count == 0

    def test_default_self_conditioning_mode_is_same_timestep(self, infer_op: Infer) -> None:
        assert infer_op.self_conditioning_mode == "same_timestep"

    def test_previous_step_feeds_stashed_clean_x1(
        self, inference_model_config: DynoPsiModelConfig, num_timesteps: int
    ) -> None:
        infer_op = Infer(
            name="infer",
            valid_steps={"coords": [0, 1]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
            self_conditioning_mode="previous_step",
        )
        sc_calls: list[dict[str, torch.Tensor | None]] = []

        def infer_cvf(x_in, feature_dict, times, self_conditioning_inputs=None):
            sc_calls.append({k: (None if v is None else v.clone()) for k, v in self_conditioning_inputs.items()})
            output = {k: v.clone() for k, v in x_in.items()}
            output["coords"] = output["coords"] + times["coords"]
            next_sc = {k: torch.full_like(v, float(len(sc_calls))) for k, v in x_in.items()}
            return output, next_sc

        inference_model_config.dynopsi_model.infer_cvf.side_effect = infer_cvf

        input_coords = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        features = {self.in_feat_key: Features(feature_dict={} | input_coords)}
        store = make_store(features=features, num_timesteps=num_timesteps)
        store.put(self.in_state_key, input_coords, 0)
        infer_op.run(store, 0)
        store.put(self.in_state_key, input_coords, 1)
        infer_op.run(store, 1)

        assert len(sc_calls) == 2
        assert sc_calls[0]["coords"] is None
        assert torch.equal(sc_calls[1]["coords"], torch.full_like(input_coords["coords"], 1.0))
        assert inference_model_config.dynopsi_model.call_count == 0

    def test_previous_step_stash_survives_clear_states(
        self, inference_model_config: DynoPsiModelConfig, num_timesteps: int
    ) -> None:
        infer_op = Infer(
            name="infer",
            valid_steps={"coords": [0, 1]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
            self_conditioning_mode="previous_step",
        )
        sc_calls: list[dict[str, torch.Tensor | None]] = []

        def infer_cvf(x_in, feature_dict, times, self_conditioning_inputs=None):
            sc_calls.append({k: (None if v is None else v.clone()) for k, v in self_conditioning_inputs.items()})
            output = {k: v.clone() for k, v in x_in.items()}
            next_sc = {k: torch.full_like(v, float(len(sc_calls))) for k, v in x_in.items()}
            return output, next_sc

        inference_model_config.dynopsi_model.infer_cvf.side_effect = infer_cvf

        input_coords = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        features = {self.in_feat_key: Features(feature_dict={} | input_coords)}
        store = make_store(features=features, num_timesteps=num_timesteps)
        store.put(self.in_state_key, input_coords, 0)
        infer_op.run(store, 0)
        store.clear_states(keep_step_num=1)
        store.put(self.in_state_key, input_coords, 1)
        infer_op.run(store, 1)

        assert len(sc_calls) == 2
        assert sc_calls[0]["coords"] is None
        assert torch.equal(sc_calls[1]["coords"], torch.full_like(input_coords["coords"], 1.0))

    def test_previous_step_stash_is_per_store(
        self, inference_model_config: DynoPsiModelConfig, num_timesteps: int
    ) -> None:
        infer_op = Infer(
            name="infer",
            valid_steps={"coords": [0, 1]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
            self_conditioning_mode="previous_step",
        )
        sc_calls: list[dict[str, torch.Tensor | None]] = []

        def infer_cvf(x_in, feature_dict, times, self_conditioning_inputs=None):
            sc_calls.append({k: (None if v is None else v.clone()) for k, v in self_conditioning_inputs.items()})
            output = {k: v.clone() for k, v in x_in.items()}
            return output, {k: v.clone() for k, v in x_in.items()}

        inference_model_config.dynopsi_model.infer_cvf.side_effect = infer_cvf

        coords_a = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        coords_b = {"coords": torch.ones(self.batch_size, 1, self.seq_len, 9)}
        store_a = make_store(
            features={self.in_feat_key: Features(feature_dict={} | coords_a)}, num_timesteps=num_timesteps
        )
        store_b = make_store(
            features={self.in_feat_key: Features(feature_dict={} | coords_b)}, num_timesteps=num_timesteps
        )
        store_a.put(self.in_state_key, coords_a, 0)
        store_b.put(self.in_state_key, coords_b, 0)
        infer_op.run(store_a, 0)
        infer_op.run(store_b, 0)
        store_a.put(self.in_state_key, coords_a, 1)
        infer_op.run(store_a, 1)

        assert sc_calls[0]["coords"] is None
        assert sc_calls[1]["coords"] is None
        assert torch.equal(sc_calls[2]["coords"], coords_a["coords"])
        assert not torch.equal(sc_calls[2]["coords"], coords_b["coords"])

    def test_previous_step_raises_when_stash_empty(
        self, inference_model_config: DynoPsiModelConfig, num_timesteps: int
    ) -> None:
        infer_op = Infer(
            name="infer",
            valid_steps={"coords": [0, 1]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
            self_conditioning_mode="previous_step",
        )
        input_coords = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        features = {self.in_feat_key: Features(feature_dict={} | input_coords)}
        store = make_store(features=features, num_timesteps=num_timesteps)
        store.put(self.in_state_key, input_coords, 1)
        with pytest.raises(RuntimeError, match="earlier active step"):
            infer_op.run(store, 1)

    def test_previous_step_requires_self_conditioning_checkpoint(
        self, inference_model_config: DynoPsiModelConfig, num_timesteps: int
    ) -> None:
        infer_op = Infer(
            name="infer",
            valid_steps={"coords": [0]},
            inference_model_config=inference_model_config,
            in_x_loc_store=self.in_state_key,
            in_feat_loc_store=self.in_feat_key,
            out_loc_store=self.out_state_key,
            self_conditioning_mode="previous_step",
        )
        inference_model_config.dynopsi_model.use_self_cond = False
        input_coords = {"coords": torch.zeros(self.batch_size, 1, self.seq_len, 9)}
        features = {self.in_feat_key: Features(feature_dict={} | input_coords)}
        store = make_store(features=features, num_timesteps=num_timesteps)
        store.put(self.in_state_key, input_coords, 0)
        with pytest.raises(ValueError, match="use_self_cond"):
            infer_op.run(store, 0)
