from unittest import mock

import torch

from dynopsi.model.model_config import DynoPsiModelConfig, HuggingFaceLoaderConfig
from dynopsi.simulation.build_initial_state import InitializeFeatures, InitializeState
from dynopsi.simulation.observable import ObservableOutputConfig
from dynopsi.simulation.ops import Infer, ODEStepUpdate
from dynopsi.simulation.simulation_pipeline.ode_simulation import ODESimulation
from dynopsi.simulation.time_sampler import LinearTimeSampler

NUM_STEPS = 5

# A lightweight DynoPsiModelConfig — no model is loaded unless .dynopsi_model is accessed.
_FAKE_CFG = DynoPsiModelConfig(loader=HuggingFaceLoaderConfig(repo_id="fake/repo", filename="fake.pt", revision="0"))


def make_sim(num_steps: int = NUM_STEPS, observable_output_configs=None, **overrides) -> ODESimulation:
    """Construct an ODESimulation with a lightweight inference_model_config."""
    kwargs = dict(
        name="test_ode",
        inference_model_config=_FAKE_CFG,
        time_sampler=LinearTimeSampler(num_steps=num_steps, modalities=["coords"]),
    )
    if observable_output_configs is not None:
        kwargs["observable_output_configs"] = observable_output_configs
    kwargs.update(overrides)
    return ODESimulation(**kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestODESimulationRegistration:
    def test_type_string(self) -> None:
        sim = make_sim()
        assert sim.type == "ode_simulation"


# ---------------------------------------------------------------------------
# plan_init structure
# ---------------------------------------------------------------------------


class TestODESimulationPlanInit:
    def test_plan_init_has_two_steps(self) -> None:
        assert len(make_sim().plan_init) == 2

    def test_plan_init_first_step_is_initialize_features(self) -> None:
        assert isinstance(make_sim().plan_init[0], InitializeFeatures)

    def test_plan_init_second_step_is_initialize_state(self) -> None:
        assert isinstance(make_sim().plan_init[1], InitializeState)

    def test_plan_init_features_keys_wired_correctly(self) -> None:
        sim = make_sim()
        init_feat: InitializeFeatures = sim.plan_init[0]  # type: ignore[assignment]
        assert init_feat.in_feat_loc_store == "features"
        assert init_feat.out_feat_loc_store == "batch_features"

    def test_plan_init_state_reads_batch_features(self) -> None:
        sim = make_sim()
        init_state: InitializeState = sim.plan_init[1]  # type: ignore[assignment]
        assert init_state.in_feat_loc_store == "batch_features"

    def test_plan_init_state_writes_X_T(self) -> None:
        sim = make_sim()
        init_state: InitializeState = sim.plan_init[1]  # type: ignore[assignment]
        assert init_state.out_loc_store == "X_T"


# ---------------------------------------------------------------------------
# plan structure
# ---------------------------------------------------------------------------


class TestODESimulationPlan:
    def test_plan_has_two_ops(self) -> None:
        assert len(make_sim().plan) == 2

    def test_plan_first_op_is_infer(self) -> None:
        assert isinstance(make_sim().plan[0], Infer)

    def test_plan_second_op_is_ode_step_update(self) -> None:
        assert isinstance(make_sim().plan[1], ODEStepUpdate)

    def test_infer_reads_X_T_and_batch_features(self) -> None:
        infer: Infer = make_sim().plan[0]  # type: ignore[assignment]
        assert infer.in_x_loc_store == "X_T"
        assert infer.in_feat_loc_store == "batch_features"

    def test_infer_writes_CVF_T(self) -> None:
        infer: Infer = make_sim().plan[0]  # type: ignore[assignment]
        assert infer.out_loc_store == "CVF_T"

    def test_ode_update_reads_X_T_and_CVF_T(self) -> None:
        ode: ODEStepUpdate = make_sim().plan[1]  # type: ignore[assignment]
        assert ode.in_x_loc_store == "X_T"
        assert ode.in_cvf_loc_store == "CVF_T"

    def test_ode_update_writes_X_T(self) -> None:
        ode: ODEStepUpdate = make_sim().plan[1]  # type: ignore[assignment]
        assert ode.out_x_loc_store == "X_T"

    def test_ops_active_at_all_steps(self) -> None:
        sim = make_sim(num_steps=NUM_STEPS)
        infer: Infer = sim.plan[0]  # type: ignore[assignment]
        expected = list(range(NUM_STEPS))
        assert infer.valid_steps["coords"] == expected

    def test_infer_defaults_to_same_timestep_self_conditioning(self) -> None:
        infer: Infer = make_sim().plan[0]  # type: ignore[assignment]
        assert infer.self_conditioning_mode == "same_timestep"

    def test_infer_threads_previous_step_self_conditioning(self) -> None:
        infer: Infer = make_sim(self_conditioning_mode="previous_step").plan[0]  # type: ignore[assignment]
        assert infer.self_conditioning_mode == "previous_step"


# ---------------------------------------------------------------------------
# observable_output_configs defaults
# ---------------------------------------------------------------------------


class TestODESimulationObservableOutputConfigs:
    def test_X_T_present_by_default(self) -> None:
        assert "X_T" in make_sim().observable_output_configs

    def test_X_T_added_when_absent_in_input(self) -> None:
        sim = make_sim(observable_output_configs={})
        assert "X_T" in sim.observable_output_configs

    def test_custom_config_preserved(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim = make_sim(observable_output_configs={"MY_KEY": cfg})
        assert "MY_KEY" in sim.observable_output_configs

    def test_features_key_registered_on_configs(self) -> None:
        sim = make_sim()
        for config in sim.observable_output_configs.values():
            assert "features" in config.features_keys


# ---------------------------------------------------------------------------
# ensure_observable_output_config_features_keys
# ---------------------------------------------------------------------------


class TestEnsureObservableOutputConfigFeaturesKeys:
    def test_adds_input_features_key_to_configs(self) -> None:
        sim = make_sim()
        # Clear any previously registered keys
        for config in sim.observable_output_configs.values():
            config.features_keys = []
        sim.ensure_observable_output_config_features_keys()
        for config in sim.observable_output_configs.values():
            assert "features" in config.features_keys

    def test_adds_input_features_key_to_intermediate_observables(self) -> None:

        sim = make_sim()
        sim.setup_observable(total_batch_size=2)
        # intermediate_observables is populated after setup_observable; just verify method runs
        sim.ensure_observable_output_config_features_keys()  # should not raise


# ---------------------------------------------------------------------------
# expected_feature_keys
# ---------------------------------------------------------------------------


class TestODESimulationExpectedFeatureKeys:
    def test_input_features_key_required(self) -> None:
        sim = make_sim()
        assert "features" in sim.expected_feature_keys


# ---------------------------------------------------------------------------
# setup_models propagates inference_model_config to ops
# ---------------------------------------------------------------------------


class TestODESimulationSetupModels:
    def test_setup_models_propagates_new_inference_model_config_to_ops(self) -> None:
        sim = make_sim()
        new_cfg = DynoPsiModelConfig(
            loader=HuggingFaceLoaderConfig(repo_id="new/repo", filename="new.pt", revision="0")
        )
        assert sim.plan[0].inference_model_config is not new_cfg  # sanity: old config before

        with mock.patch("dynopsi.model.model_config.DynoPsiModelConfig.to_device"):
            sim.inference_model_config = new_cfg
            sim.setup_models(torch.device("cpu"))

        infer_op: Infer = sim.plan[0]  # type: ignore[assignment]
        assert infer_op.inference_model_config is new_cfg
        init_state: InitializeState = sim.plan_init[1]  # type: ignore[assignment]
        assert init_state.inference_model_config is new_cfg
