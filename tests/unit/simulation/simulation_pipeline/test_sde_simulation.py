from unittest import mock

import pytest
import torch

from dynopsi.design_pipeline_parser import DesignLanguageParser
from dynopsi.model.model_config import DynoPsiModelConfig, HuggingFaceLoaderConfig
from dynopsi.simulation.build_initial_state import InitializeFeatures, InitializeState
from dynopsi.simulation.observable import ObservableOutputConfig
from dynopsi.simulation.ops import (
    GaussianScoreWeightedODEStepUpdate,
    GaussianSDEStepUpdate,
    Infer,
)
from dynopsi.simulation.simulation_pipeline.registry import SIMULATION_PIPELINE_REGISTRY
from dynopsi.simulation.simulation_pipeline.sde_simulation import (
    BaseSDESimulation,
    SDESimulation,
)
from dynopsi.simulation.time_sampler import LinearTimeSampler
from dynopsi.utils.function_configs import ConstantFunction, InverseParameterFunction

NUM_STEPS = 11  # t=[0, 0.09, ..., 0.91] → warmup:[0,1], sde:[2..9], cooldown:[10]

_CONSTANT_FN = ConstantFunction(constant=0.5)

# A lightweight DynoPsiModelConfig — no model is loaded unless .dynopsi_model is accessed.
_FAKE_CFG = DynoPsiModelConfig(loader=HuggingFaceLoaderConfig(repo_id="fake/repo", filename="fake.pt", revision="0"))
_FAKE_PRIOR_STDS = {"coords": 1.0}


@pytest.fixture(autouse=True)
def _patch_prior_stds():
    """Prevent DynoPsiModelConfig.prior_stds from loading a real checkpoint.
    Applied to every test in this module automatically.
    """
    with mock.patch(
        "dynopsi.model.model_config.DynoPsiModelConfig.prior_stds",
        new_callable=mock.PropertyMock,
        return_value=_FAKE_PRIOR_STDS,
    ):
        yield


def _base_kwargs(num_steps: int = NUM_STEPS, **overrides):
    kwargs = dict(
        inference_model_config=_FAKE_CFG,
        time_sampler=LinearTimeSampler(num_steps=num_steps, modalities=["coords"]),
        diffusion_coefficient_fn=_CONSTANT_FN,
        noise_scale_fn=_CONSTANT_FN,
        ode_warmup_score_scale_fn=_CONSTANT_FN,
        ode_cooldown_score_scale_fn=_CONSTANT_FN,
    )
    kwargs.update(overrides)
    return kwargs


def make_sde_sim(num_steps: int = NUM_STEPS, observable_output_configs=None, **overrides) -> SDESimulation:
    """Construct an SDESimulation with a lightweight inference_model_config."""
    kwargs = _base_kwargs(num_steps=num_steps, **overrides)
    if observable_output_configs is not None:
        kwargs["observable_output_configs"] = observable_output_configs
    return SDESimulation(name="test_sde", **kwargs)


def make_base_construct(**overrides) -> BaseSDESimulation:
    """model_construct a BaseSDESimulation — bypasses model_post_init."""
    kwargs = dict(
        inference_model_config=_FAKE_CFG,
        time_sampler=LinearTimeSampler(num_steps=NUM_STEPS, modalities=["coords"]),
        diffusion_coefficient_fn=_CONSTANT_FN,
        noise_scale_fn=_CONSTANT_FN,
        ode_warmup_score_scale_fn=_CONSTANT_FN,
        ode_cooldown_score_scale_fn=_CONSTANT_FN,
        ode_warmup_max_t=0.1,
        ode_cooldown_min_t=0.9,
    )
    kwargs.update(overrides)
    return SDESimulation.model_construct(**kwargs)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSDESimulationRegistration:
    def test_sde_simulation_registered(self) -> None:
        assert "sde_simulation" in SIMULATION_PIPELINE_REGISTRY

    def test_type_string(self) -> None:
        assert make_sde_sim().type == "sde_simulation"


# ---------------------------------------------------------------------------
# BaseSDESimulation.model_post_init validation
# ---------------------------------------------------------------------------


class TestBaseSDESimulationModelPostInit:
    def test_raises_when_warmup_ge_cooldown(self) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            SDESimulation(name="bad", ode_warmup_max_t=0.9, ode_cooldown_min_t=0.5, **_base_kwargs())

    def test_raises_when_warmup_equals_cooldown(self) -> None:
        with pytest.raises(ValueError):
            SDESimulation(name="bad", ode_warmup_max_t=0.5, ode_cooldown_min_t=0.5, **_base_kwargs())


# ---------------------------------------------------------------------------
# Default SDE schedule
# ---------------------------------------------------------------------------


class TestSDESimulationDefaultSchedule:
    def test_omitted_knobs_match_release_schedule(self) -> None:
        sim = SDESimulation(name="defaults", inference_model_config=_FAKE_CFG)

        assert isinstance(sim.time_sampler, LinearTimeSampler)
        assert sim.time_sampler.num_steps == 60
        assert sim.time_sampler.start_time == 0.0
        assert sim.time_sampler.end_time == 1.0

        assert isinstance(sim.diffusion_coefficient_fn, InverseParameterFunction)
        assert sim.diffusion_coefficient_fn.eps == 0.01
        assert sim.diffusion_coefficient_fn.clamp_max == 10.0

        assert isinstance(sim.noise_scale_fn, ConstantFunction)
        assert sim.noise_scale_fn.constant == 0.3

        assert isinstance(sim.ode_warmup_score_scale_fn, ConstantFunction)
        assert sim.ode_warmup_score_scale_fn.constant == 1.0
        assert sim.ode_warmup_max_t == 0.30

        assert isinstance(sim.ode_cooldown_score_scale_fn, ConstantFunction)
        assert sim.ode_cooldown_score_scale_fn.constant == 1.3
        assert sim.ode_cooldown_min_t == 0.87

        assert sim.self_conditioning_mode == "previous_step"
        assert sim.use_legacy_cvf_score_conversion is True

    def test_parse_design_dict_omitted_knobs_match_release_schedule(self) -> None:
        design_dict = {
            "featurization_pipelines": {"feat": {}},
            "output_specifications": {"out": {}},
            "simulation_pipelines": {"sim": {"type": "sde_simulation"}},
            "design_pipelines": {
                "design": {
                    "featurization_pipeline": "feat",
                    "simulation_pipeline": "sim",
                    "output_specification": "out",
                }
            },
        }
        with (
            mock.patch(
                "dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "dynopsi.design_pipeline_parser.OutputSpecification.model_validate",
                return_value=mock.MagicMock(),
            ),
            mock.patch("dynopsi.design_pipeline_parser.DesignPipeline") as mock_design_pipeline,
        ):
            DesignLanguageParser().parse_design_dict(design_dict, inference_model_config=_FAKE_CFG)

        sim = mock_design_pipeline.call_args.kwargs["simulation_pipeline"]
        assert isinstance(sim, SDESimulation)

        assert isinstance(sim.time_sampler, LinearTimeSampler)
        assert sim.time_sampler.num_steps == 60
        assert sim.time_sampler.start_time == 0.0
        assert sim.time_sampler.end_time == 1.0

        assert isinstance(sim.diffusion_coefficient_fn, InverseParameterFunction)
        assert sim.diffusion_coefficient_fn.eps == 0.01
        assert sim.diffusion_coefficient_fn.clamp_max == 10.0

        assert isinstance(sim.noise_scale_fn, ConstantFunction)
        assert sim.noise_scale_fn.constant == 0.3

        assert isinstance(sim.ode_warmup_score_scale_fn, ConstantFunction)
        assert sim.ode_warmup_score_scale_fn.constant == 1.0
        assert sim.ode_warmup_max_t == 0.30

        assert isinstance(sim.ode_cooldown_score_scale_fn, ConstantFunction)
        assert sim.ode_cooldown_score_scale_fn.constant == 1.3
        assert sim.ode_cooldown_min_t == 0.87

        assert sim.self_conditioning_mode == "previous_step"
        assert sim.use_legacy_cvf_score_conversion is True

    def test_set_fields_override_only_themselves(self) -> None:
        sim = SDESimulation(
            name="override",
            inference_model_config=_FAKE_CFG,
            ode_warmup_max_t=0.05,
            noise_scale_fn=ConstantFunction(constant=0.5),
        )

        assert sim.ode_warmup_max_t == 0.05
        assert isinstance(sim.noise_scale_fn, ConstantFunction)
        assert sim.noise_scale_fn.constant == 0.5

        assert isinstance(sim.diffusion_coefficient_fn, InverseParameterFunction)
        assert sim.diffusion_coefficient_fn.eps == 0.01
        assert sim.ode_cooldown_min_t == 0.87
        assert isinstance(sim.ode_warmup_score_scale_fn, ConstantFunction)
        assert sim.ode_warmup_score_scale_fn.constant == 1.0


# ---------------------------------------------------------------------------
# plan_init structure
# ---------------------------------------------------------------------------


class TestSDESimulationPlanInit:
    def test_plan_init_has_two_steps(self) -> None:
        assert len(make_sde_sim().plan_init) == 2

    def test_plan_init_first_step_is_initialize_features(self) -> None:
        assert isinstance(make_sde_sim().plan_init[0], InitializeFeatures)

    def test_plan_init_second_step_is_initialize_state(self) -> None:
        assert isinstance(make_sde_sim().plan_init[1], InitializeState)

    def test_plan_init_features_keys_wired_correctly(self) -> None:
        sim = make_sde_sim()
        init_feat: InitializeFeatures = sim.plan_init[0]  # type: ignore[assignment]
        assert init_feat.in_feat_loc_store == "features"
        assert init_feat.out_feat_loc_store == "batch_features"

    def test_plan_init_state_reads_batch_features(self) -> None:
        init_state: InitializeState = make_sde_sim().plan_init[1]  # type: ignore[assignment]
        assert init_state.in_feat_loc_store == "batch_features"

    def test_plan_init_state_writes_X_T(self) -> None:
        init_state: InitializeState = make_sde_sim().plan_init[1]  # type: ignore[assignment]
        assert init_state.out_loc_store == "X_T"


# ---------------------------------------------------------------------------
# plan structure
# ---------------------------------------------------------------------------


class TestSDESimulationPlan:
    def test_plan_has_four_ops(self) -> None:
        # Infer + ODE warmup + SDE + ODE cooldown
        assert len(make_sde_sim().plan) == 4

    def test_plan_first_op_is_infer(self) -> None:
        assert isinstance(make_sde_sim().plan[0], Infer)

    def test_plan_step_update_ops_correct_types(self) -> None:
        plan = make_sde_sim().plan
        assert isinstance(plan[1], GaussianScoreWeightedODEStepUpdate)
        assert isinstance(plan[2], GaussianSDEStepUpdate)
        assert isinstance(plan[3], GaussianScoreWeightedODEStepUpdate)

    def test_infer_reads_X_T_and_batch_features(self) -> None:
        infer: Infer = make_sde_sim().plan[0]  # type: ignore[assignment]
        assert infer.in_x_loc_store == "X_T"
        assert infer.in_feat_loc_store == "batch_features"

    def test_infer_writes_CVF_T(self) -> None:
        infer: Infer = make_sde_sim().plan[0]  # type: ignore[assignment]
        assert infer.out_loc_store == "CVF_T"

    def test_ode_warmup_reads_X_T_and_CVF_T(self) -> None:
        warmup: GaussianScoreWeightedODEStepUpdate = make_sde_sim().plan[1]  # type: ignore[assignment]
        assert warmup.in_x_loc_store == "X_T"
        assert warmup.in_cvf_loc_store == "CVF_T"

    def test_sde_reads_X_T_and_CVF_T(self) -> None:
        sde: GaussianSDEStepUpdate = make_sde_sim().plan[2]  # type: ignore[assignment]
        assert sde.in_x_loc_store == "X_T"
        assert sde.in_cvf_loc_store == "CVF_T"

    def test_ode_cooldown_reads_X_T_and_CVF_T(self) -> None:
        cooldown: GaussianScoreWeightedODEStepUpdate = make_sde_sim().plan[3]  # type: ignore[assignment]
        assert cooldown.in_x_loc_store == "X_T"
        assert cooldown.in_cvf_loc_store == "CVF_T"

    def test_all_step_update_ops_write_X_T(self) -> None:
        for op in make_sde_sim().plan[1:]:
            assert op.out_x_loc_store == "X_T"

    def test_infer_active_at_all_steps(self) -> None:
        sim = make_sde_sim(num_steps=NUM_STEPS)
        infer: Infer = sim.plan[0]  # type: ignore[assignment]
        assert infer.valid_steps["coords"] == list(range(NUM_STEPS))

    def test_infer_defaults_to_previous_step_self_conditioning(self) -> None:
        infer: Infer = make_sde_sim().plan[0]  # type: ignore[assignment]
        assert infer.self_conditioning_mode == "previous_step"

    def test_infer_threads_same_timestep_self_conditioning(self) -> None:
        infer: Infer = make_sde_sim(self_conditioning_mode="same_timestep").plan[0]  # type: ignore[assignment]
        assert infer.self_conditioning_mode == "same_timestep"

    def test_infer_threads_previous_step_self_conditioning(self) -> None:
        infer: Infer = make_sde_sim(self_conditioning_mode="previous_step").plan[0]  # type: ignore[assignment]
        assert infer.self_conditioning_mode == "previous_step"


# ---------------------------------------------------------------------------
# BaseSDESimulation._compute_time_steps
# ---------------------------------------------------------------------------


class TestComputeTimeSteps:
    def _compute(self, sim: BaseSDESimulation):
        return sim._compute_time_steps()

    def test_all_steps_covers_full_range(self) -> None:
        sim = make_base_construct()
        all_steps, _, _, _ = self._compute(sim)
        assert all_steps["coords"] == list(range(NUM_STEPS))

    def test_partitions_are_disjoint_and_complete(self) -> None:
        sim = make_base_construct()
        _, warmup, sde, cooldown = self._compute(sim)
        combined = set(warmup["coords"]) | set(sde["coords"]) | set(cooldown["coords"])
        assert combined == set(range(NUM_STEPS))
        assert len(warmup["coords"]) + len(sde["coords"]) + len(cooldown["coords"]) == NUM_STEPS

    def test_warmup_steps_precede_sde_steps(self) -> None:
        sim = make_base_construct()
        _, warmup, sde, _ = self._compute(sim)
        if warmup["coords"] and sde["coords"]:
            assert max(warmup["coords"]) < min(sde["coords"])

    def test_sde_steps_precede_cooldown_steps(self) -> None:
        sim = make_base_construct()
        _, _, sde, cooldown = self._compute(sim)
        if sde["coords"] and cooldown["coords"]:
            assert max(sde["coords"]) < min(cooldown["coords"])

    def test_custom_thresholds_adjust_partition(self) -> None:
        sim = make_base_construct(
            time_sampler=LinearTimeSampler(num_steps=10, modalities=["coords"]),
            ode_warmup_max_t=0.3,
            ode_cooldown_min_t=0.7,
        )
        _, warmup, sde, cooldown = self._compute(sim)
        assert len(warmup["coords"]) > 0
        assert len(sde["coords"]) > 0
        assert len(cooldown["coords"]) > 0


# ---------------------------------------------------------------------------
# BaseSDESimulation._build_step_update_ops
# ---------------------------------------------------------------------------


class TestBuildStepUpdateOps:
    def _get_ops(self, sim: BaseSDESimulation):
        _, warmup, sde, cooldown = sim._compute_time_steps()
        return sim._build_step_update_ops("CVF_T", "feat", warmup, sde, cooldown)

    def test_returns_three_ops(self) -> None:
        sim = make_base_construct()
        assert len(self._get_ops(sim)) == 3

    def test_first_op_is_ode_warmup(self) -> None:
        sim = make_base_construct()
        assert isinstance(self._get_ops(sim)[0], GaussianScoreWeightedODEStepUpdate)

    def test_second_op_is_sde(self) -> None:
        sim = make_base_construct()
        assert isinstance(self._get_ops(sim)[1], GaussianSDEStepUpdate)

    def test_third_op_is_ode_cooldown(self) -> None:
        sim = make_base_construct()
        assert isinstance(self._get_ops(sim)[2], GaussianScoreWeightedODEStepUpdate)

    def test_warmup_op_valid_steps_match(self) -> None:
        sim = make_base_construct()
        _, warmup, sde, cooldown = sim._compute_time_steps()
        ops = sim._build_step_update_ops("CVF_T", "feat", warmup, sde, cooldown)
        assert ops[0].valid_steps == warmup

    def test_sde_op_valid_steps_match(self) -> None:
        sim = make_base_construct()
        _, warmup, sde, cooldown = sim._compute_time_steps()
        ops = sim._build_step_update_ops("CVF_T", "feat", warmup, sde, cooldown)
        assert ops[1].valid_steps == sde

    def test_cooldown_op_valid_steps_match(self) -> None:
        sim = make_base_construct()
        _, warmup, sde, cooldown = sim._compute_time_steps()
        ops = sim._build_step_update_ops("CVF_T", "feat", warmup, sde, cooldown)
        assert ops[2].valid_steps == cooldown

    def test_all_ops_write_to_X_T(self) -> None:
        sim = make_base_construct()
        for op in self._get_ops(sim):
            assert op.out_x_loc_store == "X_T"


# ---------------------------------------------------------------------------
# BaseSDESimulation._setup_observable_output
# ---------------------------------------------------------------------------


class TestSetupObservableOutput:
    def test_X_T_added_when_absent(self) -> None:
        sim = make_base_construct()
        sim.observable_output_configs = {}
        sim._setup_observable_output("features")
        assert "X_T" in sim.observable_output_configs

    def test_X_T_not_duplicated_when_present(self) -> None:
        sim = make_base_construct()
        existing = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim.observable_output_configs = {"X_T": existing}
        sim._setup_observable_output("features")
        assert sim.observable_output_configs["X_T"] is existing

    def test_features_key_registered_on_all_configs(self) -> None:
        sim = make_base_construct()
        extra = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim.observable_output_configs = {"X_T": extra}
        sim._setup_observable_output("my_features")
        for config in sim.observable_output_configs.values():
            assert "my_features" in config.features_keys

    def test_multiple_feature_keys_all_registered(self) -> None:
        sim = make_base_construct()
        extra = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim.observable_output_configs = {"X_T": extra}
        sim._setup_observable_output("key_a", "key_b")
        for config in sim.observable_output_configs.values():
            assert "key_a" in config.features_keys
            assert "key_b" in config.features_keys


# ---------------------------------------------------------------------------
# observable_output_configs defaults
# ---------------------------------------------------------------------------


class TestSDESimulationObservableOutputConfigs:
    def test_X_T_present_by_default(self) -> None:
        assert "X_T" in make_sde_sim().observable_output_configs

    def test_X_T_added_when_absent_in_input(self) -> None:
        sim = make_sde_sim(observable_output_configs={})
        assert "X_T" in sim.observable_output_configs

    def test_custom_config_preserved(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim = make_sde_sim(observable_output_configs={"MY_KEY": cfg})
        assert "MY_KEY" in sim.observable_output_configs

    def test_features_key_registered_on_configs(self) -> None:
        sim = make_sde_sim()
        for config in sim.observable_output_configs.values():
            assert "features" in config.features_keys


# ---------------------------------------------------------------------------
# ensure_observable_output_config_features_keys
# ---------------------------------------------------------------------------


class TestEnsureObservableOutputConfigFeaturesKeys:
    def test_adds_input_features_key_to_configs(self) -> None:
        sim = make_sde_sim()
        for config in sim.observable_output_configs.values():
            config.features_keys = []
        sim.ensure_observable_output_config_features_keys()
        for config in sim.observable_output_configs.values():
            assert "features" in config.features_keys

    def test_adds_input_features_key_to_intermediate_observables(self) -> None:
        sim = make_sde_sim()
        sim.setup_observable(total_batch_size=2)
        sim.ensure_observable_output_config_features_keys()  # should not raise


# ---------------------------------------------------------------------------
# expected_feature_keys
# ---------------------------------------------------------------------------


class TestSDESimulationExpectedFeatureKeys:
    def test_input_features_key_required(self) -> None:
        assert "features" in make_sde_sim().expected_feature_keys


# ---------------------------------------------------------------------------
# setup_models propagates inference_model_config to ops
# ---------------------------------------------------------------------------


class TestSDESimulationSetupModels:
    def test_setup_models_propagates_new_inference_model_config_to_ops(self) -> None:
        sim = make_sde_sim()
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

    def test_setup_models_updates_sde_op_prior_stds_from_new_config(self) -> None:
        sim = make_sde_sim()
        new_prior_stds = {"coords": 99.0}
        new_cfg = DynoPsiModelConfig(
            loader=HuggingFaceLoaderConfig(repo_id="new/repo", filename="new.pt", revision="0")
        )

        with (
            mock.patch("dynopsi.model.model_config.DynoPsiModelConfig.to_device"),
            mock.patch(
                "dynopsi.model.model_config.DynoPsiModelConfig.prior_stds",
                new_callable=mock.PropertyMock,
                return_value=new_prior_stds,
            ),
        ):
            sim.inference_model_config = new_cfg
            sim.setup_models(torch.device("cpu"))

        sde_op: GaussianSDEStepUpdate = sim.plan[2]  # type: ignore[assignment]
        assert sde_op.prior_stds == new_prior_stds


# ---------------------------------------------------------------------------
