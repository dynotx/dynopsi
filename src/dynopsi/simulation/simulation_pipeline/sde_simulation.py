from typing import Any, Literal

import torch
from pydantic import Field

from dynopsi.data.featurization.base import Features
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.build_initial_state import (
    InitializeFeatures,
    InitializeState,
    InitializeStepType,
)
from dynopsi.simulation.observable import ObservableOutputConfig
from dynopsi.simulation.ops import (
    GaussianScoreWeightedODEStepUpdate,
    GaussianSDEStepUpdate,
    Infer,
    Op,
)
from dynopsi.simulation.ops.infer import SelfConditioningMode
from dynopsi.simulation.simulation import Simulation
from dynopsi.simulation.simulation_pipeline.registry import register_simulation_pipeline
from dynopsi.simulation.time_sampler import LinearTimeSampler, TimeSamplerType
from dynopsi.simulation.utils import all_steps_for_modalities
from dynopsi.utils import ConstantFunction, InverseParameterFunction, SingleParameterFunctionType


class BaseSDESimulation(Simulation):
    """Shared fields and helpers for SDE-based simulations."""

    inference_model_config: DynoPsiModelConfig
    time_sampler: TimeSamplerType = Field(default_factory=lambda: LinearTimeSampler(num_steps=60))

    # SDE parameters
    diffusion_coefficient_fn: SingleParameterFunctionType = Field(
        default_factory=lambda: InverseParameterFunction(eps=0.01, clamp_max=10.0)
    )
    noise_scale_fn: SingleParameterFunctionType = Field(default_factory=lambda: ConstantFunction(constant=0.3))
    use_legacy_cvf_score_conversion: bool = True
    seed_sde_noise_with_step_num: bool = False

    # ODE parameters
    ode_warmup_score_scale_fn: SingleParameterFunctionType = Field(
        default_factory=lambda: ConstantFunction(constant=1.0)
    )
    ode_warmup_max_t: float = 0.30
    ode_cooldown_score_scale_fn: SingleParameterFunctionType = Field(
        default_factory=lambda: ConstantFunction(constant=1.3)
    )
    ode_cooldown_min_t: float = 0.87
    self_conditioning_mode: SelfConditioningMode = "previous_step"

    # Saving and validation
    save_intermediate_terms: bool = False
    clear_states_each_step: bool = True
    validate_before_run: bool = False
    prior_seed: int | None = None
    observable_output_configs: dict[str, ObservableOutputConfig] = {
        "X_T": ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": ("pdb", "npz")})
    }

    plan_init: list[InitializeStepType] = Field(default_factory=list)
    plan: list[Op] = Field(default_factory=list)

    def declared_categorical_keys(self) -> set[str]:
        return self.inference_model_config.declared_categorical_keys()

    def _compute_time_steps(self) -> tuple[dict, dict, dict, dict]:
        """Return (all_steps, ode_warmup_steps, sde_steps, ode_cooldown_steps)."""
        all_steps = all_steps_for_modalities(["coords"], self.time_sampler.num_steps)
        ts, _ = self.time_sampler.get_full_schedule(device=torch.device("cpu"))
        t = ts["coords"][0, :, 0, 0]
        ode_warmup_steps = {"coords": torch.where(t <= self.ode_warmup_max_t)[0].tolist()}
        sde_steps = {"coords": torch.where((t > self.ode_warmup_max_t) & (t <= self.ode_cooldown_min_t))[0].tolist()}
        ode_cooldown_steps = {"coords": torch.where(t > self.ode_cooldown_min_t)[0].tolist()}
        return all_steps, ode_warmup_steps, sde_steps, ode_cooldown_steps

    def _build_step_update_ops(
        self,
        cvf_key: str,
        feat_key: str,
        ode_warmup_steps: dict,
        sde_steps: dict,
        ode_cooldown_steps: dict,
    ) -> list[Op]:
        """Return the three step-update ops (ODE warmup, SDE, ODE cooldown)."""
        state_key = "X_T"
        return [
            # Score-weighted ODE update for t < ode_warmup_max_t
            GaussianScoreWeightedODEStepUpdate(
                name="ODE_warmup_update",
                valid_steps=ode_warmup_steps,
                in_feat_loc_store=feat_key,
                in_x_loc_store=state_key,
                in_cvf_loc_store=cvf_key,
                out_x_loc_store=state_key,
                score_scale_fn=self.ode_warmup_score_scale_fn,
            ),
            # SDE update for ode_warmup_max_t <= t <= ode_cooldown_min_t
            GaussianSDEStepUpdate(
                name="SDE_update",
                valid_steps=sde_steps,
                in_feat_loc_store=feat_key,
                in_x_loc_store=state_key,
                in_cvf_loc_store=cvf_key,
                diffusion_coefficient_fn=self.diffusion_coefficient_fn,
                noise_scale_fn=self.noise_scale_fn,
                out_x_loc_store=state_key,
                prior_stds=self.inference_model_config.prior_stds,
                use_legacy_cvf_score_conversion=self.use_legacy_cvf_score_conversion,
                save_intermediate_terms=self.save_intermediate_terms,
                drift_terms_loc_store="SDE_drift_terms",
                score_terms_loc_store="SDE_score_terms",
                noise_terms_loc_store="SDE_noise_terms",
                manually_seed_step=self.seed_sde_noise_with_step_num,
            ),
            # Score-weighted ODE update for t > ode_cooldown_min_t
            GaussianScoreWeightedODEStepUpdate(
                name="ODE_cooldown_update",
                valid_steps=ode_cooldown_steps,
                in_feat_loc_store=feat_key,
                in_x_loc_store=state_key,
                in_cvf_loc_store=cvf_key,
                out_x_loc_store=state_key,
                score_scale_fn=self.ode_cooldown_score_scale_fn,
            ),
        ]

    def _setup_observable_output(self, *features_keys: str) -> None:
        """Ensure X_T is in observable_output_configs and register features keys.

        Each key in ``features_keys`` is added to every config that does not
        already contain it, producing one set of output files per key.
        """
        if "X_T" not in self.observable_output_configs:
            self.observable_output_configs["X_T"] = ObservableOutputConfig(
                state_type="clean_data_coords", outputs={"final": ("pdb", "npz")}
            )
        for config in self.observable_output_configs.values():
            for key in features_keys:
                config.add_features_key(key)

    def model_post_init(self, __context: Any) -> None:
        """Run before subclass ``model_post_init`` builds ``plan`` / ``plan_init``."""
        super().model_post_init(__context)
        if self.ode_warmup_max_t >= self.ode_cooldown_min_t:
            raise ValueError(
                "ode_warmup_max_t must be strictly less than ode_cooldown_min_t "
                f"(got ode_warmup_max_t={self.ode_warmup_max_t}, "
                f"ode_cooldown_min_t={self.ode_cooldown_min_t})."
            )


@register_simulation_pipeline()
class SDESimulation(BaseSDESimulation):
    """SDE simulation of a trajectory from a single model with a single set of conditioning features."""

    type: Literal["sde_simulation"] = "sde_simulation"

    _input_features_key: str = "features"
    _batch_features_key: str = "batch_features"
    _state_key: str = "X_T"
    _cvf_key: str = "CVF_T"

    def validate_configuration(self, features_input: dict[str, Features]) -> None:
        """Validate the configuration of the simulation pipeline."""
        self._validate(features_input)

    def ensure_observable_output_config_features_keys(self) -> None:
        """Set features_key on configs so save can resolve reference features."""
        for config in self.observable_output_configs.values():
            config.add_features_key(self._input_features_key)
        for config in self.intermediate_observables.values():
            config.add_features_key(self._input_features_key)

    def _rebuild_plan(self) -> None:
        input_features_key = self._input_features_key
        batch_features_key = self._batch_features_key
        state_key = self._state_key
        cvf_key = self._cvf_key

        all_steps, ode_warmup_steps, sde_steps, ode_cooldown_steps = self._compute_time_steps()

        self.plan_init = [
            InitializeFeatures(
                name="initialize_features",
                in_feat_loc_store=input_features_key,
                out_feat_loc_store=batch_features_key,
            ),
            InitializeState(
                name="initialize_gaussian_euclidean_interpolant",
                inference_model_config=self.inference_model_config,
                in_feat_loc_store=batch_features_key,
                out_loc_store=state_key,
                seed=self.prior_seed,
            ),
        ]
        self.plan = [
            Infer(
                name="infer_CVF_T",
                inference_model_config=self.inference_model_config,
                in_x_loc_store=state_key,
                in_feat_loc_store=batch_features_key,
                out_loc_store=cvf_key,
                valid_steps=all_steps,
                self_conditioning_mode=self.self_conditioning_mode,
            ),
            *self._build_step_update_ops(
                cvf_key,
                batch_features_key,
                ode_warmup_steps,
                sde_steps,
                ode_cooldown_steps,
            ),
        ]
        self._setup_observable_output(input_features_key)
        super()._rebuild_plan()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self._rebuild_plan()
