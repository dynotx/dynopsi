import typing
from typing import Literal

from pydantic import Field

from dynopsi.data.featurization import Features
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.build_initial_state import InitializeFeatures, InitializeState, InitializeStepType
from dynopsi.simulation.observable import ObservableOutputConfig
from dynopsi.simulation.ops import Infer, ODEStepUpdate, Op
from dynopsi.simulation.simulation import Simulation
from dynopsi.simulation.simulation_pipeline.registry import register_simulation_pipeline
from dynopsi.simulation.time_sampler import TimeSamplerType
from dynopsi.simulation.utils import all_steps_for_modalities


@register_simulation_pipeline()
class ODESimulation(Simulation):
    """Basic ODE simulation of a trajectory from a single model with a single set of conditioning features."""

    type: Literal["ode_simulation"] = "ode_simulation"
    name: str
    inference_model_config: DynoPsiModelConfig
    time_sampler: TimeSamplerType
    clear_states_each_step: bool = True
    validate_before_run: bool = True
    observable_output_configs: dict[str, ObservableOutputConfig] = {
        "X_T": ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": ("pdb", "npz")})
    }

    _input_features_key: str = "features"
    _batch_features_key: str = "batch_features"
    _state_key: str = "X_T"
    _cvf_key: str = "CVF_T"

    # Do not set, these will be overridden
    plan_init: list[InitializeStepType] = Field(default_factory=list)
    plan: list[Op] = Field(default_factory=list)

    def validate_configuration(self, features_input: dict[str, Features]) -> None:
        """Validate the configuration of the simulation pipeline."""
        self._validate(features_input)

    def ensure_observable_output_config_features_keys(self) -> None:
        """Set features_key on configs so save can resolve reference features."""
        for config in self.observable_output_configs.values():
            config.set_features_key(self._input_features_key)

    def model_post_init(self, __context: typing.Any) -> None:
        input_features_key = self._input_features_key
        batch_features_key = self._batch_features_key
        state_key = self._state_key
        cvf_key = self._cvf_key

        num_steps = self.time_sampler.num_steps
        valid_steps = all_steps_for_modalities(["coords"], num_steps)

        self.plan_init = [
            InitializeFeatures(
                name="initialize_features", in_feat_loc_store=input_features_key, out_feat_loc_store=batch_features_key
            ),
            InitializeState(
                name="initialize_gaussian_euclidean_interpolant",
                inference_model_config=self.inference_model_config,
                in_feat_loc_store=batch_features_key,
                out_loc_store=state_key,
            ),
        ]
        self.plan = [
            Infer(
                name="infer_CVF_T",
                inference_model_config=self.inference_model_config,
                in_x_loc_store=state_key,
                in_feat_loc_store=batch_features_key,
                out_loc_store=cvf_key,
                valid_steps=valid_steps,
            ),
            ODEStepUpdate(
                name="X_T_step_update",
                in_x_loc_store=state_key,
                in_cvf_loc_store=cvf_key,
                out_x_loc_store=state_key,
                valid_steps=valid_steps,
            ),
        ]

        # update the observable output in case it is missing X_T
        if "X_T" not in self.observable_output_configs.keys():
            self.observable_output_configs["X_T"] = ObservableOutputConfig(
                state_type="clean_data_coords", outputs={"final": ("pdb", "npz")}
            )
