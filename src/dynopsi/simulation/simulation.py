import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any, Self, cast

import torch
from pydantic import Field, field_serializer, field_validator, model_validator
from tqdm import tqdm  # type: ignore[import-untyped]

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.sample_centers import (
    SAMPLE_CENTERS_OF_MASS_METADATA_KEY,
    narrow_per_sample_features,
)
from dynopsi.simulation.build_initial_state import InitializeStepType
from dynopsi.simulation.observable import FileOutputConfig, Observable, ObservableOutputConfig
from dynopsi.simulation.ops import Infer, Op, get_op_from_config
from dynopsi.simulation.store import Store
from dynopsi.simulation.time_sampler import TimeSamplerType
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)


def _require_unique_previous_step_infer_names(plan: list[Op]) -> None:
    """``previous_sc`` is keyed by Infer name, so a shared name overwrites the earlier stash."""
    seen: set[str] = set()
    for op in plan:
        if not isinstance(op, Infer) or op.self_conditioning_mode != "previous_step":
            continue
        if op.name in seen:
            raise ValueError(f"previous_step Infer ops must have distinct names; {op.name!r} is used more than once")
        seen.add(op.name)


class Simulation(StrictConfig):
    """
    Run a plan of operations over time.

    Executes each Op in a plan at each step from the time sampler.
    State advances via the store (e.g. CVFStepUpdate writes to step_num + 1).

    Batching is inferred from tensor shape: features and state tensors typically
    have a leading batch dimension (e.g. [batch_size, seq_len, ...]).
    run takes batch_size and loops over batches. run_batch is the step loop:
    it advances one batch tensor and does not index samples inside the batch.
    Each op receives and writes that full batch tensor.

    Attributes
    ----------
    name: str
        Name of the simulation, used for naming outputs automatically.
    plan_init: list[InitializeStepType]
        Ordered list of operations to run at step 0.
    plan : list[Op]
        Ordered list of operations to run at each step.
    time_sampler : TimeSampler
        Provides get_steps() and get(step_num) -> (t, dt).
    clear_states_each_step : bool
        If True, call store.clear_states(keep_step_num=step_num + 1) at the end of each step
        to trim working state (keeps only the advanced step for non-save_states
        keys). Default True to avoid unbounded memory on long runs.
    validate_before_run : bool
        If True, run() calls _validate() before the step loop to check that
        the store has required state at step 0 and required features. Default True.
    observable_keys : list[tuple[str, Literal["any", "final"]]]
        State keys to record. When non-empty, run() creates an Observable and
        attaches it to the store; recording is then via observable.keys_to_record.
    skip_distogram_head : bool
        If True, setup_models() removes distogram_head from loaded generate models.
    compile_denoiser : bool
        If True, setup_models() compiles the denoiser with torch.compile on CUDA.
        CPU generate does not compile.
    tf32_matmul : bool
        If True, setup_models() sets float32 matmul precision to ``high`` on CUDA.
        Independent of ``compile_denoiser``. This is process-global.

    NOTE: Subclasses must define a type field as a Literal["subclass_type"] = "subclass_type"
    this is used for serialization/deserialization.
    """

    name: str
    # Checkpoint short name the parser resolved into ``inference_model_config``.
    # Recorded so ``resolved_design_config.json`` names the requested weights.
    checkpoint: str | None = None

    plan_init: list[InitializeStepType]
    plan: list[Op]
    time_sampler: TimeSamplerType
    clear_states_each_step: bool = True
    validate_before_run: bool = True

    observable_output_configs: dict[str, ObservableOutputConfig] = Field(default_factory=dict)
    intermediate_observables: dict[str, ObservableOutputConfig] = Field(default_factory=dict)
    observable: Observable = Field(default_factory=Observable)
    skip_distogram_head: bool = True
    compile_denoiser: bool = True
    tf32_matmul: bool = True

    @property
    def expected_feature_keys(self) -> set[str]:
        """Get the set of feature keys that are expected by the plan."""
        feature_keys = set()
        for init_step in self.plan_init:
            feature_keys.update(init_step.input_feature_keys())
        for op in self.plan:
            feature_keys.update(op.input_feature_keys())
        return feature_keys

    def declared_categorical_keys(self) -> set[str]:
        """Return the registered keys this simulation's checkpoint declares. The base returns an empty set."""
        return set()

    @field_validator("plan", mode="before")
    @classmethod
    def validate_plan(cls, plan: list[Op] | dict[str, Any]) -> list[Op]:
        result: list[Op] = []
        for op_cfg in plan:
            if isinstance(op_cfg, dict):
                op = get_op_from_config(op_cfg)
            else:
                op = cast(Op, op_cfg)
            result.append(op)
        _require_unique_previous_step_infer_names(result)
        return result

    @model_validator(mode="after")
    def get_intermediate_observables(self) -> Self:
        """Get any intermediate values if defined by the ops"""
        all_intermediate_keys: set[str] = set()
        new_observables: dict[str, ObservableOutputConfig] = {}
        for op in self.plan_init + self.plan:
            intermediate_keys = op.intermediate_observable_keys()
            if set(intermediate_keys) & all_intermediate_keys != set():
                raise ValueError("Cannot have overlapping intermediate keys")
            if set(intermediate_keys) & set(self.observable_output_configs.keys()) != set():
                raise ValueError("Intermediate keys cannot overlap with other state keys.")
            all_intermediate_keys.update(set(intermediate_keys))
            for key in intermediate_keys:
                new_observables[key] = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        self.intermediate_observables = new_observables
        return self

    @field_serializer("plan")
    def serialize_plan(self, plan: list[Op]) -> list[dict[str, Any]]:
        return [op.model_dump(serialize_as_any=True) for op in plan]

    def validate_configuration(self, features_input: dict[str, Features]) -> None:
        """Validate the configuration of the simulation pipeline."""
        raise NotImplementedError("validate_configuration is not implemented for the base Simulation class.")

    def ensure_observable_output_config_features_keys(self) -> None:
        """Set features_key on observable_output_configs that have None. Override in subclasses."""
        pass

    def setup_observable(self, total_batch_size: int) -> None:
        """Set up the observables for total_batch_size."""
        self.ensure_observable_output_config_features_keys()

        self.observable = Observable(
            total_batch_size=total_batch_size,
            observable_output_configs=self.observable_output_configs | self.intermediate_observables,
        )

    def _rebuild_plan(self) -> None:
        """Check plan invariants after a subclass assigns ``plan``.

        Subclasses that override this must call ``super()._rebuild_plan()`` after
        the plan is assigned. Direct ``plan`` assignment does not re-run field
        validators.
        """
        _require_unique_previous_step_infer_names(self.plan)

    def setup_models(self, device: torch.device) -> None:
        self._rebuild_plan()
        for step in self.plan_init + self.plan:
            for model_config in step.model_configs():
                model_config.to_device(device)
        self._apply_generate_defaults(use_cuda=device.type == "cuda")

    def _apply_generate_defaults(self, use_cuda: bool) -> None:
        if use_cuda and self.tf32_matmul:
            torch.set_float32_matmul_precision("high")
        seen: set[int] = set()
        for step in self.plan_init + self.plan:
            for model_config in step.model_configs():
                model = model_config.loaded_model
                if model is None:
                    continue
                mid = id(model)
                if mid in seen:
                    continue
                seen.add(mid)
                if self.skip_distogram_head:
                    model.skip_distogram_head()
                if self.compile_denoiser and use_cuda:
                    model.compile_denoiser()

    def run_batch(
        self,
        features_input: dict[str, Features],
        batch_slice: slice,
        device: torch.device,
        show_progress: bool = False,
        sample_offset: int = 0,
    ) -> None:
        """
        Execute the plan over all steps, no batch splitting. Outputs specified by observable_output_configs
        are recorded in CPU memory.

        Specifically:
        1. Initialize the store with the features input.
        2. Runs the plan initialization steps a single time.
        3. Runs the plan steps for each step returned by the time sampler, clearing
        the state from GPU memory after each step when clear_states_each_step is True. Depending on
        the observable_output configuration, intermediate steps may be recorded in CPU memory.
        4. Records the final timestep for any observable outputs marked as "final".

        Parameters
        ----------
        features_input : dict[str, Features]
            The input features to the simulation.
        batch_slice : slice
            The slice of the batch to run. This specifies where to store the output in the observable.
        device : torch.device
            The device to run the simulation on.
        show_progress : bool, optional
            If True, wrap the step loop with tqdm. Default False.
        sample_offset : int, optional
            Global sample index of observable slot 0 in this disk-write chunk.
            Narrows per-sample feature metadata and offsets the prior RNG.
        """
        time_schedule = self.time_sampler.get_full_schedule(device)

        sample_slice = slice(sample_offset + batch_slice.start, sample_offset + batch_slice.stop)
        narrowed_features = {
            key: narrow_per_sample_features(feat, sample_slice) for key, feat in features_input.items()
        }

        store = Store(
            features=deepcopy(narrowed_features),
            observable=self.observable,
            batch_slice=batch_slice,
            sample_offset=sample_offset,
            time_schedule=time_schedule,
        )

        store.to_device(device)

        for init_step in self.plan_init:
            init_step.run(store)

        steps = list(self.time_sampler.get_steps())
        step_iter = tqdm(steps, desc="steps", unit="step") if show_progress else steps
        for step_num in step_iter:
            for op in self.plan:
                op.run(store, step_num)
            if self.clear_states_each_step:
                store.clear_states(keep_step_num=step_num + 1)
        for key in self.observable.keys_to_record_final_timestep:
            store.record_final_timestep(key, step_num + 1)

    def _validate(self, features_input: dict[str, Features]) -> None:
        """Check to make sure the simulation will run without errors. Checks include:
        - Features input has all the expected feature keys.
        - There are not state key collisions.
        - Desired observable outputs are compatible with the state keys.

        Raises
        ------
        KeyError
            If the input features are missing expected feature keys.
            If there are state key collisions (e.g. two ops write to the same state key at the same step).
            If the desired observable outputs are not compatible with the states created by the plan.
        """
        state_keys: set[str] = set()
        feature_keys: set[str] = set(features_input.keys())
        output_state_keys_by_step: dict[int, set[str]] = defaultdict(set)

        for init_step in self.plan_init:
            init_step_input_feature_keys = init_step.input_feature_keys()
            if len(init_step_input_feature_keys - feature_keys) > 0:
                raise KeyError(
                    f"Features input missing keys. Init step expects: {init_step_input_feature_keys}, but features input has: {feature_keys}"
                    f"Please ensure that the features input has all the expected feature keys."
                )
            init_step_output_feature_keys = init_step.output_feature_keys()
            if len(init_step_output_feature_keys & feature_keys) > 0:
                raise KeyError(f"Feature key collision: {init_step_output_feature_keys & feature_keys}")
            feature_keys.update(init_step_output_feature_keys)

            init_step_input_state_keys = init_step.input_state_keys()
            if len(init_step_input_state_keys - state_keys) > 0:
                raise KeyError(f"State key not found: {init_step_input_state_keys - state_keys}")

            step_state_keys_out = init_step.output_state_keys(0)
            if len(state_keys & step_state_keys_out) > 0:
                raise KeyError(f"State key collision: {state_keys & step_state_keys_out}")
            state_keys.update(step_state_keys_out)
            output_state_keys_by_step[0].update(step_state_keys_out)

        if len(self.expected_feature_keys - feature_keys) > 0:
            raise KeyError(
                f"Features input missing expected feature keys for plan to run. "
                f"The following keys were input or created by the plan initialization step: {feature_keys}, "
                f"but the following keys are expected: {self.expected_feature_keys}"
                f"Please ensure that the features input has all the expected feature keys."
            )

        total_steps = len(self.time_sampler.get_steps())
        for step_num in range(total_steps + 1):
            for op in self.plan:
                step_state_keys_in = op.input_state_keys(step_num)
                if step_state_keys_in - output_state_keys_by_step[step_num] != set():
                    raise KeyError(
                        f"Input state keys not found: {step_state_keys_in - output_state_keys_by_step[step_num]}"
                    )

                output_step_num = op.get_output_step(input_step_num=step_num)
                if output_step_num is None:
                    pass
                else:
                    step_state_keys_out = op.output_state_keys(output_step_num)
                    if len(step_state_keys_out & output_state_keys_by_step[output_step_num]) > 0:
                        raise KeyError(
                            f"State key collision: {step_state_keys_out & output_state_keys_by_step[output_step_num]}"
                        )
                    state_keys.update(step_state_keys_out)
                    output_state_keys_by_step[output_step_num].update(step_state_keys_out)

        final_step_observable_output_keys = {
            k
            for k in self.observable_output_configs.keys()
            if "final" in self.observable_output_configs[k].outputs.keys()
        }
        any_step_observable_output_keys = {
            k
            for k in self.observable_output_configs.keys()
            if "any" in self.observable_output_configs[k].outputs.keys()
        }

        state_keys_step_num_final = output_state_keys_by_step[step_num]
        if set(state_keys_step_num_final) & set(final_step_observable_output_keys) != set(
            final_step_observable_output_keys
        ):
            raise KeyError(
                f"The following state keys are recorded at the final step: {set(state_keys_step_num_final)}, "
                f"but the outputs expected are: {final_step_observable_output_keys}. "
                f"Please ensure that the state keys are compatible with the expected outputs."
            )
        state_keys_any_steps = state_keys
        if set(state_keys_any_steps) & set(any_step_observable_output_keys) != set(any_step_observable_output_keys):
            raise KeyError(
                f"The following state keys are recorded intermediate steps: {set(state_keys_any_steps)}"
                f"but the outputs expected are: {any_step_observable_output_keys}."
                f"Please ensure that the state keys are compatible with the outputs expected."
            )

    def run(
        self,
        features_input: dict[str, Features],
        output_dir: str | None,
        num_samples: int,
        batch_size: int,
        save_every_n_batches: int = 1,
        device: torch.device | None = None,
    ) -> list[FileOutputConfig]:
        """
        Run the simulation and save observable outputs to the output directory if specified.

        The simulation is run in chunks of batch_size on one device. Batches run sequentially.

        See run_batch() for more details on the simulation loop.

        Parameters
        ----------
        features_input : dict[str, Features]
            The input features to the simulation.
        output_dir : str | None
            The directory to save the outputs to. If None, no outputs are saved.
        num_samples : int
            The number of samples to run the simulation for.
        batch_size : int
            The batch size for model inference.
        save_every_n_batches : int
            The number of batches to accumulate in memory before saving the outputs
            to disk. Higher values reduce I/O frequency but increase memory usage.
        device : torch.device or None
            The device to run on. If None, use ``cuda:0`` when a CUDA device is present.

        Returns
        -------
        list[FileOutputConfig]
            The list of output files created by the simulation.

        Raises
        ------
        Exception
            If validate_before_run is True and _validate() fails. See _validate() for more details.
        ValueError
            If ``device`` is None and no CUDA device is found.
        """
        logger.info(
            f"Running simulation {self.name} with {num_samples} samples, {batch_size} batch size, "
            f"save every {save_every_n_batches} batches, device {device}"
        )
        if self.validate_before_run:
            self._validate(features_input)
            logger.info(f"Simulation {self.name} validated")

        for key, features in features_input.items():
            if not isinstance(features.atoms_to_include, AtomFilter):
                raise ValueError(f"Features {key} must have atoms_to_include set as an AtomFilter.")
            sample_centers = features.featurization_metadata.get(SAMPLE_CENTERS_OF_MASS_METADATA_KEY)
            if sample_centers is not None and len(sample_centers) < num_samples:
                raise ValueError(
                    f"sample_centers_of_mass has {len(sample_centers)} rows but num_samples is {num_samples}"
                )

        if device is None:
            if torch.cuda.device_count() == 0:
                raise ValueError("No devices found. Please set device to a CUDA device.")
            device = torch.device("cuda:0")

        observable_size = batch_size * save_every_n_batches

        logger.info(f"Setting up models on {device}")
        self.setup_models(device)
        logger.info(f"Models setup on {device}")

        all_output_files: list[FileOutputConfig] = []

        # this index keeps track of the total number of samples written to disk
        total_samples = 0

        # outer loop represents a single disk write
        logger.info(f"Chunking simulation into {observable_size} samples per disk write")
        for i in range(0, num_samples, observable_size):
            true_observable_size = min(observable_size, num_samples - total_samples)
            self.setup_observable(true_observable_size)

            batches = [
                slice(start, min(start + batch_size, true_observable_size))
                for start in range(0, true_observable_size, batch_size)
            ]
            for batch in batches:
                self.run_batch(
                    features_input,
                    batch,
                    show_progress=True,
                    device=device,
                    sample_offset=total_samples,
                )

            if output_dir is not None:
                logger.info(f"Saving observable state keys to output files in {output_dir}")
                output_files = self.observable.save(
                    output_dir,
                    start_index=total_samples,
                    reference_features=features_input,
                )
                all_output_files.extend(output_files)
            total_samples += true_observable_size
        logger.info(f"Simulation {self.name} completed")
        return all_output_files
