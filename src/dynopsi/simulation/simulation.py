from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, cast

import torch
from pydantic import BaseModel, Field, field_serializer, field_validator
from tqdm import tqdm  # type: ignore

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization import Features
from dynopsi.simulation.build_initial_state import InitializeStepType
from dynopsi.simulation.observable import FileOutputConfig, Observable, ObservableOutputConfig
from dynopsi.simulation.ops import Op, get_op_from_config
from dynopsi.simulation.store import Store
from dynopsi.simulation.time_sampler import TimeSamplerType


class Simulation(BaseModel):
    """
    Run a plan of operations over time.

    Executes each Op in a plan at each step from the time sampler.
    State advances via the store (e.g. CVFStepUpdate writes to step_num + 1).

    Batching is inferred from tensor shape: features and state tensors typically
    have a leading batch dimension (e.g. [batch_size, seq_len, ...]). The
    simulation loop runs over steps only; it does not loop over batch indices.
    Each op receives and writes full tensors (with batch dim) preserves batch structure.
    No separate batch loop or batch_size parameter is used in Simulation.

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

    NOTE: Subclasses must define a type field as a Literal["subclass_type"] = "subclass_type"
    this is used for serialization/deserialization.
    """

    name: str

    plan_init: list[InitializeStepType]
    plan: list[Op]
    time_sampler: TimeSamplerType
    clear_states_each_step: bool = True
    validate_before_run: bool = True

    observable_output_configs: dict[str, ObservableOutputConfig] = Field(default_factory=dict)
    observable: Observable = Field(default_factory=Observable)

    @property
    def expected_feature_keys(self) -> set[str]:
        """Get the set of feature keys that are expected by the plan."""
        feature_keys = set()
        for init_step in self.plan_init:
            feature_keys.update(init_step.input_feature_keys())
        for op in self.plan:
            feature_keys.update(op.input_feature_keys())
        return feature_keys

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
        return result

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
            observable_output_configs=self.observable_output_configs,
        )

    def setup_models(self, devices: list[torch.device]) -> None:
        for device in devices:
            for step in self.plan_init + self.plan:
                for model_config in step.model_configs():
                    model_config.to_device(device)

    def run_batch(
        self,
        features_input: dict[str, Features],
        batch_slice: slice,
        device: torch.device,
        show_progress: bool = False,
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
        """
        # get the time schedule
        time_schedule = self.time_sampler.get_full_schedule(device)

        # initialize a store
        store = Store(
            features=deepcopy(features_input),
            observable=self.observable,
            batch_slice=batch_slice,
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
        # check that the features input has all the expected feature keys
        input_features_keys = set(features_input.keys())
        # first, check that plan init (which can create new feature keys) is valid
        for init_step in self.plan_init:
            if len(init_step.input_feature_keys() - input_features_keys) > 0:
                raise KeyError(
                    f"Features input missing keys. Init step expects: {init_step.input_feature_keys()}, but features input has: {input_features_keys}"
                    f"Please ensure that the features input has all the expected feature keys."
                )
            else:
                input_features_keys.update(init_step.output_feature_keys())

        # now check that the rest of the plan is valid
        if len(self.expected_feature_keys - input_features_keys) > 0:
            raise KeyError(
                f"Features input missing expected feature keys for plan to run. "
                f"The following keys were input or created by the plan initialization step: {input_features_keys}, "
                f"but the following keys are expected: {self.expected_feature_keys}"
                f"Please ensure that the features input has all the expected feature keys."
            )

        # check that there are not state key collisions
        state_keys: set[str] = set()
        for init_step in self.plan_init:
            new_state_keys = init_step.output_state_keys(0)
            if len(new_state_keys & state_keys) > 0:
                raise KeyError(f"State key collision: {new_state_keys & state_keys}")
            state_keys.update(new_state_keys)
        for op in self.plan:
            state_keys_step_0 = op.output_state_keys(0)
            if len(state_keys_step_0 & state_keys) > 0:
                raise KeyError(f"State key collision: {state_keys_step_0 & state_keys}")
            state_keys.update(state_keys_step_0)
        # now clear and test other steps
        total_steps = len(self.time_sampler.get_steps())
        for step_num in range(1, total_steps + 1):
            state_keys_step_num: set[str] = set()
            for op in self.plan:
                op_state_keys = op.output_state_keys(step_num)
                if len(state_keys_step_num & op_state_keys) > 0:
                    raise KeyError(f"State key collision: {state_keys_step_num & op_state_keys}")
                state_keys_step_num.update(op_state_keys)
                state_keys.update(state_keys_step_num)

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

        # check final step
        state_keys_step_num_final = state_keys_step_num
        if set(state_keys_step_num_final) & set(final_step_observable_output_keys) != set(
            final_step_observable_output_keys
        ):
            raise KeyError(
                f"The following state keys are recorded at the final step: {set(state_keys_step_num_final)}, "
                f"but the outputs expected are: {final_step_observable_output_keys}. "
                f"Please ensure that the state keys are compatible with the expected outputs."
            )
        # check any steps
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
        devices: list[torch.device] | None = None,
    ) -> list[FileOutputConfig]:
        """
        Run the simulation and save observable outputs to the output directory if specified.

        The simulation is run in chunks of batch_size, and is split in parallel across the available
        devices. On a single device, the batches are run sequentially. On multiple devices, the batches
        are dispatched in parallel using a ThreadPoolExecutor. This is single-machine multi-GPU parallelism only.

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
            The number of batches per device to accumulate in memory before saving
            the outputs to disk. Higher values reduce I/O frequency but increase memory usage.
        devices : list[torch.device] | None
            The CUDA devices to run the simulation on. If None,
            devices are inferred with torch.cuda.device_count().


        Returns
        -------
        list[FileOutputConfig]
            The list of output files created by the simulation.

        Raises
        ------
        Exception
            If validate_before_run is True and _validate() fails. See _validate() for more details.
        ValueError
            If devices is set to None and no CUDA devices are found.
        """
        if self.validate_before_run:
            self._validate(features_input)

        # check that features all have atoms_to_include set as an AtomFilter
        for key, features in features_input.items():
            if not isinstance(features.atoms_to_include, AtomFilter):
                raise ValueError(f"Features {key} must have atoms_to_include set as an AtomFilter.")

        if devices is None:
            if torch.cuda.device_count() == 0:
                raise ValueError("No devices found. Please set devices to a list of GPU-enabled devices.")
            devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        # calculate the number of samples per disk write
        observable_size = batch_size * len(devices) * save_every_n_batches
        assert len(devices) > 0, "Devices must be set"

        self.setup_models(devices)

        # initialize a list of output files
        all_output_files: list[FileOutputConfig] = []

        # this index keeps track of the total number of samples written to disk
        total_samples = 0

        # outer loop represents a single disk write
        # inner loop breaks each disk write into batches
        # and runs the simulation on each batch in parallel across the devices
        for i in range(0, num_samples, observable_size):
            true_observable_size = min(observable_size, num_samples - total_samples)
            self.setup_observable(true_observable_size)

            batches = [
                slice(start, min(start + batch_size, true_observable_size))
                for start in range(0, true_observable_size, batch_size)
            ]
            if len(devices) == 1:
                for batch in batches:
                    self.run_batch(
                        features_input,
                        batch,
                        show_progress=True,
                        device=devices[0],
                    )
            else:
                with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                    futures = [
                        executor.submit(
                            self.run_batch,
                            features_input,
                            batch,
                            show_progress=False,
                            device=devices[idx % len(devices)],
                        )
                        for idx, batch in enumerate(batches)
                    ]
                    for future in as_completed(futures):
                        future.result()

            # Save the observable state keys to output files
            if output_dir is not None:
                output_files = self.observable.save(
                    output_dir,
                    start_index=total_samples,
                    reference_features=features_input,
                )
                all_output_files.extend(output_files)
            total_samples += true_observable_size

        return all_output_files
