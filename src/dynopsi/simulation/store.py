import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynopsi.data.featurization import Features
from dynopsi.simulation.observable import Observable
from dynopsi.simulation.time_sampler import TimeSchedule


class Store(BaseModel):
    """
    Stores information for a sampling run.

    Stores the following: features (model inputs), time schedule, current state (e.g. X_T, conditional vector field), and states along
    the trajectory (e.g. X_T at each step).

    Feature keys and state keys must be disjoint.

    Attributes
    ----------
    features : dict[str, Features]
        Feature objects keyed by a readable name (e.g. "conditioned_features", "unconditioned_features")
    time_schedule : TimeSchedule
        Time schedule for the simulation that specifies the time and dt for each step for each modality.
    observable : Observable or None, default=None
        Observable object for recording a subset of state keys at each step. If None, batch_slice is ignored.
    batch_slice : slice or None
        Slice along the batch dimension that corresponds to the position in the observable
        that this store writes to. Used with observable.record(). If the observable is None, this is ignored.
    states : dict[str, dict[int, dict[str, torch.Tensor]]]
        State at each step: states[store_key][step_num][modality] -> tensor.
        Not a constructor argument; set by store.put() in Simulation.run()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: dict[str, Features]
    time_schedule: TimeSchedule
    observable: Observable | None = Field(default=None)
    batch_slice: slice | None = None

    states: dict[str, dict[int, dict[str, torch.Tensor]]] = Field(default_factory=dict, init=False)

    @model_validator(mode="after")
    def _validate_store_keys(self) -> "Store":
        feat = set(self.features)
        if self.observable is not None:
            obs_in_feat = (
                set(self.observable.keys_to_record_any_timestep) | set(self.observable.keys_to_record_final_timestep)
            ) & feat
            if obs_in_feat:
                raise ValueError(
                    f"observable.keys_to_record_any_timestep and observable.keys_to_record_final_timestep must only list state keys. Keys in features: {sorted(obs_in_feat)}."
                )
        return self

    @field_validator("time_schedule", mode="after")
    @classmethod
    def _validate_time_schedule(cls, time_schedule: TimeSchedule) -> TimeSchedule:
        times, dts = time_schedule
        # check for correct formatting, tensors should be size [1 or bs, T, 1, 1]
        error_msg = f"Incorrect size for {0} {1}, must be (bs or 1, T, 1, 1)"
        for mod, time in times.items():
            if len(time.shape) != 4 or time.shape[2] != 1 or time.shape[3] != 1:
                raise ValueError(error_msg.format(mod, "ts"))
        for mod, dt in dts.items():
            if len(dt.shape) != 4 or dt.shape[2] != 1 or dt.shape[3] != 1:
                raise ValueError(error_msg.format(mod, "dts"))
        return time_schedule

    def get_time(self, step_num: int, modality: str, batch_size: int) -> torch.Tensor:
        """Return a view of the time schedule."""
        ts = self.time_schedule[0][modality][:, step_num, :, :]
        if ts.shape[0] == 1:
            return ts.expand(batch_size, -1, -1)
        elif ts.shape[0] == batch_size:
            return ts
        else:
            raise ValueError(
                f"Mismatch in time tensor in batch dimension: ts: {ts.shape[0]}, expected: {batch_size} or 1"
            )

    def get_timestep_size(self, step_num: int, modality: str, batch_size: int) -> torch.Tensor:
        """Return a view of the dt schedule."""
        dts = self.time_schedule[1][modality][:, step_num, :, :]
        if dts.shape[0] == 1:
            return dts.expand(batch_size, -1, -1)
        elif dts.shape[0] == batch_size:
            return dts
        else:
            raise ValueError(
                f"Mismatch in time step tensor in batch dimension: dts: {dts.shape[0]}, expected: {batch_size} or 1"
            )

    def get_features(self, key: str) -> Features:
        """Get a features configuration from the store by key."""
        if key not in self.features:
            raise KeyError(f"Unknown feature key: {key}")
        return self.features[key]

    def put_features(self, key: str, features: Features) -> None:
        if key in self.features:
            raise ValueError(f"Key {key} already in features. Cannot overwrite existing features.")

        self.features[key] = features

    def get_state(self, key: str, step_num: int) -> dict[str, torch.Tensor]:
        """
        Get state at a key and step.

        Parameters
        ----------
        key : str
            State key (e.g. X, CVF/cond).
        step_num : int
            Step index.

        Returns
        -------
        dict[str, torch.Tensor]
            Modality -> tensor at that step. Same as states[key][step_num][modality]

        Raises
        ------
        KeyError
            If key or step_num is missing.
        """
        if key not in self.states:
            raise KeyError(f"Unknown store key: {key}")
        steps = self.states[key]
        if step_num not in steps:
            raise KeyError(f"No state for key {key!r} at step {step_num}")
        return steps[step_num]

    def put(self, key: str, value: dict[str, torch.Tensor], step_num: int) -> None:
        """
        Put state at the given key and step (state only). Possibly save a copy to the observable.

        Specifically: states[key][step_num] = value

        If the key is in the observable.keys_to_record_any_timestep, a copy is saved to the observable.

        Parameters
        ----------
        key : str
            State key (e.g. X, CVF/cond).
        value : dict[str, torch.Tensor]
            Modality -> tensor to put at the given step.
        step_num : int
            Step index.

        Raises
        ------
        ValueError
            If key is already a feature key (feature and state keys must be disjoint).
        """
        if key in self.features:
            raise ValueError(
                f"Key {key!r} is a feature key; cannot use for state. Feature and state keys must be disjoint."
            )
        if key not in self.states:
            self.states[key] = {}
        steps = self.states[key]
        if step_num not in steps:
            steps[step_num] = value
        else:
            raise ValueError(f"State for key {key!r} already exists at step {step_num}")

        if self.observable is not None and key in self.observable.keys_to_record_any_timestep:
            snapshot = {key: value.detach().clone().cpu() for key, value in steps[step_num].items()}
            self.observable.record(key, step_num, snapshot, self.batch_slice)

    def record_final_timestep(self, key: str, step_num: int) -> None:
        """Record the final timestep of a state at a key."""
        if self.observable is None:
            return

        # check if in observable already
        key_observable = self.observable.get_observable_step(key, step_num)
        if key_observable is not None:
            return

        # otherwise obtain from states
        if key not in self.states:
            raise KeyError(f"No state for key {key!r} at step {step_num}")
        steps = self.states[key]
        if step_num not in steps:
            raise KeyError(f"No state for key {key!r} at step {step_num}")
        snapshot = {key: value.detach().clone().cpu() for key, value in steps[step_num].items()}
        self.observable.record(key, step_num, snapshot, self.batch_slice)

    def _infer_batch_size(self) -> int:
        """
        Infer batch size from the first available feature or state tensor.

        Uses the leading dimension of the first feature tensor, or the batch
        dimension (index 1 for state tensors with put-history) of the first
        state tensor.

        Returns
        -------
        int
            Batch size (leading or batch dimension size).

        Raises
        ------
        ValueError
            If no feature or state tensor with a suitable shape exists.
        """
        for feat in self.features.values():
            for v in feat.feature_dict.values():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    return v.shape[0]
        for key, steps in self.states.items():
            for step_data in steps.values():
                for v in step_data.values():
                    if v.dim() > 1:
                        return v.shape[1]
                    if v.dim() == 1:
                        return v.shape[0]
        raise ValueError("Cannot infer batch size: no feature or state tensors with leading dim.")

    def to_device(self, device: torch.device) -> None:
        """Move all tensors to the given device, except for the observable."""
        for key, feat in self.features.items():
            feat.feature_dict = {k: v.to(device) for k, v in feat.feature_dict.items()}
        for key, steps in self.states.items():
            for step, step_data in steps.items():
                step_data = {modality: value.to(device) for modality, value in step_data.items()}

    def clear_states(self, keep_step_num: int) -> None:
        """
        Clear all states except for the given step number.

        Parameters
        ----------
        keep_step_num : int
            The step number to keep.

        Returns
        -------
        None
            Clears the states in place.
        """
        for key in list(self.states.keys()):
            if keep_step_num in self.states.get(key, {}):
                kept = self.states[key][keep_step_num]
                del self.states[key]
                self.states[key] = {keep_step_num: kept}
