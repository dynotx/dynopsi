from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.store import Store
from dynopsi.utils.strict_config import StrictConfig


class Op(StrictConfig):
    """
    Abstract base for an operation in a simulation plan.

    Subclassed ops must implement run method. Each op reads and writes shared
    state via the store and gets time information from the time sampler.

    Notes
    -----
    Subclasses must implement `run(store, time_sampler, step_num)`. Each op:
    - Reads from the store via `store.get(key)` for features or
      `store.get(key, step_num)` for state.
    - Writes to the store via `store.put(key, value, step_num)`.
    - Gets time info via `time_sampler.get(step_num)`, returning ``(t, dt)``.
    State and feature tensors are typically batched (leading batch dimension);
    ops must preserve batch structure in reads and writes.

    NOTE: Subclasses must define a type field as a Literal["subclass_type"]
    this is used for serialization/deserialization.
    """

    name: str
    valid_steps: dict[str, list[int]]

    def input_state_keys(self, step_num: int) -> set[str]:
        """
        State keys this op reads at the given step (for validation).

        Parameters
        ----------
        step_num : int
            Current step index.

        Returns
        -------
        set[str]
            Store keys for state that this op reads at step_num. Empty by default.
        """
        return set()

    def input_feature_keys(self) -> set[str]:
        """
        Feature keys this op reads (for validation).

        Returns
        -------
        set[str]
            Store keys for features that this op reads. Empty by default.
        """
        return set()

    def output_state_keys(self, step_num: int) -> set[str]:
        """
        State keys this op writes at the given step (for validation).

        Used to avoid requiring keys at step start that are produced earlier in the same step.

        Parameters
        ----------
        step_num : int
            Current step index.

        Returns
        -------
        set[str]
            Store keys for state that this op writes at step_num. Empty by default.
        """
        return set()

    def get_output_step(self, input_step_num: int) -> int | None:
        """
        Get the output step number given the input step number.

        Arguments
        ----------
        input_step_num : int
            The input step number to get the output step for.

        Returns
        -------
        int | None
            The output step number, or None if not active at that step.
        """
        return None

    def intermediate_observable_keys(self) -> set[str]:
        """
        Intermediate state keys that this op writes, step agnostic. This is used for observation only.

        Returns
        -------
        set[str]
            Store keys for intermediate state(s) that this op writes to at any step num. Empty by default.
        """
        return set()

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return []

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:
        """
        Execute this op for the given step.

        Parameters
        ----------
        store : Store
            Shared state. Read via get(key) or get(key, step_num); write via
            put(key, value, step_num). Tensors are batched; op must preserve
            batch structure.
        step_num : int
            Current simulation step index.

        Returns
        -------
        None
            Modifies store in place.
        """
        raise NotImplementedError("Subclasses must implement this method")
