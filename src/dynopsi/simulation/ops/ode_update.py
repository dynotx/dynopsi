import logging
from typing import Literal

import torch

from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.ops.registry import register_op
from dynopsi.simulation.store import Store
from dynopsi.simulation.utils import is_active_at, valid_modalities_at
from dynopsi.utils import SingleParameterFunctionType

logger = logging.getLogger(__name__)


@register_op()
class ODEStepUpdate(Op):
    """
    Represents a step update for a CVF: x_out = x_in + cvf * dt.

    Reads current state and CVF from the store, gets (t, dt) from the time
    sampler, and writes the updated state to the store.
    Operates on batched tensors (leading batch dim); x_out has same batch shape as x_in.

    Attributes
    ----------
    name : str
        Name of the operation.
    valid_steps : dict[str, list[int]]
        Keys are modalities; values are step indices at which this op runs.
    in_x_loc_store : str
        Store key for input state (read at step_num).
    in_cvf_loc_store : str
        Store key for input CVF (read at step_num).
    out_x_loc_store : str
        Store key for output state (written at step_num + 1; advanced state for next step).
    """

    type: Literal["ode_update"] = "ode_update"

    name: str
    valid_steps: dict[str, list[int]]
    in_x_loc_store: str
    in_cvf_loc_store: str
    out_x_loc_store: str

    def input_state_keys(self, step_num: int) -> set[str]:
        if is_active_at(self.valid_steps, step_num):
            return {self.in_x_loc_store, self.in_cvf_loc_store}
        return set()

    def output_state_keys(self, step_num: int) -> set[str]:
        # We write to step_num + 1, but is_active_at tracks step_num. Therefore,
        # need to check that this op runs at step_num - 1.
        if is_active_at(self.valid_steps, step_num - 1):
            return {self.out_x_loc_store}
        return set()

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:
        """
        Run the CVF step update: x_out = x_in + cvf * dt; write at step_num + 1.

        Parameters
        ----------
        store : Store
            Read in_x_loc_store and in_cvf_loc_store at step_num; write
            x_out = x_in + cvf * dt to out_x_loc_store at step_num + 1.
        step_num : int
            Current simulation step index.

        Returns
        -------
        None
            Modifies store in place.
        """
        valid_modalities = valid_modalities_at(self.valid_steps, step_num)
        if len(valid_modalities) == 0:
            return
        cvf_in = store.get_state(self.in_cvf_loc_store, step_num)
        x_in = store.get_state(self.in_x_loc_store, step_num)

        batch_size = x_in[valid_modalities[0]].shape[0]
        dts = {modality: store.get_timestep_size(step_num, modality, batch_size) for modality in valid_modalities}
        x_out = {modality: x_in[modality] + cvf_in[modality] * dts[modality] for modality in valid_modalities}
        store.put(self.out_x_loc_store, x_out, step_num + 1)


@register_op()
class GaussianScoreWeightedODEStepUpdate(Op):
    """
    Represents a step update for a CVF that is scaled by the score:

    (1) Convert the CVF to a score
    score(x_t, t) = (t * cvf(x_t, t) - x_t) / (1 - t)

    (2) Scale the score by the score_scale_fn
    scaled_score(x_t, t) = score_scale_fn(t) * score(x_t, t)

    (3) Convert the scaled score back to a CVF
    cvf_scaled(x_t, t) = (scaled_score(x_t, t) * (1 - t) + x_t) / t

    (4) Update the state
    x_out = x_in + cvf_scaled(x_t, t) * dt

    Scale update is only applied to the noised atoms;
    the unnoised atoms are updated based on the base CVF update
    (i.e. x_out = x_in + cvf * dt)

    Attributes
    ----------
    valid_steps : dict[str, list[int]]
        Keys are modalities; values are step indices at which this op runs.
    in_feat_loc_store : str
        Store key for input features (read without step_num). Used to access the noised mask.
    in_x_loc_store : str
        Store key for input state (read at step_num).
    in_cvf_loc_store : str
        Store key for input CVF (read at step_num). Should come from the Infer op.
    out_x_loc_store : str
        Store key for output state (written at step_num + 1; advanced state for next step).
    score_scale_fn : SingleParameterFunctionType
        Function to scale the score by.
        See dynopsi.utils.function_configs SingleParameterFunctionType for available functions.
    """

    type: Literal["gaussian_score_weighted_ode_update"] = "gaussian_score_weighted_ode_update"

    name: str
    valid_steps: dict[str, list[int]]
    in_feat_loc_store: str
    in_x_loc_store: str
    in_cvf_loc_store: str
    out_x_loc_store: str
    score_scale_fn: SingleParameterFunctionType

    def output_state_keys(self, step_num: int) -> set[str]:
        # We write to step_num + 1, but is_active_at tracks step_num. Therefore,
        # need to check that this op runs at step_num - 1.
        if is_active_at(self.valid_steps, step_num - 1):
            return {self.out_x_loc_store}
        return set()

    def scale_cvf_by_score(
        self,
        current_state: torch.Tensor,
        cvf: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplify the score weighting operation:

        (1) s(x_t, t) = (t * v(x_t, t) - x_t) / (1 - t)
        (2) s'(x_t, t) = score_scale_fn(t) * s(x_t, t)
        (3) v'(x_t, t) = (s'(x_t, t) * (1 - t) + x_t) / t

        --> simplifies to
        score_scale_fn(t) * v(x_t, t) + (1-score_scale_fn(t)) * x_t / t
        """
        assert torch.all(time > 0.0)
        score_scale = self.score_scale_fn(time)
        return (score_scale * cvf) + (1 - score_scale) * current_state / time

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:
        valid_modalities = valid_modalities_at(self.valid_steps, step_num)
        if len(valid_modalities) == 0:
            return
        valid_modalities = valid_modalities_at(self.valid_steps, step_num)
        cvf_in = store.get_state(self.in_cvf_loc_store, step_num)
        x_in = store.get_state(self.in_x_loc_store, step_num)
        feat_in = store.get_features(self.in_feat_loc_store)
        noised_mask = feat_in["noised_mask"]

        batch_size = x_in[valid_modalities[0]].shape[0]
        times = {modality: store.get_time(step_num, modality, batch_size) for modality in valid_modalities}
        dts = {modality: store.get_timestep_size(step_num, modality, batch_size) for modality in valid_modalities}

        x_out = {}
        for modality in valid_modalities:
            x_step = cvf_in[modality] * dts[modality]
            # only apply the scaled update to the noised atoms
            scaled_x_step = self.scale_cvf_by_score(x_in[modality], cvf_in[modality], times[modality]) * dts[modality]
            x_step[noised_mask] = scaled_x_step[noised_mask]

            x_out[modality] = x_in[modality] + x_step

        store.put(self.out_x_loc_store, x_out, step_num + 1)
