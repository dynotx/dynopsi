import logging
from typing import Literal

import torch
from pydantic import Field

from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.ops.registry import register_op
from dynopsi.simulation.store import Store
from dynopsi.simulation.utils import is_active_at, valid_modalities_at
from dynopsi.utils.function_configs import SingleParameterFunctionType

logger = logging.getLogger(__name__)


@register_op()
class GaussianSDEStepUpdate(Op):
    """
    Represents a step update for a SDE for a Gaussian flow-matching model:

    x_out = cvf * dt + sigma(t) / 2 * score(x_in, cvf, t) * dt + sqrt(sigma(t)) * noise_scale_fn(t) * dW
    where dW is a Wiener process and sigma(t) is the diffusion coefficient as a function of time.

    Reads current state, CVF, time, and dt from the store, and updates step+1 with
    the new state defined by the SDE update.

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
    diffusion_coefficient_fn_config: SingleParameterFunctionType
        Configuration parameters defining the diffusion coefficient as a function of time.
        See dynopsi.utils.function_configs SingleParameterFunctionType for available functions.
    noise_scale_fn_config: SingleParameterFunctionType
        Configuration parameters defining the noise scale as a function of time. This is specifically
        applied to the stochastic term of the SDE update, and values < 1 correspond to
        low-temperature sampling. See dynopsi.utils.function_configs SingleParameterFunctionType for available functions.
    out_x_loc_store : str
        Store key for output state (written at step_num + 1; advanced state for next step).
    """

    type: Literal["gaussian_sde_update"] = "gaussian_sde_update"

    name: str
    valid_steps: dict[str, list[int]]
    in_feat_loc_store: str
    in_x_loc_store: str
    in_cvf_loc_store: str
    diffusion_coefficient_fn: SingleParameterFunctionType
    noise_scale_fn: SingleParameterFunctionType
    out_x_loc_store: str
    # for saving intermediate terms
    save_intermediate_terms: bool
    drift_terms_loc_store: str = Field(default="drift_terms")
    score_terms_loc_store: str = Field(default="score_terms")
    noise_terms_loc_store: str = Field(default="noise_terms")

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

    def brownian_increment(self, dt: torch.Tensor) -> torch.Tensor:
        """
        Sample a Brownian increment dw_t ~ N(0, dt*I) with the Euler-Maruyama method:

        sqrt(dt) * eps, eps ~ N(0, I)

        Parameters
        ----------
        dt: torch.Tensor, the dt, shape (bs, L, d)
        """
        eps = torch.randn(dt.shape, dtype=dt.dtype, device=dt.device)

        return torch.sqrt(dt) * eps

    def vector_field_to_score(
        self,
        current_state: torch.Tensor,
        cvf: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score given the vector field learned by flow matching

        s(x_t, t) = (t * v(x_t, t) - x_t) / (1 - t)

        Parameters
        ----------
        current_state: torch.Tensor, the x_t, shape (bs, L, d)
        cvf: torch.Tensor, prediction of learned vector field v(x_t, t), shape (bs, L, d)
        time: torch.Tensor, shape (bs, 1, 1)
        """
        assert torch.all(time < 1.0)
        return (time * cvf - current_state) / (1.0 - time)

    def get_sde_update_terms(
        self,
        current_state: torch.Tensor,
        cvf: torch.Tensor,
        time: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the state using the SDE update."""
        score = self.vector_field_to_score(current_state, cvf, time)  # bs, L, d
        diffusion_coefficient_term = self.diffusion_coefficient_fn(time)  # bs, 1, 1
        noise_scale = self.noise_scale_fn(time)  # bs, 1, 1

        dt_expanded = dt.expand(-1, current_state.shape[1], current_state.shape[2])  # bs, L, d
        brownian_increment = self.brownian_increment(dt_expanded)  # bs, L, d

        drift_term = cvf * dt
        score_term = diffusion_coefficient_term * score * dt
        noise_term = torch.sqrt(2 * diffusion_coefficient_term * noise_scale) * brownian_increment
        return drift_term, score_term, noise_term

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:

        valid_modalities = valid_modalities_at(self.valid_steps, step_num)
        if len(valid_modalities) == 0:
            return

        cvf_in = store.get_state(self.in_cvf_loc_store, step_num)
        x_in = store.get_state(self.in_x_loc_store, step_num)

        # get the noised mask from the features
        # this mask is needed so that we only update the noised atoms
        feat_in = store.get_features(self.in_feat_loc_store)
        noised_mask = feat_in["noised_mask"]

        batch_size = x_in[valid_modalities[0]].shape[0]
        times = {modality: store.get_time(step_num, modality, batch_size) for modality in valid_modalities}
        dts = {modality: store.get_timestep_size(step_num, modality, batch_size) for modality in valid_modalities}

        x_out = {}
        if self.save_intermediate_terms:
            drift_terms = {}
            score_terms = {}
            noise_terms = {}

        for modality in valid_modalities:
            drift_term, score_term, noise_term = self.get_sde_update_terms(
                x_in[modality], cvf_in[modality], times[modality], dts[modality]
            )
            x_step = drift_term
            x_step[noised_mask] = (drift_term + score_term + noise_term)[noised_mask]

            x_out[modality] = x_in[modality] + x_step

            if self.save_intermediate_terms:
                drift_terms[modality] = drift_term
                score_terms[modality] = score_term
                noise_terms[modality] = noise_term

        store.put(self.out_x_loc_store, x_out, step_num + 1)
        if self.save_intermediate_terms:
            store.put(self.drift_terms_loc_store, drift_terms, step_num + 1)
            store.put(self.score_terms_loc_store, score_terms, step_num + 1)
            store.put(self.noise_terms_loc_store, noise_terms, step_num + 1)
