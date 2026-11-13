from typing import Literal

import torch

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.ops.registry import register_op
from dynopsi.simulation.store import Store
from dynopsi.simulation.utils import is_active_at, valid_modalities_at

SelfConditioningMode = Literal["same_timestep", "previous_step"]


@register_op()
class Infer(Op):
    """
    Represents a basic inference step for a DynoPsi model.

    Reads state and features from the store, gets (t, dt) from the time sampler,
    runs ``infer_cvf``, and writes the output (e.g. CVF) to the store.

    Attributes
    ----------
    name : str
        Name of the operation.
    valid_steps : dict[str, list[int]]
        Keys are modalities; values are step indices at which this op runs.
    inference_model_config : DynoPsiModelConfig
        Model config whose live model runs ``infer_cvf``.
    in_x_loc_store : str
        Store key for input state (read at step_num).
    in_feat_loc_store : str
        Store key for input features (read without step_num).
    out_loc_store : str
        Store key for output state (e.g. X/... or CVF/...).
    self_conditioning_mode : SelfConditioningMode
        ``same_timestep`` runs the model's dummy same-t self-conditioning pass.
        ``previous_step`` feeds the previous Infer's clean x1 from
        ``Store.previous_sc`` (empty on the first active step) and skips that
        pass. Requires ``use_self_cond``. Keyed by this op's ``name``.
    """

    type: Literal["infer"] = "infer"

    inference_model_config: DynoPsiModelConfig
    in_x_loc_store: str
    in_feat_loc_store: str
    out_loc_store: str
    self_conditioning_mode: SelfConditioningMode = "same_timestep"

    def input_state_keys(self, step_num: int) -> set[str]:
        if is_active_at(self.valid_steps, step_num):
            return {self.in_x_loc_store}
        return set()

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def output_state_keys(self, step_num: int) -> set[str]:
        if is_active_at(self.valid_steps, step_num):
            return {self.out_loc_store}
        return set()

    def get_output_step(self, input_step_num: int) -> int | None:
        if is_active_at(self.valid_steps, input_step_num):
            return input_step_num
        return None

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return [self.inference_model_config]

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:
        """
        Run the inference step: model.infer_cvf(x_in, feat_in, t) -> out; store.put(out).

        Parameters
        ----------
        store : Store
            Read in_x_loc_store at step_num, in_feat_loc_store (no step); write
            to out_loc_store at step_num.
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
        x_in = store.get_state(self.in_x_loc_store, step_num)
        feat_in = store.get_features(self.in_feat_loc_store)

        batch_size = x_in[valid_modalities[0]].shape[0]
        times = {modality: store.get_time(step_num, modality, batch_size) for modality in valid_modalities}
        model = self.inference_model_config.dynopsi_model
        sc_inputs = self._self_conditioning_inputs(model, store, step_num, valid_modalities)
        model_out, next_sc = model.infer_cvf(x_in, feat_in.feature_dict, times, self_conditioning_inputs=sc_inputs)
        if self.self_conditioning_mode == "previous_step":
            store.previous_sc[self.name] = next_sc
        store.put(
            self.out_loc_store,
            {modality: model_out[modality].detach() for modality in valid_modalities},
            step_num,
        )

    def _self_conditioning_inputs(
        self,
        model: DynoPsiModel,
        store: Store,
        step_num: int,
        valid_modalities: list[str],
    ) -> dict[str, torch.Tensor | None] | None:
        """Return SC inputs for this step. ``None`` selects the dummy same-timestep pass."""
        if self.self_conditioning_mode == "same_timestep":
            return None
        if not model.use_self_cond:
            raise ValueError("self_conditioning_mode='previous_step' requires use_self_cond on the loaded checkpoint")
        return self._read_previous_step_sc(store, step_num, valid_modalities)

    def _read_previous_step_sc(
        self,
        store: Store,
        step_num: int,
        valid_modalities: list[str],
    ) -> dict[str, torch.Tensor | None]:
        first_step = min(step for modality in valid_modalities for step in self.valid_steps[modality])
        if step_num == first_step:
            return {modality: None for modality in valid_modalities}
        stashed = store.previous_sc.get(self.name)
        if stashed is None:
            raise RuntimeError(
                "previous_step self-conditioning is missing; this Infer has not run on an earlier active step"
            )
        return stashed
