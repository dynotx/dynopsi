from typing import Literal

from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.ops.registry import register_op
from dynopsi.simulation.store import Store
from dynopsi.simulation.utils import is_active_at, valid_modalities_at


@register_op()
class Infer(Op):
    """
    Represents a basic inference step for a DynoPsi model.

    Reads state and features from the store, gets (t, dt) from the time sampler,
    runs the model forward, and writes the output (e.g. CVF) to the store.

    Attributes
    ----------
    name : str
        Name of the operation.
    valid_steps : dict[str, list[int]]
        Keys are modalities; values are step indices at which this op runs.
    model : nn.Module
        Model used for the forward pass.
    in_x_loc_store : str
        Store key for input state (read at step_num).
    in_feat_loc_store : str
        Store key for input features (read without step_num).
    out_loc_store : str
        Store key for output state (e.g. X/... or CVF/...).
    """

    type: Literal["infer"] = "infer"

    name: str
    valid_steps: dict[str, list[int]]
    inference_model_config: DynoPsiModelConfig
    in_x_loc_store: str
    in_feat_loc_store: str
    out_loc_store: str

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

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return [self.inference_model_config]

    def run(
        self,
        store: Store,
        step_num: int,
    ) -> None:
        """
        Run the inference step: model(x_in, feat_in, t) -> out; store.put(out).

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
        model_out = self.inference_model_config.dynopsi_model(x_in, feat_in.feature_dict, times=times)
        store.put(
            self.out_loc_store,
            {modality: model_out[modality].detach() for modality in valid_modalities},
            step_num,
        )
