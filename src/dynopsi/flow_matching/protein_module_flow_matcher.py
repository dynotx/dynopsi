from typing import Any

import torch

from dynopsi.flow_matching.flow_matcher import FlowMatcher


class ProteinModuleFlowMatcher(FlowMatcher):
    """Flow matching over backbone atoms using a `ProteinModule` for the denoising model."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        for modality in self.modalities:
            if modality not in ["coords"]:  # Other modalities supported in future releases
                raise ValueError(f"Unsupported modality: {modality}")

        # The primary modality is defined as the one that handles coordinates, which is mandatory.
        self.modality = "coords"

    def _construct_model_inputs(
        self,
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        self_conditioning_inputs: dict[str, torch.Tensor | None],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        other_denoiser_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct the model inputs for the denoising model.

        The protein module expects `coords` and `sc_coords` to be of shape (batch_size, L, n_atoms, 3).
        The current state tensors need to be converted from shape (batch_size, L, d), where d = n_atoms * 3,
        while the self-conditioning inputs are already in the correct shape thanks to the `_convert_raw_predictions_to_sc_inputs` method.
        """
        batch_size = current_states[self.modality].shape[0]
        seq_len = current_states[self.modality].shape[1]

        model_inputs = other_denoiser_inputs.copy()
        if "true_coords" in model_inputs:
            model_inputs["true_coords"] = model_inputs["true_coords"].reshape(batch_size, seq_len, -1, 3)
        model_inputs.update(
            {
                # (batch_size, L, d) -> (batch_size, L, n_atoms, 3)
                "coords": current_states[self.modality].reshape(batch_size, seq_len, -1, 3),
                "t": times[self.modality].reshape(-1),  # (batch_size,)
                "mask": resolved_mask,  # (batch_size, L)
                "noised_mask": noised_mask,  # (batch_size, L)
                "sc_coords": self_conditioning_inputs[self.modality],  # (batch_size, L, n_atoms, 3) or None
            }
        )

        return model_inputs

    def _convert_raw_predictions_to_sc_inputs(
        self,
        raw_predictions: dict[str, torch.Tensor],
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        """Convert the raw predictions to self-conditioning inputs.

        The main thing that has to happen is to ensure that the self-conditioning inputs
        are clean data predictions, not CVF predictions.
        """
        for k in raw_predictions:
            if isinstance(raw_predictions[k], torch.Tensor):
                raw_predictions[k] = raw_predictions[k].detach()
        processed_predictions, _ = self._postprocess_predictions(raw_predictions)
        sc_preds: dict[str, torch.Tensor | None] = {}
        for modality in self.modalities:
            if self.is_cvf_prediction[modality]:
                clean_data_preds = self.interpolants[modality].apply_conditional_vector_field(
                    xt=current_states[modality], ut=processed_predictions[f"out_{modality}"], dt=1 - times[modality]
                )
            else:
                clean_data_preds = processed_predictions[f"out_{modality}"]

            if modality == self.modality:  # add num_atoms dimension
                bs, L, d = clean_data_preds.shape
                num_atoms = int(d // 3)
                clean_data_preds = clean_data_preds.reshape(bs, L, num_atoms, 3)

            sc_preds[modality] = clean_data_preds
        return sc_preds

    def _postprocess_predictions(
        self, raw_predictions: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Postprocess the predictions from the denoising model to the format expected by the flow matching model,
        which is a dictionary of {out_{modality} : torch.Tensor} for each modality.

        The output of the protein module is of shape (batch_size, L, n_atoms, 3).
        We need to reshape it to (batch_size, L, d), where d = n_atoms * 3.
        """
        batch_size = raw_predictions["out_coords"].shape[0]
        seq_len = raw_predictions["out_coords"].shape[1]

        # Add flow predictions to the output
        flow_predictions = {
            f"out_{modality}": raw_predictions["out_coords"].reshape(batch_size, seq_len, -1)
            if modality == self.modality
            else raw_predictions[f"out_{modality}"].reshape(batch_size, seq_len, -1)
            for modality in self.modalities
        }

        # Add distogram predictions, optionally
        if hasattr(self.denoiser.output_head, "distogram_head"):
            distogram_predictions = {"distogram_prediction": raw_predictions["distogram_prediction"]}
        else:
            distogram_predictions = {}

        return flow_predictions, distogram_predictions
