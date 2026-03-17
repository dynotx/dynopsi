from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynopsi.data.featurization import Features
from dynopsi.flow_matching.interpolant import (
    EuclideanInterpolant,
    Interpolant,
    LinearInterpolantSchedule,
)
from dynopsi.flow_matching.priors import IsotropicGaussian, Sampleable
from dynopsi.flow_matching.protein_module_flow_matcher import ProteinModuleFlowMatcher
from dynopsi.model.modules.embedders import DynoPsiSingleConditionalPairEmbedder
from dynopsi.model.modules.heads import DynoPsiOutputHead
from dynopsi.model.modules.protein_module import ProteinModule
from dynopsi.model.modules.trunks import DynoPsiTransformerTrunk

_REQUIRED_FEAT_KEYS = ("frame_resolved_mask", "noised_mask", "token_index")


def _validate_inference_inputs(
    feat: dict[str, Features],
    x_t: dict[str, torch.Tensor],
    modalities: list[str],
) -> None:
    missing_feat = [key for key in _REQUIRED_FEAT_KEYS if key not in feat]
    if missing_feat:
        raise KeyError(f"DynoPsiModel requires feat to contain {_REQUIRED_FEAT_KEYS}; missing: {missing_feat}.")
    missing_mod = [modality for modality in modalities if modality not in x_t]
    if missing_mod:
        raise KeyError(f"x_t missing state for modality/modalities {missing_mod!r}. Required modalities: {modalities}.")


class DynoPsiModel(nn.Module):
    """DynoPsi flow matching model for inference.

    Attributes
    ----------
    config_priors : dict[str, dict[str, Any]]
        A dictionary of prior configurations.
        Keys are the modalities.
        Values are dictionaries containing the prior type and parameters.
    config_interpolants : dict[str, dict[str, Any]]
        A dictionary of interpolant configurations.
        Keys are the modalities.
        Values are dictionaries containing the interpolant type and parameters.
    config_denoiser : dict[str, Any]
        A dictionary of denoiser configurations.
        Must have keys 'embedder', 'trunk' and 'output_head'.
    is_cvf_prediction : dict[str, bool]
        A dictionary of booleans for each modality.
        If True, the denoising model is assumed to predict the conditional vector field.
        If False, the denoising model is assumed to predict the clean data point.
        Keys are the modalities.
        Values are the booleans.
    cvf_eps : float
        Small value used when converting predictions so denominators are never zero (default 1e-1).
    """

    def __init__(
        self,
        config_priors: dict[str, dict[str, Any]],
        config_interpolants: dict[str, dict[str, Any]],
        config_denoiser: dict[str, Any],
        num_atoms: int,
        is_cvf_prediction: dict[str, bool],
        cvf_eps: float = 1e-1,
    ) -> None:
        super().__init__()

        # Store externally used parameters at the top level
        self.num_atoms = num_atoms
        self.use_self_cond = config_denoiser["params"]["embedder"]["params"]["use_self_cond"]
        self.cvf_eps = cvf_eps

        # Build objects from configs to send to the flow matcher
        self.priors = self._build_priors(config_priors)
        self.interpolants = self._build_interpolants(config_interpolants)
        denoiser = self._build_denoiser(config_denoiser, num_time_embeddings=1)

        self.flow_matcher = ProteinModuleFlowMatcher(
            priors=self.priors,
            interpolants=self.interpolants,
            denoiser=denoiser,
            is_cvf_prediction=is_cvf_prediction,
        )

    @property
    def modalities(self) -> list[str]:
        """Data modality names.
        Dyno Psi-v1 only supports the 'coords' modality.
        """
        return list(self.flow_matcher.modalities)

    def forward(
        self, x_t: dict[str, torch.Tensor], feat: dict[str, torch.Tensor], times: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Compute CVF predictions for the current state and time (Infer op interface).

        Parameters
        ----------
        x_t : dict[str, torch.Tensor]
            Current state per modality (e.g. ``coords``), each shape ``(bs, L, d)``.
        feat : dict[str, torch.Tensor]
            Feature dict from the store. Must contain ``frame_resolved_mask``,
            ``noised_mask``, and ``token_index``; plus any other keys required by
            the denoiser.
        times : dict[str, torch.Tensor]
            Time per modality (e.g. from the time sampler), typically shape ``(bs, 1, 1)`` per key.

        Returns
        -------
        dict[str, torch.Tensor]
            Conditional vector field per modality, same keys as modalities.
        """
        current_states = {modality: x_t[modality] for modality in self.modalities}

        other_denoiser_inputs = feat.copy()
        noised_mask = other_denoiser_inputs.pop("noised_mask")
        resolved_mask = other_denoiser_inputs.pop("resolved_mask")

        with torch.no_grad():
            predictions = self.predict(
                current_states=current_states,
                times=times,
                use_self_conditioning=self.use_self_cond,
                return_cvf_prediction=True,
                cvf_eps=self.cvf_eps,
                other_denoiser_inputs=other_denoiser_inputs,
                noised_mask=noised_mask,
                resolved_mask=resolved_mask,
            )
        return {m: predictions[f"pred_{m}"] for m in self.modalities}

    def to_device(self, device: torch.device) -> None:
        self.flow_matcher.denoiser.to(device)

    def _build_priors(self, config_priors: dict[str, dict[str, Any]]) -> dict[str, Sampleable]:

        priors: dict[str, Sampleable] = {}
        for m, config in config_priors.items():
            if config["type"] == "isotropic_gaussian":
                priors[m] = IsotropicGaussian(**config["params"])
            else:
                raise ValueError(f"Unsupported prior type: {config['type']}")

        return priors

    def _build_interpolants(self, config_interpolants: dict[str, dict[str, Any]]) -> dict[str, Interpolant]:

        interpolants: dict[str, Interpolant] = {}
        for m, config in config_interpolants.items():
            if config["type"] == "euclidean_interpolant":
                interpolant_schedule_config = config["params"]["interpolant_schedule"]
                assert interpolant_schedule_config.get("type", "") == "linear_interpolant_schedule"
                interpolants[m] = EuclideanInterpolant(
                    interpolant_schedule=LinearInterpolantSchedule(**interpolant_schedule_config["params"]),
                )
            else:
                raise ValueError(f"Unsupported interpolant type: {config['type']}")

        return interpolants

    def _build_embedder(self, config_embedder: dict[str, Any], num_time_embeddings: int) -> nn.Module:

        if config_embedder["type"] == "dyno_psi_single_cond_pair_embedder":
            return DynoPsiSingleConditionalPairEmbedder(
                **config_embedder["params"],
                num_atoms=self.num_atoms,
                num_time_embeddings=num_time_embeddings,
            )
        else:
            raise ValueError(f"Invalid embedder type '{config_embedder['type']}'.")

    def _build_trunk(self, config_trunk: dict[str, Any]) -> nn.Module:

        if config_trunk["type"] == "dyno_psi_transformer_trunk":
            return DynoPsiTransformerTrunk(**config_trunk["params"])
        else:
            raise ValueError(f"Invalid trunk type '{config_trunk['type']}'.")

    def _build_output_head(self, config_output_head: dict[str, Any]) -> nn.Module:

        if config_output_head["type"] == "dyno_psi_output_head":
            return DynoPsiOutputHead(**config_output_head["params"], num_atoms=self.num_atoms)
        else:
            raise ValueError(f"Invalid output_head type '{config_output_head['type']}'.")

    def _build_denoiser(self, config_denoiser: dict[str, Any], num_time_embeddings: int) -> nn.Module:

        assert config_denoiser["type"] == "ProteinModule", "Only `ProteinModule` is supported as a denoiser"
        protein_module_params = config_denoiser["params"]
        embedder = self._build_embedder(protein_module_params["embedder"], num_time_embeddings=num_time_embeddings)
        trunk = self._build_trunk(protein_module_params["trunk"])
        output_head = self._build_output_head(protein_module_params["output_head"])
        return ProteinModule(embedder=embedder, main_trunk=trunk, output_head=output_head)

    def predict(
        self,
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        use_self_conditioning: bool,
        return_cvf_prediction: bool,
        other_denoiser_inputs: dict[str, Any] = {},
        cvf_eps: float = 1e-1,
    ) -> dict[str, torch.Tensor]:
        """Return predictions from the flow matcher in standard dict[str, torch.Tensor] format.

        The output contains both the flow predictions (key = modality) and the auxiliary predictions
        (key = "distogram_prediction"). The FlowMatcher returns both the conditional vector field and the
        clean data point, but this method requires the user to specify which one to return.

        Parameters
        ----------
        current_states : dict[str, torch.Tensor]
            A dictionary of current states.
            Keys are the modalities.
            Values are the current state `xt` tensors, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            A dictionary of times.
            Keys are the modalities.
            Values are the times, shape (bs, 1, 1).
        resolved_mask : torch.Tensor
            The resolved mask, shape (bs, L).
        noised_mask : torch.Tensor
            The mask indicating which positions are noised in the prior, shape (bs, L).
        use_self_conditioning : bool
            Whether to use self-conditioning in the prediction.
            If False, the self-conditioning input will be set to None for each modality.
        return_cvf_prediction : bool
            Whether to return the conditional vector field or the clean data point.
        cvf_eps : float
            Small value used when converting predictions so denominators are never zero (default 1e-1).
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
            Keys are the parameter names.
            Values are the input tensors or None.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of predictions
        """
        flow_predictions, aux_predictions = self.flow_matcher.denoise(
            current_states=current_states,
            times=times,
            resolved_mask=resolved_mask,
            noised_mask=noised_mask,
            use_self_conditioning=use_self_conditioning,
            other_denoiser_inputs=other_denoiser_inputs,
            cvf_eps=cvf_eps,
        )
        if return_cvf_prediction:
            predictions = {f"pred_{k}": v.conditional_vector_field for k, v in flow_predictions.items()}
        else:
            predictions = {f"pred_{k}": v.clean_data_point for k, v in flow_predictions.items()}
        # Add all aux predictions to the output
        predictions.update(aux_predictions)

        return predictions
