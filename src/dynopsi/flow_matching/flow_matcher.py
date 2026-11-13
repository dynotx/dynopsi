from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from dynopsi.flow_matching.interpolant import Interpolant
from dynopsi.flow_matching.priors import Sampleable


@dataclass
class FlowMatcherOutput:
    """Conditional vector field and clean data point from one flow-matcher step.

    Attributes
    ----------
    conditional_vector_field : torch.Tensor
        The conditional vector field predictions, shape (bs, L, d).
    clean_data_point : torch.Tensor
        The clean data point predictions, shape (bs, L, d).
    """

    conditional_vector_field: torch.Tensor
    clean_data_point: torch.Tensor


class FlowMatcher(ABC, nn.Module):
    """Flow matching over data modalities.

    This module produces denoising predictions (conditional vector fields or clean data points) for each modality,
    as well as the associated targets in the same format.

    This class is designed to be subclassed for a particular denoising model by implementing
    `_construct_model_inputs`, `_convert_raw_predictions_to_sc_inputs`, and `_postprocess_predictions`.

    Each modality's conditional probability path is implicitly defined using a `prior`, which governs the initial noisy
    sample at time = 0, and an `interpolant`, which governs the computation of intermediate states (xt) between samples
    from the prior (x0) and from the clean data distribution (x1). The conditional probability path is defined as p_t(x | x1),
    marginalizing over samples from the prior distribution, while the interpolant computes p_t(x | x0, x1) for a particular
    prior sample as a delta function at xt.

    Parameters
    ----------
    priors : dict[str, Sampleable]
        A dictionary of prior objects.
        Keys: modalities. Values: sampleable prior distributions.
    interpolants : dict[str, Interpolant]
        A dictionary of interpolant objects.
        Keys: modalities. Values: interpolants.
    denoiser : nn.Module
        The denoising model.
    is_cvf_prediction : dict[str, bool]
        A dictionary of booleans for each modality.
        Keys: modalities. Values: booleans.
        If True, the denoising model is assumed to predict the conditional vector field.
        If False, the denoising model is assumed to predict the clean data point.
    explicit_mask_fixed_positions : bool
        If True, positions that are fixed (noised_mask==False) are set to exactly equal the target
        for clean data point and zero for conditional vector field.
    """

    def __init__(
        self,
        priors: dict[str, Sampleable],
        interpolants: dict[str, Interpolant],
        denoiser: nn.Module,
        is_cvf_prediction: dict[str, bool],
        explicit_mask_fixed_positions: bool = False,
    ) -> None:
        super().__init__()

        assert sorted(priors) == sorted(interpolants)
        self.priors = priors
        self.interpolants = interpolants
        self.modalities = sorted(list(priors.keys()))
        self.is_cvf_prediction = is_cvf_prediction
        self.denoiser = denoiser
        self.explicit_mask_fixed_positions = explicit_mask_fixed_positions

    @abstractmethod
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

        Converts standard dictionary key names and tensor shapes to the specific
        model inputs required by the denoising model.

        Parameters
        ----------
        current_states : dict[str, torch.Tensor]
            Current state `xt` tensors per modality, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            Time tensors per modality, shape (bs, 1, 1).
        self_conditioning_inputs : dict[str, torch.Tensor | None]
            Self-conditioning inputs per modality, shape (bs, L, d) or None.
        resolved_mask : torch.Tensor
            Shape (bs, L).
        noised_mask : torch.Tensor
            Mask indicating which positions are noised in the prior, shape (bs, L).
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.

        Returns
        -------
        dict[str, Any]
            Model inputs keyed by parameter name.
        """
        raise NotImplementedError

    @abstractmethod
    def _convert_raw_predictions_to_sc_inputs(
        self,
        raw_predictions: dict[str, torch.Tensor],
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        postprocess_args: dict[str, Any] = {},
    ) -> dict[str, torch.Tensor | None]:
        """Convert raw denoiser predictions to self-conditioning inputs.

        Takes the raw output of the denoising model and converts it to a dictionary
        that can be fed back into the denoising model as self-conditioning inputs.

        Parameters
        ----------
        raw_predictions : dict[str, torch.Tensor]
            Raw predictions from the denoising model.
        current_states : dict[str, torch.Tensor]
            Current state `xt` tensors per modality, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            Time tensors per modality, shape (bs, 1, 1).
        postprocess_args : dict[str, Any]
            Additional arguments to the self-conditioning postprocessing function.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Self-conditioning inputs per modality in the shape required by the denoising model.
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess_predictions(
        self, raw_predictions: dict[str, torch.Tensor], **kwargs: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Postprocess predictions from the denoising model.

        Converts the custom model output to standard dictionary key names and tensor shapes
        expected by the flow matching model. Primary predictions must be in the format
        ``{out_{modality}: torch.Tensor}`` for each modality. Additional outputs (e.g. distogram
        logits) are returned as auxiliary predictions.

        Parameters
        ----------
        raw_predictions : dict[str, torch.Tensor]
            Raw predictions from the denoising model.
        **kwargs
            Additional arguments forwarded from postprocess_args.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            Primary predictions keyed as ``out_{modality}``, and auxiliary predictions.
        """
        raise NotImplementedError

    def _apply_explicit_mask_fixed_positions(
        self,
        predictions: FlowMatcherOutput,
        data_sample: torch.Tensor,
        noised_mask: torch.Tensor,
    ) -> FlowMatcherOutput:
        """Force fixed (un-noised) positions to exactly equal the target.

        Updates predictions so that positions where noised_mask==False have
        clean_data_point set to the target and conditional_vector_field set to zero.

        Parameters
        ----------
        predictions : FlowMatcherOutput
            Model predictions.
        data_sample : torch.Tensor
            Shape (bs, L, d). Desired values for fixed positions — either a fully clean
            data point or an interpolated state generated with data-dependent coupling.
        noised_mask : torch.Tensor
            Shape (bs, L). True for positions that are noised.

        Returns
        -------
        FlowMatcherOutput
            Predictions with fixed positions overwritten.
        """
        cvf_pred = predictions.conditional_vector_field
        x1_pred = predictions.clean_data_point
        cvf_pred = torch.where(noised_mask.unsqueeze(-1), cvf_pred, torch.zeros_like(cvf_pred))
        x1_pred = torch.where(noised_mask.unsqueeze(-1), x1_pred, data_sample)
        return FlowMatcherOutput(conditional_vector_field=cvf_pred, clean_data_point=x1_pred)

    def _compute_self_conditioning_inputs(
        self,
        interpolated_states: dict[str, torch.Tensor],
        time_samples: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        other_denoiser_inputs: dict[str, Any],
        postprocess_args: dict[str, Any] = {},
    ) -> dict[str, torch.Tensor | None]:
        """Compute self-conditioning inputs for each modality.

        Makes an initial no-grad denoiser pass with no self-conditioning and converts
        those predictions into inputs for the main denoiser pass.

        Parameters
        ----------
        interpolated_states : dict[str, torch.Tensor]
            Interpolated states per modality, shape (bs, L, d).
        time_samples : dict[str, torch.Tensor]
            Time samples per modality, shape (bs, 1, 1).
        resolved_mask : torch.Tensor
            Shape (bs, L).
        noised_mask : torch.Tensor
            Mask indicating which positions are noised in the prior, shape (bs, L).
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
        postprocess_args : dict[str, Any]
            Additional arguments to the self-conditioning postprocessing function.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Self-conditioning inputs per modality in the shape required by the denoising model.
        """
        model_inputs = self._construct_model_inputs(
            current_states=interpolated_states,
            times=time_samples,
            self_conditioning_inputs={m: None for m in self.modalities},
            resolved_mask=resolved_mask,
            noised_mask=noised_mask,
            other_denoiser_inputs=other_denoiser_inputs,
        )
        with torch.no_grad():
            raw_predictions = self.denoiser(model_inputs)
        return self._convert_raw_predictions_to_sc_inputs(
            raw_predictions, interpolated_states, time_samples, postprocess_args
        )

    def sc_inputs_from_flow_output(
        self, flow_predictions: dict[str, FlowMatcherOutput]
    ) -> dict[str, torch.Tensor | None]:
        """Convert denoise clean-x1 outputs into the next step's self-conditioning inputs.

        The default keeps ``clean_data_point`` shape ``(bs, L, d)``. Subclasses reshape
        when the denoiser expects a different SC layout.
        """
        return {modality: output.clean_data_point.detach() for modality, output in flow_predictions.items()}

    def denoise(
        self,
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        use_self_conditioning: bool,
        other_denoiser_inputs: dict[str, Any],
        postprocess_args: dict[str, Any] = {},
        postprocess_args_sc: dict[str, Any] = {},
        prior_samples: dict[str, torch.Tensor] | None = None,
        cvf_eps: float = 1e-1,
        *,
        self_conditioning_inputs: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[dict[str, FlowMatcherOutput], dict[str, torch.Tensor]]:
        """Predict the conditional vector field and clean data point given current states and times.

        Uses ``self.is_cvf_prediction`` to determine whether the denoiser predicts the conditional
        vector field or the clean data point, then derives the other from that prediction.

        Parameters
        ----------
        current_states : dict[str, torch.Tensor]
            Current state `xt` tensors per modality, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            Time tensors per modality, shape (bs, 1, 1).
        resolved_mask : torch.Tensor
            Shape (bs, L).
        noised_mask : torch.Tensor
            Mask indicating which positions are noised in the prior, shape (bs, L).
        use_self_conditioning : bool
            If False, self-conditioning inputs are set to None for all modalities.
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
        postprocess_args : dict[str, Any]
            Additional arguments forwarded to _postprocess_predictions.
        postprocess_args_sc : dict[str, Any]
            Additional arguments forwarded to _postprocess_predictions for the SC pass.
        prior_samples : dict[str, torch.Tensor] | None
            Prior samples `x0` per modality. If provided and is_cvf_prediction is False,
            used to compute the CVF from x1 predictions for numerical stability wrt t.
        cvf_eps : float
            Denominator clamp for CVF derivation from x1, by default 1e-1.
        self_conditioning_inputs : dict[str, torch.Tensor | None] | None
            Precomputed SC tensors per modality. When provided (including all-None),
            skips the dummy same-timestep pass. When omitted, ``use_self_conditioning``
            selects dummy SC vs all-None.

        Returns
        -------
        predictions : dict[str, FlowMatcherOutput]
        aux_predictions : dict[str, torch.Tensor]
        """
        if self_conditioning_inputs is not None:
            sc_inputs = self_conditioning_inputs
        elif use_self_conditioning:
            sc_inputs = self._compute_self_conditioning_inputs(
                current_states, times, resolved_mask, noised_mask, other_denoiser_inputs, postprocess_args_sc
            )
        else:
            sc_inputs = {m: None for m in self.modalities}

        model_inputs = self._construct_model_inputs(
            current_states=current_states,
            times=times,
            self_conditioning_inputs=sc_inputs,
            resolved_mask=resolved_mask,
            noised_mask=noised_mask,
            other_denoiser_inputs=other_denoiser_inputs,
        )
        raw_predictions = self.denoiser(model_inputs)
        predictions, aux_predictions = self._postprocess_predictions(raw_predictions, **postprocess_args)

        output: dict[str, FlowMatcherOutput] = {}
        for m in self.modalities:
            if self.is_cvf_prediction[m]:
                cvf_pred = predictions[f"out_{m}"]
                x1_pred = self.interpolants[m].apply_conditional_vector_field(
                    xt=current_states[m], ut=cvf_pred, dt=1 - times[m]
                )
            else:
                x1_pred = predictions[f"out_{m}"]
                x0 = prior_samples.get(m) if prior_samples is not None else None
                cvf_pred = self.interpolants[m].conditional_vector_field(
                    xt=current_states[m], x1=x1_pred, t=times[m], x0=x0, eps=cvf_eps
                )
            output[m] = FlowMatcherOutput(conditional_vector_field=cvf_pred, clean_data_point=x1_pred)

        if self.explicit_mask_fixed_positions:
            for m in self.modalities:
                output[m] = self._apply_explicit_mask_fixed_positions(output[m], current_states[m], noised_mask)

        return output, aux_predictions
