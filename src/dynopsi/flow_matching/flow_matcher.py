from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from dynopsi.flow_matching.interpolant import Interpolant
from dynopsi.flow_matching.priors import Sampleable


@dataclass
class FlowMatcherOutput:
    """Output of the flow matcher (conditional vector field and clean data point per modality).

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
    as well as the associated targets in the same format. Currently only the backbone coordinates modality is supported.

    This class is designed to be subclassed for a particular denoising model by implementing
    `_construct_model_inputs` and `_postprocess_predictions`.

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
        The denoising model
    is_cvf_prediction : dict[str, bool]
        A dictionary of booleans for each modality.
        Keys: modalities. Values: booleans.
        If True, the denoising model is assumed to predict the conditional vector field.
        If False, the denoising model is assumed to predict the clean data point.
    """

    def __init__(
        self,
        priors: dict[str, Sampleable],
        interpolants: dict[str, Interpolant],
        denoiser: nn.Module,
        is_cvf_prediction: dict[str, bool],
    ) -> None:
        super().__init__()

        self.priors = priors
        self.interpolants = interpolants
        assert sorted(self.priors) == sorted(self.interpolants)
        self.modalities = sorted(list(self.priors.keys()))
        self.is_cvf_prediction = is_cvf_prediction
        self.denoiser = denoiser

    def _compute_self_conditioning_inputs(
        self,
        interpolated_states: dict[str, torch.Tensor],
        time_samples: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        other_denoiser_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor | None]:
        """Compute the self-conditioning inputs for each modality.

        Self-conditioning is a technique where the denoising model makes an initial prediction
        and these predictions are used as inputs for the final prediction for the denoising step.

        Parameters
        ----------
        interpolated_states : dict[str, torch.Tensor]
            A dictionary of interpolated states.
            Keys: modalities. Values: interpolated states, shape (bs, L, d).
        time_samples : dict[str, torch.Tensor]
            A dictionary of time samples.
            Keys: modalities. Values: time sample tensors, shape (bs, 1, 1).
        resolved_mask : torch.Tensor
            The resolved mask, shape (bs, L).
        noised_mask : torch.Tensor
            The mask indicating which positions are noised in the prior, shape (bs, L).
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
            Keys: parameter names. Values: inputs.

        Returns
        -------
        dict[str, torch.Tensor | None]
            A dictionary of self-conditioning inputs.
            Keys: modalities. Values: self-conditioning inputs in the shape required by the denoising model.

        """
        # Inputs to the denoising model with no self-conditioning
        model_inputs = self._construct_model_inputs(
            current_states=interpolated_states,
            times=time_samples,
            self_conditioning_inputs={m: None for m in self.modalities},
            resolved_mask=resolved_mask,
            noised_mask=noised_mask,
            other_denoiser_inputs=other_denoiser_inputs,
        )
        # Skip gradients for first pass for efficiency
        with torch.no_grad():
            raw_predictions = self.denoiser(model_inputs)
        self_conditioning_inputs: dict[str, torch.Tensor | None] = self._convert_raw_predictions_to_sc_inputs(
            raw_predictions, interpolated_states, time_samples
        )

        return self_conditioning_inputs

    @abstractmethod
    def _convert_raw_predictions_to_sc_inputs(
        self,
        raw_predictions: dict[str, torch.Tensor],
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        """Convert the raw predictions to self-conditioning inputs.

        This method takes the raw output of the denoising model and converts it to
        a dictionary with standard keys that can be fed back into the denoising model as self-conditioning inputs.

        Parameters
        ----------
        raw_predictions : dict[str, torch.Tensor]
            A dictionary of raw predictions from the denoising model.
        current_states : dict[str, torch.Tensor]
            A dictionary of current states.
            Keys: modalities. Values: current state `xt` tensors, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            A dictionary of times.
            Keys: modalities. Values: times, shape (bs, 1, 1).

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of self-conditioning inputs.
            Keys: modalities. Values: self-conditioning inputs in the shape required by the denoising model.
        """
        raise NotImplementedError("Prediction conversion to self-conditioning inputs must be implemented in subclass.")

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

        This method is designed to convert the standard dictionary key names and tensor shapes to
        the specific model inputs required by the denoising model.

        Parameters
        ----------
        current_states : dict[str, torch.Tensor]
            A dictionary of current states.
            Keys: modalities. Values: current state `xt` tensors, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            A dictionary of times.
            Keys: modalities. Values: times, shape (bs, 1, 1).
        self_conditioning_inputs : dict[str, torch.Tensor | None]
            A dictionary of self-conditioning inputs.
            Keys: modalities. Values: self-conditioning inputs, shape (bs, L, d) or None.
        resolved_mask : torch.Tensor
            The resolved mask, shape (bs, L).
        noised_mask : torch.Tensor
            The mask indicating which positions are noised in the prior, shape (bs, L).
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
            Keys: parameter names.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary of model inputs.
            Keys: parameter names. Values: inputs.
        """
        raise NotImplementedError("Model inputs construction must be implemented in subclass")

    @abstractmethod
    def _postprocess_predictions(
        self, raw_predictions: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Postprocess the predictions from the denoising model.

        This method is designed to convert the custom model output to the standard dictionary key names and tensor shapes
        that are expected by the flow matching model and can be used as self-conditioning inputs. In particular,
        the output should be converted such that the primary predictions are in the format {out_{modality} : torch.Tensor} for each modality.

        Parameters
        ----------
        raw_predictions : dict[str, torch.Tensor]
            A dictionary of raw predictions from the denoising model.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            A tuple containing the primary predictions and auxiliary predictions.
            The primary predictions are in the format {out_{modality} : torch.Tensor} for each modality.
        """
        raise NotImplementedError("Prediction postprocessing must be implemented in subclass.")

    def denoise(
        self,
        current_states: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        resolved_mask: torch.Tensor,
        noised_mask: torch.Tensor,
        use_self_conditioning: bool,
        other_denoiser_inputs: dict[str, Any],
        prior_samples: dict[str, torch.Tensor] | None = None,
        cvf_eps: float = 1e-1,
    ) -> tuple[dict[str, FlowMatcherOutput], dict[str, torch.Tensor]]:
        """Predict the conditional vector field AND the clean data point given current states (xt) and times (t).
        Note that self.is_cvf_prediction is used to determine whether we assume that the denoiser predicts the
        conditional vector field or the clean data point, and then the other is computed from that.

        Parameters
        ----------
        current_states : dict[str, torch.Tensor]
            A dictionary of current states.
            Keys: modalities. Values: current state `xt` tensors, shape (bs, L, d).
        times : dict[str, torch.Tensor]
            A dictionary of times.
            Keys: modalities. Values: times, shape (bs, 1, 1).
        resolved_mask : torch.Tensor
            The resolved mask, shape (bs, L).
        noised_mask : torch.Tensor
            The mask indicating which positions are noised in the prior, shape (bs, L).
        use_self_conditioning : bool
            Whether to use self-conditioning in the prediction.
            If False, the self-conditioning input will be set to None for each modality.
        other_denoiser_inputs : dict[str, Any]
            Additional inputs to the denoising model.
            Keys: parameter names. Values: input tensors or None.
        prior_samples : dict[str, torch.Tensor]
            A dictionary of prior samples.
            If provided, prior samples are used instead of interpolated_states to compute the
            conditional vector field for modalities where is_cvf_prediction is False.
            Keys: modalities. Values: prior samples, shape (bs, L, d).
        cvf_eps : float
            Small value to avoid division by zero in interpolant.conditional_vector_field
            conversion from clean data point (x1) to cvf, by default 1e-1.

        Returns
        -------
        tuple[dict[str, FlowMatcherOutput], dict[str, torch.Tensor]]
            A tuple containing the primary predictions and auxiliary predictions.
            The primary predictions are in the format {out_{modality} : FlowMatcherOutput} for each modality.
        """
        # Compute self-conditioning inputs
        self_conditioning_inputs: dict[str, torch.Tensor | None]
        if use_self_conditioning:
            self_conditioning_inputs = self._compute_self_conditioning_inputs(
                current_states,
                times,
                resolved_mask,
                noised_mask,
                other_denoiser_inputs,
            )
        else:
            self_conditioning_inputs = {m: None for m in self.modalities}

        # Construct the model inputs for the denoising model
        model_inputs = self._construct_model_inputs(
            current_states=current_states,
            times=times,
            self_conditioning_inputs=self_conditioning_inputs,
            resolved_mask=resolved_mask,
            noised_mask=noised_mask,
            other_denoiser_inputs=other_denoiser_inputs,
        )
        raw_predictions = self.denoiser(model_inputs)
        predictions, aux_predictions = self._postprocess_predictions(raw_predictions)

        output: dict[str, FlowMatcherOutput] = {}
        for m in self.modalities:
            if self.is_cvf_prediction[m]:
                cvf_pred = predictions[f"out_{m}"]
                x1_pred = self.interpolants[m].apply_conditional_vector_field(
                    xt=current_states[m], ut=cvf_pred, dt=1 - times[m]
                )
            else:
                x1_pred = predictions[f"out_{m}"]
                x0 = prior_samples.get(m, None) if prior_samples is not None else None
                cvf_pred = self.interpolants[m].conditional_vector_field(
                    xt=current_states[m],
                    x1=x1_pred,
                    t=times[m],
                    x0=x0,
                    eps=cvf_eps,
                )

            output[m] = FlowMatcherOutput(
                conditional_vector_field=cvf_pred,
                clean_data_point=x1_pred,
            )

        return output, aux_predictions
