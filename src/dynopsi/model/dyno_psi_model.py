from __future__ import annotations

from typing import Any

import torch
from torch import nn

from dynopsi.flow_matching.interpolant import (
    EuclideanInterpolant,
    Interpolant,
    LinearInterpolantSchedule,
)
from dynopsi.flow_matching.priors import DirichletPrior, HarmonicPrior, IsotropicGaussian, Sampleable
from dynopsi.flow_matching.protein_module_flow_matcher import ProteinModuleFlowMatcher
from dynopsi.model.modules.embedders import (
    DynoPsiSingleConditionalEmbedder,
    DynoPsiSingleConditionalPairEmbedder,
)
from dynopsi.model.modules.heads import DynoPsiOutputHead
from dynopsi.model.modules.protein_module import ProteinModule
from dynopsi.model.modules.trunks import DynoPsiTransformerTrunk

_REQUIRED_FEAT_KEYS = ("frame_resolved_mask", "noised_mask", "token_index")


def _validate_inference_inputs(
    feat: dict[str, torch.Tensor],
    x_t: dict[str, torch.Tensor],
    modalities: list[str],
) -> None:
    missing_feat = [key for key in _REQUIRED_FEAT_KEYS if key not in feat]
    if missing_feat:
        raise KeyError(f"DynoPsiModel requires feat to contain {_REQUIRED_FEAT_KEYS}; missing: {missing_feat}.")
    missing_mod = [m for m in modalities if m not in x_t]
    if missing_mod:
        raise KeyError(f"x_t missing state for modalities {missing_mod!r}.")


class DynoPsiModel(nn.Module):
    """DynoPsi flow-matching model for inference.

    Parameters
    ----------
    config_priors : dict[str, dict[str, Any]]
        Prior configuration per modality. Each value has ``type`` and ``params``.
    config_interpolants : dict[str, dict[str, Any]]
        Interpolant configuration per modality.
    config_denoiser : dict[str, Any]
        Denoiser configuration. Must have keys ``embedder``, ``trunk``, and ``output_head``.
    num_atoms : int
        Number of atoms per residue (1 for CA-only, 3 for backbone).
    is_cvf_prediction : dict[str, bool]
        Whether each modality's denoiser predicts the conditional vector field.
    cvf_eps : float
        Small value used when converting predictions so denominators are never zero.
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

        self.num_atoms = num_atoms
        self.use_self_cond = config_denoiser["params"]["embedder"]["params"]["use_self_cond"]
        self.cvf_eps = cvf_eps

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
        Psi models support only the 'coords' modality.
        """
        return list(self.flow_matcher.modalities)

    def infer_cvf(
        self,
        x_t: dict[str, torch.Tensor],
        feat: dict[str, torch.Tensor],
        times: dict[str, torch.Tensor],
        self_conditioning_inputs: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """Compute CVF and the next-step SC tensors for the Infer op.

        Parameters
        ----------
        x_t : dict[str, torch.Tensor]
            Current state per modality, shape (bs, L, d).
        feat : dict[str, torch.Tensor]
            Must include 'frame_resolved_mask', 'noised_mask', 'token_index'.
        times : dict[str, torch.Tensor]
            Shape (bs, 1, 1) per modality.
        self_conditioning_inputs : dict[str, torch.Tensor | None] | None
            Precomputed SC tensors. ``None`` runs the dummy same-timestep pass when
            ``use_self_cond`` is set. A dict (including all-None) skips that pass.
        """
        _validate_inference_inputs(feat, x_t, self.modalities)
        current_states = {m: x_t[m] for m in self.modalities}
        other_denoiser_inputs = feat.copy()
        noised_mask = other_denoiser_inputs.pop("noised_mask")
        resolved_mask = other_denoiser_inputs.pop("frame_resolved_mask")
        # The embedder reads ``indices``; featurization stores that tensor as ``token_index``.
        other_denoiser_inputs["indices"] = other_denoiser_inputs["token_index"]

        with torch.no_grad():
            flow_predictions, _ = self.flow_matcher.denoise(
                current_states=current_states,
                times=times,
                resolved_mask=resolved_mask,
                noised_mask=noised_mask,
                use_self_conditioning=self.use_self_cond,
                other_denoiser_inputs=other_denoiser_inputs,
                cvf_eps=self.cvf_eps,
                self_conditioning_inputs=self_conditioning_inputs,
            )
        cvf = {m: flow_predictions[m].conditional_vector_field for m in self.modalities}
        return cvf, self.flow_matcher.sc_inputs_from_flow_output(flow_predictions)

    def to_device(self, device: torch.device) -> None:
        self.flow_matcher.denoiser.to(device)

    def skip_distogram_head(self) -> None:
        """Remove ``distogram_head`` so generate skips it and postprocess omits ``distogram_prediction``."""
        head = self.flow_matcher.denoiser.output_head
        if hasattr(head, "distogram_head"):
            del head.distogram_head

    def compile_denoiser(self) -> None:
        """Compile the denoiser that sampling calls via ``flow_matcher.denoise``."""
        if not hasattr(self.flow_matcher.denoiser, "_orig_mod"):
            self.flow_matcher.denoiser = torch.compile(self.flow_matcher.denoiser)  # type: ignore[assignment]

    def compile_model(self) -> DynoPsiModel:
        """Compile the flow-matcher module in-place and return ``self``."""
        if not hasattr(self.flow_matcher, "_orig_mod"):
            self.flow_matcher = torch.compile(self.flow_matcher)  # type: ignore[assignment]
        return self

    def _build_priors(self, config_priors: dict[str, dict[str, Any]]) -> dict[str, Sampleable]:
        priors: dict[str, Sampleable] = {}
        for m, cfg in config_priors.items():
            t = cfg["type"]
            if t == "isotropic_gaussian":
                priors[m] = IsotropicGaussian(**cfg["params"])
            elif t == "harmonic_prior":
                priors[m] = HarmonicPrior(**cfg["params"])
            elif t == "dirichlet":
                priors[m] = DirichletPrior(**cfg["params"])
            else:
                raise ValueError(f"Unsupported prior type: {t!r}")
        return priors

    def _build_interpolants(self, config_interpolants: dict[str, dict[str, Any]]) -> dict[str, Interpolant]:
        interpolants: dict[str, Interpolant] = {}
        for m, cfg in config_interpolants.items():
            if cfg["type"] == "euclidean_interpolant":
                sched_cfg = cfg["params"]["interpolant_schedule"]
                assert sched_cfg.get("type", "") == "linear_interpolant_schedule"
                interpolants[m] = EuclideanInterpolant(
                    interpolant_schedule=LinearInterpolantSchedule(**sched_cfg["params"])
                )
            else:
                raise ValueError(f"Unsupported interpolant type: {cfg['type']!r}")
        return interpolants

    def _build_embedder(self, config_embedder: dict[str, Any], num_time_embeddings: int) -> nn.Module:
        t = config_embedder["type"]
        if t == "dyno_psi_single_cond_embedder":
            return DynoPsiSingleConditionalEmbedder(
                **config_embedder["params"], num_atoms=self.num_atoms, num_time_embeddings=num_time_embeddings
            )
        elif t == "dyno_psi_single_cond_pair_embedder":
            return DynoPsiSingleConditionalPairEmbedder(
                **config_embedder["params"], num_atoms=self.num_atoms, num_time_embeddings=num_time_embeddings
            )
        else:
            raise ValueError(f"Unsupported embedder type: {t!r}")

    def _build_trunk(self, config_trunk: dict[str, Any]) -> nn.Module:
        if config_trunk["type"] == "dyno_psi_transformer_trunk":
            return DynoPsiTransformerTrunk(**config_trunk["params"])
        raise ValueError(f"Unsupported trunk type: {config_trunk['type']!r}")

    def _build_output_head(self, config_output_head: dict[str, Any]) -> nn.Module:
        if config_output_head["type"] == "dyno_psi_output_head":
            return DynoPsiOutputHead(**config_output_head["params"], num_atoms=self.num_atoms)
        raise ValueError(f"Unsupported output_head type: {config_output_head['type']!r}")

    def _build_denoiser(self, config_denoiser: dict[str, Any], num_time_embeddings: int) -> nn.Module:
        assert config_denoiser["type"] == "ProteinModule", "Only 'ProteinModule' is supported as a denoiser"
        p = config_denoiser["params"]
        embedder = self._build_embedder(p["embedder"], num_time_embeddings=num_time_embeddings)
        trunk = self._build_trunk(p["trunk"])
        output_head = self._build_output_head(p["output_head"])
        return ProteinModule(embedder=embedder, main_trunk=trunk, output_head=output_head)
