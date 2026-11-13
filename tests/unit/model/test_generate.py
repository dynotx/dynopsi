from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from dynopsi.flow_matching.interpolant import EuclideanInterpolant, LinearInterpolantSchedule
from dynopsi.flow_matching.priors import IsotropicGaussian
from dynopsi.flow_matching.protein_module_flow_matcher import ProteinModuleFlowMatcher
from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.modules.heads import DynoPsiOutputHead


def _output_head() -> DynoPsiOutputHead:
    return DynoPsiOutputHead(
        num_atoms=3,
        config_coords_decoder_head={"dim_single": 8, "use_layer_norm": False},
        config_distogram_head={
            "dim_pair": 8,
            "no_bins": 4,
            "use_layer_norm": False,
        },
    )


class _Denoiser(nn.Module):
    def __init__(self, output_head: nn.Module) -> None:
        super().__init__()
        self.output_head = output_head


class _GenerateModel(DynoPsiModel):
    def __init__(self, denoiser: nn.Module) -> None:  # type: ignore[override]
        nn.Module.__init__(self)
        self.flow_matcher = SimpleNamespace(denoiser=denoiser)  # type: ignore[assignment]


def _model_with_head(head: DynoPsiOutputHead) -> _GenerateModel:
    return _GenerateModel(_Denoiser(head))


def _head_forward_batch(seq_len: int = 4, dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "single_rep": torch.randn(1, seq_len, dim),
        "mask": torch.ones(1, seq_len),
        "pair_rep": torch.randn(1, seq_len, seq_len, dim),
    }


class TestSkipDistogramHead:
    def test_removes_attr_and_keeps_coords(self) -> None:
        head = _output_head()
        assert hasattr(head, "distogram_head")
        _model_with_head(head).skip_distogram_head()
        assert not hasattr(head, "distogram_head")

        out = head(_head_forward_batch())
        assert "out_coords" in out
        assert out["out_coords"].shape == (1, 4, 3, 3)
        assert "distogram_prediction" not in out

    def test_idempotent_when_already_absent(self) -> None:
        head = _output_head()
        model = _model_with_head(head)
        model.skip_distogram_head()
        model.skip_distogram_head()
        assert not hasattr(head, "distogram_head")

    def test_postprocess_does_not_require_distogram_prediction(self) -> None:
        head = _output_head()
        denoiser = _Denoiser(head)
        flow = ProteinModuleFlowMatcher(
            priors={"coords": IsotropicGaussian(std=1.0, sample_shape=None)},
            interpolants={"coords": EuclideanInterpolant(interpolant_schedule=LinearInterpolantSchedule())},
            denoiser=denoiser,
            is_cvf_prediction={"coords": False},
        )
        _GenerateModel(flow.denoiser).skip_distogram_head()
        assert not hasattr(head, "distogram_head")

        raw = {"out_coords": torch.zeros(1, 4, 3, 3)}
        flow_predictions, aux = flow._postprocess_predictions(raw)
        assert "out_coords" in flow_predictions
        assert "distogram_prediction" not in aux


class TestCompileDenoiser:
    def test_wraps_denoiser_not_flow_matcher(self) -> None:
        denoiser = _Denoiser(_output_head())
        model = _GenerateModel(denoiser)
        compiled = object()

        with mock.patch("dynopsi.model.dyno_psi_model.torch.compile", return_value=compiled) as compile_mock:
            model.compile_denoiser()

        compile_mock.assert_called_once_with(denoiser)
        assert model.flow_matcher.denoiser is compiled

    def test_skips_when_already_compiled(self) -> None:
        denoiser = _Denoiser(_output_head())
        denoiser._orig_mod = denoiser
        model = _GenerateModel(denoiser)

        with mock.patch("dynopsi.model.dyno_psi_model.torch.compile") as compile_mock:
            model.compile_denoiser()

        compile_mock.assert_not_called()
        assert model.flow_matcher.denoiser is denoiser
