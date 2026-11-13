from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.model_config import DynoPsiModelConfig, HuggingFaceLoaderConfig
from dynopsi.model.modules.heads import DynoPsiOutputHead
from dynopsi.simulation.simulation_pipeline.ode_simulation import ODESimulation
from dynopsi.simulation.time_sampler import LinearTimeSampler


class _Denoiser(nn.Module):
    def __init__(self, output_head: nn.Module) -> None:
        super().__init__()
        self.output_head = output_head


class _GenerateModel(DynoPsiModel):
    def __init__(self) -> None:  # type: ignore[override]
        nn.Module.__init__(self)
        self.denoiser = _Denoiser(
            DynoPsiOutputHead(
                num_atoms=3,
                config_coords_decoder_head={"dim_single": 8, "use_layer_norm": False},
                config_distogram_head={
                    "dim_pair": 8,
                    "no_bins": 4,
                    "use_layer_norm": False,
                },
            )
        )
        self.flow_matcher = SimpleNamespace(denoiser=self.denoiser)  # type: ignore[assignment]
        self.devices: list[torch.device] = []

    def to_device(self, device: torch.device) -> None:
        # Recording instead of moving keeps the CUDA cases runnable on hosts without an NVIDIA driver.
        self.devices.append(device)


def _make_sim(
    model: DynoPsiModel,
    device_name: str = "cpu",
    *,
    skip_distogram_head: bool = True,
    compile_denoiser: bool = True,
    tf32_matmul: bool = True,
) -> ODESimulation:
    cfg = DynoPsiModelConfig(
        loader=HuggingFaceLoaderConfig(repo_id="fake/repo", filename="fake.pt", revision="0"),
        device_name=device_name,
    )
    cfg._dynopsi_model = model
    return ODESimulation(
        name="test_generate_defaults",
        inference_model_config=cfg,
        time_sampler=LinearTimeSampler(num_steps=3, modalities=("coords",)),
        skip_distogram_head=skip_distogram_head,
        compile_denoiser=compile_denoiser,
        tf32_matmul=tf32_matmul,
    )


class TestSetupModelsGenerateDefaults:
    def test_cpu_drops_distogram_and_does_not_compile(self) -> None:
        model = _GenerateModel()
        assert hasattr(model.denoiser.output_head, "distogram_head")
        sim = _make_sim(model)

        with mock.patch("dynopsi.model.dyno_psi_model.torch.compile") as compile_mock:
            sim.setup_models(torch.device("cpu"))

        assert not hasattr(model.denoiser.output_head, "distogram_head")
        compile_mock.assert_not_called()

    def test_cpu_opt_out_keeps_distogram(self) -> None:
        model = _GenerateModel()
        sim = _make_sim(model, skip_distogram_head=False, compile_denoiser=False)
        sim.setup_models(torch.device("cpu"))
        assert hasattr(model.denoiser.output_head, "distogram_head")

    def test_cuda_sets_tf32_and_compiles_denoiser(self) -> None:
        model = _GenerateModel()
        sim = _make_sim(model, device_name="cuda")
        compiled = object()

        with (
            mock.patch("dynopsi.simulation.simulation.torch.set_float32_matmul_precision") as tf32_mock,
            mock.patch("dynopsi.model.dyno_psi_model.torch.compile", return_value=compiled) as compile_mock,
        ):
            sim.setup_models(torch.device("cuda"))

        assert set(model.devices) == {torch.device("cuda")}
        tf32_mock.assert_called_once_with("high")
        compile_mock.assert_called_once_with(model.denoiser)
        assert model.flow_matcher.denoiser is compiled
        assert not hasattr(model.denoiser.output_head, "distogram_head")

    def test_cuda_tf32_opt_out_still_compiles(self) -> None:
        model = _GenerateModel()
        sim = _make_sim(model, device_name="cuda", tf32_matmul=False)
        compiled = object()

        with (
            mock.patch("dynopsi.simulation.simulation.torch.set_float32_matmul_precision") as tf32_mock,
            mock.patch("dynopsi.model.dyno_psi_model.torch.compile", return_value=compiled) as compile_mock,
        ):
            sim.setup_models(torch.device("cuda"))

        tf32_mock.assert_not_called()
        compile_mock.assert_called_once_with(model.denoiser)
        assert model.flow_matcher.denoiser is compiled

    def test_cuda_compile_opt_out_still_sets_tf32(self) -> None:
        model = _GenerateModel()
        sim = _make_sim(model, device_name="cuda", compile_denoiser=False)
        denoiser = model.denoiser

        with (
            mock.patch("dynopsi.simulation.simulation.torch.set_float32_matmul_precision") as tf32_mock,
            mock.patch("dynopsi.model.dyno_psi_model.torch.compile") as compile_mock,
        ):
            sim.setup_models(torch.device("cuda"))

        tf32_mock.assert_called_once_with("high")
        compile_mock.assert_not_called()
        assert model.flow_matcher.denoiser is denoiser
        assert not hasattr(model.denoiser.output_head, "distogram_head")


class TestGenerateDefaultYamlRoundTrip:
    def test_defaults_on_when_omitted(self) -> None:
        sim = ODESimulation.model_validate(
            {
                "type": "ode_simulation",
                "name": "yaml_defaults",
                "inference_model_config": {
                    "loader": {"type": "huggingface", "repo_id": "fake/repo", "filename": "fake.pt", "revision": "0"}
                },
                "time_sampler": {"type": "linear_time_sampler", "num_steps": 3, "modalities": ["coords"]},
            }
        )
        assert sim.skip_distogram_head is True
        assert sim.compile_denoiser is True
        assert sim.tf32_matmul is True

    def test_opt_out_round_trips(self) -> None:
        payload = {
            "type": "ode_simulation",
            "name": "yaml_opt_out",
            "inference_model_config": {
                "loader": {"type": "huggingface", "repo_id": "fake/repo", "filename": "fake.pt", "revision": "0"}
            },
            "time_sampler": {"type": "linear_time_sampler", "num_steps": 3, "modalities": ["coords"]},
            "skip_distogram_head": False,
            "compile_denoiser": False,
            "tf32_matmul": False,
        }
        sim = ODESimulation.model_validate(payload)
        dumped = sim.model_dump(mode="json", exclude={"plan", "plan_init", "observable"})
        reparsed = ODESimulation.model_validate(dumped)
        assert reparsed.skip_distogram_head is False
        assert reparsed.compile_denoiser is False
        assert reparsed.tf32_matmul is False
        assert dumped["skip_distogram_head"] is False
        assert dumped["compile_denoiser"] is False
        assert dumped["tf32_matmul"] is False
