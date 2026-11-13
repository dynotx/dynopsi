import json
import os
from pathlib import Path
from unittest import mock

import pytest

from dynopsi import cli
from dynopsi.data.conditioning import ConditioningCheckpointError
from dynopsi.data.constants import AtomFilter, DSSPLoopFeature
from dynopsi.data.featurization.inference_featurization_pipeline.binder_design import (
    DynoPsiBinderDesignFeaturizationPipeline,
)
from dynopsi.data.featurization.primitives import PredefinedCenterOfMassEstimator
from dynopsi.design_pipeline import DesignPipeline, OutputSpecification, resolved_output_dir
from dynopsi.model.model_config import DynoPsiModelConfig, HuggingFaceLoaderConfig
from dynopsi.simulation.simulation_pipeline.sde_simulation import SDESimulation
from dynopsi.simulation.time_sampler import LinearTimeSampler
from dynopsi.utils.function_configs import ConstantFunction

NUM_STEPS = 11
_CONSTANT_FN = ConstantFunction(constant=0.5)
_FAKE_PRIOR_STDS = {"coords": 1.0}


@pytest.fixture(autouse=True)
def _patch_prior_stds():
    """Serve fake checkpoint metadata so pipeline tests do not load a checkpoint."""
    with (
        mock.patch(
            "dynopsi.model.model_config.DynoPsiModelConfig.prior_stds",
            new_callable=mock.PropertyMock,
            return_value=_FAKE_PRIOR_STDS,
        ),
        mock.patch.object(
            DynoPsiModelConfig,
            "declared_categorical_keys",
            return_value={"chain_is_loopy"},
        ),
    ):
        yield


def _make_featurization_pipeline(**overrides) -> DynoPsiBinderDesignFeaturizationPipeline:
    kwargs = dict(
        name="feat_test",
        index_type="original_residue_index",
        structure_path="path/to/structure.pdb",
        hotspots={"hs": "A5,A8"},
        new_chains_lengths=[10],
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=((25.0, 40.0, 30.0),))},
        binder_atoms_to_include=AtomFilter.CA,
        design_conditioning={"chain_is_loopy": {"design_is_loopy": DSSPLoopFeature.LOOP_OR_OTHER}},
    )
    kwargs.update(overrides)
    return DynoPsiBinderDesignFeaturizationPipeline(**kwargs)


@pytest.mark.parametrize("prior_centering_strategy", ["origin", "chain_and_noise_separated"])
def test_binder_design_prior_centering_strategy_is_resolved(prior_centering_strategy: str) -> None:
    pipeline = DynoPsiBinderDesignFeaturizationPipeline(
        name="feat_test",
        index_type="original_residue_index",
        structure_path="path/to/structure.pdb",
        hotspots={"hs": "A5,A8"},
        new_chains_lengths=[10],
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=((25.0, 40.0, 30.0),))},
        binder_atoms_to_include=AtomFilter.CA,
        prior_centering_strategy=prior_centering_strategy,
    )

    assert (
        pipeline._center_of_mass_featurizer[0].prior_means_config.prior_centering_strategy == prior_centering_strategy
    )


def _make_simulation_pipeline(**overrides) -> SDESimulation:
    kwargs = dict(
        name="sim_test",
        inference_model_config=DynoPsiModelConfig(
            loader=HuggingFaceLoaderConfig(repo_id="fake/repo", filename="fake.ckpt", revision="0")
        ),
        time_sampler=LinearTimeSampler(num_steps=NUM_STEPS, modalities=["coords"]),
        diffusion_coefficient_fn=_CONSTANT_FN,
        noise_scale_fn=_CONSTANT_FN,
        ode_warmup_score_scale_fn=_CONSTANT_FN,
        ode_cooldown_score_scale_fn=_CONSTANT_FN,
    )
    kwargs.update(overrides)
    return SDESimulation(**kwargs)


def _make_design_pipeline(
    tmp_path,
    featurization_pipeline: DynoPsiBinderDesignFeaturizationPipeline | None = None,
    **sim_overrides,
) -> DesignPipeline:
    return DesignPipeline(
        name="design_test",
        featurization_pipeline=featurization_pipeline or _make_featurization_pipeline(),
        simulation_pipeline=_make_simulation_pipeline(**sim_overrides),
        output_specification=OutputSpecification.model_validate(
            {"output_dir": str(tmp_path), "formats": ["pdb"], "save_trajectory": False}
        ),
    )


def _read_config(save_dir: str) -> dict:
    config_path = os.path.join(save_dir, "resolved_design_config.json")
    with open(config_path) as f:
        return json.load(f)


class TestWriteResolvedConfig:
    def test_file_written_and_valid_json(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)
        pipeline._write_resolved_config(str(tmp_path), num_samples=8, batch_size=4)

        config_path = os.path.join(str(tmp_path), "resolved_design_config.json")
        assert os.path.exists(config_path)
        payload = _read_config(str(tmp_path))
        assert payload["num_samples"] == 8
        assert payload["batch_size"] == 4
        assert payload["design_pipeline"]["name"] == "design_test"

    def test_records_effective_sde_params(self, tmp_path) -> None:
        # use_legacy_cvf_score_conversion is left at its default.
        pipeline = _make_design_pipeline(tmp_path, ode_warmup_max_t=0.2)
        pipeline._write_resolved_config(str(tmp_path), num_samples=1, batch_size=1)

        sim = _read_config(str(tmp_path))["design_pipeline"]["simulation_pipeline"]
        assert sim["ode_warmup_max_t"] == 0.2
        assert sim["use_legacy_cvf_score_conversion"] is True

    def test_excludes_runtime_and_derived_fields(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)
        pipeline._write_resolved_config(str(tmp_path), num_samples=1, batch_size=1)

        sim = _read_config(str(tmp_path))["design_pipeline"]["simulation_pipeline"]
        assert "plan" not in sim
        assert "plan_init" not in sim
        assert "observable" not in sim

    def test_roundtrips_for_checkpoint_loader(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path, ode_warmup_max_t=0.15)
        pipeline._write_resolved_config(str(tmp_path), num_samples=2, batch_size=2)
        dumped = _read_config(str(tmp_path))["design_pipeline"]

        roundtrip_dir = tmp_path / "roundtrip"
        roundtrip_dir.mkdir()
        reparsed = DesignPipeline.model_validate(dumped)
        reparsed._write_resolved_config(str(roundtrip_dir), num_samples=2, batch_size=2)
        redumped = _read_config(str(roundtrip_dir))["design_pipeline"]

        assert redumped == dumped

    def test_checkpoint_short_name_survives_roundtrip(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path, checkpoint="v1")
        pipeline._write_resolved_config(str(tmp_path), num_samples=2, batch_size=2)
        dumped = _read_config(str(tmp_path))["design_pipeline"]
        assert dumped["simulation_pipeline"]["checkpoint"] == "v1"

        roundtrip_dir = tmp_path / "roundtrip"
        roundtrip_dir.mkdir()
        reparsed = DesignPipeline.model_validate(dumped)
        assert reparsed.simulation_pipeline.checkpoint == "v1"
        reparsed._write_resolved_config(str(roundtrip_dir), num_samples=2, batch_size=2)
        redumped = _read_config(str(roundtrip_dir))["design_pipeline"]
        assert redumped == dumped

    def test_no_model_load_side_effect(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)
        with mock.patch(
            "dynopsi.model.model_config.DynoPsiModelConfig._ensure_loaded",
            side_effect=AssertionError("serialization must not trigger a model load"),
        ):
            pipeline._write_resolved_config(str(tmp_path), num_samples=1, batch_size=1)
        assert os.path.exists(os.path.join(str(tmp_path), "resolved_design_config.json"))


class TestSampleWritesResolvedConfig:
    def test_rejects_conditioning_mismatch_before_creating_outputs(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)

        with (
            mock.patch.object(DynoPsiModelConfig, "declared_categorical_keys", return_value=set()),
            pytest.raises(ConditioningCheckpointError, match="design_conditioning.chain_is_loopy"),
        ):
            pipeline.sample(num_samples=1, batch_size=1)

        assert not Path(pipeline.simulation_output_dir()).exists()

    def test_config_written_before_featurization_loop(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)
        simulation_save_dir = os.path.join(str(tmp_path), pipeline.name, pipeline.simulation_pipeline.name)

        with (
            mock.patch("dynopsi.design_pipeline.torch.cuda.device_count", return_value=1),
            mock.patch.object(
                type(pipeline.featurization_pipeline),
                "__call__",
                side_effect=RuntimeError("featurization boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="featurization boom"):
                pipeline.sample(num_samples=4, batch_size=2)

        # The sidecar is written before the loop, so it exists even though the loop body raised.
        payload = _read_config(simulation_save_dir)
        assert payload["num_samples"] == 4
        assert payload["batch_size"] == 2

    def test_records_inferred_batch_size(self, tmp_path) -> None:
        pipeline = _make_design_pipeline(tmp_path)
        simulation_save_dir = os.path.join(str(tmp_path), pipeline.name, pipeline.simulation_pipeline.name)

        with (
            mock.patch("dynopsi.design_pipeline.torch.cuda.device_count", return_value=1),
            mock.patch("dynopsi.design_pipeline.infer_batch_size", return_value=7),
            mock.patch.object(
                type(pipeline.featurization_pipeline),
                "__call__",
                side_effect=RuntimeError("featurization boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="featurization boom"):
                pipeline.sample(num_samples=4, batch_size=None)

        assert _read_config(simulation_save_dir)["batch_size"] == 7


class TestSampleDispatchesPriorComDraw:
    def test_sample_uses_featurize_for_sampling_seam(self, tmp_path) -> None:
        featurization = _make_featurization_pipeline(
            prior_com_translation_std_angstrom=2.0,
            prior_com_translation_seed=3,
        )
        pipeline = _make_design_pipeline(tmp_path, featurization_pipeline=featurization)
        simulation_save_dir = os.path.join(str(tmp_path), pipeline.name, pipeline.simulation_pipeline.name)

        with (
            mock.patch("dynopsi.design_pipeline.torch.cuda.device_count", return_value=1),
            mock.patch.object(
                type(featurization),
                "featurize_for_sampling",
                side_effect=RuntimeError("draw boom"),
            ) as drawn,
        ):
            with pytest.raises(RuntimeError, match="draw boom"):
                pipeline.sample(num_samples=4, batch_size=2)

        drawn.assert_called_once()
        assert drawn.call_args.kwargs["num_samples"] == 4

        featurization_config = _read_config(simulation_save_dir)["design_pipeline"]["featurization_pipeline"]
        assert featurization_config["prior_com_translation_std_angstrom"] == 2.0
        assert featurization_config["prior_com_translation_seed"] == 3
        assert featurization_config["sample_centers_of_mass"] is None

    def test_sigma_zero_delegates_to_scalar_call(self, tmp_path) -> None:
        # The base seam always runs; with sigma == 0 it delegates to the scalar __call__ path.
        pipeline = _make_design_pipeline(tmp_path)

        with (
            mock.patch("dynopsi.design_pipeline.torch.cuda.device_count", return_value=1),
            mock.patch.object(
                type(pipeline.featurization_pipeline),
                "__call__",
                side_effect=RuntimeError("scalar featurization boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="scalar featurization boom"):
                pipeline.sample(num_samples=4, batch_size=2)


def test_valid_nested_output_path(tmp_path: Path) -> None:
    resolved = resolved_output_dir(str(tmp_path), "pipeline", "simulation")
    assert resolved == str((tmp_path / "pipeline" / "simulation").resolve())


@pytest.mark.parametrize("component", ["..", ".", "/tmp/elsewhere", "nested/name"])
def test_rejects_path_components_that_leave_the_output_root(tmp_path: Path, component: str) -> None:
    with pytest.raises(ValueError, match="single name"):
        resolved_output_dir(str(tmp_path), component)


def test_symlink_component_cannot_leave_the_output_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(outside)
    with pytest.raises(ValueError, match="must stay inside"):
        resolved_output_dir(str(root), "escape")


def test_sample_overwrite_removes_the_simulation_output_dir(tmp_path: Path) -> None:
    pipeline = _make_design_pipeline(tmp_path)
    target = Path(pipeline.simulation_output_dir())
    target.mkdir(parents=True)
    (target / "old.txt").write_text("stale")
    config = tmp_path / "design.yaml"
    config.write_text("dummy: true")

    with (
        mock.patch("dynopsi.cli._load_design_pipelines", return_value=[pipeline]),
        mock.patch.object(DesignPipeline, "sample", return_value=[]),
    ):
        cli._cmd_sample(config_path=str(config), num_samples=1, batch_size=1, overwrite=True)

    assert not target.exists()
    assert pipeline.simulation_output_dir() == str((tmp_path / "design_test" / "sim_test").resolve())
    assert pipeline.verification_output_dir() == str((tmp_path / "design_test" / "verify_configuration").resolve())
