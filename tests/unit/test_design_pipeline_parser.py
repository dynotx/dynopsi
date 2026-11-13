import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from dynopsi.data.constants import DSSPLoopFeature, HomooligomerPartnerFeature
from dynopsi.data.featurization import FixedHotspotOffsetEstimator
from dynopsi.data.featurization.inference_featurization_pipeline import (
    get_featurization_pipeline_from_config,
)
from dynopsi.design_pipeline import OutputSpecification
from dynopsi.design_pipeline_parser import DesignLanguageParser
from dynopsi.model.loaders import CheckpointSelectionError
from dynopsi.model.model_config import DEFAULT_DYNOPSI_MODEL_CONFIG, DynoPsiModelConfig, HuggingFaceLoaderConfig

_EXAMPLE_CONFIG_DIR = Path(__file__).resolve().parents[2] / "example" / "configurations"
EXAMPLE_DESIGNED_OUTPUT_FORMATS: tuple[str, ...] = ("mmcif", "npz")


def make_parser_with_mocks(mock_feat, mock_sim, mock_out, mock_design):
    """Return a parser with all external dependencies patched."""
    patches = [
        patch(
            "dynopsi.design_pipeline_parser.OutputSpecification.model_validate",
            return_value=mock_out,
        ),
        patch(
            "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
            return_value=mock_sim,
        ),
        patch(
            "dynopsi.design_pipeline_parser.DesignPipeline",
            return_value=mock_design,
        ),
        patch(
            "dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config",
            return_value=mock_feat,
        ),
    ]
    return patches


class TestParse:
    """Test the parse method of the DesignLanguageParser class."""

    def test_loads_yaml_file(self, minimal_design_dict, tmp_path):
        yaml_file = tmp_path / "design.yaml"
        yaml_file.write_text(yaml.dump(minimal_design_dict))

        with patch.object(DesignLanguageParser, "parse_design_dict", return_value=[]) as mock_pdd:
            DesignLanguageParser().parse(str(yaml_file))
            mock_pdd.assert_called_once_with(minimal_design_dict, inference_model_config=None, checkpoint=None)

    def test_raises_on_unsupported_extension(self, tmp_path):
        parser = DesignLanguageParser()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parser.parse(str(tmp_path / "design.json"))

    def test_raises_on_missing_file(self):
        parser = DesignLanguageParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/design.yaml")

    def test_returns_design_pipelines_from_yaml(self, minimal_design_dict, tmp_path):
        yaml_file = tmp_path / "design.yaml"
        yaml_file.write_text(yaml.dump(minimal_design_dict))

        mock_dp = MagicMock()
        with patch.object(DesignLanguageParser, "parse_design_dict", return_value=[mock_dp]):
            parser = DesignLanguageParser()
            result = parser.parse(str(yaml_file))
        assert result == [mock_dp]


class TestParseDesignDict:
    """Test the parse_design_dict method of the DesignLanguageParser class."""

    def test_returns_one_design_pipeline(
        self,
        minimal_design_dict,
        mock_featurization_pipeline,
        mock_simulation_pipeline,
        mock_output_specification,
        mock_design_pipeline,
    ):
        patches = make_parser_with_mocks(
            mock_featurization_pipeline,
            mock_simulation_pipeline,
            mock_output_specification,
            mock_design_pipeline,
        )
        with patches[0], patches[1], patches[2], patches[3]:
            result = DesignLanguageParser().parse_design_dict(minimal_design_dict)
        assert len(result) == 1
        assert result[0] is mock_design_pipeline

    def test_returns_multiple_design_pipelines(
        self,
        multi_pipeline_design_dict,
        mock_featurization_pipeline,
        mock_simulation_pipeline,
        mock_output_specification,
        mock_design_pipeline,
    ):
        patches = make_parser_with_mocks(
            mock_featurization_pipeline,
            mock_simulation_pipeline,
            mock_output_specification,
            mock_design_pipeline,
        )
        with patches[0], patches[1], patches[2], patches[3]:
            result = DesignLanguageParser().parse_design_dict(multi_pipeline_design_dict)
        assert len(result) == 2

    def test_featurization_pipeline_name_injected(self, minimal_design_dict):
        """Ensures the pipeline name key is added before validate_python is called."""
        captured = {}

        def capture_feat(d):
            captured.update(d)
            return MagicMock()

        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", side_effect=capture_feat),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.DesignPipeline", return_value=MagicMock()),
        ):
            DesignLanguageParser().parse_design_dict(minimal_design_dict)

        assert captured.get("name") == "feat_a"

    def test_simulation_pipeline_gets_default_model_config(self, minimal_design_dict):
        """Ensures DEFAULT_DYNOPSI_MODEL_CONFIG is injected into simulation pipeline dicts."""
        captured = {}

        def capture_sim(sim_dict):
            captured.update(sim_dict)
            return MagicMock()

        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                side_effect=capture_sim,
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline", return_value=MagicMock()),
        ):
            DesignLanguageParser().parse_design_dict(minimal_design_dict)

        assert "inference_model_config" in captured
        assert captured.get("name") == "sim_a"

    def test_simulation_pipeline_uses_provided_inference_model_config(self, minimal_design_dict):
        custom_config = DynoPsiModelConfig(
            loader=HuggingFaceLoaderConfig(repo_id="my/custom-model", filename="custom.ckpt", revision="0")
        )
        captured = {}

        def capture_sim(sim_dict):
            captured.update(sim_dict)
            return MagicMock()

        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                side_effect=capture_sim,
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline", return_value=MagicMock()),
        ):
            DesignLanguageParser().parse_design_dict(minimal_design_dict, inference_model_config=custom_config)

        assert captured.get("inference_model_config") == custom_config

    def test_design_pipeline_wired_with_correct_components(
        self,
        minimal_design_dict,
        mock_featurization_pipeline,
        mock_simulation_pipeline,
        mock_output_specification,
    ):
        """DesignPipeline constructor receives the correct resolved objects."""
        with (
            patch(
                "dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config",
                return_value=mock_featurization_pipeline,
            ),
            patch(
                "dynopsi.design_pipeline_parser.OutputSpecification.model_validate",
                return_value=mock_output_specification,
            ),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=mock_simulation_pipeline,
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline") as MockDesignPipeline,
        ):
            DesignLanguageParser().parse_design_dict(minimal_design_dict)

        MockDesignPipeline.assert_called_once_with(
            name="design_a",
            featurization_pipeline=mock_featurization_pipeline,
            simulation_pipeline=mock_simulation_pipeline,
            output_specification=mock_output_specification,
        )

    def test_unknown_featurization_pipeline_ref_raises(self, minimal_design_dict):
        minimal_design_dict["design_pipelines"]["design_a"]["featurization_pipeline"] = "nonexistent"
        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
        ):
            with pytest.raises(KeyError):
                DesignLanguageParser().parse_design_dict(minimal_design_dict)

    def test_unknown_simulation_pipeline_ref_raises(self, minimal_design_dict):
        minimal_design_dict["design_pipelines"]["design_a"]["simulation_pipeline"] = "nonexistent"
        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
        ):
            with pytest.raises(KeyError):
                DesignLanguageParser().parse_design_dict(minimal_design_dict)

    def test_empty_design_pipelines_returns_empty_list(self, minimal_design_dict):
        minimal_design_dict["design_pipelines"] = {}
        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
        ):
            result = DesignLanguageParser().parse_design_dict(minimal_design_dict)
        assert result == []

    def test_missing_top_level_key_raises(self):
        with pytest.raises((KeyError, TypeError)):
            DesignLanguageParser().parse_design_dict({"featurization_pipelines": {}})


class TestCheckpointSelection:
    def _parse_capturing_sim(self, design_dict, **kwargs):
        captured: dict = {}

        def capture_sim(sim_dict):
            captured.update(sim_dict)
            return MagicMock()

        with (
            patch("dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config", side_effect=capture_sim),
            patch("dynopsi.design_pipeline_parser.DesignPipeline", return_value=MagicMock()),
        ):
            DesignLanguageParser().parse_design_dict(design_dict, **kwargs)
        return captured

    def test_rejects_inference_model_config_block(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["inference_model_config"] = {
            "loader": {"type": "huggingface", "repo_id": "x/y", "filename": "z.ckpt"}
        }
        with pytest.raises(CheckpointSelectionError, match="checkpoint:"):
            self._parse_capturing_sim(minimal_design_dict)

    def test_non_string_config_checkpoint_refused(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = 12
        with pytest.raises(CheckpointSelectionError, match="must be a string"):
            self._parse_capturing_sim(minimal_design_dict)

    def test_unknown_checkpoint_name_refused(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = "v99"
        with pytest.raises(CheckpointSelectionError, match="Unknown checkpoint"):
            self._parse_capturing_sim(minimal_design_dict)

    def test_flag_alone_selects_short_name(self, minimal_design_dict):
        captured = self._parse_capturing_sim(minimal_design_dict, checkpoint="v1")
        config = captured["inference_model_config"]
        assert config.loader.filename == "dynopsi-1.ckpt"
        assert config.loader.revision == "346e377be69ca2a8a78e6f921c08e2f3aa7b6e18"
        assert captured["checkpoint"] == "v1"

    def test_flag_and_config_disagree_refused(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = "v1"
        with pytest.raises(CheckpointSelectionError, match="different weights"):
            self._parse_capturing_sim(minimal_design_dict, checkpoint="other")

    def test_flag_and_config_agree_ok(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = "v1"
        captured = self._parse_capturing_sim(minimal_design_dict, checkpoint="v1")
        assert captured["inference_model_config"].loader.filename == "dynopsi-1.ckpt"

    def test_config_checkpoint_with_python_config_refused(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = "v1"
        custom = DynoPsiModelConfig(loader=HuggingFaceLoaderConfig(repo_id="my/repo", filename="c.ckpt", revision="0"))
        with pytest.raises(CheckpointSelectionError, match="passed from Python"):
            self._parse_capturing_sim(minimal_design_dict, inference_model_config=custom)

    def test_flag_and_python_config_refused(self, minimal_design_dict):
        custom = DynoPsiModelConfig(loader=HuggingFaceLoaderConfig(repo_id="my/repo", filename="c.ckpt", revision="0"))
        with pytest.raises(CheckpointSelectionError, match="provide only one"):
            self._parse_capturing_sim(minimal_design_dict, checkpoint="v1", inference_model_config=custom)

    def test_omitted_checkpoint_keeps_default(self, minimal_design_dict):
        captured = self._parse_capturing_sim(minimal_design_dict)
        assert captured["inference_model_config"].loader.filename == DEFAULT_DYNOPSI_MODEL_CONFIG.loader.filename
        assert captured["inference_model_config"].loader.revision == DEFAULT_DYNOPSI_MODEL_CONFIG.loader.revision
        assert captured.get("checkpoint") is None

    def test_reused_dict_parses_identically(self, minimal_design_dict):
        minimal_design_dict["simulation_pipelines"]["sim_a"]["checkpoint"] = "v1"
        first = self._parse_capturing_sim(minimal_design_dict)
        assert "inference_model_config" not in minimal_design_dict["simulation_pipelines"]["sim_a"]
        second = self._parse_capturing_sim(minimal_design_dict)
        assert first["inference_model_config"] == second["inference_model_config"]
        assert first["checkpoint"] == second["checkpoint"] == "v1"

    def test_flag_forwarded_to_parse_design_dict_as_raw_name(self, minimal_design_dict, tmp_path):
        yaml_file = tmp_path / "design.yaml"
        yaml_file.write_text(yaml.dump(minimal_design_dict))
        with patch.object(DesignLanguageParser, "parse_design_dict", return_value=[]) as mock_pdd:
            DesignLanguageParser().parse(str(yaml_file), checkpoint="v1")
        mock_pdd.assert_called_once_with(minimal_design_dict, inference_model_config=None, checkpoint="v1")


@pytest.fixture
def binder_design_feat_dict():
    """Featurization pipeline dict matching a binder-design YAML structure."""
    return {
        "type": "dynopsi_binder_design_featurization_pipeline",
        "index_type": "original_residue_index",
        "structure_path": "path/to/structure.pdb",
        "hotspots": {"hs": "A5,A8"},
        "new_chains_lengths": [10],
        "new_chains_centers_of_mass": {
            "center": {
                "type": "predefined_center_of_mass",
                "centers_of_mass": [[25.0, 40.0, 30.0]],
            }
        },
        "binder_atoms_to_include": "ca",
    }


class TestBinderDesignParserPath:
    """Parser-path tests for binder-design YAML configs.

    These tests exercise DesignLanguageParser default injection and registry
    deserialization, complementing the direct-construction tests in
    test_design_pipeline_resolved_config.py.
    """

    def test_registry_deserializes_sphere_exclusion_center(self, binder_design_feat_dict):
        binder_design_feat_dict["name"] = "test_binder_design"
        binder_design_feat_dict["new_chains_centers_of_mass"] = {"sphere": {"type": "sphere_exclusion"}}
        pipeline = get_featurization_pipeline_from_config(binder_design_feat_dict)
        estimator = next(iter(pipeline.new_chains_centers_of_mass.values()))
        assert estimator.type == "sphere_exclusion"
        assert estimator.exclusion_offset == 0.0
        assert estimator.exclusion_radius is None

    def test_registry_deserializes_fixed_hotspot_offset(self, binder_design_feat_dict):
        binder_design_feat_dict["name"] = "test_binder_design"
        binder_design_feat_dict["new_chains_centers_of_mass"] = {
            "fixed_hotspot_offset": {"type": "fixed_hotspot_offset"}
        }
        pipeline = get_featurization_pipeline_from_config(binder_design_feat_dict)
        estimator = next(iter(pipeline.new_chains_centers_of_mass.values()))
        assert isinstance(estimator, FixedHotspotOffsetEstimator)
        assert estimator.type == "fixed_hotspot_offset"

    def test_registry_deserializes_sphere_exclusion_knobs(self, binder_design_feat_dict):
        binder_design_feat_dict["name"] = "test_binder_design"
        binder_design_feat_dict["new_chains_centers_of_mass"] = {
            "sphere": {"type": "sphere_exclusion", "exclusion_offset": -3.0, "exclusion_radius": 5.0}
        }
        pipeline = get_featurization_pipeline_from_config(binder_design_feat_dict)
        estimator = next(iter(pipeline.new_chains_centers_of_mass.values()))
        assert estimator.exclusion_offset == -3.0
        assert estimator.exclusion_radius == 5.0

    def test_registry_deserializes_origin_prior(self, binder_design_feat_dict):
        binder_design_feat_dict.update(
            name="test_binder_design",
            prior_centering_strategy="origin",
        )

        pipeline = get_featurization_pipeline_from_config(binder_design_feat_dict)

        assert pipeline.prior_centering_strategy == "origin"

    def test_yaml_file_keeps_published_il7ra_defaults(self, tmp_path):
        """The published il7ra example keeps the multimer binder-design defaults."""
        design_dict = yaml.safe_load((_EXAMPLE_CONFIG_DIR / "il7ra_3di3_config.yaml").read_text())

        with (
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline") as MockDesignPipeline,
        ):
            MockDesignPipeline.return_value = MagicMock()
            DesignLanguageParser().parse_design_dict(design_dict)

        feat_pipeline = MockDesignPipeline.call_args.kwargs["featurization_pipeline"]
        assert feat_pipeline.design_task == "MULTIMER_BINDER_DESIGN"
        assert feat_pipeline.prior_centering_strategy == "chain_and_noise_separated"

    def test_parser_path_preserves_binder_design_noise_all(self, binder_design_feat_dict):
        binder_design_feat_dict["name"] = "test_binder_design"
        binder_design_feat_dict["design_task"] = "MULTIMER_BINDER_DESIGN_NOISE_ALL"
        design_dict = {
            "featurization_pipelines": {"feat": binder_design_feat_dict},
            "output_specifications": {"out": {"output_dir": "./out", "formats": ["pdb"], "save_trajectory": False}},
            "simulation_pipelines": {"sim": {"type": "sde_simulation"}},
            "design_pipelines": {
                "design": {
                    "featurization_pipeline": "feat",
                    "simulation_pipeline": "sim",
                    "output_specification": "out",
                }
            },
        }
        with (
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline") as MockDesignPipeline,
        ):
            MockDesignPipeline.return_value = MagicMock()
            DesignLanguageParser().parse_design_dict(design_dict)

        feat_pipeline = MockDesignPipeline.call_args.kwargs["featurization_pipeline"]
        assert feat_pipeline.design_task == "MULTIMER_BINDER_DESIGN_NOISE_ALL"


class TestDefaultDesignedFormats:
    def test_example_designed_output_formats_constant_is_mmcif_npz(self) -> None:
        assert EXAMPLE_DESIGNED_OUTPUT_FORMATS == ("mmcif", "npz")
        assert "cif" not in EXAMPLE_DESIGNED_OUTPUT_FORMATS

    def test_example_yaml_simple_style_formats_are_mmcif_npz(self) -> None:
        yaml_paths = sorted(_EXAMPLE_CONFIG_DIR.glob("*.yaml"))
        assert yaml_paths
        for yaml_path in yaml_paths:
            data = yaml.safe_load(yaml_path.read_text())
            for spec_data in data["output_specifications"].values():
                spec = OutputSpecification.model_validate(spec_data)
                formats = spec.observable_output_configs["X_T"].outputs["final"]
                assert list(formats) == list(EXAMPLE_DESIGNED_OUTPUT_FORMATS), yaml_path
                assert "mmcif" in formats
                assert "cif" not in formats
                assert "pdb" not in formats

    def test_simple_style_formats_mmcif_npz_parse(self, tmp_path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["mmcif", "npz"],
                "save_trajectory": False,
            }
        )
        assert list(spec.observable_output_configs["X_T"].outputs["final"]) == ["mmcif", "npz"]

    def test_explicit_formats_pdb_still_parses(self, tmp_path) -> None:
        spec = OutputSpecification.model_validate(
            {"output_dir": str(tmp_path), "formats": ["pdb"], "save_trajectory": False}
        )
        assert spec.observable_output_configs["X_T"].outputs["final"] == ("pdb",)

    def test_cif_token_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="mmcif"):
            OutputSpecification.model_validate(
                {"output_dir": str(tmp_path), "formats": ["cif"], "save_trajectory": False}
            )

    def test_omitted_formats_does_not_inject_default(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="formats and save_trajectory must both be present"):
            OutputSpecification.model_validate({"output_dir": str(tmp_path), "save_trajectory": False})

    def test_parser_path_example_yaml_threads_mmcif_npz(self) -> None:
        design_dict = yaml.safe_load((_EXAMPLE_CONFIG_DIR / "il7ra_3di3_config.yaml").read_text())
        with (
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline") as mock_design,
        ):
            mock_design.return_value = MagicMock()
            DesignLanguageParser().parse_design_dict(design_dict)

        spec = mock_design.call_args.kwargs["output_specification"]
        assert isinstance(spec, OutputSpecification)
        assert list(spec.observable_output_configs["X_T"].outputs["final"]) == list(EXAMPLE_DESIGNED_OUTPUT_FORMATS)
        assert spec.observable_output_configs["X_T"].primary_residue_index_key == "res_idx"


RELEASE_BOUND_CONFIGS: dict[str, int] = {
    "il7ra_3di3_config.yaml": 1,
    "alternative_time_sampler.yaml": 1,
    "multiple_designs_multiple_targets_pipeline.yaml": 2,
    "multiple_residue_specifications.yaml": 2,
    "vegf_1bj1_multichain_target.yaml": 1,
    "il7ra_3di3_loop_conditioning_config.yaml": 1,
}


@contextmanager
def _mocked_release_bound_parse() -> Iterator[MagicMock]:
    """Patch the simulation and design dependencies for release-bound config parsing.

    Yields the ``DesignPipeline`` mock so a test can assert on its call arguments.
    """
    with (
        patch(
            "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
            return_value=MagicMock(),
        ),
        patch("dynopsi.design_pipeline_parser.DesignPipeline") as mock_design,
    ):
        mock_design.return_value = MagicMock()
        yield mock_design


class TestReleaseBoundExampleParsing:
    def test_release_bound_configs_match_readme(self) -> None:
        """YAML names before the notebooks list are the release-bound set."""
        readme = (_EXAMPLE_CONFIG_DIR.parent / "README.md").read_text()
        before_notebooks: list[str] = []
        for line in readme.splitlines():
            if line.startswith("- notebooks"):
                break
            before_notebooks.append(line)
        yaml_tokens = {
            token for token in re.findall(r"`([^`]*)`", "\n".join(before_notebooks)) if token.endswith(".yaml")
        }
        assert yaml_tokens == set(RELEASE_BOUND_CONFIGS)

    @pytest.mark.parametrize(("config_name", "expected_count"), RELEASE_BOUND_CONFIGS.items())
    def test_release_bound_config_parses(self, config_name, expected_count):
        design_dict = yaml.safe_load((_EXAMPLE_CONFIG_DIR / config_name).read_text())
        with _mocked_release_bound_parse():
            result = DesignLanguageParser().parse_design_dict(design_dict)
        assert len(result) == expected_count

    @pytest.mark.parametrize("config_name", RELEASE_BOUND_CONFIGS)
    def test_release_bound_configs_set_required_design_conditioning(self, config_name):
        design_dict = yaml.safe_load((_EXAMPLE_CONFIG_DIR / config_name).read_text())
        with _mocked_release_bound_parse() as mock_design:
            DesignLanguageParser().parse_design_dict(design_dict)
        assert mock_design.call_count == RELEASE_BOUND_CONFIGS[config_name]
        if config_name == "il7ra_3di3_loop_conditioning_config.yaml":
            expected_loop = {"loop_or_other": DSSPLoopFeature.LOOP_OR_OTHER}
        else:
            expected_loop = {"non_loopy": DSSPLoopFeature.NON_LOOP}
        for call in mock_design.call_args_list:
            featurization_pipeline = call.kwargs["featurization_pipeline"]
            assert featurization_pipeline.design_conditioning == {
                "chain_is_loopy": expected_loop,
                "has_homooligomer_partner": {"no_partner": HomooligomerPartnerFeature.NO_PARTNER},
            }


class TestRemovedDesignIsLoopyName:
    """design_is_loopy is rejected as an unknown field.

    InferenceFeaturizationPipeline rejects it through StrictConfig (extra="forbid").
    """

    def test_design_is_loopy_rejected_as_extra(self, binder_design_feat_dict):
        binder_design_feat_dict["name"] = "test_binder_design"
        binder_design_feat_dict["design_is_loopy"] = {"non_loopy": 1}
        with pytest.raises(ValidationError) as exc_info:
            get_featurization_pipeline_from_config(binder_design_feat_dict)
        assert any(
            error["type"] == "extra_forbidden" and error["loc"] == ("design_is_loopy",)
            for error in exc_info.value.errors()
        )
