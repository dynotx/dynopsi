from unittest.mock import MagicMock, patch

import pytest
import yaml

from dynopsi.design_pipeline_parser import DesignLanguageParser


def make_parser_with_mocks(mock_feat, mock_sim, mock_out, mock_design):
    """Return a parser with all external dependencies patched."""
    patches = [
        patch(
            "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
            return_value=mock_feat,
        ),
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
    ]
    return patches


class TestParse:
    """Test the parse method of the DesignLanguageParser class."""

    def test_loads_yaml_file(self, minimal_design_dict, tmp_path):
        yaml_file = tmp_path / "design.yaml"
        yaml_file.write_text(yaml.dump(minimal_design_dict))

        with patch.object(DesignLanguageParser, "parse_design_dict", return_value=[]) as mock_pdd:
            DesignLanguageParser().parse(str(yaml_file))
            mock_pdd.assert_called_once_with(minimal_design_dict)

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
        """Ensures the pipeline name key is added before model_validate is called."""
        captured = {}

        def capture_validate(d):
            captured.update(d)
            return MagicMock()

        with (
            patch(
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
                side_effect=capture_validate,
            ),
            patch("dynopsi.design_pipeline_parser.OutputSpecification.model_validate", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config", return_value=MagicMock()),
            patch("dynopsi.design_pipeline_parser.DesignPipeline", return_value=MagicMock()),
        ):
            DesignLanguageParser().parse_design_dict(minimal_design_dict)

        assert captured.get("name") == "feat_a"

    def test_simulation_pipeline_gets_default_model_config(self, minimal_design_dict):
        """Ensures DEFAULT_DYNOPSI_MODEL_CONFIG is injected into simulation pipeline dicts."""
        captured = {}

        def capture_sim(d):
            captured.update(d)
            return MagicMock()

        with (
            patch(
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
                return_value=MagicMock(),
            ),
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
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
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
            inference_featurization_pipeline=mock_featurization_pipeline,
            simulation_pipeline=mock_simulation_pipeline,
            output_specification=mock_output_specification,
        )

    def test_unknown_featurization_pipeline_ref_raises(self, minimal_design_dict):
        minimal_design_dict["design_pipelines"]["design_a"]["featurization_pipeline"] = "nonexistent"
        with (
            patch(
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
                return_value=MagicMock(),
            ),
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
            patch(
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
                return_value=MagicMock(),
            ),
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
            patch(
                "dynopsi.design_pipeline_parser.DynoPsiBinderDesignFeaturizationPipeline.model_validate",
                return_value=MagicMock(),
            ),
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
