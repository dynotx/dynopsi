from unittest.mock import patch

from dynopsi import cli


class TestCheckpointFlagThreading:
    """The --checkpoint value reaches the parser as a raw name for both subcommands."""

    def test_sample_forwards_checkpoint(self, tmp_path):
        config = tmp_path / "design.yaml"
        config.write_text("dummy: true")
        with patch("dynopsi.cli.DesignLanguageParser.parse", return_value=[]) as mock_parse:
            cli._cmd_sample(
                config_path=str(config),
                num_samples=1,
                batch_size=1,
                overwrite=True,
                checkpoint="v1",
            )
        mock_parse.assert_called_once_with(str(config), checkpoint="v1")

    def test_check_forwards_checkpoint(self, tmp_path):
        config = tmp_path / "design.yaml"
        config.write_text("dummy: true")
        with patch("dynopsi.cli.DesignLanguageParser.parse", return_value=[]) as mock_parse:
            cli._cmd_check(
                config_path=str(config),
                output_format="pdb",
                overwrite=True,
                checkpoint="v1",
            )
        mock_parse.assert_called_once_with(str(config), checkpoint="v1")

    def test_sample_defaults_to_no_checkpoint(self, tmp_path):
        config = tmp_path / "design.yaml"
        config.write_text("dummy: true")
        with patch("dynopsi.cli.DesignLanguageParser.parse", return_value=[]) as mock_parse:
            cli._cmd_sample(config_path=str(config), num_samples=1, batch_size=1, overwrite=True)
        mock_parse.assert_called_once_with(str(config), checkpoint=None)

    def test_check_defaults_to_no_checkpoint(self, tmp_path):
        config = tmp_path / "design.yaml"
        config.write_text("dummy: true")
        with patch("dynopsi.cli.DesignLanguageParser.parse", return_value=[]) as mock_parse:
            cli._cmd_check(config_path=str(config), output_format="pdb", overwrite=True)
        mock_parse.assert_called_once_with(str(config), checkpoint=None)
