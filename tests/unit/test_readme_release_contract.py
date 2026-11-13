from pathlib import Path


def test_pip_install_journey_downloads_the_cli_example() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()

    assert "pip install dynopsi==2.0.0" in readme
    assert "--output il7ra_3di3_config.yaml" in readme
    assert (
        "https://raw.githubusercontent.com/dynotx/dynopsi/v2.0.0/example/configurations/il7ra_3di3_config.yaml"
    ) in readme
    assert "dynopsi check il7ra_3di3_config.yaml" in readme
    assert "dynopsi sample il7ra_3di3_config.yaml --num_samples 10" in readme
    assert "dynopsi check example/configurations/" not in readme
    assert "dynopsi sample example/configurations/" not in readme
