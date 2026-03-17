from typing import Any

import smart_open
import yaml
from pydantic import BaseModel

from dynopsi.data.featurization.inference_featurization_pipeline import DynoPsiBinderDesignFeaturizationPipeline
from dynopsi.design_pipeline import DesignPipeline, OutputSpecification
from dynopsi.model.model_config import DEFAULT_DYNOPSI_MODEL_CONFIG
from dynopsi.simulation.simulation_pipeline.registry import get_simulation_pipeline_from_config


class DesignLanguageParser(BaseModel):
    """Build design pipelines from a YAML configuration file.

    Reads a single YAML that defines featurization pipelines, simulation pipelines,
    and output options, and returns a list of `DesignPipeline` instances ready to run.
    Can be used on either a config file path (`parse`) or a loaded
    dictionary (`parse_design_dict`).

    See Also
    --------
    examples/il7ra_3di3_config.yaml : Example configuration file for a design pipeline.
    """

    def parse(self, path: str) -> list[DesignPipeline]:
        """Load and parse a YAML file into a list of design pipelines.

        Parameters
        ----------
        path : str
            Path to a YAML configuration file that follows the
            design language schema.

        Returns
        -------
        list[DesignPipeline]
            Design pipelines constructed from the file.

        Raises
        ------
        ValueError
            If the file extension is not `.yaml` or `.yml`.

        See Also
        --------
        parse_design_dict : underlying parse function for design dictionary.
        """
        if path.endswith((".yaml", ".yml")):
            with smart_open.open(path) as f:
                design_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

        return self.parse_design_dict(design_dict)

    def parse_design_dict(self, design_dict: dict[str, Any]) -> list[DesignPipeline]:
        """Parse a dictionary following design language schema into a list of design pipelines.

        Expects keys: `featurization_pipelines`, `simulation_pipelines`,
        `design_pipelines`, and `output_configuration`. Builds featurization and
        simulation pipelines from the first two, applies output settings to simulation
        outputs, then pairs featurizations and simulation in `design_pipelines`.

        Parameters
        ----------
        design_dict : dict[str, Any]
            Dictionary following design language schema (e.g. parsed from config).

        Returns
        -------
        list[DesignPipeline]
            A list of `DesignPipeline` instances; pipeline per entry in `"design_pipelines"`.
        """
        featurization_pipelines = {}
        simulation_pipelines = {}
        output_specifications = {}
        design_pipelines = []

        # parse featurization pipelines
        for featurization_pipeline_name, featurization_pipeline_dict in design_dict["featurization_pipelines"].items():
            featurization_pipeline_dict["name"] = featurization_pipeline_name
            featurization_pipeline = DynoPsiBinderDesignFeaturizationPipeline.model_validate(
                featurization_pipeline_dict
            )
            featurization_pipelines[featurization_pipeline_name] = featurization_pipeline

        # parse output configuration
        for output_specification_name, output_specification_dict in design_dict["output_specifications"].items():
            output_specification = OutputSpecification.model_validate(output_specification_dict)
            output_specifications[output_specification_name] = output_specification

        # parse simulation pipelines
        for simulation_pipeline_name, simulation_pipeline_dict in design_dict["simulation_pipelines"].items():
            # add the inference model
            simulation_pipeline_dict["inference_model_config"] = DEFAULT_DYNOPSI_MODEL_CONFIG.model_copy()
            simulation_pipeline_dict["name"] = simulation_pipeline_name
            simulation_pipeline = get_simulation_pipeline_from_config(simulation_pipeline_dict)
            simulation_pipelines[simulation_pipeline_name] = simulation_pipeline

        # combine featurization and simulation pipelines into design pipelines
        for design_pipeline_name, design_pipeline_dict in design_dict["design_pipelines"].items():
            featurization_pipeline = featurization_pipelines[design_pipeline_dict["featurization_pipeline"]]
            simulation_pipeline = simulation_pipelines[design_pipeline_dict["simulation_pipeline"]]
            output_specification = output_specifications[design_pipeline_dict["output_specification"]]

            design_pipeline = DesignPipeline(
                name=design_pipeline_name,
                inference_featurization_pipeline=featurization_pipeline,
                simulation_pipeline=simulation_pipeline,
                output_specification=output_specification,
            )
            design_pipelines.append(design_pipeline)

        return design_pipelines
