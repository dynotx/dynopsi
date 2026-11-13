from typing import Any

import smart_open
import yaml
from pydantic import BaseModel

from dynopsi.data.featurization.inference_featurization_pipeline import (
    get_featurization_pipeline_from_config,
)
from dynopsi.design_pipeline import DesignPipeline, OutputSpecification
from dynopsi.model.loaders import CheckpointSelectionError
from dynopsi.model.model_config import (
    DEFAULT_DYNOPSI_MODEL_CONFIG,
    DynoPsiModelConfig,
    resolve_checkpoint,
)
from dynopsi.simulation.simulation_pipeline.registry import (
    get_simulation_pipeline_from_config,
)

_CONFIG_CHECKPOINT_ORIGIN = "the checkpoint: config key"
_FLAG_CHECKPOINT_ORIGIN = "the --checkpoint flag"


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

    def parse(
        self,
        path: str,
        inference_model_config: DynoPsiModelConfig | None = None,
        checkpoint: str | None = None,
    ) -> list[DesignPipeline]:
        """Load and parse a YAML file into a list of design pipelines.

        Parameters
        ----------
        path : str
            Path to a YAML configuration file that follows the
            design language schema.
        inference_model_config : DynoPsiModelConfig or None
            Model config to use for all simulation pipelines. When ``None``,
            ``DEFAULT_DYNOPSI_MODEL_CONFIG`` is used.
        checkpoint : str or None
            Checkpoint short name from ``--checkpoint``. It is not a pre-built model config.

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

        return self.parse_design_dict(design_dict, inference_model_config=inference_model_config, checkpoint=checkpoint)

    def parse_design_dict(
        self,
        design_dict: dict[str, Any],
        inference_model_config: DynoPsiModelConfig | None = None,
        checkpoint: str | None = None,
    ) -> list[DesignPipeline]:
        """Parse a dictionary following design language schema into a list of design pipelines.

        Expects keys: `featurization_pipelines`, `simulation_pipelines`,
        `design_pipelines`, and `output_configuration`. Builds featurization and
        simulation pipelines from the first two, applies output settings to simulation
        outputs, then pairs featurizations and simulation in `design_pipelines`.

        Parameters
        ----------
        design_dict : dict[str, Any]
            Dictionary following design language schema (e.g. parsed from config).
        inference_model_config : DynoPsiModelConfig or None
            Model config for every simulation pipeline. When ``None``,
            ``DEFAULT_DYNOPSI_MODEL_CONFIG`` is used.
        checkpoint : str or None
            Short name passed as ``--checkpoint``.

        Returns
        -------
        list[DesignPipeline]
            A list of `DesignPipeline` instances; pipeline per entry in `"design_pipelines"`.

        Raises
        ------
        CheckpointSelectionError
            If a simulation pipeline still carries an ``inference_model_config``
            block, if the ``checkpoint:`` config value and ``--checkpoint`` name
            different weights, if a checkpoint request is combined with a Python
            ``inference_model_config``, or if the short name is unknown.
        """
        featurization_pipelines = {}
        simulation_pipelines = {}
        output_specifications = {}
        design_pipelines = []

        # parse featurization pipelines
        for featurization_pipeline_name, featurization_pipeline_dict in design_dict["featurization_pipelines"].items():
            featurization_pipeline_dict["name"] = featurization_pipeline_name
            featurization_pipeline = get_featurization_pipeline_from_config(featurization_pipeline_dict)
            featurization_pipelines[featurization_pipeline_name] = featurization_pipeline

        # parse output configuration
        for output_specification_name, output_specification_dict in design_dict["output_specifications"].items():
            output_specification = OutputSpecification.model_validate(output_specification_dict)
            output_specifications[output_specification_name] = output_specification

        # parse simulation pipelines
        for simulation_pipeline_name, simulation_pipeline_dict in design_dict["simulation_pipelines"].items():
            sim_dict = self._select_checkpoint(
                simulation_pipeline_dict,
                pipeline_name=simulation_pipeline_name,
                flag_checkpoint=checkpoint,
                python_model_config=inference_model_config,
            )
            sim_dict["name"] = simulation_pipeline_name
            simulation_pipeline = get_simulation_pipeline_from_config(sim_dict)
            simulation_pipelines[simulation_pipeline_name] = simulation_pipeline

        # combine featurization and simulation pipelines into design pipelines
        for design_pipeline_name, design_pipeline_dict in design_dict["design_pipelines"].items():
            featurization_pipeline = featurization_pipelines[design_pipeline_dict["featurization_pipeline"]]
            simulation_pipeline = simulation_pipelines[design_pipeline_dict["simulation_pipeline"]]
            output_specification = output_specifications[design_pipeline_dict["output_specification"]]

            design_pipeline = DesignPipeline(
                name=design_pipeline_name,
                featurization_pipeline=featurization_pipeline,
                simulation_pipeline=simulation_pipeline,
                output_specification=output_specification,
            )
            design_pipelines.append(design_pipeline)

        return design_pipelines

    @staticmethod
    def _select_checkpoint(
        sim_dict: dict[str, Any],
        *,
        pipeline_name: str,
        flag_checkpoint: str | None,
        python_model_config: DynoPsiModelConfig | None,
    ) -> dict[str, Any]:
        """Return a copy that keeps the checkpoint request so ``resolved_design_config.json`` names the weights."""
        # Copy so the caller's dict is unchanged and a second parse sees the same input.
        selected = dict(sim_dict)
        if "inference_model_config" in selected:
            raise CheckpointSelectionError(
                f"Simulation pipeline {pipeline_name!r} sets 'inference_model_config', which is no longer accepted. "
                "Name the weights with 'checkpoint:' instead."
            )

        yaml_checkpoint = selected.get("checkpoint")
        if yaml_checkpoint is not None and not isinstance(yaml_checkpoint, str):
            raise CheckpointSelectionError(
                f"checkpoint: {yaml_checkpoint!r} in simulation pipeline {pipeline_name!r} must be a string "
                "such as 'v1'."
            )
        if python_model_config is not None and (yaml_checkpoint is not None or flag_checkpoint is not None):
            raise CheckpointSelectionError(
                f"Simulation pipeline {pipeline_name!r} receives a checkpoint request while a model config was "
                "passed from Python; provide only one."
            )
        if yaml_checkpoint is not None and flag_checkpoint is not None and yaml_checkpoint != flag_checkpoint:
            raise CheckpointSelectionError(
                f"checkpoint: {yaml_checkpoint!r} in simulation pipeline {pipeline_name!r} and "
                f"--checkpoint {flag_checkpoint!r} name different weights; provide matching values or only one."
            )
        if flag_checkpoint is not None:
            request = flag_checkpoint
            origin = _FLAG_CHECKPOINT_ORIGIN
        elif yaml_checkpoint is not None:
            request = yaml_checkpoint
            origin = _CONFIG_CHECKPOINT_ORIGIN
        else:
            base = python_model_config or DEFAULT_DYNOPSI_MODEL_CONFIG
            selected["inference_model_config"] = base.model_copy()
            return selected
        selected["checkpoint"] = request
        selected["inference_model_config"] = resolve_checkpoint(request, origin=origin)
        return selected
