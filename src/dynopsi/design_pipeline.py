import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Self

import smart_open
import torch
from pydantic import Field, field_serializer, field_validator, model_validator

from dynopsi.data.conditioning import assert_registered_conditioning_matches
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.inference_featurization_pipeline import (
    InferenceFeaturizationPipeline,
    get_featurization_pipeline_from_config,
)
from dynopsi.model.model_config import default_checkpoint_label
from dynopsi.simulation import Simulation
from dynopsi.simulation.observable import FileOutputConfig, ObservableOutputConfig
from dynopsi.simulation.simulation_pipeline import get_simulation_pipeline_from_config
from dynopsi.simulation.utils import infer_batch_size
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)


def resolved_output_dir(output_dir: str, *components: str) -> str:
    """Resolve a directory under a local ``output_dir``.

    ``output_dir`` must be a local path; a URI is not a containment root.
    Each component must be one path segment and must not be ``.`` or ``..``.
    The resolved path must be a descendant of the resolved output directory.
    """
    if not components:
        raise ValueError("An output path needs at least one component under the output directory.")
    root = Path(output_dir).expanduser().resolve()
    for component in components:
        if component in {"", ".", ".."} or "/" in component or "\\" in component or Path(component).name != component:
            raise ValueError(
                f"Output path component {component!r} must be a single name, not an absolute path, '.', or '..'."
            )
    target = root.joinpath(*components).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Output path {target} must stay inside {root}.")
    return str(target)


class OutputSpecification(StrictConfig):
    """Basic output specification for the design pipelines

    Attributes
    ----------
    output_dir : str
        The directory to save the output files.
    observable_output_configs : dict[str, ObservableOutputConfig]
        The observable output configurations to save.
    """

    output_dir: str
    observable_output_configs: dict[str, ObservableOutputConfig] = Field(default_factory=dict)

    def resolved_dir(self, *components: str) -> str:
        return resolved_output_dir(self.output_dir, *components)

    @model_validator(mode="before")
    @classmethod
    def from_yaml_inputs(cls, data: Any) -> Any:
        """Allow parsing of yaml inputs into a OutputSpecification object.

        Parameters
        ----------
        data : Any
            The inputs to parse. Expected inputs are mutually exclusive and must be one of the following:
            1. Instance of OutputSpecification, will just be returned unchanged.
            2. Dictionary with the following keys (likely parsed from a yaml file)
                - formats: list[Literal["pdb", "npz", "mmcif"]]
                - save_trajectory: bool
                - output_dir: str
                - primary_residue_index_key: Literal["res_idx", "auth_res_idx"] (optional)
                - save_raw_context_output: bool (optional)
            3. Dictionary with the following keys:
                - output_dir: str
                - observable_output_configs: dict[str, ObservableOutputConfig]

        Returns
        -------
        Any
            The parsed OutputSpecification object.

        Raises
        ------
        ValueError
            If the yaml inputs are invalid.
        """
        if not isinstance(data, dict):
            return data

        formats = data.pop("formats", None)
        save_trajectory = data.pop("save_trajectory", None)
        primary_residue_index_key = data.pop("primary_residue_index_key", None)
        save_raw_context_output = data.pop("save_raw_context_output", None)
        valid_formats = ["pdb", "npz", "mmcif"]

        # check that formats and save_trajectory are either both present and valid or both absent
        if formats is not None and save_trajectory is not None:
            if not isinstance(formats, (list, tuple)):
                raise ValueError("formats must be a list or tuple")
            if len(formats) > 0:
                if not all(fmt in valid_formats for fmt in formats):
                    raise ValueError("formats must contain only 'pdb', 'npz', or 'mmcif'")
            else:
                raise ValueError("formats must be non-empty if save_trajectory is True")
            if not isinstance(save_trajectory, bool):
                raise ValueError("save_trajectory must be a boolean")

            save_clean_data_coords = True
        elif (formats is not None and save_trajectory is None) or (formats is None and save_trajectory is not None):
            raise ValueError("formats and save_trajectory must both be present or both absent")
        else:
            save_clean_data_coords = False

        if (
            primary_residue_index_key is not None or save_raw_context_output is not None
        ) and not save_clean_data_coords:
            raise ValueError(
                "primary_residue_index_key and save_raw_context_output are only supported alongside "
                "formats and save_trajectory; set them directly on each ObservableOutputConfig when using "
                "observable_output_configs."
            )
        if primary_residue_index_key is None:
            primary_residue_index_key = "res_idx"
        if save_raw_context_output is None:
            save_raw_context_output = False

        # build observable output configs and potentially replace
        configs: dict[str, ObservableOutputConfig] = dict(data.get("observable_output_configs", {}))
        if len(configs) > 0 and save_clean_data_coords:
            raise ValueError(
                "Observable output configurations already exist; cannot overwrite. Please verify that the inputs either include formats and save_trajectory or observable_output_configs."
            )
        x_t_keys = [k for k in configs.keys() if k.startswith("X_T")]
        if len(x_t_keys) == 0 and not save_clean_data_coords:
            raise ValueError(
                "X_T observable output configurations not found; Please verify that the inputs either include formats and save_trajectory or an X_T<suffix> observable output configuration."
            )

        if save_clean_data_coords:
            if save_trajectory:
                configs["X_T"] = ObservableOutputConfig(
                    state_type="clean_data_coords",
                    outputs={"final": tuple(formats), "any": tuple(formats)},
                    primary_residue_index_key=primary_residue_index_key,
                    save_raw_context_output=save_raw_context_output,
                )
            else:
                configs["X_T"] = ObservableOutputConfig(
                    state_type="clean_data_coords",
                    outputs={"final": tuple(formats)},
                    primary_residue_index_key=primary_residue_index_key,
                    save_raw_context_output=save_raw_context_output,
                )

        data["observable_output_configs"] = configs
        return data


class DesignPipeline(StrictConfig):
    """End-to-end design pipeline: featurization, simulation, and saving outputs.

    `sample` (1) builds features, (2) runs the simulation on each feature
    set, and (3) writes outputs under `save_dir`. Each featurization result
    is ``{"features": Features}``.

    Attributes
    ----------
    name : str
        Name of the pipeline.
    featurization_pipeline : InferenceFeaturizationPipelineType
        Pipeline that produces one feature set per design task. Each feature set
        is run through simulation separately.
    simulation_pipeline : Simulation
        Defines the simulation/sampling steps (e.g. time schedule,
        flow model, etc.). Runs on each featurization produced by the
        `featurization_pipeline`.
    output_specification : OutputSpecification
        Specification for where and how to write design pipeline outputs.
    """

    name: str
    featurization_pipeline: InferenceFeaturizationPipeline
    simulation_pipeline: Simulation
    output_specification: OutputSpecification

    @field_validator("featurization_pipeline", mode="before")
    @classmethod
    def validate_featurization_pipeline(
        cls, featurization_pipeline: InferenceFeaturizationPipeline | dict[str, Any]
    ) -> InferenceFeaturizationPipeline:
        if isinstance(featurization_pipeline, dict):
            featurization_pipeline = get_featurization_pipeline_from_config(featurization_pipeline)
        return featurization_pipeline

    @field_validator("simulation_pipeline", mode="before")
    @classmethod
    def validate_simulation_pipeline(cls, simulation_pipeline: Simulation | dict[str, Any]) -> Simulation:
        if isinstance(simulation_pipeline, dict):
            simulation_pipeline = get_simulation_pipeline_from_config(simulation_pipeline)
        return simulation_pipeline

    @field_serializer("featurization_pipeline")
    def serialize_featurization_pipeline(
        self, featurization_pipeline: InferenceFeaturizationPipeline
    ) -> dict[str, Any]:
        return featurization_pipeline.model_dump(serialize_as_any=True)

    @field_serializer("simulation_pipeline")
    def serialize_simulation_pipeline(self, simulation_pipeline: Simulation) -> dict[str, Any]:
        return simulation_pipeline.model_dump(serialize_as_any=True)

    @model_validator(mode="after")
    def update_simulation_pipeline_with_output_specification(self) -> Self:
        """Update the simulation pipeline's observable output configurations with the output specification."""
        logger.info(
            f"Updating simulation pipeline's observable output configurations with the output specification: {self.output_specification}"
        )
        self.simulation_pipeline.observable_output_configs = self.output_specification.observable_output_configs
        return self

    def simulation_output_dir(self) -> str:
        return self.output_specification.resolved_dir(self.name, self.simulation_pipeline.name)

    def verification_output_dir(self) -> str:
        return self.output_specification.resolved_dir(self.name, "verify_configuration")

    def sample(
        self,
        num_samples: int,
        batch_size: int | None = None,
    ) -> list[FileOutputConfig]:
        """Run the full design pipeline and write outputs to disk.

        Runs the featurization pipeline to get one or more feature sets,
        then runs the simulation pipeline separately on each feature set with
        the requested `batch_size` to produce `num_samples` designs.
        All outputs are saved under `save_dir`, and a list of output file configs
        is returned.

        Parameters
        ----------
        num_samples : int
            Total number of designs to generate per featurization.
        batch_size : int | None
            Batch size for simulation.
            Default is None, in which case batch size is inferred with `simulation.utils.infer_batch_size`.

        Returns
        -------
        list[FileOutputConfig]
            Configs describing each saved output (paths, metadata, etc.).
            Use these to load or post-process results.

        Raises
        ------
        AssertionError
            If `batch_size` is provided and is not greater than 0.

        Notes
        -----
        Batch size only needs to fit on one device. Sampling uses ``cuda:0``
        when CUDA is available, and the CPU otherwise.
        """
        self.assert_conditioning_matches_checkpoint()

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        simulation_save_dir = self.simulation_output_dir()
        os.makedirs(simulation_save_dir, exist_ok=True)
        logger.info(f"Saving simulation outputs to {simulation_save_dir}")

        if batch_size is None:
            actual_batch_size = infer_batch_size(device)
        else:
            assert batch_size > 0, "Batch size must be greater than 0"
            actual_batch_size = batch_size

        self._write_resolved_config(simulation_save_dir, num_samples, actual_batch_size)

        featurization_pipeline = self.featurization_pipeline
        features_input = [Features(atoms_to_include=featurization_pipeline.atoms_to_include)]
        feature_iterator = featurization_pipeline.featurize_for_sampling(features_input, num_samples=num_samples)

        all_output_files: list[FileOutputConfig] = []
        for features_dict in feature_iterator:
            features_save_dir = resolved_output_dir(
                simulation_save_dir,
                self.featurization_pipeline.get_featurization_name(features_dict),
            )
            os.makedirs(features_save_dir, exist_ok=True)
            logger.info(f"Saving simulation outputs for features to {features_save_dir}")

            output_files = self.simulation_pipeline.run(
                features_dict,
                features_save_dir,
                num_samples=num_samples,
                batch_size=actual_batch_size,
                device=device,
            )
            logger.info("Finished sampling for group.")
            all_output_files.extend(output_files)
        return all_output_files

    def _write_resolved_config(self, save_dir: str, num_samples: int, batch_size: int) -> None:
        """Write the resolved pipeline config and runtime args to ``save_dir/resolved_design_config.json``."""
        payload = {
            "num_samples": num_samples,
            "batch_size": batch_size,
            "design_pipeline": {
                "name": self.name,
                "featurization_pipeline": self.featurization_pipeline.model_dump(mode="json", serialize_as_any=True),
                # Exclude derived plan/plan_init (rebuilt by _rebuild_plan on re-parse) and the
                # observable tensor accumulator (no JSON encoder); the remaining fields round-trip.
                "simulation_pipeline": self.simulation_pipeline.model_dump(
                    mode="json",
                    serialize_as_any=True,
                    exclude={"plan", "plan_init", "observable"},
                ),
                "output_specification": self.output_specification.model_dump(mode="json"),
            },
        }
        config_path = os.path.join(save_dir, "resolved_design_config.json")
        with smart_open.open(config_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote resolved design config to {config_path}")

    def assert_conditioning_matches_checkpoint(self) -> None:
        """Raise ConditioningCheckpointError when supplied and declared keys differ."""
        simulation = self.simulation_pipeline
        checkpoint_label = simulation.checkpoint if simulation.checkpoint is not None else default_checkpoint_label()
        assert_registered_conditioning_matches(
            checkpoint_label=checkpoint_label,
            declared_keys=simulation.declared_categorical_keys(),
            design_conditioning=self.featurization_pipeline.supplied_design_conditioning(),
        )

    def verify_configuration(self, output_format: Literal["pdb", "mmcif", "npz"] = "pdb") -> None:
        """Verify the configuration of the design pipeline.

        First, create all featurization outputs for the design pipeline.
        Then, verify the configuration of the simulation pipeline.
        """
        outdir = self.verification_output_dir()
        logger.info(f"Saving verification outputs to {outdir}")
        features_list = self.featurization_pipeline.visualize_configuration(outdir, output_format)

        logger.info(f"Verifying configuration of the design pipeline: {self}")
        for features_dict in features_list:
            self.simulation_pipeline.validate_configuration(features_dict)
