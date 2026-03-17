import logging
import os
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator
from typing_extensions import Self

from dynopsi.data.featurization import Features
from dynopsi.data.featurization.inference_featurization_pipeline import (
    DynoPsiBinderDesignFeaturizationPipeline,
    InferenceFeaturizationPipeline,
)
from dynopsi.simulation import Simulation
from dynopsi.simulation.observable import FileOutputConfig, ObservableOutputConfig
from dynopsi.simulation.simulation_pipeline import get_simulation_pipeline_from_config
from dynopsi.simulation.utils import infer_batch_size

logger = logging.getLogger(__name__)


class OutputSpecification(BaseModel):
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
                - formats: list[Literal["pdb", "npz", "cif"]]
                - save_trajectory: bool
                - output_dir: str
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

        # build observable output configs and potentially replace
        configs: dict[str, ObservableOutputConfig] = dict(data.get("observable_output_configs", {}))
        if len(configs) > 0 and save_clean_data_coords:
            raise ValueError(
                "Observable output configurations already exist; cannot overwrite. Please verify that the inputs either include formats and save_trajectory or observable_output_configs."
            )

        if save_trajectory:
            configs["X_T"] = ObservableOutputConfig(
                state_type="clean_data_coords", outputs={"final": tuple(formats), "any": tuple(formats)}
            )
        else:
            configs["X_T"] = ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": tuple(formats)})

        data["observable_output_configs"] = configs
        return data


class DesignPipeline(BaseModel):
    """End-to-end design pipeline: featurization, simulation, and saving outputs.

    Main entry point for running a design job. It (1) builds features
    using the inference featurization pipeline, (2) runs the simulation pipeline
    for each featurization (e.g. different design tasks), and (3) writes all
    outputs under `save_dir`.

    Use the `sample` method to execute the design pipeline.

    Attributes
    ----------
    name : str
        Name of the pipeline.
    inference_featurization_pipeline : InferenceFeaturizationPipeline
        Pipeline that produces the feature sets used for design. Each produced
        feature set is run through simulation separately.
    simulation_pipeline : Simulation
        Defines the simulation/sampling steps (e.g. time schedule,
        flow model, etc.). Runs on each featurization produced by the
        `inference_featurization_pipeline`.
    output_specification : OutputSpecification
        Specification for where and how to write design pipeline outputs.
    """

    name: str
    inference_featurization_pipeline: InferenceFeaturizationPipeline
    simulation_pipeline: Simulation
    output_specification: OutputSpecification

    @model_validator(mode="after")
    def update_simulation_pipeline_with_output_specification(self) -> Self:
        """Update the simulation pipeline's observable output configurations with the output specification."""
        logger.info(
            f"Updating simulation pipeline's observable output configurations with the output specification: {self.output_specification}"
        )
        self.simulation_pipeline.observable_output_configs = self.output_specification.observable_output_configs
        return self

    def sample(
        self,
        num_samples: int,
        batch_size: int | None = None,
    ) -> list[FileOutputConfig]:
        """Run the full design pipeline and write outputs to disk.

        Runs the inference featurization pipeline to get one or more feature sets,
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
            Default is None, in which case batch size is inferred from
            available CUDA devices.

        Returns
        -------
        list[FileOutputConfig]
            Configs describing each saved output (paths, metadata, etc.).
            Use these to load or post-process results.

        Raises
        ------
        ValueError
            If no CUDA devices are available.
        AssertionError
            If `batch_size` is provided and is not greater than 0.

        Notes
        -----
        Batch size only needs to fit on one device; all available
        CUDA devices are used.
        """
        device_names = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

        if len(device_names) == 0:
            raise ValueError("No devices found. Please set devices to a list of CUDA devices.")
        else:
            devices = [torch.device(device_name) for device_name in device_names]

        simulation_save_dir = os.path.join(
            self.output_specification.output_dir, self.name, self.simulation_pipeline.name
        )
        os.makedirs(simulation_save_dir, exist_ok=True)
        logger.info(f"Saving simulation outputs to {simulation_save_dir}")

        if batch_size is None:
            actual_batch_size = infer_batch_size(devices)
        else:
            assert batch_size > 0, "Batch size must be greater than 0"
            actual_batch_size = batch_size

        all_output_files: list[FileOutputConfig] = []
        for feature in self.inference_featurization_pipeline(
            [Features(atoms_to_include=self.inference_featurization_pipeline.atoms_to_include)]
        ):
            features_save_dir = os.path.join(
                simulation_save_dir, self.inference_featurization_pipeline.get_featurization_name(feature)
            )
            os.makedirs(features_save_dir, exist_ok=True)
            logger.info(
                f"Saving simulation outputs for features {self.inference_featurization_pipeline.get_featurization_name(feature)} to {features_save_dir}"
            )

            output_files = self.simulation_pipeline.run(
                {"features": feature},
                features_save_dir,
                num_samples=num_samples,
                batch_size=actual_batch_size,
                devices=devices,
            )
            logger.info("Finished sampling for group.")
            all_output_files.extend(output_files)
        return all_output_files

    def verify_configuration(self, output_format: Literal["pdb", "mmcif", "npz"] = "pdb") -> None:
        """Verify the configuration of the design pipeline.

        First, create all featurization outputs for the design pipeline.
        Then, verify the configuration of the simulation pipeline.
        """
        outdir = os.path.join(self.output_specification.output_dir, self.name, "verify_configuration")
        logger.info(f"Saving verification outputs to {outdir}")
        features_list = self.inference_featurization_pipeline.visualize_configuration(outdir, output_format)

        logger.info(f"Verifying configuration of the design pipeline: {self}")
        for features in features_list:
            self.simulation_pipeline.validate_configuration(features)


class DynoPsiBinderDesignPipeline(DesignPipeline):
    """Design pipeline for binder design with DynoPsi.

     Main design pipeline for binder design with DynoPsi.
     This pipeline pairs the DynoPsi binder design featurization with
     a choice of simulation (e.g an SDE simulator). Use this to design binders
     against a target structure with a DynoPsi model.

     The pipeline can be build using code or from a config.
     When loading from config, the `simulation_pipeline` is
     automatically converted to a `Simulation` instance.

    Attributes
    ----------
     inference_featurization_pipeline : DynoPsiBinderDesignFeaturizationPipeline
         Binder design featurization (e.g. reference structure, chain designations,
         hotspot residues, etc.) compatible with a DynoPsi model.
     simulation_pipeline : Simulation | dict[str, Any]
         Simulation definition. Can be a `Simulation` instance or a dictionary
         to be parsed from config.
     save_dir : str
         Root directory where run outputs are saved.
         Subdirectories are created per simulation and per featurization
    output_specification : OutputSpecification
         Specification for where and how to write design pipeline outputs.
         Subdirectories are created per simulation and per featurization
         under the root directory defined by `output_specification`.

    See Also
    --------
     DesignPipeline : Base pipeline; describes `run` and output layout.
     get_simulation_pipeline_from_config : Used to build `simulation_pipeline` from a dictionary.
    """

    inference_featurization_pipeline: DynoPsiBinderDesignFeaturizationPipeline
    simulation_pipeline: Simulation
    output_specification: OutputSpecification

    @field_validator("simulation_pipeline", mode="before")
    @classmethod
    def validate_simulation_pipeline(cls, simulation_pipeline: Simulation | dict[str, Any]) -> Simulation:
        if isinstance(simulation_pipeline, dict):
            simulation_pipeline = get_simulation_pipeline_from_config(simulation_pipeline)
        return simulation_pipeline

    @field_serializer("simulation_pipeline")
    def serialize_simulation_pipeline(self, simulation_pipeline: Simulation) -> dict[str, Any]:
        return simulation_pipeline.model_dump(serialize_as_any=True)
