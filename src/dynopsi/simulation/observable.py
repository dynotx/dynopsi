import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, Self, TypedDict

from dynopsi.data.featurization import Features
from dynopsi.data.featurization.utils import crop_features
from dynopsi.data.structure import MultiModelStructure, Structure
from dynopsi.data.structure_io import save_structures_to_file
from dynopsi.data.utils import align_by_mask

logger = logging.getLogger(__name__)

OutputTimestep = Literal["any", "final"]
OutputFormat = Literal["pdb", "mmcif", "npz"]


class ObservableOutputConfig(BaseModel):
    """Define the output configuration for an observable state key.

    During pipeline setup, an ObservableOutputConfig is created for each state key
    that will be saved to disk (e.g., "X_T", "CVF_T") to specify output formats.

    Attributes
    ----------
    state_type : Literal["clean_data_coords", "conditional_vector_fields"]
        The type of state key to output.
        The type determines the valid output formats.
    outputs : dict[Timestep, tuple[OutputFormat, ...]]
        The output formats and timesteps to record an observable.
        The OutputFormat is restricted by the state key type.
        Only coordinates can be output in pdb and mmcif formats.
        All other state keys can only be output in npz format.
    features_key : str or None, optional
        The key of the features to use for the observable. If None, this this field must
        be set by the simulation pipeline before use.
    """

    state_type: Literal["clean_data_coords", "conditional_vector_fields"]
    outputs: dict[OutputTimestep, tuple[OutputFormat, ...]]

    features_key: str | None = None

    @model_validator(mode="after")
    def _check_output_formats(self) -> Self:
        if self.state_type != "clean_data_coords":
            for output_formats in self.outputs.values():
                if any(fmt != "npz" for fmt in output_formats):
                    raise ValueError(f"Only npz format is supported for state keys of type: {self.state_type}")
        return self

    def set_features_key(self, new_key: str) -> None:
        if self.features_key is None:
            self.features_key = new_key
        else:
            if self.features_key != new_key:
                raise ValueError(
                    f"Trying to set `features_key` in ObservableOutputConfig, but there is a mismatch. "
                    f"Current value: {self.features_key}, new value {new_key}"
                )


class FileOutputConfig(TypedDict):
    """Configuration for a single output file."""

    state_key: str
    features_key: NotRequired[str]
    timestep: OutputTimestep
    output_format: OutputFormat
    path: Path | str
    metadata: NotRequired[dict[str, Any]]


class Observable(BaseModel):
    """Record aspects of the state during simulation for export to disk.

    This class tracks the values of the state keys to be saved to disk through the simulation
    and writes to disk once the simulation is complete.

    Store.put() forwards to record() when a key is in keys_to_record. Supports chunked
    execution: record(key, step_num, value, batch_slice) writes into the
    given batch indices so a single Trajectory accumulates from multiple
    chunk runs.

    Attributes
    ----------
    total_batch_size : int or None
        Sum of the lengths of the batch slices. If None, chunked execution is not supported and the value is equal to the batch dimension of the first record.
    observable_output_configs : dict[str, ObservableOutputConfig]
        The output configurations for the observable state keys.
    observable_outputs : dict[str, dict[int, dict[str, torch.Tensor]]]
        The outputs of the observable state keys.
        The keys are the state keys, the values are the outputs at each step.
        The per-step outputs are a dictionary of output tensors, keyed by the modality.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_batch_size: int | None = None
    observable_output_configs: dict[str, ObservableOutputConfig] = Field(default_factory=dict)
    observable_outputs: dict[str, dict[int, dict[str, torch.Tensor]]] = Field(default_factory=dict)

    @property
    def keys_to_record_any_timestep(self) -> list[str]:
        return [
            key
            for key in self.observable_output_configs.keys()
            if "any" in self.observable_output_configs[key].outputs.keys()
        ]

    @property
    def keys_to_record_final_timestep(self) -> list[str]:
        return [
            key
            for key in self.observable_output_configs.keys()
            if "final" in self.observable_output_configs[key].outputs.keys()
        ]

    def record(
        self,
        key: str,
        step_num: int,
        value: dict[str, torch.Tensor],
        batch_slice: slice | None = None,
    ) -> None:
        """Record state at key and step for the given batch indices.

        On first record for (key, step_num), allocates tensors of shape
        (total_batch_size, *value.shape[1:]) per modality. Then copies
        value into [batch_slice]. If total_batch_size was None, it is set
        from batch_slice (batch_slice.stop) or value.shape[0].

        Parameters
        ----------
        key : str
            State key (e.g. "X").
        step_num : int
            Step index.
        value : dict[str, torch.Tensor]
            Modality -> tensor, shape (chunk_batch, ...) or (full_batch, ...).
        batch_slice : slice or None, optional
            Indices into the full batch. If None, treated as slice(0, value.shape[0]).
        """
        if batch_slice is None:
            batch_slice = slice(0, next(iter(value.values())).shape[0])

        # First record for this trajectory sets total batch size (e.g. from chunked run).
        if self.total_batch_size is None:
            self.total_batch_size = batch_slice.stop
        n = self.total_batch_size

        if key not in self.observable_outputs:
            self.observable_outputs[key] = {}
        steps = self.observable_outputs[key]
        if step_num not in steps:
            steps[step_num] = {}
            # Allocate full-batch buffers per modality for this step.
            for modality, tensor in value.items():
                device = tensor.device
                dtype = tensor.dtype
                shape = (n,) + tensor.shape[1:]
                buf = torch.zeros(shape, device=device, dtype=dtype)
                steps[step_num][modality] = buf

        # Write incoming chunk (or full batch) into the correct batch indices.
        for modality, tensor in value.items():
            buf = steps[step_num][modality]
            steps[step_num][modality][batch_slice] = tensor.detach().clone().to(buf.device)

    def get_observable(self, key: str) -> dict[int, dict[str, torch.Tensor]] | None:
        """Get the observable output for a key in the format of dict[step_num, dict[modality, tensor]]."""
        if key not in self.observable_outputs:
            return None
        return self.observable_outputs[key]

    def get_observable_step(self, key: str, step_num: int) -> dict[str, torch.Tensor] | None:
        """Get the observable output for a key at a specific step."""
        if key not in self.observable_outputs:
            return None
        if step_num not in self.observable_outputs[key]:
            return None
        return self.observable_outputs[key][step_num]

    def save(
        self, save_dir: str | Path, start_index: int, reference_features: dict[str, Features]
    ) -> list[FileOutputConfig]:
        """Save the observable state keys to a directory for a batch of samples.

        Each sample receives a unique index equal to its position in the batch, plus `start_index`.
        This index is used to name the output files for a given sample.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        all_output_files: list[FileOutputConfig] = []

        # save the observable
        self.save_observable_to_npz(save_dir=save_dir, start_index=start_index)

        # save clean data coords as other file formats
        for state_key, config in self.observable_output_configs.items():
            for timesteps, output_formats in config.outputs.items():
                # this is checked earlier, but just in case throw an error
                if config.features_key is None:
                    raise ValueError(
                        f"features_key definition in observable_output_configs is required, "
                        f"but is missing for {state_key}, timestep {timesteps}, state type {config.state_type}"
                    )
                else:
                    features_key = config.features_key
                if config.state_type == "clean_data_coords":
                    output_files_by_timestep: list[FileOutputConfig] = self.save_clean_data_coords(
                        save_dir=save_dir,
                        timesteps=timesteps,
                        output_formats=output_formats,
                        observable_key=state_key,
                        reference_features_key=features_key,
                        reference_features=reference_features[features_key],
                        start_index=start_index,
                    )
                    all_output_files.extend(output_files_by_timestep)
                else:
                    continue
        return all_output_files

    def save_observable_to_npz(
        self,
        save_dir: str | Path,
        start_index: int,
    ) -> list[FileOutputConfig]:
        """Save state tensors associated with observable keys in a npz file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        output_files: list[FileOutputConfig] = []

        # batch: output_key/step_num/modality: tensor
        flat_outputs: dict[int, dict[str, Any]] = defaultdict(dict[str, Any])
        for state_key in self.observable_output_configs.keys():
            observable = self.get_observable(state_key)
            if observable is None:
                logger.warning(f"Observable doesn't have values for state key: {state_key}")
                continue
            for step_num, modality_dict in observable.items():
                for modality, tensor in modality_dict.items():
                    out_arr = tensor.cpu().numpy()
                    batch_size = out_arr.shape[0]
                    for batch_idx in range(batch_size):
                        flat_outputs[batch_idx][f"{state_key}/{step_num}/{modality}"] = out_arr[batch_idx]

        for batch_idx, output_dict in flat_outputs.items():
            npz_file = save_dir / f"observable_outputs_{batch_idx + start_index}.npz"
            np.savez_compressed(file=str(npz_file), **output_dict)
            output_files.append(
                FileOutputConfig(
                    state_key="all_observable_outputs",
                    timestep="any",
                    output_format="npz",
                    path=npz_file,
                )
            )

        return output_files

    def save_clean_data_coords(
        self,
        save_dir: str | Path,
        timesteps: Literal["any", "final"],
        output_formats: tuple[Literal["pdb", "mmcif", "npz"], ...],
        observable_key: str,
        reference_features_key: str,
        reference_features: Features,
        start_index: int,
    ) -> list[FileOutputConfig]:
        """Save predicted final structure coordinates with relevant features in the specified formats.

        Notes
        -----
        Coordinates for each timestep are aligned to the unnoised positions, if possible.
        Otherwise, coordinates for each intermediate step are aligned to the coordinates at the final timestep..
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        observable = self.get_observable(observable_key)
        if observable is None:
            return []

        reference_feature_dict = {k: v.clone() for k, v in reference_features.feature_dict.items()}
        features_metadata = reference_features.get_metadata(verbose=False)

        if timesteps == "final":
            step = max(observable.keys())
            coords_stack = observable[step]["coords"].cpu()  # shape (batch_size, seq_len, num_atoms*3)
            batch_size = coords_stack.shape[0]
            coords_stack = coords_stack.reshape(batch_size, -1, 3)
            coords_stack = coords_stack.unsqueeze(1)
        else:
            steps_sorted = sorted(observable.keys())
            first = observable[steps_sorted[0]]
            coords_per_step = first["coords"].cpu()
            batch_size = coords_per_step.shape[0]
            seq_len = coords_per_step.shape[1]
            flat = coords_per_step.shape[2]
            num_timesteps = len(steps_sorted)
            coords_stack = torch.zeros(batch_size, num_timesteps, seq_len, flat)
            for t, step in enumerate(steps_sorted):
                coords_stack[:, t] = observable[step]["coords"].cpu()

            # Reshape to (batch_size, num_timesteps, num_atoms, 3)
            coords_stack = coords_stack.reshape(batch_size, num_timesteps, -1, 3)

        # Strip padding
        atom_pad_mask = torch.zeros(coords_stack.shape[2], dtype=torch.bool, device=coords_stack.device)
        atom_pad_mask[torch.where(reference_feature_dict["atom_pad_mask"].bool())[0]] = True
        coords_stack = coords_stack[:, :, atom_pad_mask, :]

        # Output is a dictionary of output files for each output format
        output_files: list[FileOutputConfig] = []

        # Construct reference coords if available
        non_noised_atom_mask = reference_feature_dict["atom_to_token"][:, ~reference_feature_dict["noised_mask"]].any(
            dim=1
        )
        ref_mask = reference_feature_dict["atom_resolved_mask"].bool() & non_noised_atom_mask
        if ref_mask.any():
            use_input_features_alignment = True
        else:
            use_input_features_alignment = False

        for batch_idx in range(coords_stack.shape[0]):
            structures: list[Structure] = []
            coords = coords_stack[batch_idx]  # (num_timesteps, n_atoms, 3)
            num_timesteps = coords.shape[0]

            if use_input_features_alignment:
                reference_coords = reference_feature_dict["coords"][0]  # (n_atoms, 3)
                reference_mask = ref_mask  # n_atoms
            else:
                # Align to the final state
                reference_coords = coords[-1]  # (n_atoms, 3)
                reference_mask = reference_feature_dict["atom_resolved_mask"].bool()  # n_atoms

            # The structure for each timestep is a different model in the output file
            for t in range(num_timesteps):
                coords_t = align_by_mask(coords[t], reference_coords, reference_mask, reference_mask)
                reference_feature_dict["coords"] = coords_t.unsqueeze(0)

                if features_metadata["reference_structure_input"] is None:
                    structure = Features(
                        feature_dict=reference_feature_dict, atoms_to_include=reference_features.atoms_to_include
                    ).to_structure()
                else:
                    # Only retain the denoised positions
                    cropped_ref_feat = crop_features(
                        reference_feature_dict, token_mask=reference_feature_dict["noised_mask"].clone()
                    )
                    structure = Features(
                        feature_dict=cropped_ref_feat, atoms_to_include=reference_features.atoms_to_include
                    ).to_structure()

                    # Add the reference structure to the denoised structure
                    # This step adds side chains to the reference structure
                    # The noised coordinates are already aligned to the reference features
                    structure = structure.concat(features_metadata["reference_structure_input"])

                structures.append(structure)

            state_name = "final_state" if timesteps == "final" else "trajectory"
            sample_name = f"{observable_key}_{state_name}_{start_index + batch_idx}"

            if "npz" in output_formats:
                npz_file = save_dir / f"{sample_name}.npz"
                if len(structures) == 1:
                    structures[0].dump_to_npz(npz_file)
                else:
                    multi_model_structure = MultiModelStructure.from_structures(structures)
                    multi_model_structure.dump_to_npz(npz_file)
                output_files.append(
                    FileOutputConfig(
                        state_key=observable_key,
                        features_key=reference_features_key,
                        timestep=timesteps,
                        output_format="npz",
                        path=npz_file,
                        metadata=features_metadata,
                    )
                )

            output_formats_to_check: list[OutputFormat] = ["pdb", "mmcif"]
            for output_format in output_formats_to_check:
                if output_format in output_formats:
                    file_extension = "pdb" if output_format == "pdb" else "cif"
                    file_name = save_dir / f"{sample_name}.{file_extension}"
                    save_structures_to_file(structures, file_name)
                    output_files.append(
                        FileOutputConfig(
                            state_key=observable_key,
                            features_key=reference_features_key,
                            timestep=timesteps,
                            output_format=output_format,
                            path=file_name,
                            metadata=features_metadata,
                        )
                    )

        return output_files
