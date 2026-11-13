import logging
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, NotRequired, Self, TypedDict

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynopsi.data.constants import InpaintFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.sample_centers import reference_features_for_sample
from dynopsi.data.structure import MultiModelStructure
from dynopsi.data.structure_io import save_structures_to_file
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)

OutputTimestep = Literal["any", "final"]
OutputFormat = Literal["pdb", "mmcif", "npz"]
PrimaryResidueIndexKey = Literal["res_idx", "auth_res_idx"]


class ObservableOutputConfig(StrictConfig):
    """Define the output configuration for an observable state key.

    During pipeline setup, an ObservableOutputConfig is created for each state key
    that will be saved to disk (e.g., "X_T", "CVF_T") to specify output formats.

    Attributes
    ----------
    state_type : Literal["clean_data_coords", "conditional_vector_fields"]
        The type of state to output.
        The type determines the valid output formats.
    outputs : dict[Timestep, tuple[OutputFormat, ...]]
        The output formats and timesteps to record an observable.
        The OutputFormat is restricted by the state key type.
        Only coordinates can be output in pdb and mmcif formats.
        All other state keys can only be output in npz format.
    features_keys : list[str]
        The keys of the features sets to use when saving this observable. Each key
        produces its own set of output files. If empty, this field must be populated
        via ``add_features_key`` by the simulation pipeline before use. When more than
        one key is present, each key is appended to the output filename to disambiguate.
    primary_residue_index_key : Literal["res_idx", "auth_res_idx"]
        Which residue numbers to write in designed PDB/mmCIF
        files. ``res_idx`` is sequential 0-based ``residue_index`` (the default).
        ``auth_res_idx`` writes ``auth_residue_index`` as seqid.
    save_raw_context_output : bool
        Save model-generated coordinates for context replaced by the reference structure.
        When that context was noised, each requested pdb, mmcif, or npz output also
        writes those coordinates to a ``rawcontext_`` file.
    """

    state_type: Literal["clean_data_coords", "conditional_vector_fields", "step_updates", "any"]
    outputs: dict[OutputTimestep, tuple[OutputFormat, ...]]

    features_keys: list[str] = Field(default_factory=list)
    primary_residue_index_key: PrimaryResidueIndexKey = "res_idx"
    save_raw_context_output: bool = False

    @model_validator(mode="after")
    def _check_output_formats(self) -> Self:
        if self.state_type != "clean_data_coords":
            for output_formats in self.outputs.values():
                if any(fmt != "npz" for fmt in output_formats):
                    raise ValueError(f"Only npz format is supported for state keys of type: {self.state_type}")
        return self

    def add_features_key(self, key: str) -> None:
        """Add a features key if not already present."""
        if key not in self.features_keys:
            self.features_keys.append(key)


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

    _record_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

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
        Concurrent callers share one buffer: allocate-and-write holds one lock.

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

        with self._record_lock:
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
                if tensor.shape[0] == n:
                    value_slice = batch_slice
                else:
                    value_slice = slice(batch_slice.start, batch_slice.start + tensor.shape[0])
                steps[step_num][modality][value_slice] = tensor.detach().clone().to(buf.device)

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
        npz_output_files = self.save_observable_to_npz(save_dir=save_dir, start_index=start_index)
        all_output_files.extend(npz_output_files)

        # save clean data coords as other file formats
        for state_key, config in self.observable_output_configs.items():
            if not config.features_keys:
                raise ValueError(
                    f"features_keys in observable_output_configs is empty for {state_key}. "
                    f"The simulation pipeline must call add_features_key before saving."
                )
            disambiguate = len(config.features_keys) > 1
            for features_key in config.features_keys:
                for timesteps, output_formats in config.outputs.items():
                    if config.state_type == "clean_data_coords":
                        output_files_by_timestep: list[FileOutputConfig] = self.save_clean_data_coords(
                            save_dir=save_dir,
                            timesteps=timesteps,
                            output_formats=output_formats,
                            observable_key=state_key,
                            reference_features_key=features_key,
                            reference_features=reference_features[features_key],
                            start_index=start_index,
                            features_label=features_key if disambiguate else None,
                            primary_residue_index_key=config.primary_residue_index_key,
                            save_raw_context_output=config.save_raw_context_output,
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
        features_label: str | None = None,
        primary_residue_index_key: PrimaryResidueIndexKey = "res_idx",
        save_raw_context_output: bool = False,
    ) -> list[FileOutputConfig]:
        """Save predicted final structure coordinates with relevant features in the specified formats.

        Notes
        -----
        Coordinates for each timestep are aligned to the unnoised positions, if possible.
        Otherwise, coordinates for each intermediate step are aligned to the coordinates at the final timestep.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        observable = self.get_observable(observable_key)
        if observable is None:
            return []

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

        # Output is a dictionary of output files for each output format
        output_files: list[FileOutputConfig] = []

        features_metadata = reference_features.get_metadata(verbose=False)
        raw_structure_output_requested = save_raw_context_output and any(
            output_format in output_formats for output_format in ("pdb", "mmcif", "npz")
        )

        for batch_idx in range(coords_stack.shape[0]):
            coords = coords_stack[batch_idx]  # (num_timesteps, n_atoms, 3)

            sample_features = reference_features_for_sample(reference_features, start_index + batch_idx)
            structures = sample_features.design_coordinates_to_structure(coords, use_reference_for_noised_context=True)
            raw_structures = (
                sample_features.design_coordinates_to_structure(coords, use_reference_for_noised_context=False)
                if raw_structure_output_requested and _has_noised_reference_context(sample_features)
                else None
            )

            state_name = "final_state" if timesteps == "final" else "trajectory"
            name_parts = [observable_key]
            if features_label is not None:
                name_parts.append(features_label)
            name_parts.append(f"{state_name}_{start_index + batch_idx}")
            sample_name = "_".join(name_parts)

            if "npz" in output_formats:
                npz_outputs = [(structures, save_dir / f"{sample_name}.npz")]
                if raw_structures is not None:
                    npz_outputs.append((raw_structures, save_dir / f"rawcontext_{sample_name}.npz"))
                for structures_to_save, npz_file in npz_outputs:
                    sequential_structures = [
                        structure.with_sequential_auth_res_idx() for structure in structures_to_save
                    ]
                    if len(sequential_structures) == 1:
                        sequential_structures[0].dump_to_npz(npz_file)
                    else:
                        multi_model_structure = MultiModelStructure.from_structures(sequential_structures)
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
                    logger.info(
                        "Writing designed structure residue seqids from %s.",
                        primary_residue_index_key,
                    )
                    save_structures_to_file(structures, file_name, primary_residue_index_key=primary_residue_index_key)
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
                    if raw_structures is not None:
                        raw_file_name = save_dir / f"rawcontext_{sample_name}.{file_extension}"
                        save_structures_to_file(
                            raw_structures,
                            raw_file_name,
                            primary_residue_index_key=primary_residue_index_key,
                        )
                        output_files.append(
                            FileOutputConfig(
                                state_key=observable_key,
                                features_key=reference_features_key,
                                timestep=timesteps,
                                output_format=output_format,
                                path=raw_file_name,
                                metadata=features_metadata,
                            )
                        )

        return output_files


def _has_noised_reference_context(features: Features) -> bool:
    if features.input_reference_structure is None:
        return False

    inpaint_feature = features.feature_dict["inpaint_feature"]
    context_mask = (
        (inpaint_feature == InpaintFeature.BINDER_CONTEXT) | (inpaint_feature == InpaintFeature.TARGET_CONTEXT)
    ) & features.feature_dict["token_pad_mask"].bool()
    return bool((context_mask & features.feature_dict["noised_mask"].bool()).any().item())
