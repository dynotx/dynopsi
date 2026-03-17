import uuid
from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from dynopsi.data.constants import TOKENS, AtomFilter, PocketFeature
from dynopsi.data.featurization.utils import add_oxygen_to_bb_features, strip_chain_padding
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Structure,
)
from dynopsi.data.structure_io import save_structures_to_file
from dynopsi.data.utils import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP


####################
### Features
###################
class Features(BaseModel, Mapping):
    """Holds and manages input features and corresponding metadata.

    A Features object holds a dictionary of features that are input to a DynoPsi model (e.g. residue types and indices) and metadata about the steps taken to generate them (e.g. names and outputs of individual Featurizers used in the featurization pipeline).

    Attributes
    ----------
    feature_dict: dict[str, torch.Tensor], the dictionary of features that will be input to the dynopsi model.
    atoms_to_include: AtomFilter | None, the atoms to include in the features. If None, this value
       will be set by featurizers/featurization pipelines. If provided and if there is a mismatch
       with a featurization step, an error will be raised.
    featurization_steps: list[tuple[str, str]], the list of steps taken to generate the features. Each tuple contains the type and name of the featurizer. Managed internally; not set by the user.
    featurization_metadata: dict[str, Any], the dictionary of metadata about the features, keyed by the name of the featurizer. Managed internally; not set by the user.
    input_reference_structure: Structure | None, the full-atom input reference structure. This will be
       modified to reflect cropping, rotations, translations that are performed on the input. Managed internally; not set by the user.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # feature dict ultimately input to the model
    feature_dict: dict[str, torch.Tensor] = Field(default_factory=dict)

    # atoms to include in the features
    atoms_to_include: AtomFilter | None = None

    # list of steps taken to generate the features
    featurization_steps: list[tuple[str, str]] = Field(default_factory=list, init=False)
    # tracks information output by featurizers
    # key is the name of the featurization_step, value is the metadata
    featurization_metadata: dict[str, Any] = Field(default_factory=dict, init=False)
    # specific tracking of the full-atom input reference structure
    # which will be modified to reflect cropping, rotations, translations
    # that are performed on the input
    input_reference_structure: Structure | None = Field(default=None, init=False)

    def add_step(self, featurizer_type: str, featurizer_name: str) -> None:
        """Add a featurization step to the processing pipeline.

        Parameters
        ----------
        featurizer_type : str
            The type/category of the featurizer (e.g., 'structure', 'sequence').
        featurizer_name : str
            The specific name or identifier of the featurizer.
        """
        self.featurization_steps.append((featurizer_type, featurizer_name))

    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update a metadata entry for the featurization process.

        Parameters
        ----------
        key : str
            The metadata key.
        value : Any
            The metadata value to associate with the key.
        """
        self.featurization_metadata[key] = value

    def get_metadata(self, verbose: bool = True) -> dict[str, Any]:
        """Retrieve featurization metadata augmented with binder and target chain information.

        Parameters
        ----------
        verbose : bool, default=True
            If True, include the full featurization metadata. If False, return only
            the binder and target chain IDs.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'binder_chains': list of single-letter chain IDs for binder residues
            - 'target_chains': list of single-letter chain IDs for target residues
            - Additional featurization metadata entries (if verbose=True)
        """
        if verbose:
            metadata = self.featurization_metadata.copy()
        else:
            metadata = {}

        binder_asym_ids = (
            self.feature_dict["asym_id"][
                (self.feature_dict["binder_mask"].bool() & self.feature_dict["token_pad_mask"].bool())
            ]
            .unique()
            .tolist()
        )
        binder_auth_asym_ids = (
            self.feature_dict["auth_asym_id"][
                (self.feature_dict["binder_mask"].bool() & self.feature_dict["token_pad_mask"].bool())
            ]
            .unique()
            .tolist()
        )
        target_asym_ids = (
            self.feature_dict["asym_id"][
                (self.feature_dict["target_mask"].bool() & self.feature_dict["token_pad_mask"].bool())
            ]
            .unique()
            .tolist()
        )
        target_auth_asym_ids = (
            self.feature_dict["auth_asym_id"][
                (self.feature_dict["target_mask"].bool() & self.feature_dict["token_pad_mask"].bool())
            ]
            .unique()
            .tolist()
        )

        binder_chains = [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in binder_asym_ids]
        target_chains = [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in target_asym_ids]
        binder_auth_chains = [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id] for auth_asym_id in binder_auth_asym_ids]
        target_auth_chains = [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id] for auth_asym_id in target_auth_asym_ids]

        augmented_metadata = {
            **metadata,
            "binder_chains": binder_chains,
            "target_chains": target_chains,
            "binder_auth_chains": binder_auth_chains,
            "target_auth_chains": target_auth_chains,
            "reference_structure_input": self.input_reference_structure,
        }

        return augmented_metadata

    def safe_copy(self) -> Self:
        """Return a deep copy of this Features object.

        Tensors in feature_dict are cloned via torch.Tensor.clone(); all other
        fields are deep copied via copy.deepcopy.
        """
        feature_dict_copy = {
            k: v.clone() if isinstance(v, torch.Tensor) else deepcopy(v) for k, v in self.feature_dict.items()
        }

        features_copy = self.__class__(feature_dict=feature_dict_copy, atoms_to_include=deepcopy(self.atoms_to_include))
        features_copy.featurization_steps = deepcopy(self.featurization_steps)
        features_copy.featurization_metadata = deepcopy(self.featurization_metadata)
        features_copy.input_reference_structure = deepcopy(self.input_reference_structure)
        return features_copy

    def to_structure(self) -> Structure:
        """Convert features dictionary to a Structure object.

        Transforms internal tensor-based features in the feature dictionary
        into a Structure that can be saved to a file.
        For backbone-only representations (N, CA, C), oxygen
        atoms are added to complete the backbone.

        The pocket_feature values are encoded in the B-factor field:
        - POCKET residues: 100.0
        - BINDER residues: 50.0
        - Other residues: 0.0 (e.g. UNSELECTED or UNSPECIFIED residues)

        Returns
        -------
        Structure
            A Structure object containing atoms, residues, chains, and features.
            Interface arrays are returned empty.

        Raises
        ------
        NotImplementedError
            If `atoms_to_include` is not `AtomFilter.BB`. Only backbone atoms
            are currently supported.

        Notes
        -----
        Interface computation is not currently implemented; the returned Structure
        will have empty `interfaces` and `residue_interfaces` arrays.
        """
        feature_dict_nopad_tensor = strip_chain_padding(self.feature_dict)

        if self.atoms_to_include == AtomFilter.BB:
            feature_dict_nopad_tensor = add_oxygen_to_bb_features(feature_dict_nopad_tensor)
            n_residues = len(feature_dict_nopad_tensor["token_index"])
            bb_atom_names = AtomFilter.BB.to_atom_names() + ("O",)
            atom_names: np.ndarray = np.tile(np.array(bb_atom_names, dtype=np.str_), n_residues)
        else:
            raise NotImplementedError("Only BB atoms are supported")

        feature_dict_nopad = {key: value.numpy() for key, value in feature_dict_nopad_tensor.items()}

        atom_to_token_idx = feature_dict_nopad["atom_to_token"].argmax(
            axis=1
        )  # shape (n_atoms,) w/ index as the token index

        # use hotspot to encode b factor
        if "pocket_feature" in feature_dict_nopad.keys():
            pocket_feature_to_bfactor = np.zeros(len(feature_dict_nopad["pocket_feature"]))
            pocket_feature_to_bfactor[feature_dict_nopad["pocket_feature"] == PocketFeature.POCKET] = 100.0
            pocket_feature_to_bfactor[feature_dict_nopad["pocket_feature"] == PocketFeature.BINDER] = 50.0
            bfactor = pocket_feature_to_bfactor[atom_to_token_idx]
        else:
            bfactor = torch.ones(len(feature_dict_nopad["atom_resolved_mask"])) * 100.0

        atom_dtype_input = {
            "asym_id": feature_dict_nopad["asym_id"][atom_to_token_idx],
            "res_idx": feature_dict_nopad["residue_index"][atom_to_token_idx],
            "atom_name": atom_names,
            "coords": feature_dict_nopad["coords"][0],
            "is_present": feature_dict_nopad["atom_resolved_mask"],
            "bfactor": bfactor,
        }
        atoms = np.empty(len(atom_to_token_idx), dtype=ATOMS_DTYPE)
        for key, value in atom_dtype_input.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            atoms[key] = value

        # Assign the residue information to the residues array
        token_to_atom_max_idx = (len(feature_dict_nopad["atom_to_token"]) - 1) - np.flip(
            feature_dict_nopad["atom_to_token"], axis=0
        ).argmax(axis=0)
        token_to_atom_min_idx = feature_dict_nopad["atom_to_token"].argmax(axis=0)

        num_atoms_per_residue = token_to_atom_max_idx - token_to_atom_min_idx + 1

        if self.atoms_to_include == AtomFilter.BB:
            atom_center = token_to_atom_min_idx + 1
            is_present = (atoms["is_present"][:, None] & (feature_dict_nopad["atom_to_token"]).astype(bool)).any(axis=0)
            ca_present = feature_dict_nopad["token_resolved_mask"]
            backbone_present = feature_dict_nopad["frame_resolved_mask"]
        else:
            raise NotImplementedError("Only BB atoms are supported")

        residue_dtype_input = {
            "res_type": feature_dict_nopad["res_type"],
            "asym_id": feature_dict_nopad["asym_id"],
            "res_idx": feature_dict_nopad["residue_index"],
            "auth_res_idx": feature_dict_nopad["auth_residue_index"],
            "atom_start_idx": token_to_atom_min_idx,
            "num_atoms": num_atoms_per_residue,
            "atom_center": atom_center,
            "is_standard": np.ones(len(feature_dict_nopad["residue_index"]), dtype=bool),
            "is_present": is_present,
            "ca_present": ca_present,
            "backbone_present": backbone_present,
        }
        residue_dtype_input["res_name"] = np.array(
            [TOKENS[res_type] for res_type in feature_dict_nopad["res_type"]], dtype=str
        )

        residues = np.empty(len(feature_dict_nopad["residue_index"]), dtype=RESIDUES_DTYPE)
        for key, value in residue_dtype_input.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            residues[key] = value

        # Assign the chain information to the chains array
        unique_asym_ids = list(np.unique(feature_dict_nopad["asym_id"]))
        asym_id_to_auth_asym_id = {
            asym_id: auth_asym_id
            for asym_id, auth_asym_id in zip(
                feature_dict_nopad["asym_id"].tolist(), feature_dict_nopad["auth_asym_id"].tolist()
            )
        }

        chains_dtype_input = {
            "asym_id": np.array([asym_id for asym_id in unique_asym_ids]),
            "auth_asym_id": np.array([asym_id_to_auth_asym_id[asym_id] for asym_id in unique_asym_ids]),
            "chain_id": np.array([ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in unique_asym_ids]),
            "auth_chain_id": np.array(
                [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id_to_auth_asym_id[asym_id]] for asym_id in unique_asym_ids]
            ),
            "sym_id": np.array(
                [
                    np.unique(feature_dict_nopad["sym_id"][feature_dict_nopad["asym_id"] == asym_id])[0]
                    for asym_id in unique_asym_ids
                ]
            ),
            "entity_id": np.array(
                [
                    np.unique(feature_dict_nopad["entity_id"][feature_dict_nopad["asym_id"] == asym_id])[0]
                    for asym_id in unique_asym_ids
                ]
            ),
            "atom_start_idx": np.array(
                [token_to_atom_min_idx[feature_dict_nopad["asym_id"] == asym_id].min() for asym_id in unique_asym_ids]
            ),
            "num_atoms": np.array(
                [num_atoms_per_residue[feature_dict_nopad["asym_id"] == asym_id].sum() for asym_id in unique_asym_ids]
            ),
            "res_start_idx": np.array(
                [
                    feature_dict_nopad["token_index"][feature_dict_nopad["asym_id"] == asym_id].min()
                    for asym_id in unique_asym_ids
                ]
            ),
            "num_residues": np.array([(feature_dict_nopad["asym_id"] == asym_id).sum() for asym_id in unique_asym_ids]),
        }

        chains = np.empty(len(unique_asym_ids), dtype=CHAINS_DTYPE)
        for key, value in chains_dtype_input.items():
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            chains[key] = value

        # features
        if (
            len(
                set(feature_dict_nopad.keys())
                & set(["binder_mask", "noised_mask", "pocket_feature", "inpaint_feature"])
            )
            == 4
        ):
            features_dtype_input = {
                "is_designed": feature_dict_nopad["binder_mask"],
                "is_noised": feature_dict_nopad["noised_mask"],
                "pocket_feature": feature_dict_nopad["pocket_feature"],
                "inpaint_feature": feature_dict_nopad["inpaint_feature"],
            }
            features = np.empty(len(feature_dict_nopad["binder_mask"]), dtype=DYNO_PSI_FEATURES_DTYPE)
            for key, value in features_dtype_input.items():
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                features[key] = value
        else:
            features = np.empty(len(feature_dict_nopad["token_index"]), dtype=DYNO_PSI_FEATURES_DTYPE)

        return Structure(
            atoms=atoms,
            residues=residues,
            chains=chains,
            interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
            residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
            features=features,
        )

    def save_to_pdb(self, file_path: Path | str) -> None:
        """Save the features to a PDB file.

        Converts the internal feature dictionary to a Structure object
        and writes the Structure in PDB format to the specified file path.

        Parameters
        ----------
        file_path : Path | str
            Destination file path. Must have a `.pdb` extension.

        Raises
        ------
        ValueError
            If the output `file_path` provided is not a valid .pdb path.

        Notes
        -----
        This will exclude sequence metadata for missing residues.

        See Also
        --------
        to_structure : Converts features to a Structure object.
        """
        if file_path.lower().endswith(".pdb"):
            structure = self.to_structure()
            save_structures_to_file([structure], file_path)
        else:
            raise ValueError(f"Invalid file path: {file_path}")

    def save_to_mmcif(self, file_path: Path | str) -> None:
        """Save the features to an mmCIF file.

        Converts the internal feature representation to a Structure object
        and writes it to the specified file path.

        Parameters
        ----------
        file_path : Path | str
            Destination file path. Must have a `.mmcif` or `.cif` extension.

        Raises
        ------
        ValueError
            If `file_path` does not end with `.mmcif` or `.cif`.

        See Also
        --------
        to_structure : Converts features to a Structure object.
        data.structure.Structure.save_to_mmcif : Underlying save implementation.
        """
        if file_path.lower().endswith(".mmcif") or file_path.lower().endswith(".cif"):
            structure = self.to_structure()
            save_structures_to_file([structure], file_path)
        else:
            raise ValueError(f"Invalid file path: {file_path}")

    def save_to_npz(self, file_path: Path | str) -> None:
        """Save the features to a .npz file.

        Converts the internal feature representation to a Structure object
        and writes it to the specified file path.

        Parameters
        ----------
        file_path : Path | str
            Destination file path. Must have a `.npz` extension.

        Raises
        ------
        ValueError
            If `file_path` does not end with `.npz`.

        See Also
        --------
        to_structure : Converts features to a Structure object.
        data.structure.Structure.dump_to_npz : Underlying save implementation.
        """
        if file_path.lower().endswith(".npz"):
            structure = self.to_structure()
            structure.dump_to_npz(file_path)
        else:
            raise ValueError(f"Invalid file path: {file_path}")

    def to_dict(self) -> dict[str, Any]:
        """Return the underlying feature dictionary.

        Returns
        -------
        dict[str, Any]
            The feature dictionary containing all featurized data.
        """
        return self.feature_dict

    def keys(self) -> KeysView[str]:
        """Return a view of the feature dictionary keys.

        Returns
        -------
        KeysView[str]
            A view of the feature names.
        """
        return self.feature_dict.keys()

    def values(self) -> ValuesView[Any]:
        """Return a view of the feature dictionary keys.

        Returns
        -------
        KeysView[str]
            A view of the feature names.
        """
        return self.feature_dict.values()

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the feature dictionary items.

        Returns
        -------
        ItemsView[str, Any]
            A view of (feature_name, feature_value) pairs.
        """
        return self.feature_dict.items()

    def __getitem__(self, key: str) -> Any:
        return self.feature_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.feature_dict[key] = value

    def __len__(self) -> int:
        # asym_id must be in the dict
        if "asym_id" not in self.feature_dict.keys():
            if len(self.feature_dict.keys()) > 0:
                raise ValueError(
                    f"Keys in the feature dict must include asym_id, but {self.feature_dict.keys()} are present"
                )
            else:
                return 0
        else:
            return len(self.feature_dict["asym_id"])


class Featurizer(BaseModel):
    """Base class for runtime featurization steps.

    Featurizer classes actually perform the featurization steps by
    transforming Features objects. Specifically, Featurizers both
    applying specific processing steps and record provenance by
    annotating each Features object with the step applied and
    associated metadata (e.g., updated asym_id mappings).
    They also include validation logic to prevent invalid featurization
    steps.

    The input/output contract:
    - Input: Iterable[Features]
    - Output: Iterator[Features]
    - Features are treated as immutable, so new Features objects must be created for any modifications. See Features.safe_copy().
    - May produce one or many outputs per input (branching).
    - Must record provenance via `Features.add_step(...)` and optionally metadata via `Features.add_metadata(...)`.

    NOTE: To enable serialization/deserialization, featurizer subclasses must define a type attribute that is a literal
    (e.g. type: Literal["my_featurizer_subclass"] = "my_featurizer_subclass") and must be registered as a featurizer using
    the @register_featurizer decorator. See dynopsi.data.featurization.registry for more details.

    See Also
    --------
    Features.add_step : Method to record featurization provenance.
    Features.add_metadata : Method to record featurization metadata.
    """

    # name is used to identify the featurizer in the features metadata
    # useful where there are multiple featurizers of the same class
    name: str = Field(default_factory=str)

    verbose_metadata: bool = False

    @model_validator(mode="after")
    def set_name(self) -> Self:
        # if the name is not provided, make it a unique id
        if self.name == "":
            self.name = uuid.uuid4().hex[:6]
        return self

    def _update_features_metadata(self, features: Features, more_metadata: dict[str, Any] = {}) -> None:
        """Record the featurization step and associated metadata on a Features object.

        Verifies this featurizer is registered as one of the `features` featurization steps
        and updates `features` step metadata with `more_metadata` associated with
        this featurization step.

        Arguments
        ----------
        features: Features
            The features object to update.
        more_metadata: dict[str, Any]
            Additional metadata to add to the features step metadata. It will be merged with the existing metadata,
            so any keys in `more_metadata` will override the config metadata.

        Raises
        ------
        ValueError:
            If the featurizer type is not set. The type attribute should be set for all subclasses of Featurizer.
        """
        featurizer_type = getattr(self, "type", None)
        if featurizer_type is None or self.model_fields.get("type", None) is None:
            raise ValueError(
                "Featurizer type is not set, this is required for provenance tracking. Please define a type attribute in the featurizer subclass."
            )
        featurizer_type = str(featurizer_type)
        # add the step and metadata
        features.add_step(featurizer_type=featurizer_type, featurizer_name=self.name)
        # add the config metadata
        features.add_metadata(
            self.name,
            {
                **self.model_dump(),
                **more_metadata,
            },
        )

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        raise NotImplementedError("Subclasses must implement this method")
