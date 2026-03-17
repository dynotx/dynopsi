import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import smart_open
import torch
from torch import from_numpy
from typing_extensions import Self

from dynopsi.data.constants import AtomFilter
from dynopsi.data.utils import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP, ChainResidueRepr, align_by_mask

logger = logging.getLogger(__name__)

CHAINS_DTYPE = np.dtype(
    [
        # alpha chain id, we only allow single letter chain ids
        ("chain_id", "<U1"),
        # auth_chain_id: the author-provided chain_id of the chain
        ("auth_chain_id", "<U1"),
        # asym_id; we will only allow asym_ids 0-25
        ("asym_id", "i1"),
        # auth_asym_id: the author-provided asym_id of the chain
        ("auth_asym_id", "i1"),
        # entity_id: the number is constrained by the asym_id
        ("entity_id", "i1"),
        # sym_id: the number is constrained by the asym_id
        ("sym_id", "i1"),
        # atom_start_idx: the index of the first atom in the chain
        ("atom_start_idx", "i4"),
        # atom_num: the number of atoms in the chain
        ("num_atoms", "i4"),
        # res_start_idx: the index of the first residue in the chain
        ("res_start_idx", "i4"),
        # res_num: the number of residues in the chain
        ("num_residues", "i4"),
    ],
    align=False,
)

RESIDUES_DTYPE = np.dtype(
    [
        # name: the residue 3-letter code
        ("res_name", "<U3"),
        # res_type: integer code for the residue type, see TOKEN_IDS in constants.py
        ("res_type", "i1"),
        # asym_id: the asym_id of the chain
        ("asym_id", "i1"),
        # res_idx: the index of the residue in the chain, starting from 0 with respect to the chain
        ("res_idx", "i4"),
        # attempts to pull from the structure's residue index mapping
        # mmCif is lenient on author residue index mapping (doesn't require ints)
        ("auth_res_idx", "i4"),
        # atom_idx: the index of the first atom in the residue, 0-indexed with respect to the entire structure
        ("atom_start_idx", "i4"),
        # num_atoms: the number of atoms in the residue
        ("num_atoms", "i4"),
        # atom_center: the index of the atom that is the center of the residue
        ("atom_center", "i4"),
        # is_standard: whether the residue is a standard residue
        ("is_standard", "?"),
        # is_present: whether the residue is present in the structure at all
        ("is_present", "?"),
        # ca_present: whether the residue has a CA atom present in the structure
        ("ca_present", "?"),
        # backbone_present: whether the residue has backbone atoms N, CA, C, and O present in the structure
        ("backbone_present", "?"),
    ],
    align=False,
)

ATOMS_DTYPE = np.dtype(
    [
        # asym_id: the asym_id of the chain
        ("asym_id", "i1"),
        # res_idx: the index of the residue in the chain
        ("res_idx", "i4"),
        # name: the atom name
        ("atom_name", "<U4"),
        # coords: the atom coordinates
        ("coords", "3f4"),
        # is_present: whether the atom is present in the structure
        ("is_present", "?"),
        # bfactor: the atom B-factor or nan if not experimentally determined
        ("bfactor", "f4"),
        # plddt: the atom pLDDT or nan if not predicted
        ("plddt", "f4"),
    ],
    align=False,
)

CHAIN_INTERFACE_DTYPE = np.dtype(
    [
        ("chain_1", np.dtype("i4")),
        ("chain_2", np.dtype("i4")),
    ],
    align=False,
)

RESIDUE_INTERFACE_DTYPE = np.dtype(
    [
        # chain_1 asym_id
        ("chain_1", np.dtype("i4")),
        # chain_2 asym_id
        ("chain_2", np.dtype("i4")),
        # residue from chain_1 res_idx in chain_1
        ("res_idx_1", np.dtype("i4")),
        # residue from chain_2 res_idx in chain_2
        ("res_idx_2", np.dtype("i4")),
    ],
    align=False,
)

DYNO_PSI_FEATURES_DTYPE = np.dtype(
    [
        # is_designed (residue can be either fixed or designed)
        ("is_designed", "?"),
        # is_noised (residue can be either non-noised or noised)
        ("is_noised", "?"),
        # hotspot -> pocket_feature
        ("pocket_feature", "i1"),
        # inpainting -> inpaint_feature
        ("inpaint_feature", "i1"),
    ],
    align=False,
)


@dataclass(frozen=True, slots=True)
class Chain:
    """A chain object.

    Attributes
    ----------
    name: str, the PDB-standard chain name
    asym_id: int, the PDB-standard asym_id of the chain
    sym_id: int, the sym_id of the chain
    entity_id: int, the entity_id of the chain
    ungapped_sequence: list[str], the ungapped sequence of the chain
    """

    name: str
    asym_id: int
    sym_id: int
    entity_id: int
    ungapped_sequence: list[str]


@dataclass(frozen=True, slots=True)
class Structure:
    """Structure datatype.

    Attributes
    ----------
    atoms: np.ndarray of ATOMS_DTYPE, shape (n_atoms, )
    residues: np.ndarray of RESIDUES_DTYPE, shape (n_residues, )
    chains: np.ndarray of CHAINS_DTYPE, shape (n_chains, )
    interfaces: np.ndarray of CHAIN_INTERFACE_DTYPE, shape (n_chain_interfaces, 2)
    residue_interfaces: np.ndarray of RESIDUE_INTERFACE_DTYPE, shape (n_residue_interfaces, 4)
    features: np.ndarray of FEATURES_DTYPE, shape (n_residues, )

    NOTE: atoms, residues, and chains are expected to be non-empty.
    Interfaces, residue_interfaces, and features can be empty arrays.
    """

    atoms: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    interfaces: np.ndarray
    residue_interfaces: np.ndarray
    features: np.ndarray

    @classmethod
    def load_from_npz(cls: type[Self], path: Path | str) -> Self:
        """Load the structure from an NPZ file."""
        with smart_open.open(path, "rb") as f:
            structure = np.load(f)
            if "atoms" not in structure.keys():
                if "atoms_0" in structure.keys():
                    raise ValueError(
                        "This is likely a MultiModelStructure, load with MultiModelStructure.load_from_npz."
                    )
                else:
                    raise ValueError("Invalid input, no 'atoms' attribute")

            kwargs = {
                "atoms": structure["atoms"],
                "residues": structure["residues"],
                "chains": structure["chains"],
            }
            kwargs["interfaces"] = np.empty(0, dtype=CHAIN_INTERFACE_DTYPE)
            kwargs["residue_interfaces"] = np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE)
            kwargs["features"] = np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE)

            return cls(**kwargs)

    def dump_to_npz(self, path: Path | str) -> None:
        """Dump the structure to an NPZ file."""
        np.savez_compressed(str(path), **asdict(self))

    def get_auth_chain_id_to_pdb_chain_id_map(self) -> dict[str, str]:
        """Get a map from auth_chain_id to pdb_chain_id."""
        return {str(chain["auth_chain_id"]): str(chain["chain_id"]) for chain in self.chains}

    def get_pdb_chain_id_to_auth_chain_id_map(self) -> dict[str, str]:
        """Get a map from pdb_chain_id to auth_chain_id."""
        return {str(chain["chain_id"]): str(chain["auth_chain_id"]) for chain in self.chains}

    @staticmethod
    def update_start_indices(
        chains: np.ndarray, residues: np.ndarray, atoms: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update fields in the chains, residues, and atoms arrays after resizing (crop or concatenation).

        Specifically, update the following fields:
        - `residues.atom_start_idx`, `residues.atom_center` to reflect the changes in `atom` array
        - `chains.atom_start_idx`, `chains.num_atoms` to reflect the changes in `atom` array
        - `chains.res_start_idx`, `chains.num_residues` to reflect the changes in `residue` array
        - `residues.res_idx` and `atoms.res_idx` to reset res_idx to 0 at the start of each chain

        Parameters
        ----------
        chains: np.ndarray of CHAINS_DTYPE
        residues: np.ndarray of RESIDUES_DTYPE
        atoms: np.ndarray of ATOMS_DTYPE

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The updated chains, residues, and atoms arrays (same dtypes as inputs)
        """
        for asym_id in chains["asym_id"]:
            # update absolute positions within the arrays
            start_atom_idx = np.argmax(atoms["asym_id"] == asym_id)
            start_residue_idx = np.argmax(residues["asym_id"] == asym_id)

            asym_id_atoms = atoms[atoms["asym_id"] == asym_id]
            asym_id_residues = residues[residues["asym_id"] == asym_id]

            center_minus_start = asym_id_residues["atom_center"] - asym_id_residues["atom_start_idx"]
            residue_arr_atom_starts = np.unique(asym_id_atoms["res_idx"], return_index=True)[1] + start_atom_idx
            residue_arr_atom_centers = residue_arr_atom_starts + center_minus_start

            residues["atom_center"][residues["asym_id"] == asym_id] = residue_arr_atom_centers
            residues["atom_start_idx"][residues["asym_id"] == asym_id] = residue_arr_atom_starts

            chains["atom_start_idx"][chains["asym_id"] == asym_id] = start_atom_idx
            chains["res_start_idx"][chains["asym_id"] == asym_id] = start_residue_idx

            # update the number of atoms and residues
            chains["num_atoms"][chains["asym_id"] == asym_id] = len(asym_id_atoms)
            chains["num_residues"][chains["asym_id"] == asym_id] = len(asym_id_residues)

            # update the res_idx to start from 0 for the chain
            chain_min_res_idx = np.min(asym_id_residues["res_idx"])
            residues["res_idx"][residues["asym_id"] == asym_id] -= chain_min_res_idx
            atoms["res_idx"][atoms["asym_id"] == asym_id] -= chain_min_res_idx

        return chains, residues, atoms

    def update_asym_ids(self, ref_asym_id_map: dict[int, int], auth_asym_id_map: dict[int, int]) -> Self:
        """Update the asym_ids and chains according to the provided maps.

        Parameters
        ----------
        ref_asym_id_map: dict[int, int]
            A map from the reference asym_ids to the new asym_ids.
        auth_asym_id_map: dict[int, int]
            A map from the reference auth_asym_ids to the new auth_asym_ids.
        """
        # auth_asym_id updates are only performed on the chains
        new_chains = self.chains.copy()
        new_chains["asym_id"] = [ref_asym_id_map[asym_id] for asym_id in self.chains["asym_id"]]
        new_chains["auth_asym_id"] = [auth_asym_id_map[auth_asym_id] for auth_asym_id in self.chains["auth_asym_id"]]
        new_chains["chain_id"] = [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in new_chains["asym_id"]]
        new_chains["auth_chain_id"] = [
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id] for auth_asym_id in new_chains["auth_asym_id"]
        ]

        new_residues = self.residues.copy()
        new_residues["asym_id"] = [ref_asym_id_map[asym_id] for asym_id in self.residues["asym_id"]]

        new_atoms = self.atoms.copy()
        new_atoms["asym_id"] = [ref_asym_id_map[asym_id] for asym_id in self.atoms["asym_id"]]

        new_interfaces = self.interfaces.copy()
        new_interfaces["chain_1"] = [
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[ref_asym_id_map[asym_id]] for asym_id in self.interfaces["chain_1"]
        ]
        new_interfaces["chain_2"] = [
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[ref_asym_id_map[asym_id]] for asym_id in self.interfaces["chain_2"]
        ]

        new_residue_interfaces = self.residue_interfaces.copy()
        new_residue_interfaces["chain_1"] = [
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[ref_asym_id_map[asym_id]]
            for asym_id in self.residue_interfaces["chain_1"]
        ]
        new_residue_interfaces["chain_2"] = [
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[ref_asym_id_map[asym_id]]
            for asym_id in self.residue_interfaces["chain_2"]
        ]

        return self.__class__(
            atoms=new_atoms,
            residues=new_residues,
            chains=new_chains,
            interfaces=new_interfaces,
            residue_interfaces=new_residue_interfaces,
            features=self.features.copy(),
        )

    def crop(
        self,
        residue_index_chain_spec: ChainResidueRepr | dict[str, Any],
        index_type: Literal["residue_index", "auth_residue_index"] = "residue_index",
        inverse_crop: bool = False,
    ) -> Self:
        """Crop the structure based on the residue index chain specification and return a new structure.

        This sets interface and other features information to empty arrays of the appropriate dtype.

        Parameters
        ----------
        residue_index_chain_spec : ChainResidueRepr| dict[str, list[int] | None]
            The residue index chain specification to crop the structure to, either as a ChainResidueRepr or a dictionary
            representation of a ChainResidueRepr.
        index_type : Literal["residue_index", "auth_residue_index"]
            The type of index to use for the residue index chain specification.
        inverse_crop : bool, default False
            Whether to inverse the crop, i.e. crop the structure to the complement of the residue index chain specification.

        Returns
        -------
        A new Structure with the cropped residues, atoms, and chains. The res_idx field will be reset to start at 0 for each asym_id.
        """
        if isinstance(residue_index_chain_spec, dict):
            chain_residue_rep = ChainResidueRepr.model_validate(residue_index_chain_spec)
        else:
            chain_residue_rep = residue_index_chain_spec

        # entrypoint is residues because this is defined by the residue index chain specification
        if index_type == "auth_residue_index":
            idx_attr = "auth_res_idx"
            asym_id_attr = "auth_asym_id"
        else:
            idx_attr = "res_idx"
            asym_id_attr = "asym_id"

        res_idx_tensor = from_numpy(self.residues[idx_attr].copy())
        asym_id_tensor = from_numpy(np.repeat(self.chains[asym_id_attr], self.chains["num_residues"]))

        residue_keep_mask = chain_residue_rep.to_mask(
            asym_id=asym_id_tensor,
            residue_index=res_idx_tensor,
            pad_mask=torch.ones(len(self.residues), dtype=torch.bool),
            index_type=index_type,
        )
        if inverse_crop:
            residue_keep_mask = ~residue_keep_mask
        residues = self.residues[residue_keep_mask.numpy()]

        # now atoms based on present residues
        atom_keep_mask = np.zeros(self.atoms.shape[0]).astype(bool)
        for asym_id in np.unique(residues["asym_id"]):
            res_present = residues[residues["asym_id"] == asym_id]["res_idx"]
            atom_asym_id_mask = self.atoms["asym_id"] == asym_id
            atom_keep_mask[atom_asym_id_mask] = np.isin(self.atoms["res_idx"][atom_asym_id_mask], res_present)

        atoms = self.atoms[atom_keep_mask]

        chains = self.chains[np.isin(self.chains["asym_id"], residues["asym_id"])]

        chains, residues, atoms = self.update_start_indices(chains, residues, atoms)

        if len(self.features) == len(residues):
            # features are either empty or same size as residues
            features = self.features[residue_keep_mask.numpy()]
        else:
            features = np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE)

        return self.__class__(
            atoms=atoms,
            residues=residues,
            chains=chains,
            interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
            residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
            features=features,
        )

    def concat(self, other: Self) -> Self:
        """Concatenate this structure with another structure.

        Parameters
        ----------
        other : Self
            The other structure to concatenate with.

        Returns
        -------
        A new

        Raises
        ------
        NotImplementedError
            If concatenating structures with the same asym_id. There is not currently support for
            checking for collisions at the residue_index level, so we check that asym_ids are unique.
        """
        if np.intersect1d(self.chains["asym_id"], other.chains["asym_id"]).shape[0] > 0:
            raise NotImplementedError("Concatenating Structures with the same asym_id is not currently supported.")
        atoms = np.concatenate([self.atoms, other.atoms], axis=0)
        residues = np.concatenate([self.residues, other.residues], axis=0)
        chains = np.concatenate([self.chains, other.chains], axis=0)

        # only concat features if both are present, otherwise set to empty
        if len(self.features) == len(self.residues) and len(other.features) == len(other.residues):
            features = np.concatenate([self.features, other.features])
        else:
            features = np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE)

        chains_out, residues_out, atoms_out = self.update_start_indices(chains=chains, residues=residues, atoms=atoms)

        return self.__class__(
            atoms=atoms_out,
            residues=residues_out,
            chains=chains_out,
            interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
            residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
            features=features,
        )

    def align_to_feature_dict(self, feature_dict: dict[str, torch.Tensor], atoms_to_include: AtomFilter) -> Self:
        """
        Align this Structure to the atomic positions defined by feature_dict['coords']

        Parameters
        ----------
        feature_dict : dict[str, torch.Tensor]
            A dictionary containing the features to align to. Required keys: `coords`, `atom_to_token`, `residue_index`, `asym_id`
        atoms_to_include : AtomFilter
            AtomFilter specifying which atoms to include in the alignment.

        Returns
        -------
        A new Structure object, aligned to atoms shared with feature_dict['coords']

        NOTE: Structure or feature_dict['coords'] must contain matching chains and residues in the same order,
        even if there are more atoms in one than the other (for example, the Structure can be full-atom, but
        feature_dict['coords'] can contain only backbone N, CA, C atoms)
        """

        def construct_unique_keys(
            asym_ids: np.ndarray, residue_indices: np.ndarray, atom_names: np.ndarray
        ) -> np.ndarray:
            return np.array([f"{a}_{r}_{n}".format(a, r, n) for a, r, n in zip(asym_ids, residue_indices, atom_names)])

        # match structure input to this
        # use numpy strings because doing this just a few times and safer
        struct_keys = construct_unique_keys(
            self.atoms["asym_id"].copy(), self.atoms["res_idx"].copy(), self.atoms["atom_name"].copy()
        )

        if atoms_to_include == AtomFilter.FULL:
            raise NotImplementedError("Alignment not implemented for full atom features.")
        else:
            atom_features_names = np.tile(atoms_to_include.to_atom_names(), feature_dict["token_index"].shape[0])
            atom_features_token_index = feature_dict["atom_to_token"].argmax(dim=1)
            atom_features_residue_index = feature_dict["residue_index"][atom_features_token_index]
            atom_features_asym_id = feature_dict["asym_id"][atom_features_token_index]
            atom_features_key = construct_unique_keys(
                atom_features_asym_id.numpy(), atom_features_residue_index.numpy(), atom_features_names
            )

        # identify shared keys
        atom_features_shared_keys_mask = np.isin(atom_features_key, struct_keys)
        struct_shared_keys_mask = np.isin(struct_keys, atom_features_key)

        # make sure they are identical order
        if not np.array_equal(struct_keys[struct_shared_keys_mask], atom_features_key[atom_features_shared_keys_mask]):
            raise ValueError("Shared Structure atoms and feature atoms are not in the same order")

        ref_coords = feature_dict["coords"][0]

        struct_coords_aligned = align_by_mask(
            coords1=from_numpy(self.atoms["coords"].copy()),  # [N, 3]
            coords2=ref_coords,  # [M, 3]
            mask1=from_numpy(struct_shared_keys_mask.copy()),  # [N]
            mask2=from_numpy(atom_features_shared_keys_mask.copy()),  # [M]
        )

        new_atoms = self.atoms.copy()
        new_atoms["coords"] = struct_coords_aligned

        return self.__class__(
            chains=self.chains.copy(),
            residues=self.residues.copy(),
            atoms=new_atoms,
            interfaces=self.interfaces.copy(),
            residue_interfaces=self.residue_interfaces.copy(),
            features=self.features.copy(),
        )


@dataclass
class MultiModelStructure:
    """A structure with multiple models for the same coordinates.
    This is used for trajectory saving.

    Attributes
    ----------
    atoms: list[np.ndarray] of ATOMS_DTYPE, shape (n_atoms, )
    residues: np.ndarray of RESIDUES_DTYPE, shape (n_residues, )
    chains: np.ndarray of CHAINS_DTYPE, shape (n_chains, )
    features: np.ndarray of FEATURES_DTYPE, shape (n_residues, )

    NOTE: unlike the Structure class, this does not have interfaces or residue_interfaces
    """

    atoms: list[np.ndarray]
    residues: np.ndarray
    chains: np.ndarray
    features: np.ndarray

    @classmethod
    def load_from_npz(cls: type[Self], path: Path | str) -> Self:
        with smart_open.open(path, "rb") as f:
            multi_model_structure = np.load(f)
            atom_keys_and_idx: list[tuple[str, int]] = [
                (key, int(key.split("_")[1])) for key in multi_model_structure.keys() if "atoms" in key
            ]
            atom_keys_and_idx = sorted(atom_keys_and_idx, key=lambda x: x[1])

            if len(atom_keys_and_idx) == 0:
                raise ValueError("Invalid input, expected 'atoms_idx' fields")

            kwargs = {key: multi_model_structure[key] for key, _ in atom_keys_and_idx} | {
                "residues": multi_model_structure["residues"],
                "chains": multi_model_structure["chains"],
            }
            if multi_model_structure.get("features", None) is None:
                kwargs["features"] = np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE)
            else:
                kwargs["features"] = multi_model_structure["features"]

            return cls(**kwargs)

    def dump_to_npz(self, path: Path | str) -> None:

        struct_dict = asdict(self)
        for model_idx, atom_entry in enumerate(self.atoms):
            struct_dict[f"atoms_{model_idx}"] = atom_entry
        struct_dict.pop("atoms")
        np.savez_compressed(str(path), **struct_dict)

    def get_structures(self) -> list[Structure]:
        """Get the structures for each model."""
        structures: list[Structure] = []
        for model_idx in range(len(self.atoms)):
            structure = Structure(
                atoms=self.atoms[model_idx],
                residues=self.residues,
                chains=self.chains,
                features=self.features,
                interfaces=np.empty(0),
                residue_interfaces=np.empty(0),
            )
            structures.append(structure)
        return structures

    @classmethod
    def from_structures(cls, structures: list[Structure]) -> Self:
        """Create a MultiModelStructure from a list of structures."""
        if len(structures) > 1:
            for structure in structures:
                if not np.array_equal(structure.residues, structures[0].residues):
                    raise ValueError("All structures must have the same residues.")
                if not np.array_equal(structure.chains, structures[0].chains):
                    raise ValueError("All structures must have the same chains.")
                if not np.array_equal(structure.features, structures[0].features):
                    raise ValueError("All structures must have the same features.")

        return cls(
            atoms=[structure.atoms for structure in structures],
            residues=structures[0].residues,
            chains=structures[0].chains,
            features=structures[0].features,
        )
