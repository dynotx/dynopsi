from typing import Any, Literal

import gemmi
import numpy as np
import pytest
import torch

from dynopsi.data.constants import TOKEN_IDS, AtomFilter
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    MultiModelStructure,
    Structure,
)
from dynopsi.data.structure_io import parse_mmcif, save_structures_to_file
from dynopsi.data.utils import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP, ChainResidueRepr

from .conftest import make_two_chain_structure

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_struct_info_chain_A(coordinates_chain_A: torch.Tensor) -> dict[str, Any]:
    return {
        "asym_id": torch.tensor([0, 0, 0, 0]),
        "entity_id": torch.tensor([0, 0, 0, 0]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2, 3]),
        "auth_residue_index": torch.tensor([1, 2, 3, 4]),
    }


@pytest.fixture
def base_struct_info() -> dict[str, Any]:
    return {
        "asym_id": torch.tensor([3, 3, 1, 4]),
        "entity_id": torch.tensor([0, 0, 1, 2]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 0, 0]),
        "auth_residue_index": torch.tensor([12, 13, 5, 8]),
        "coords": torch.randn(4 * 3, 3),  # 3 atoms per residue
        "atom_resolved_mask": torch.ones(4 * 3, dtype=torch.bool),
        "token_resolved_mask": torch.ones(4, dtype=torch.bool),
        "resname": np.array(["ALA", "LEU", "PHE", "MET"]),
    }


@pytest.fixture
def base_struct_info2() -> dict[str, Any]:
    return {
        "asym_id": torch.tensor([5, 5, 6, 6]),
        "entity_id": torch.tensor([0, 0, 1, 1]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 0, 1]),
        "auth_residue_index": torch.tensor([4, 5, 1, 2]),
        "coords": torch.randn(4 * 3, 3),  # 3 atoms per residue
        "atom_resolved_mask": torch.ones(4 * 3, dtype=torch.bool),
        "token_resolved_mask": torch.ones(4, dtype=torch.bool),
        "resname": np.array(["ARG", "PRO", "PHE", "ILE"]),
    }


@pytest.fixture
def feature_dict(base_struct_info: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "token_index": torch.arange(len(base_struct_info["asym_id"])),
        "residue_index": base_struct_info["residue_index"],
        "auth_residue_index": base_struct_info["auth_residue_index"],
        "asym_id": base_struct_info["asym_id"],
        "entity_id": base_struct_info["entity_id"],
        "sym_id": base_struct_info["sym_id"],
        "token_resolved_mask": base_struct_info["token_resolved_mask"],
        "token_pad_mask": torch.ones(4, dtype=torch.float),
        "coords": base_struct_info["coords"].unsqueeze(0),
        "frames_idx": torch.arange(4 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(4), 3, dim=0),
        "frame_resolved_mask": base_struct_info["token_resolved_mask"],
        "atom_resolved_mask": base_struct_info["atom_resolved_mask"],
        "atom_pad_mask": torch.ones_like(base_struct_info["atom_resolved_mask"], dtype=torch.float),
    }


def structure_from_base_struct_info(input_dict: dict[str, Any]) -> Structure:
    atoms = np.empty(
        len(input_dict["atom_resolved_mask"]),
        dtype=ATOMS_DTYPE,
    )
    atoms["asym_id"] = np.repeat(input_dict["asym_id"].numpy(), 3)
    atoms["res_idx"] = np.repeat(input_dict["residue_index"].numpy(), 3)
    atoms["atom_name"] = np.tile(["N", "CA", "C"], len(input_dict["asym_id"]))
    atoms["coords"] = input_dict["coords"].numpy()
    atoms["is_present"] = np.full(len(input_dict["atom_resolved_mask"]), True)
    atoms["bfactor"] = np.full(len(input_dict["atom_resolved_mask"]), 100.0)
    atoms["plddt"] = np.full(len(input_dict["atom_resolved_mask"]), np.nan)

    residues = np.empty(
        len(input_dict["residue_index"]),
        dtype=RESIDUES_DTYPE,
    )

    residues["res_name"] = input_dict["resname"].copy()
    residues["res_type"] = [TOKEN_IDS[resname] for resname in input_dict["resname"].copy()]
    residues["asym_id"] = input_dict["asym_id"].numpy()
    residues["res_idx"] = input_dict["residue_index"].numpy()
    residues["auth_res_idx"] = input_dict["auth_residue_index"].numpy()
    residues["atom_start_idx"] = np.arange(len(input_dict["asym_id"])) * 3
    residues["num_atoms"] = np.full(len(input_dict["residue_index"]), 3)
    residues["atom_center"] = residues["atom_start_idx"] + 1
    residues["is_standard"] = np.full(len(input_dict["residue_index"]), True)
    residues["is_present"] = np.full(len(input_dict["residue_index"]), True)
    residues["ca_present"] = np.full(len(input_dict["residue_index"]), True)
    residues["backbone_present"] = np.full(len(input_dict["residue_index"]), True)

    num_asym_ids = len(input_dict["asym_id"].unique())
    chains = np.empty(
        num_asym_ids,
        dtype=CHAINS_DTYPE,
    )
    asym_ids = list(dict.fromkeys(input_dict["asym_id"].tolist()))
    chains["chain_id"] = np.array([ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in asym_ids])
    chains["asym_id"] = np.array(asym_ids)
    chains["entity_id"] = np.arange(num_asym_ids)
    chains["sym_id"] = np.zeros(num_asym_ids)
    _, idx, counts = np.unique(input_dict["asym_id"].numpy(), return_index=True, return_counts=True)
    chains["res_start_idx"] = np.sort(idx)
    chains["num_residues"] = counts[np.argsort(idx)]
    chains["atom_start_idx"] = chains["res_start_idx"] * 3
    chains["num_atoms"] = chains["num_residues"] * 3

    return Structure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


@pytest.fixture
def structure(base_struct_info: dict[str, Any]) -> Structure:
    return structure_from_base_struct_info(base_struct_info)


@pytest.fixture
def structure2(base_struct_info2: dict[str, Any]) -> Structure:
    return structure_from_base_struct_info(base_struct_info2)


@pytest.fixture
def structure_concat_structure2(base_struct_info: dict[str, Any], base_struct_info2: dict[str, Any]) -> Structure:
    concat_struct_info = {}
    # direct concat some tensors
    direct_concat_keys = [
        "asym_id",
        "residue_index",
        "auth_residue_index",
        "coords",
        "atom_resolved_mask",
        "token_resolved_mask",
        "sym_id",
    ]
    for key in direct_concat_keys:
        concat_struct_info[key] = torch.cat([base_struct_info[key], base_struct_info2[key]])

    concat_struct_info["resname"] = np.concatenate([base_struct_info["resname"], base_struct_info2["resname"]])

    # entity_id cannot be concat directly
    concat_struct_info["entity_id"] = torch.cat(
        [
            base_struct_info["entity_id"],
            base_struct_info2["entity_id"]
            + base_struct_info["entity_id"].max()
            - base_struct_info["entity_id"].min()
            + 1,
        ]
    )

    return structure_from_base_struct_info(concat_struct_info)


@pytest.fixture
def empty_structure() -> Structure:
    return Structure(
        atoms=np.empty(0, dtype=ATOMS_DTYPE),
        residues=np.empty(0, dtype=RESIDUES_DTYPE),
        chains=np.empty(0, dtype=CHAINS_DTYPE),
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


def assert_structure_equality(struct1: Structure, struct2: Structure):
    """Check that two structures are equal."""
    attribs = ["chains", "features", "residues", "atoms", "interfaces", "residue_interfaces"]
    for attr in attribs:
        attr1 = getattr(struct1, attr)
        attr2 = getattr(struct2, attr)
        assert attr1.shape == attr2.shape, f"Attribute {attr} has different shape: {attr1.shape} != {attr2.shape}"
        for field_name in attr1.dtype.names:
            if field_name == "plddt":
                continue
            assert np.array_equal(attr1[field_name], attr2[field_name]), (
                f"Attribute {attr} field {field_name} has different values: {attr1[field_name]} != {attr2[field_name]}"
            )


# ---------------------------------------------------------------------------
# TestStructureBase
# ---------------------------------------------------------------------------


class TestStructureBase:
    def test_get_auth_chain_id_to_pdb_chain_id_map(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        # chain A: pdb='A', auth='B'; chain B: pdb='C', auth='D'
        result = struct.get_auth_chain_id_to_pdb_chain_id_map()
        assert result == {
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[1]: ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[0],
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[3]: ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[2],
        }

    def test_get_pdb_chain_id_to_auth_chain_id_map(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        result = struct.get_pdb_chain_id_to_auth_chain_id_map()
        assert result == {
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[0]: ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[1],
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[2]: ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[3],
        }

    def test_chain_id_maps_are_inverses(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        auth_to_pdb = struct.get_auth_chain_id_to_pdb_chain_id_map()
        pdb_to_auth = struct.get_pdb_chain_id_to_auth_chain_id_map()
        for auth_id, pdb_id in auth_to_pdb.items():
            assert pdb_to_auth[pdb_id] == auth_id


# ---------------------------------------------------------------------------
# TestStructureSerialization
# ---------------------------------------------------------------------------


class TestStructureSerialization:
    def test_with_sequential_auth_res_idx_returns_modified_copy(self, structure: Structure) -> None:
        original_auth_res_idx = structure.residues["auth_res_idx"].copy()

        sequential = structure.with_sequential_auth_res_idx()

        assert sequential is not structure
        np.testing.assert_array_equal(sequential.residues["auth_res_idx"], structure.residues["res_idx"])
        np.testing.assert_array_equal(structure.residues["auth_res_idx"], original_auth_res_idx)
        assert sequential.atoms is structure.atoms
        assert sequential.chains is structure.chains

    def test_dump_and_load_roundtrip(self, tmp_path, structure: Structure) -> None:
        path = tmp_path / "test_structure.npz"
        structure.dump_to_npz(path)
        loaded = Structure.load_from_npz(path)
        for field in ["atoms", "residues", "chains"]:
            arr_orig = getattr(structure, field)
            arr_loaded = getattr(loaded, field)
            assert arr_orig.shape == arr_loaded.shape
            for name in arr_orig.dtype.names:
                if name == "plddt":
                    continue
                assert np.array_equal(arr_orig[name], arr_loaded[name]), f"Mismatch in {field}.{name}"

    def test_load_raises_for_multimodel_file(self, tmp_path) -> None:
        """A file containing atoms_0 should raise a descriptive error."""
        atoms = np.empty(3, dtype=ATOMS_DTYPE)
        atoms["asym_id"] = 0
        np.savez_compressed(str(tmp_path / "multi.npz"), atoms_0=atoms)
        with pytest.raises(ValueError, match="MultiModelStructure"):
            Structure.load_from_npz(tmp_path / "multi.npz")

    def test_load_raises_for_missing_atoms_key(self, tmp_path) -> None:
        """A file with no atoms key should raise a descriptive error."""
        np.savez_compressed(str(tmp_path / "bad.npz"), something_else=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="no 'atoms'"):
            Structure.load_from_npz(tmp_path / "bad.npz")


# ---------------------------------------------------------------------------
# TestUpdateArrays
# ---------------------------------------------------------------------------


class TestUpdateArrays:
    def test_res_idx_preserved_when_reset_disabled(self) -> None:
        """Residue and atom res_idx values are preserved."""
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=3, num_res_b=2)
        chains = struct.chains.copy()
        residues = struct.residues.copy()
        atoms = struct.atoms.copy()

        residues["res_idx"][residues["asym_id"] == 0] = [5, 6, 7]
        atoms["res_idx"][atoms["asym_id"] == 0] = np.repeat([5, 6, 7], 3)
        expected_res_idx = residues["res_idx"].copy()
        expected_atom_res_idx = atoms["res_idx"].copy()

        _, res_out, atoms_out = Structure.rebuild_structure_arrays(chains, residues, atoms, reset_residue_index=False)

        np.testing.assert_array_equal(res_out["res_idx"], expected_res_idx)
        np.testing.assert_array_equal(atoms_out["res_idx"], expected_atom_res_idx)

    def test_updates_start_indices_even_when_reset_disabled(self) -> None:
        """Start/count fields are recomputed."""
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=2)
        chains = struct.chains.copy()
        residues = struct.residues.copy()
        atoms = struct.atoms.copy()

        # Corrupt index bookkeeping fields to make sure rebuild_structure_arrays recomputes them.
        chains["atom_start_idx"] = -1
        chains["res_start_idx"] = -1
        chains["num_atoms"] = -1
        chains["num_residues"] = -1
        residues["atom_start_idx"] = -1
        residues["atom_center"] = 0

        chains_out, residues_out, _ = Structure.rebuild_structure_arrays(
            chains, residues, atoms, reset_residue_index=False
        )

        np.testing.assert_array_equal(residues_out["atom_start_idx"], np.array([0, 3, 6, 9]))
        np.testing.assert_array_equal(residues_out["atom_center"], np.array([1, 4, 7, 10]))
        np.testing.assert_array_equal(chains_out["atom_start_idx"], np.array([0, 6]))
        np.testing.assert_array_equal(chains_out["res_start_idx"], np.array([0, 2]))
        np.testing.assert_array_equal(chains_out["num_atoms"], np.array([6, 6]))
        np.testing.assert_array_equal(chains_out["num_residues"], np.array([2, 2]))

    @pytest.mark.parametrize("reset_residue_index", [True, False])
    def test_chain_counts_match_actual_data(self, reset_residue_index: bool) -> None:
        """chains.num_atoms and num_residues must reflect the actual slices."""
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=3, num_res_b=2)
        chains, residues, atoms = Structure.rebuild_structure_arrays(
            struct.chains.copy(), struct.residues.copy(), struct.atoms.copy(), reset_residue_index=reset_residue_index
        )
        chain_a = chains[chains["asym_id"] == 0]
        chain_b = chains[chains["asym_id"] == 2]
        assert chain_a["num_residues"][0] == 3
        assert chain_a["num_atoms"][0] == 9
        assert chain_b["num_residues"][0] == 2
        assert chain_b["num_atoms"][0] == 6

    @pytest.mark.parametrize("reset_residue_index", [True, False])
    def test_atom_start_idx_reflects_position_in_atom_array(self, reset_residue_index: bool) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=2)
        chains, residues, atoms = Structure.rebuild_structure_arrays(
            struct.chains.copy(), struct.residues.copy(), struct.atoms.copy(), reset_residue_index=reset_residue_index
        )
        # residues for chain A sit at positions 0..5 in atoms; chain B at 6..11
        assert residues["atom_start_idx"][0] == 0
        assert residues["atom_start_idx"][1] == 3
        assert residues["atom_start_idx"][2] == 6
        assert residues["atom_start_idx"][3] == 9

    @pytest.mark.parametrize("reset_residue_index", [True, False])
    def test_idempotent_on_already_correct_structure(self, reset_residue_index: bool) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=3)
        chains_in = struct.chains.copy()
        residues_in = struct.residues.copy()
        atoms_in = struct.atoms.copy()
        chains_out, residues_out, atoms_out = Structure.rebuild_structure_arrays(
            chains_in, residues_in, atoms_in, reset_residue_index=reset_residue_index
        )
        np.testing.assert_array_equal(chains_out["num_residues"], struct.chains["num_residues"])
        np.testing.assert_array_equal(residues_out["atom_start_idx"], struct.residues["atom_start_idx"])

    def test_resets_residue_index(self) -> None:
        struct = make_two_chain_structure(
            asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=3, residue_index_a=[3, 5], residue_index_b=[1, 2, 4]
        )
        chains_in = struct.chains.copy()
        residues_in = struct.residues.copy()
        atoms_in = struct.atoms.copy()
        chains_out, residues_out, atoms_out = Structure.rebuild_structure_arrays(
            chains_in, residues_in, atoms_in, reset_residue_index=True
        )
        expected_residues_idx = np.array([0, 2, 0, 1, 3])
        np.testing.assert_array_equal(residues_out["res_idx"], expected_residues_idx)
        expected_atoms_idx = np.repeat(expected_residues_idx, 3)
        np.testing.assert_array_equal(atoms_out["res_idx"], expected_atoms_idx)


# ---------------------------------------------------------------------------
# TestUpdateAsymIds
# ---------------------------------------------------------------------------


class TestUpdateAsymIds:
    def test_remaps_asym_ids_in_chains_residues_atoms(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        ref_map = {0: 5, 2: 7}
        auth_map = {1: 0, 3: 2}
        new_struct = struct.update_asym_ids(ref_map, auth_map)

        assert set(new_struct.chains["asym_id"]) == {5, 7}
        assert set(new_struct.residues["asym_id"]) == {5, 7}
        assert set(new_struct.atoms["asym_id"]) == {5, 7}

    def test_updates_chain_id_strings(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        ref_map = {0: 5, 2: 7}
        auth_map = {1: 0, 3: 2}
        new_struct = struct.update_asym_ids(ref_map, auth_map)

        expected_chain_ids = {
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[5],
            ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[7],
        }
        assert set(new_struct.chains["chain_id"]) == expected_chain_ids

    def test_does_not_mutate_original(self) -> None:
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, auth_asym_id_a=1, auth_asym_id_b=3)
        original_asym_ids = struct.chains["asym_id"].copy()
        struct.update_asym_ids({0: 5, 2: 7}, {1: 0, 3: 2})
        np.testing.assert_array_equal(struct.chains["asym_id"], original_asym_ids)


# ---------------------------------------------------------------------------
# TestStructure Transformations (crop, merge, inpaint, align_to_feature_dict)
# ---------------------------------------------------------------------------


class TestStructureCrop:
    @pytest.mark.parametrize(
        ["crop_spec", "index_type", "inverse_crop"],
        [
            (ChainResidueRepr.from_string("D2,B"), "residue_index", False),
            (ChainResidueRepr.from_string("D1,E"), "residue_index", True),
        ],
    )
    def test_crop_structure(
        self,
        base_struct_info: dict[str, Any],
        crop_spec: ChainResidueRepr,
        index_type: Literal["residue_index", "auth_residue_index"],
        inverse_crop: bool,
    ) -> None:

        crop_struct_info = {}
        token_mask = torch.tensor([False, True, True, False])
        atom_mask = token_mask.repeat_interleave(3)
        direct_token_crop_keys = [
            "asym_id",
            "entity_id",
            "sym_id",
            "auth_residue_index",
            "token_resolved_mask",
            "residue_index",
        ]
        direct_atom_crop_keys = ["coords", "atom_resolved_mask"]
        for key in direct_token_crop_keys:
            crop_struct_info[key] = base_struct_info[key][token_mask]
        for key in direct_atom_crop_keys:
            crop_struct_info[key] = base_struct_info[key][atom_mask]

        crop_struct_info["resname"] = base_struct_info["resname"][token_mask.numpy()]

        expected_crop_struct = structure_from_base_struct_info(crop_struct_info)
        original_structure = structure_from_base_struct_info(base_struct_info)
        cropped_structure = original_structure.crop(
            crop_spec, reset_residue_index=False, index_type=index_type, inverse_crop=inverse_crop
        )
        assert_structure_equality(cropped_structure, expected_crop_struct)

    def test_crop_returns_correct_residue_count(self) -> None:
        """Crop a single chain to one residue and check counts."""
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=3, num_res_b=2)
        # chain A has asym_id=0 → 'A' in chain_id; keep only res_idx=0
        chain_letters = (ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[0], ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[2])
        spec = ChainResidueRepr.from_string(f"{chain_letters[0]}1,{chain_letters[1]}")
        cropped = struct.crop(spec, index_type="residue_index", reset_residue_index=False)
        assert len(cropped.residues[cropped.residues["asym_id"] == 0]) == 1
        # chain B untouched
        assert len(cropped.residues[cropped.residues["asym_id"] == 2]) == 2

    def test_crop_as_dict_spec(self, base_struct_info: dict[str, Any]) -> None:
        """crop() accepts dict representations of ChainResidueRepr."""
        original = structure_from_base_struct_info(base_struct_info)
        spec = ChainResidueRepr.from_string("D2,B")
        result_pydantic = original.crop(spec, index_type="residue_index", reset_residue_index=False)
        result_dict = original.crop(spec.model_dump(), index_type="residue_index", reset_residue_index=False)
        assert_structure_equality(result_pydantic, result_dict)

    def test_crop_interfaces_and_features_cleared(self) -> None:
        """crop() always returns empty interfaces."""
        struct = make_two_chain_structure(asym_id_a=0, asym_id_b=2)
        chain_letter = ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[0]
        spec = ChainResidueRepr.from_string(f"{chain_letter}1")
        cropped = struct.crop(spec, index_type="residue_index", reset_residue_index=False)
        assert len(cropped.interfaces) == 0
        assert len(cropped.residue_interfaces) == 0
        assert len(cropped.features) == 0


class TestStructureMerge:
    def test_merge_structures_concatenate_raises_error(self, base_struct_info: dict[str, Any]) -> None:
        """Confirm that we do not currently support concatenating structures with the same asym_id."""
        with pytest.raises(
            ValueError,
            match="Concatenating Structures with the same asym_id is not valid for 'concatenate' mode. Use 'inpaint' mode instead.",
        ):
            structure = structure_from_base_struct_info(base_struct_info)
            structure2 = structure_from_base_struct_info(base_struct_info)
            structure.merge(structure2, mode="concatenate", reset_residue_index=False)

    def test_merge_structures(self, base_struct_info: dict[str, Any], base_struct_info2: dict[str, Any]) -> None:

        concat_struct_info = {}
        # direct concat some tensors
        direct_concat_keys = [
            "asym_id",
            "residue_index",
            "auth_residue_index",
            "coords",
            "atom_resolved_mask",
            "token_resolved_mask",
            "sym_id",
        ]
        for key in direct_concat_keys:
            concat_struct_info[key] = torch.cat([base_struct_info[key], base_struct_info2[key]])

        concat_struct_info["resname"] = np.concatenate([base_struct_info["resname"], base_struct_info2["resname"]])

        # entity_id cannot be concat directly
        concat_struct_info["entity_id"] = torch.cat(
            [
                base_struct_info["entity_id"],
                base_struct_info2["entity_id"]
                + base_struct_info["entity_id"].max()
                - base_struct_info["entity_id"].min()
                + 1,
            ]
        )

        expected_concat_struct = structure_from_base_struct_info(concat_struct_info)

        structure = structure_from_base_struct_info(base_struct_info)
        structure2 = structure_from_base_struct_info(base_struct_info2)
        structure2.chains["entity_id"] = (
            structure2.chains["entity_id"]
            + structure.chains["entity_id"].max()
            - structure.chains["entity_id"].min()
            + 1
        )
        concatenated_structure = structure.merge(structure2, mode="concatenate", reset_residue_index=False)

        assert_structure_equality(concatenated_structure, expected_concat_struct)

    def test_merge_total_residue_and_atom_counts(self) -> None:
        a = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=3, entity_id_a=0, entity_id_b=1)
        b = make_two_chain_structure(asym_id_a=4, asym_id_b=6, num_res_a=1, num_res_b=2, entity_id_a=2, entity_id_b=3)
        result = a.merge(b, mode="concatenate", reset_residue_index=False)
        assert len(result.residues) == 8  # 2+3+1+2
        assert len(result.atoms) == 24  # 8 residues × 3 atoms
        assert len(result.interfaces) == 0
        assert len(result.residue_interfaces) == 0
        assert len(result.features) == 0

    def test_merge_raises_when_same_asym_id(self) -> None:
        a = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=3, entity_id_a=0, entity_id_b=1)
        b = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=1, num_res_b=2, entity_id_a=2, entity_id_b=3)
        with pytest.raises(
            ValueError,
            match="Concatenating Structures with the same asym_id is not valid for 'concatenate' mode. Use 'inpaint' mode instead.",
        ):
            a.merge(b, mode="concatenate", reset_residue_index=False)


class TestStructureAlignToFeatureDict:
    def test_full_atom_filter_raises(self, structure: Structure, feature_dict: dict[str, torch.Tensor]) -> None:
        with pytest.raises(NotImplementedError, match="full atom"):
            structure.align_to_feature_dict(feature_dict, AtomFilter.FULL)

    def test_aligned_structure_has_same_residues_and_chains(
        self, structure: Structure, feature_dict: dict[str, torch.Tensor]
    ) -> None:
        """Alignment must not change residues or chains."""
        aligned = structure.align_to_feature_dict(feature_dict, AtomFilter.BB)
        np.testing.assert_array_equal(aligned.residues["res_idx"], structure.residues["res_idx"])
        np.testing.assert_array_equal(aligned.chains["asym_id"], structure.chains["asym_id"])

    def test_aligned_coords_when_already_aligned(
        self, structure: Structure, feature_dict: dict[str, torch.Tensor]
    ) -> None:
        """When the structure and feature_dict share the same coordinates, alignment should be (near) identity."""
        aligned = structure.align_to_feature_dict(feature_dict, AtomFilter.BB)
        np.testing.assert_allclose(
            aligned.atoms["coords"],
            structure.atoms["coords"],
            atol=1e-4,
        )

    def test_raises_when_atom_order_mismatched(
        self, structure: Structure, feature_dict: dict[str, torch.Tensor]
    ) -> None:
        """Mismatched atom ordering should raise a ValueError."""
        # Shuffle the asym_id so that key ordering no longer agrees
        bad_feature_dict = {**feature_dict, "asym_id": feature_dict["asym_id"].flip(0)}
        with pytest.raises(ValueError, match="same order"):
            structure.align_to_feature_dict(bad_feature_dict, AtomFilter.BB)


class TestStructureInpaint:
    @pytest.fixture
    def inpainted_cropped_structure(
        self,
        chains_dtype: np.ndarray,  # fixture
        residues_dtype: np.ndarray,  # fixture
        atoms_dtype: np.ndarray,  # fixture
    ) -> Structure:
        """
        Input structure is
        A1,A2,B1,B2,C1,C2 in original residue index or U1,U2,V2,V3,W4,W5 in author residue index.

        inpaint pos: B2,C1
        inpaint sizes: 3, 4
        """
        inpaint_positions = [2, 5]

        chains_dtype = chains_dtype.copy()
        residues_dtype = residues_dtype.copy()
        atoms_dtype = atoms_dtype.copy()

        residues_cropped = residues_dtype[np.array(inpaint_positions)]
        atoms_cropped = np.concatenate(
            [
                atoms_dtype[
                    residues_dtype["atom_start_idx"][i] : residues_dtype["atom_start_idx"][i]
                    + residues_dtype["num_atoms"][i]
                ]
                for i in inpaint_positions
            ]
        )

        # update atom start idx after assigning atoms_cropped
        residues_cropped["atom_start_idx"] = np.cumsum([0] + residues_cropped["num_atoms"].tolist()[:-1])
        residues_cropped["atom_center"] = residues_cropped["atom_start_idx"] + 1

        chains_cropped = chains_dtype[np.isin(chains_dtype["asym_id"], residues_cropped["asym_id"])]
        chains_cropped["num_residues"] = [
            residues_cropped[residues_cropped["asym_id"] == asym_id].shape[0] for asym_id in chains_cropped["asym_id"]
        ]
        chains_cropped["res_start_idx"] = np.cumsum([0] + chains_cropped["num_residues"].tolist()[:-1])
        chains_cropped["num_atoms"] = [
            np.sum(residues_cropped[residues_cropped["asym_id"] == asym_id]["num_atoms"])
            for asym_id in chains_cropped["asym_id"]
        ]
        chains_cropped["atom_start_idx"] = np.cumsum([0] + chains_cropped["num_atoms"].tolist()[:-1])

        # update the res_idx
        residues_cropped["res_idx"] = np.array([0, 4])
        atoms_cropped["res_idx"] = np.repeat(residues_cropped["res_idx"], residues_cropped["num_atoms"])

        return Structure(
            residues=residues_cropped,
            atoms=atoms_cropped,
            chains=chains_cropped,
            features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
            interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
            residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        )

    @pytest.mark.parametrize(
        "inpaint_regions, inpaint_lengths, index_type",
        [
            ((ChainResidueRepr.from_string("B2"), ChainResidueRepr.from_string("C1")), (3, 4), "residue_index"),
            ((ChainResidueRepr.from_string("W4"), ChainResidueRepr.from_string("V3")), (4, 3), "auth_residue_index"),
        ],
    )
    def test_inpaint_basic(
        self,
        inpainted_cropped_structure: Structure,
        structure_chain_B_C_only: Structure,
        inpaint_regions: tuple[ChainResidueRepr, ...],
        inpaint_lengths: tuple[int, ...],
        index_type: str,
    ) -> None:

        result = structure_chain_B_C_only.inpaint(inpaint_regions, inpaint_lengths, index_type=index_type)

        assert_structure_equality(result, inpainted_cropped_structure)

    def test_inpaint_raises_when_lengths_mismatch_regions(self, structure_chain_B_C_only: Structure) -> None:
        with pytest.raises(ValueError, match="must have the same length"):
            structure_chain_B_C_only.inpaint(
                (ChainResidueRepr.from_string("B2"), ChainResidueRepr.from_string("C1")),
                (3,),
                index_type="residue_index",
            )

    def test_inpaint_raises_on_empty_regions_or_empty_lengths(self, structure_chain_B_C_only: Structure) -> None:
        with pytest.raises(ValueError, match="No inpainting regions provided."):
            structure_chain_B_C_only.inpaint((), (), index_type="residue_index")

    def test_inpaint_order_independent_for_multiple_regions_same_chain(
        self, structure_chain_B_C_only: Structure
    ) -> None:
        regions_forward = (ChainResidueRepr.from_string("C1"), ChainResidueRepr.from_string("C2"))
        regions_reverse = (ChainResidueRepr.from_string("C2"), ChainResidueRepr.from_string("C1"))
        lengths_forward = (3, 4)
        lengths_reverse = (4, 3)

        out_forward = structure_chain_B_C_only.inpaint(regions_forward, lengths_forward, index_type="residue_index")
        out_reverse = structure_chain_B_C_only.inpaint(regions_reverse, lengths_reverse, index_type="residue_index")

        assert_structure_equality(out_forward, out_reverse)

    def test_inpaint_overlapping_regions_union_behavior(self, structure_chain_B_C_only: Structure) -> None:
        # B1-2 and B2 overlap; union should drop all B-chain residues.
        result = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B1-2"), ChainResidueRepr.from_string("B2")),
            (3, 4),
            index_type="residue_index",
        )

        assert len(result.residues) == 2
        assert np.all(result.residues["asym_id"] == 2)

    def test_inpaint_full_chain_region_returns_empty_chain_content(self, structure_chain_B_C_only: Structure) -> None:
        result = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B1-2"), ChainResidueRepr.from_string("C1-2")),
            (3, 4),
            index_type="residue_index",
        )

        assert len(result.residues) == 0
        assert len(result.atoms) == 0

    def test_inpaint_auth_residue_index_matches_residue_index_for_equivalent_regions(
        self, structure_chain_B_C_only: Structure
    ) -> None:
        result_res_idx = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B2"), ChainResidueRepr.from_string("C1")),
            (3, 4),
            index_type="residue_index",
        )
        result_auth_idx = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("V3"), ChainResidueRepr.from_string("W4")),
            (3, 4),
            index_type="auth_residue_index",
        )

        assert_structure_equality(result_res_idx, result_auth_idx)

    def test_inpaint_updates_atom_res_idx_consistent_with_residue_res_idx(
        self, structure_chain_B_C_only: Structure
    ) -> None:
        result = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B2"), ChainResidueRepr.from_string("C1")),
            (3, 4),
            index_type="residue_index",
        )
        expected_atom_res_idx = np.repeat(result.residues["res_idx"], result.residues["num_atoms"])
        np.testing.assert_array_equal(result.atoms["res_idx"], expected_atom_res_idx)

    def test_inpaint_does_not_mutate_input_chain_residue_repr_objects(
        self, structure_chain_B_C_only: Structure
    ) -> None:
        region = ChainResidueRepr.from_string("B2")
        original_repr = region.model_dump()
        structure_chain_B_C_only.inpaint((region,), (3,), index_type="residue_index")
        assert region.model_dump() == original_repr

    def test_inpaint_does_not_mutate_original_structure(self, structure_chain_B_C_only: Structure) -> None:
        original_copy = Structure(
            residues=structure_chain_B_C_only.residues.copy(),
            atoms=structure_chain_B_C_only.atoms.copy(),
            chains=structure_chain_B_C_only.chains.copy(),
            features=structure_chain_B_C_only.features.copy(),
            interfaces=structure_chain_B_C_only.interfaces.copy(),
            residue_interfaces=structure_chain_B_C_only.residue_interfaces.copy(),
        )

        _ = structure_chain_B_C_only.inpaint((ChainResidueRepr.from_string("B2"),), (3,), index_type="residue_index")

        assert_structure_equality(structure_chain_B_C_only, original_copy)

    def test_inpaint_nonexistent_region_raises_or_noops(self, structure_chain_B_C_only: Structure) -> None:
        with pytest.raises(ValueError):
            structure_chain_B_C_only.inpaint((ChainResidueRepr.from_string("B99"),), (3,), index_type="residue_index")

    def test_inpaint_duplicate_region_definition_is_handled(self, structure_chain_B_C_only: Structure) -> None:
        result_unique = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B2"),), (3,), index_type="residue_index"
        )
        result_duplicate = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B2"), ChainResidueRepr.from_string("B2")),
            (3, 3),
            index_type="residue_index",
        )
        assert_structure_equality(result_unique, result_duplicate)

    def test_inpaint_reset_residue_index_handles_leading_inpaint_segment(
        self, structure_chain_B_C_only: Structure
    ) -> None:
        """When the first region in a chain is inpainted, remaining context keeps inpaint-length offset."""
        result = structure_chain_B_C_only.inpaint(
            (ChainResidueRepr.from_string("B1"),),
            (3,),
            index_type="residue_index",
            reset_residue_index=True,
        )

        # B1 is removed; the remaining B2 context residue should be offset by the inpaint length.
        # In this fixture B2 has auth residue index 3.
        b2_mask = result.residues["auth_res_idx"] == 3
        assert b2_mask.sum() == 1
        assert result.residues["res_idx"][b2_mask][0] == 3

        # Atom residue indices should stay aligned with residue indices.
        b2_asym_id = result.residues["asym_id"][b2_mask][0]
        b2_num_atoms = result.residues["num_atoms"][b2_mask][0]
        b2_atom_mask = (result.atoms["asym_id"] == b2_asym_id) & (result.atoms["res_idx"] == 3)
        assert b2_atom_mask.sum() == b2_num_atoms


# ---------------------------------------------------------------------------
# TestMultiModelStructure
# ---------------------------------------------------------------------------


class TestMultiModelStructure:
    def _make_multi_model(self, num_models: int = 3) -> MultiModelStructure:
        base = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=2)
        atoms_list = []
        for i in range(num_models):
            a = base.atoms.copy()
            a["coords"] = float(i)
            atoms_list.append(a)
        return MultiModelStructure(
            atoms=atoms_list,
            residues=base.residues.copy(),
            chains=base.chains.copy(),
            features=base.features.copy(),
        )

    def test_get_structures_returns_one_per_model(self) -> None:
        mm = self._make_multi_model(num_models=4)
        structs = mm.get_structures()
        assert len(structs) == 4

    def test_get_structures_share_residues_and_chains(self) -> None:
        mm = self._make_multi_model(num_models=2)
        structs = mm.get_structures()
        for s in structs:
            np.testing.assert_array_equal(s.residues["res_idx"], mm.residues["res_idx"])
            np.testing.assert_array_equal(s.chains["asym_id"], mm.chains["asym_id"])

    def test_get_structures_each_model_has_distinct_coords(self) -> None:
        mm = self._make_multi_model(num_models=3)
        structs = mm.get_structures()
        for i, s in enumerate(structs):
            np.testing.assert_allclose(s.atoms["coords"], float(i))

    def test_from_structures_roundtrip(self) -> None:
        mm = self._make_multi_model(num_models=2)
        structs = mm.get_structures()
        mm2 = MultiModelStructure.from_structures(structs)
        assert len(mm2.atoms) == 2
        for i in range(2):
            np.testing.assert_array_equal(mm2.atoms[i]["coords"], mm.atoms[i]["coords"])

    def test_from_structures_raises_when_residues_differ(self) -> None:
        base = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=2, num_res_b=2)
        other = make_two_chain_structure(asym_id_a=0, asym_id_b=2, num_res_a=3, num_res_b=2)
        with pytest.raises(ValueError, match="same residues"):
            MultiModelStructure.from_structures([base, other])

    def test_from_structures_raises_when_chains_differ(self) -> None:
        base = make_two_chain_structure(asym_id_a=0, asym_id_b=2)
        other = make_two_chain_structure(asym_id_a=0, asym_id_b=4)
        # it might get caught on the same chains or same residues assertion
        # but we don't want to specify the order
        with pytest.raises(ValueError, match="All structures must have the same"):
            MultiModelStructure.from_structures([base, other])

    def test_dump_and_load_roundtrip(self, tmp_path) -> None:
        mm = self._make_multi_model(num_models=2)
        path = tmp_path / "multi.npz"
        mm.dump_to_npz(path)
        loaded = MultiModelStructure.load_from_npz(path)
        assert len(loaded.atoms) == 2
        for i in range(2):
            np.testing.assert_array_equal(loaded.atoms[i]["coords"], mm.atoms[i]["coords"])
        np.testing.assert_array_equal(loaded.residues["res_idx"], mm.residues["res_idx"])


def test_shared_entity_homooligomer_round_trips(tmp_path) -> None:
    structure = make_two_chain_structure(entity_id_a=0, entity_id_b=0, num_res_a=2, num_res_b=2)
    path = tmp_path / "homooligomer.cif"
    save_structures_to_file([structure], path, primary_residue_index_key="res_idx")

    written = gemmi.read_structure(str(path))
    assert len(written.entities) == 1
    assert list(written.entities[0].full_sequence) == ["ALA", "ALA"]

    parsed = parse_mmcif(path)
    assert len(parsed.chains) == 2
    assert list(parsed.residues["res_name"]) == ["ALA", "ALA", "ALA", "ALA"]


def test_trajectory_writes_each_entity_once(tmp_path) -> None:
    structure = make_two_chain_structure(entity_id_a=0, entity_id_b=1)
    path = tmp_path / "trajectory.cif"
    save_structures_to_file([structure, structure], path, primary_residue_index_key="res_idx")

    written = gemmi.read_structure(str(path))
    assert len(written.entities) == 2


def test_shared_entity_rejects_inconsistent_sequences(tmp_path) -> None:
    structure = make_two_chain_structure(entity_id_a=0, entity_id_b=0)
    structure.residues["res_name"][0] = "GLY"
    with pytest.raises(ValueError, match="do not have the same sequence"):
        save_structures_to_file([structure], tmp_path / "unused.cif", primary_residue_index_key="res_idx")
