"""Auth chain string identity: parse → Features → Structure round-trip."""

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import dynopsi.data.auth_chain_identity as auth_chain_identity
from dynopsi.data.auth_chain_identity import (
    AUTH_CHAIN_ID_MAX_LENGTH,
    OPAQUE_AUTH_ASYM_ID_START,
    assign_auth_asym_id,
    auth_chain_id_table_from_chains,
    checked_auth_chain_id,
    lookup_auth_chain_id,
    resolve_unique_letter_auth_int,
)
from dynopsi.data.constants import TOKEN_IDS, AtomFilter, DesignDesignation, InpaintFeature, PocketFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.primitives import (
    CropFeaturizer,
    HotspotFeaturizer,
    InpaintFeaturizer,
    NewChainsFeaturizer,
    PredefinedCenterOfMassEstimator,
    ReferenceStructureFeaturizer,
)
from dynopsi.data.featurization.utils import concat_features, concat_features_with_auth_tables, make_empty_chain
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Structure,
)
from dynopsi.data.structure_io import parse_mmcif, save_structures_to_file
from dynopsi.data.tokenizer import DynoPsiTokenizer
from dynopsi.data.utils import (
    ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP,
    SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP,
    ChainResidueRepr,
    get_asym_id_from_chain_id,
)

A7_CIF = """
data_A7
_entry.id A7
loop_
_entity.id
_entity.type
1 polymer
loop_
_entity_poly.entity_id
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.type
_entity_poly.pdbx_strand_id
1 A 'polypeptide(L)' A
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.mon_id
_entity_poly_seq.num
1 ALA 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 10 ALA A-7 N 1
ATOM 2 C CA . ALA A 1 1 ? 1.458 0.000 0.000 1.00 0.00 ? 10 ALA A-7 CA 1
ATOM 3 C C . ALA A 1 1 ? 2.009 1.420 0.000 1.00 0.00 ? 10 ALA A-7 C 1
ATOM 4 O O . ALA A 1 1 ? 1.328 2.389 0.000 1.00 0.00 ? 10 ALA A-7 O 1
"""

SHARED_AUTH_TWO_LABEL_CIF = """
data_shared_auth
_entry.id shared_auth
loop_
_entity.id
_entity.type
1 polymer
loop_
_entity_poly.entity_id
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.type
_entity_poly.pdbx_strand_id
1 AA 'polypeptide(L)' A,B
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.mon_id
_entity_poly_seq.num
1 ALA 1
1 ALA 2
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.458 0.000 0.000 1.00 0.00 ? 1 ALA A CA 1
ATOM 3 C C . ALA A 1 1 ? 2.009 1.420 0.000 1.00 0.00 ? 1 ALA A C 1
ATOM 4 O O . ALA A 1 1 ? 1.328 2.389 0.000 1.00 0.00 ? 1 ALA A O 1
ATOM 5 N N . ALA B 1 2 ? 6.000 0.000 0.000 1.00 0.00 ? 2 ALA A N 1
ATOM 6 C CA . ALA B 1 2 ? 7.458 0.000 0.000 1.00 0.00 ? 2 ALA A CA 1
ATOM 7 C C . ALA B 1 2 ? 8.009 1.420 0.000 1.00 0.00 ? 2 ALA A C 1
ATOM 8 O O . ALA B 1 2 ? 7.328 2.389 0.000 1.00 0.00 ? 2 ALA A O 1
"""

PREFIX_SHARED_CIF = """
data_prefix
_entry.id prefix
loop_
_entity.id
_entity.type
1 polymer
2 polymer
loop_
_entity_poly.entity_id
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.type
_entity_poly.pdbx_strand_id
1 A 'polypeptide(L)' A
2 A 'polypeptide(L)' B
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.mon_id
_entity_poly_seq.num
1 ALA 1
2 ALA 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.000 0.000 0.000 1.00 0.00 ? 1 ALA ABCDE N 1
ATOM 2 C CA . ALA A 1 1 ? 1.458 0.000 0.000 1.00 0.00 ? 1 ALA ABCDE CA 1
ATOM 3 C C . ALA A 1 1 ? 2.009 1.420 0.000 1.00 0.00 ? 1 ALA ABCDE C 1
ATOM 4 O O . ALA A 1 1 ? 1.328 2.389 0.000 1.00 0.00 ? 1 ALA ABCDE O 1
ATOM 5 N N . ALA B 2 1 ? 6.000 0.000 0.000 1.00 0.00 ? 1 ALA ABCDF N 1
ATOM 6 C CA . ALA B 2 1 ? 7.458 0.000 0.000 1.00 0.00 ? 1 ALA ABCDF CA 1
ATOM 7 C C . ALA B 2 1 ? 8.009 1.420 0.000 1.00 0.00 ? 1 ALA ABCDF C 1
ATOM 8 O O . ALA B 2 1 ? 7.328 2.389 0.000 1.00 0.00 ? 1 ALA ABCDF O 1
"""

TOO_LONG_AUTH_CHAIN_ID = "A" * (AUTH_CHAIN_ID_MAX_LENGTH + 1)


def _one_chain_structure(*, auth_chain_id: str, auth_asym_id: int, label_asym_id: int = 0, n_res: int = 2) -> Structure:
    atoms_per_res = 3
    n_atoms = n_res * atoms_per_res
    atoms = np.empty(n_atoms, dtype=ATOMS_DTYPE)
    atoms["asym_id"] = label_asym_id
    atoms["res_idx"] = np.repeat(np.arange(n_res), atoms_per_res)
    atoms["atom_name"] = np.tile(["N", "CA", "C"], n_res)
    atoms["coords"] = np.zeros((n_atoms, 3), dtype=np.float32)
    atoms["is_present"] = True
    atoms["bfactor"] = 100.0
    atoms["plddt"] = np.nan

    residues = np.empty(n_res, dtype=RESIDUES_DTYPE)
    residues["asym_id"] = label_asym_id
    residues["res_idx"] = np.arange(n_res)
    residues["res_name"] = "ALA"
    residues["res_type"] = TOKEN_IDS["ALA"]
    residues["auth_res_idx"] = np.arange(1, n_res + 1)
    residues["atom_start_idx"] = np.arange(n_res) * atoms_per_res
    residues["num_atoms"] = atoms_per_res
    residues["atom_center"] = residues["atom_start_idx"] + 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["ca_present"] = True
    residues["backbone_present"] = True

    chains = np.empty(1, dtype=CHAINS_DTYPE)
    chains["chain_id"] = "A"
    chains["auth_chain_id"] = auth_chain_id
    chains["asym_id"] = label_asym_id
    chains["auth_asym_id"] = auth_asym_id
    chains["entity_id"] = 0
    chains["sym_id"] = 0
    chains["atom_start_idx"] = 0
    chains["num_atoms"] = n_atoms
    chains["res_start_idx"] = 0
    chains["num_residues"] = n_res

    return Structure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


def _features_from_structure(structure: Structure) -> Features:
    feature_dict: dict[str, torch.Tensor] = {}
    for token_features in DynoPsiTokenizer(atom_filter=AtomFilter.BB).tokenize(structure):
        feature_dict.update(token_features.asdict())
    n_tokens = feature_dict["token_index"].shape[0]
    feature_dict["binder_mask"] = torch.zeros(n_tokens, dtype=torch.bool)
    feature_dict["target_mask"] = torch.ones(n_tokens, dtype=torch.bool)
    return Features(
        feature_dict=feature_dict,
        atoms_to_include=AtomFilter.BB,
        auth_chain_id_table=auth_chain_id_table_from_chains(structure.chains),
    )


def _target_chain(auth_asym_id: int, seq_length: int = 2) -> dict[str, torch.Tensor]:
    return make_empty_chain(
        seq_length=seq_length,
        atoms_to_include=AtomFilter.BB,
        res_type="ALA",
        asym_id=auth_asym_id,
        auth_asym_id=auth_asym_id,
        inpaint_feature=InpaintFeature.TARGET_CONTEXT,
    )


def _add_one_binder(features: Features) -> Features:
    featurizer = NewChainsFeaturizer(
        name="binder",
        lengths=(3,),
        estimate_centers_of_mass_function=PredefinedCenterOfMassEstimator(centers_of_mass=((0.0, 0.0, 0.0),)),
        atoms_to_include=AtomFilter.BB,
    )
    return next(featurizer([features]))


class TestIdentityOwnsLetterMaps:
    def test_identity_module_does_not_import_sequence_utils(self) -> None:
        assert "sequence_utils" not in inspect.getsource(auth_chain_identity)

    def test_mask_map_has_no_nested_identity_import(self) -> None:
        source = inspect.getsource(ChainResidueRepr._asym_id_to_residue_index_map_for_mask)
        assert "from dynopsi.data.auth_chain_identity import" not in source

    def test_data_utils_reexports_identity_maps(self) -> None:
        assert ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP is auth_chain_identity.ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP
        assert SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP is auth_chain_identity.SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP

    def test_spreadsheet_encoding_unchanged(self) -> None:
        assert get_asym_id_from_chain_id("A") == 0
        assert get_asym_id_from_chain_id("AA") == 52
        assert get_asym_id_from_chain_id("A-7") == -1


class TestAssignAuthAsymId:
    def test_unique_letters_are_letter_stable(self) -> None:
        used: set[int] = set()
        assert assign_auth_asym_id("V", used) == 21
        used.add(21)
        assert assign_auth_asym_id("W", used) == 22

    def test_a7_assigns_opaque_auth_int(self) -> None:
        assert get_asym_id_from_chain_id("A-7") == -1
        auth_int = assign_auth_asym_id("A-7", set())
        assert auth_int == OPAQUE_AUTH_ASYM_ID_START
        assert auth_int != -1


class TestParseA7Identity:
    def test_cif_parse_stores_a7_and_opaque_int(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(A7_CIF)
            temp_file.flush()
            structure = parse_mmcif(temp_file.name, compute_residue_interfaces=False)

        assert list(structure.chains["auth_chain_id"]) == ["A-7"]
        assert int(structure.chains["auth_asym_id"][0]) != -1
        assert int(structure.chains["auth_asym_id"][0]) == OPAQUE_AUTH_ASYM_ID_START

    def test_shared_author_chain_keeps_one_auth_int(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(SHARED_AUTH_TWO_LABEL_CIF)
            temp_file.flush()
            structure = parse_mmcif(temp_file.name, compute_residue_interfaces=False)

        assert list(structure.chains["auth_chain_id"]) == ["A", "A"]
        assert len(set(int(i) for i in structure.chains["auth_asym_id"])) == 1
        assert int(structure.chains["auth_asym_id"][0]) == 0

    def test_parse_features_to_structure_keeps_a7(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(A7_CIF)
            temp_file.flush()
            parsed = parse_mmcif(temp_file.name, compute_residue_interfaces=False)
        features = _features_from_structure(parsed)
        designed = features.to_structure()
        assert list(designed.chains["auth_chain_id"]) == ["A-7"]

    def test_in_memory_structure_to_structure_keeps_a7(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)
        features = _features_from_structure(structure)
        designed = features.to_structure()
        assert list(designed.chains["auth_chain_id"]) == ["A-7"]

    def test_safe_copy_and_design_coords_keep_a7(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)
        features = _features_from_structure(structure)
        copied = features.safe_copy()
        assert list(copied.to_structure().chains["auth_chain_id"]) == ["A-7"]
        designed = features.design_coordinates_to_structure(
            features.feature_dict["coords"], use_reference_for_noised_context=False
        )
        assert list(designed[0].chains["auth_chain_id"]) == ["A-7"]


class TestLongAuthChainId:
    def test_checked_auth_chain_id_rejects_over_maximum(self) -> None:
        with pytest.raises(ValueError, match="maximum"):
            checked_auth_chain_id(TOO_LONG_AUTH_CHAIN_ID)

    def test_parse_features_structure_mmcif_keeps_distinct_prefix(self, tmp_path: Path) -> None:
        source = tmp_path / "source.cif"
        source.write_text(PREFIX_SHARED_CIF)
        parsed = parse_mmcif(source, compute_residue_interfaces=False)
        assert set(parsed.chains["auth_chain_id"]) == {"ABCDE", "ABCDF"}

        designed = _features_from_structure(parsed).to_structure()
        assert set(designed.chains["auth_chain_id"]) == {"ABCDE", "ABCDF"}

        written = tmp_path / "written.cif"
        save_structures_to_file([designed], written, primary_residue_index_key="res_idx")
        again = parse_mmcif(written, compute_residue_interfaces=False)
        assert set(again.chains["auth_chain_id"]) == {"ABCDE", "ABCDF"}

    def test_parse_rejects_id_over_maximum(self, tmp_path: Path) -> None:
        path = tmp_path / "long.cif"
        path.write_text(A7_CIF.replace("A-7", TOO_LONG_AUTH_CHAIN_ID))
        with pytest.raises(ValueError, match="maximum"):
            parse_mmcif(path, compute_residue_interfaces=False)

    def test_to_structure_rejects_id_over_maximum(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)
        features = _features_from_structure(structure)
        features.auth_chain_id_table = {int(structure.chains["auth_asym_id"][0]): TOO_LONG_AUTH_CHAIN_ID}
        with pytest.raises(ValueError, match="maximum"):
            features.to_structure()

    def test_pdb_write_still_rejects_non_letter(self, tmp_path: Path) -> None:
        source = tmp_path / "source.cif"
        source.write_text(PREFIX_SHARED_CIF)
        parsed = parse_mmcif(source, compute_residue_interfaces=False)
        path = tmp_path / "designed.pdb"
        with pytest.raises(ValueError, match="ABCDE"):
            save_structures_to_file([parsed], path, primary_residue_index_key="res_idx")
        assert not path.exists()


class TestLookupAuthChainId:
    def test_missing_letter_int_raises(self) -> None:
        with pytest.raises(KeyError, match="auth_asym_id=0"):
            lookup_auth_chain_id(0, {})

    def test_missing_opaque_int_raises(self) -> None:
        with pytest.raises(KeyError, match=f"auth_asym_id={OPAQUE_AUTH_ASYM_ID_START}"):
            lookup_auth_chain_id(OPAQUE_AUTH_ASYM_ID_START, {})

    def test_table_entry_is_returned(self) -> None:
        assert lookup_auth_chain_id(0, {0: "A"}) == "A"


class TestUniqueLetterRoundTrip:
    def test_a_b_with_table_round_trip(self) -> None:
        feature_dict = concat_features(
            [_target_chain(0), _target_chain(1)],
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        features = Features(
            feature_dict=feature_dict,
            atoms_to_include=AtomFilter.BB,
            auth_chain_id_table={0: "A", 1: "B"},
        )
        structure = features.to_structure()
        assert sorted(structure.chains["auth_chain_id"].tolist()) == ["A", "B"]

    def test_v_w_with_table_round_trip(self) -> None:
        feature_dict = concat_features(
            [_target_chain(21), _target_chain(22)],
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        features = Features(
            feature_dict=feature_dict,
            atoms_to_include=AtomFilter.BB,
            auth_chain_id_table={21: "V", 22: "W"},
        )
        structure = features.to_structure()
        assert sorted(structure.chains["auth_chain_id"].tolist()) == ["V", "W"]


class TestVegfaBinderLetter:
    def test_vw_plus_one_binder_is_a(self) -> None:
        feature_dict = concat_features(
            [_target_chain(21), _target_chain(22)],
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        features = Features(
            feature_dict=feature_dict,
            atoms_to_include=AtomFilter.BB,
            auth_chain_id_table={21: "V", 22: "W"},
        )
        out = _add_one_binder(features)
        metadata = out.get_metadata(verbose=False)
        assert metadata["target_auth_chains"] == ["V", "W"]
        assert metadata["binder_auth_chains"] == ["A"]
        structure = out.to_structure()
        assert set(structure.chains["auth_chain_id"]) == {"A", "V", "W"}

    def test_sequential_ints_for_vw_binder_is_a(self) -> None:
        feature_dict = concat_features(
            [_target_chain(0), _target_chain(1)],
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        features = Features(
            feature_dict=feature_dict,
            atoms_to_include=AtomFilter.BB,
            auth_chain_id_table={0: "V", 1: "W"},
        )
        out = _add_one_binder(features)
        metadata = out.get_metadata(verbose=False)
        assert metadata["binder_auth_chains"] == ["A"]
        assert metadata["target_auth_chains"] == ["V", "W"]
        structure = out.to_structure()
        assert set(structure.chains["auth_chain_id"]) == {"A", "V", "W"}

    def test_binder_label_ids_ignore_live_auth_ints(self) -> None:
        target = _target_chain(0, seq_length=2)
        target["asym_id"][:] = 1
        features = Features(
            feature_dict=target,
            atoms_to_include=AtomFilter.BB,
            auth_chain_id_table={0: "A"},
        )
        out = _add_one_binder(features)
        binder_mask = out["binder_mask"].bool()
        assert set(out["asym_id"][binder_mask].unique().tolist()) == {0}
        assert set(out["auth_asym_id"][binder_mask].unique().tolist()) == {1}
        assert out.get_metadata(verbose=False)["binder_auth_chains"] == ["B"]


class TestConcatDoesNotCollapseAuthIdentity:
    def test_colliding_opaque_auth_stays_in_opaque_namespace(self) -> None:
        first = _target_chain(OPAQUE_AUTH_ASYM_ID_START)
        first["asym_id"][:] = 0
        second = _target_chain(OPAQUE_AUTH_ASYM_ID_START)
        second["asym_id"][:] = 0
        tables = [
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        ]
        original_tables = [dict(table) for table in tables]
        out, merged = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        assert set(out["asym_id"].tolist()) == {0, 1}
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        assert auth_ints == {OPAQUE_AUTH_ASYM_ID_START, OPAQUE_AUTH_ASYM_ID_START + 1}
        assert tables == original_tables
        assert set(merged) == auth_ints
        assert list(merged.values()) == ["A-7", "A-7"]

    def test_used_label_id_does_not_force_auth_remap(self) -> None:
        first = _target_chain(OPAQUE_AUTH_ASYM_ID_START)
        first["asym_id"][:] = 0
        second = _target_chain(0)
        second["asym_id"][:] = 1
        tables = [
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
            {0: "A"},
        ]
        original_tables = [dict(table) for table in tables]
        out, merged = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        assert set(out["asym_id"].tolist()) == {0, 1}
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        assert auth_ints == {0, OPAQUE_AUTH_ASYM_ID_START}
        assert tables == original_tables
        assert set(merged) == auth_ints
        assert set(merged.values()) == {"A-7", "A"}

    def test_colliding_letter_auth_remaps_into_opaque_namespace(self) -> None:
        letter_v = assign_auth_asym_id("V", set())
        first = _target_chain(letter_v)
        first["asym_id"][:] = 0
        second = _target_chain(letter_v)
        second["asym_id"][:] = 1
        tables = [{letter_v: "V"}, {letter_v: "V"}]
        original_tables = [dict(table) for table in tables]
        out, merged = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        assert set(out["asym_id"].tolist()) == {0, 1}
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        remapped = auth_ints - {letter_v}
        assert letter_v in auth_ints
        assert len(remapped) == 1
        assert min(remapped) >= OPAQUE_AUTH_ASYM_ID_START
        assert tables == original_tables
        assert set(merged) == auth_ints
        assert list(merged.values()) == ["V", "V"]

    def test_table_superset_does_not_overwrite_live_auth_identity(self) -> None:
        first = _target_chain(OPAQUE_AUTH_ASYM_ID_START)
        first["asym_id"][:] = 0
        incoming = _target_chain(OPAQUE_AUTH_ASYM_ID_START + 1)
        incoming["asym_id"][:] = 1
        tables = [
            {OPAQUE_AUTH_ASYM_ID_START: "H1"},
            {OPAQUE_AUTH_ASYM_ID_START: "L", OPAQUE_AUTH_ASYM_ID_START + 1: "H"},
        ]
        original_tables = [dict(table) for table in tables]
        out, merged = concat_features_with_auth_tables(
            [first, incoming],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        assert tables == original_tables
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        assert merged[OPAQUE_AUTH_ASYM_ID_START] == "H1"
        assert set(merged) == auth_ints

    def test_remapped_opaque_id_does_not_overwrite_retained_letter(self) -> None:
        upstream = _target_chain(OPAQUE_AUTH_ASYM_ID_START)
        upstream["asym_id"][:] = 1
        incoming = concat_features(
            [_target_chain(0), _target_chain(OPAQUE_AUTH_ASYM_ID_START)],
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        tables = [
            {OPAQUE_AUTH_ASYM_ID_START: "X"},
            {0: "A", OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        ]
        original_tables = [dict(table) for table in tables]
        out, merged = concat_features_with_auth_tables(
            [upstream, incoming],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        assert tables == original_tables
        assert set(merged.values()) == {"X", "A", "A-7"}
        assert len(merged) == 3
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        assert auth_ints == set(merged)
        assert all(key >= OPAQUE_AUTH_ASYM_ID_START or key == 0 for key in merged)


class TestUpdateAsymIdsKeepsAuthStrings:
    def test_remaps_ints_preserves_a7_string(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)
        remapped = structure.update_asym_ids(ref_asym_id_map={0: 5}, auth_asym_id_map={OPAQUE_AUTH_ASYM_ID_START: 7})
        assert list(remapped.chains["auth_chain_id"]) == ["A-7"]
        assert int(remapped.chains["auth_asym_id"][0]) == 7
        assert int(remapped.chains["asym_id"][0]) == 5

    def test_letter_auth_strings_also_preserved(self) -> None:
        structure = _one_chain_structure(auth_chain_id="V", auth_asym_id=21)
        remapped = structure.update_asym_ids(ref_asym_id_map={0: 3}, auth_asym_id_map={21: 8})
        assert list(remapped.chains["auth_chain_id"]) == ["V"]
        assert int(remapped.chains["auth_asym_id"][0]) == 8


class TestReferenceStructurePreservesAuthStrings:
    def test_reference_featurizer_keeps_a7(self) -> None:
        source = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)
        featurizer = ReferenceStructureFeaturizer(
            structure_path="unused.cif",
            design_designation=DesignDesignation.FIXED,
            atoms_to_include=AtomFilter.BB,
        )
        featurizer._structure = source

        features = next(featurizer([Features(atoms_to_include=AtomFilter.BB)]))
        assert features.auth_chain_id_table[OPAQUE_AUTH_ASYM_ID_START] == "A-7"
        assert list(features.to_structure().chains["auth_chain_id"]) == ["A-7"]
        assert list(features.safe_copy().to_structure().chains["auth_chain_id"]) == ["A-7"]


class TestResolveAuthIntFromTable:
    def test_a7_resolves_via_table_to_opaque_int(self) -> None:
        table = {OPAQUE_AUTH_ASYM_ID_START: "A-7"}
        assert resolve_unique_letter_auth_int("A-7", table) == OPAQUE_AUTH_ASYM_ID_START
        assert resolve_unique_letter_auth_int("A-7", table) != get_asym_id_from_chain_id("A-7")

    def test_missing_non_letter_raises(self) -> None:
        with pytest.raises(KeyError, match="A-7"):
            resolve_unique_letter_auth_int("A-7", {})

    def test_unique_letter_without_table_is_letter_stable(self) -> None:
        assert resolve_unique_letter_auth_int("V", {}) == 21

    def test_unique_letter_prefers_table(self) -> None:
        assert resolve_unique_letter_auth_int("V", {0: "V"}) == 0


class TestGetAsymIdToIndexMapAuthResolver:
    def test_auth_map_uses_table_not_letter_encoding(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A-7": [10]})
        result = crr.get_asym_id_to_index_map(
            "auth_residue_index",
            auth_chain_id_table={OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        assert result == {OPAQUE_AUTH_ASYM_ID_START: [10]}
        assert -1 not in result

    def test_auth_map_without_table_raises_for_a7(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A-7": [10]})
        with pytest.raises(ValueError, match="A-7"):
            crr.get_asym_id_to_index_map("auth_residue_index")

    def test_auth_map_unique_letter_without_table_is_letter_stable(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"V": [3]})
        assert crr.get_asym_id_to_index_map("auth_residue_index") == {21: [3]}

    def test_auth_map_prefers_table_over_letter_int(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"V": [3]})
        result = crr.get_asym_id_to_index_map(
            "auth_residue_index",
            auth_chain_id_table={0: "V"},
        )
        assert result == {0: [3]}


class TestAuthHotspotCropDictIdentity:
    def test_to_mask_dict_a7_selects_that_chain_not_letter_a(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A-7": [10]})
        mask = crr.to_mask(
            asym_id=torch.tensor(
                [0, 0, 0, OPAQUE_AUTH_ASYM_ID_START, OPAQUE_AUTH_ASYM_ID_START, OPAQUE_AUTH_ASYM_ID_START]
            ),
            residue_index=torch.tensor([8, 9, 10, 8, 9, 10]),
            pad_mask=torch.ones(6, dtype=torch.bool),
            index_type="auth_residue_index",
            auth_chain_id_table={0: "A", OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        assert mask.tolist() == [False, False, False, False, False, True]

    def test_to_mask_dict_a7_whole_chain(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A-7": None})
        mask = crr.to_mask(
            asym_id=torch.tensor([0, OPAQUE_AUTH_ASYM_ID_START, OPAQUE_AUTH_ASYM_ID_START]),
            residue_index=torch.tensor([1, 9, 10]),
            pad_mask=torch.ones(3, dtype=torch.bool),
            index_type="auth_residue_index",
            auth_chain_id_table={0: "A", OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        assert mask.tolist() == [False, True, True]

    def test_to_mask_without_table_raises_for_a7(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A-7": [10]})
        with pytest.raises(ValueError, match="A-7"):
            crr.to_mask(
                asym_id=torch.full((12,), OPAQUE_AUTH_ASYM_ID_START),
                residue_index=torch.arange(1, 13),
                pad_mask=torch.ones(12, dtype=torch.bool),
                index_type="auth_residue_index",
            )

    def test_original_residue_index_ignores_auth_table(self) -> None:
        crr = ChainResidueRepr(chain_id_to_residue_index_map={"A": [1]})
        mask = crr.to_mask(
            asym_id=torch.tensor([0, 99]),
            residue_index=torch.tensor([0, 0]),
            pad_mask=torch.ones(2, dtype=torch.bool),
            index_type="original_residue_index",
            auth_chain_id_table={99: "A"},
        )
        assert mask.tolist() == [True, False]

    def test_hotspot_dict_a7_residue_10(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        features = _features_from_structure(structure)
        featurizer = HotspotFeaturizer(
            name="hotspot",
            hotspot_positions={"A-7": [10]},
            index_type="auth_residue_index",
        )
        out = next(featurizer([features]))
        expected = torch.full((12,), int(PocketFeature.UNSELECTED))
        expected[9] = int(PocketFeature.POCKET)
        assert torch.equal(out["pocket_feature"], expected)

    def test_hotspot_dict_a7_whole_chain(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        features = _features_from_structure(structure)
        featurizer = HotspotFeaturizer(
            name="hotspot",
            hotspot_positions={"A-7": None},
            index_type="auth_residue_index",
        )
        out = next(featurizer([features]))
        assert torch.equal(out["pocket_feature"], torch.full((12,), int(PocketFeature.POCKET)))

    def test_crop_dict_a7_residue_10(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        features = _features_from_structure(structure)
        featurizer = CropFeaturizer(
            name="crop",
            residue_index_chain_spec={"A-7": [10]},
            index_type="auth_residue_index",
        )
        out = next(featurizer([features]))
        assert out["auth_residue_index"].tolist() == [10]
        assert out["auth_asym_id"].tolist() == [OPAQUE_AUTH_ASYM_ID_START]
        assert list(out.to_structure().chains["auth_chain_id"]) == ["A-7"]

    def test_crop_dict_a7_whole_chain(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        features = _features_from_structure(structure)
        featurizer = CropFeaturizer(
            name="crop",
            residue_index_chain_spec={"A-7": None},
            index_type="auth_residue_index",
        )
        out = next(featurizer([features]))
        assert out["auth_residue_index"].tolist() == list(range(1, 13))
        assert set(out["auth_asym_id"].tolist()) == {OPAQUE_AUTH_ASYM_ID_START}

    def test_structure_crop_dict_a7(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        cropped = structure.crop(
            {"A-7": [10]},
            reset_residue_index=True,
            index_type="auth_residue_index",
        )
        assert len(cropped.residues) == 1
        assert int(cropped.residues["auth_res_idx"][0]) == 10
        assert list(cropped.chains["auth_chain_id"]) == ["A-7"]

    def test_structure_inpaint_dict_a7(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        result = structure.inpaint(
            (ChainResidueRepr.model_validate({"A-7": [10]}),),
            (1,),
            index_type="auth_residue_index",
            reset_residue_index=True,
        )
        assert list(result.chains["auth_chain_id"]) == ["A-7"]
        assert len(result.residues) == 11
        assert 10 not in [int(idx) for idx in result.residues["auth_res_idx"]]

    def test_inpaint_featurizer_dict_a7_selects_residue_10(self) -> None:
        structure = _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START, n_res=12)
        features = _features_from_structure(structure)
        featurizer = InpaintFeaturizer(
            inpainting_positions=({"A-7": [10]},),
            inpainting_lengths=(1,),
            inpaint_region_centers_of_mass=(None,),
            atoms_to_include=AtomFilter.BB,
            index_type="auth_residue_index",
        )
        regions = featurizer._extract_inpaint_regions(features)
        assert len(regions) == 1
        assert regions[0].start_idx == 9
        assert regions[0].end_idx == 10
