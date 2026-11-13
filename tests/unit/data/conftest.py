from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

import gemmi
import numpy as np
import pytest
import torch

from dynopsi.data.auth_chain_identity import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP, UNIQUE_LETTER_AUTH_IDS
from dynopsi.data.constants import AA_THREE_TO_ONE_MAP, TOKEN_IDS, InpaintFeature, PocketFeature
from dynopsi.data.featurization.utils import (
    ATOM_KEYS_3D,
    ATOM_KEYS_FLAT,
    ATOM_TO_TOKEN_KEYS,
    TOKEN_KEYS_FLAT,
    TOKEN_TO_ATOM_KEYS,
)
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Structure,
)
from dynopsi.data.tokenizer import (
    FrameStructureTokenFeatures,
    StructureTokenFeatures,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol
else:
    Mol = Any


def unique_letter_auth_table(auth_asym_ids: Iterable[int]) -> dict[int, str]:
    """Test-only table for unique ``[A-Z]`` letter ints. Not a runtime fallback."""
    table: dict[int, str] = {}
    for auth_int in dict.fromkeys(int(i) for i in auth_asym_ids):
        letter = ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP.get(auth_int)
        if letter is not None and letter in UNIQUE_LETTER_AUTH_IDS:
            table[auth_int] = letter
    return table


def assert_structure_equality(
    observed: Structure, expected: Structure, float_tolerance: float = 1e-6, ignore_oxygen_atoms: bool = True
) -> None:
    attribs = ["chains", "features", "residues", "atoms", "interfaces", "residue_interfaces"]
    for attr in attribs:
        attr1 = getattr(observed, attr)
        attr2 = getattr(expected, attr)
        assert attr1.shape == attr2.shape, f"Attribute {attr} has different shape: {attr1.shape} != {attr2.shape}"
        for field_name in attr1.dtype.names:
            if field_name == "plddt":
                continue
            # check if values are ndarrays of floats
            is_float = np.issubdtype(attr1[field_name].dtype, np.floating)
            if is_float:
                equality_fn = partial(np.allclose, atol=float_tolerance)
            else:
                equality_fn = np.array_equal

            arr1 = attr1[field_name]
            arr2 = attr2[field_name]
            if ignore_oxygen_atoms and field_name == "coords":
                arr1 = arr1[attr1["atom_name"] != "O"]
                arr2 = arr2[attr2["atom_name"] != "O"]
            if not equality_fn(arr1, arr2):
                print("--------------------------------")
                print("attribute", attr)
                print("field name", field_name)
                print("observed", arr1)
                print("expected", arr2)
            assert equality_fn(arr1, arr2), f"Attribute {attr} field {field_name} has different values"


def assert_feature_dict_equality(observed: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    only_in_observed = set(observed.keys()) - set(expected.keys())
    only_in_expected = set(expected.keys()) - set(observed.keys())
    assert len(only_in_observed) == 0 and len(only_in_expected) == 0, (
        f"Keys only in observed: {only_in_observed}, keys only in expected: {only_in_expected}"
    )
    for key in observed.keys():
        if not np.array_equal(observed[key].numpy(), expected[key].numpy()):
            print("--------------------------------")
            print("feature dict key", key)
            print("observed", observed[key].numpy())
            print("expected", expected[key].numpy())
        assert np.array_equal(observed[key].numpy(), expected[key].numpy()), (
            f"Feature dict key {key} has different values"
        )


def assert_metadata_equality(observed: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]) -> None:
    if observed is None and expected is None:
        return
    if observed is None or expected is None:
        print("--------------------------------")
        print("observed", observed)
        print("expected", expected)
        raise ValueError("One of observed or expected is None")
    only_in_observed = set(observed.keys()) - set(expected.keys())
    only_in_expected = set(expected.keys()) - set(observed.keys())
    assert len(only_in_observed) == 0 and len(only_in_expected) == 0, (
        f"Keys only in observed: {only_in_observed}, keys only in expected: {only_in_expected}"
    )
    for key, value in observed.items():
        if value is None and expected[key] is None:
            continue
        if isinstance(value, dict):
            assert_metadata_equality(value, expected[key])
        elif isinstance(value, Structure):
            assert_structure_equality(value, expected[key])
        elif isinstance(value, torch.Tensor):
            if not np.array_equal(value.numpy(), expected[key].numpy()):
                print("--------------------------------")
                print("metadata key", key)
                print("observed", value.numpy())
                print("expected", expected[key].numpy())
            assert np.array_equal(value.numpy(), expected[key].numpy()), f"Metadata key {key} has different values"
        else:
            if value != expected[key]:
                print("--------------------------------")
                print("metadata key", key)
                print("observed", value)
                print("expected", expected[key])
            assert value == expected[key], f"Metadata key {key} has different values"


def check_feature_dict_lengths(feature_dict: dict[str, torch.Tensor]) -> None:
    """Check that the feature dict has the correct lengths."""
    seq_length = len(feature_dict["asym_id"])
    num_atoms = feature_dict["coords"].shape[1]
    for key in set(TOKEN_KEYS_FLAT) & set(feature_dict.keys()):
        assert len(feature_dict[key]) == seq_length, (
            f"Feature dict key {key} has length {len(feature_dict[key])} but expected {seq_length}"
        )
    for key in set(ATOM_KEYS_FLAT) & set(feature_dict.keys()):
        assert len(feature_dict[key]) == num_atoms, (
            f"Feature dict key {key} has length {len(feature_dict[key])} but expected {num_atoms}"
        )
    for key in set(ATOM_KEYS_3D) & set(feature_dict.keys()):
        assert feature_dict[key].shape == (1, num_atoms, 3), (
            f"Feature dict key {key} has shape {feature_dict[key].shape} but expected (1, {num_atoms}, 3)"
        )
    for key in set(TOKEN_TO_ATOM_KEYS) & set(feature_dict.keys()):
        assert len(feature_dict[key]) == seq_length, (
            f"Feature dict key {key} has length {len(feature_dict[key])} but expected {seq_length}"
        )
    for key in set(ATOM_TO_TOKEN_KEYS) & set(feature_dict.keys()):
        assert feature_dict[key].shape == (num_atoms, seq_length), (
            f"Feature dict key {key} has shape {feature_dict[key].shape} but expected ({num_atoms}, {seq_length})"
        )


def make_two_chain_structure(
    asym_id_a: int = 0,
    asym_id_b: int = 2,
    auth_asym_id_a: int = 1,
    auth_asym_id_b: int = 3,
    num_res_a: int = 2,
    num_res_b: int = 2,
    atoms_per_res: int = 3,
    entity_id_a: int = 0,
    entity_id_b: int = 1,
    residue_index_a: list[int] | None = None,
    residue_index_b: list[int] | None = None,
    with_features: bool = False,
) -> Structure:
    num_res = num_res_a + num_res_b
    n_atoms = num_res * atoms_per_res

    if residue_index_a is None:
        residue_index_a = np.arange(num_res_a)
    if residue_index_b is None:
        residue_index_b = np.arange(num_res_b)

    atoms = np.empty(n_atoms, dtype=ATOMS_DTYPE)
    atoms["asym_id"] = np.concatenate(
        [np.full(num_res_a * atoms_per_res, asym_id_a), np.full(num_res_b * atoms_per_res, asym_id_b)]
    )
    atoms["res_idx"] = np.concatenate(
        [np.repeat(np.array(residue_index_a), atoms_per_res), np.repeat(np.array(residue_index_b), atoms_per_res)]
    )
    atoms["atom_name"] = np.tile(["N", "CA", "C"], num_res)
    atoms["coords"] = np.zeros((n_atoms, 3), dtype=np.float32)
    atoms["is_present"] = True
    atoms["bfactor"] = 100.0
    atoms["plddt"] = np.nan

    residues = np.empty(num_res, dtype=RESIDUES_DTYPE)
    residues["asym_id"] = np.concatenate([np.full(num_res_a, asym_id_a), np.full(num_res_b, asym_id_b)])
    residues["res_idx"] = np.concatenate([np.array(residue_index_a), np.array(residue_index_b)])
    residues["res_name"] = "ALA"
    residues["res_type"] = TOKEN_IDS["ALA"]
    residues["auth_res_idx"] = np.concatenate([np.arange(num_res_a), np.arange(num_res_b)])
    residues["atom_start_idx"] = np.arange(num_res) * atoms_per_res
    residues["num_atoms"] = atoms_per_res
    residues["atom_center"] = residues["atom_start_idx"] + 1
    residues["is_standard"] = True
    residues["is_present"] = True
    residues["ca_present"] = True
    residues["backbone_present"] = True

    chains = np.empty(2, dtype=CHAINS_DTYPE)
    chains["asym_id"] = [asym_id_a, asym_id_b]
    chains["auth_asym_id"] = [auth_asym_id_a, auth_asym_id_b]
    chains["chain_id"] = [
        ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id_a],
        ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id_b],
    ]
    chains["auth_chain_id"] = [
        ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id_a],
        ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id_b],
    ]
    chains["entity_id"] = [entity_id_a, entity_id_b]
    chains["sym_id"] = [0, 0]
    chains["res_start_idx"] = [0, num_res_a]
    chains["num_residues"] = [num_res_a, num_res_b]
    chains["atom_start_idx"] = [0, num_res_a * atoms_per_res]
    chains["num_atoms"] = [num_res_a * atoms_per_res, num_res_b * atoms_per_res]

    features = np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE)
    if with_features:
        features = np.empty(num_res, dtype=DYNO_PSI_FEATURES_DTYPE)
        features["is_designed"] = True
        features["is_noised"] = False
        features["pocket_feature"] = 0
        features["inpaint_feature"] = 0

    return Structure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=features,
    )


@pytest.fixture
def cif_data() -> str:
    """Make a small CIF data string."""
    return """
        data_TEST
        #
        _entry.id TEST
        #
        loop_
        _entity.id
        _entity.type
        1 polymer
        2 polymer
        3 polymer
        #
        loop_
        _entity_poly.entity_id
        _entity_poly.pdbx_seq_one_letter_code
        _entity_poly.type
        _entity_poly.pdbx_strand_id
        1 NDPA 'polypeptide(L)' A,D
        2 FWG 'polypeptide(L)' B
        #
        loop_
        _entity_poly_seq.entity_id
        _entity_poly_seq.mon_id
        _entity_poly_seq.num
        1 ASN 1
        1 ASP 2
        1 PRO 3
        1 ALA 4
        2 PHE 1
        2 TRP 2
        2 GLY 3
        #
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
        ATOM 1     N N   . ASN A 1 1   ? 161.767 176.410 123.966 1.00 167.91 ? 18   ASN A N   1
        ATOM 2     C CA  . ASN A 1 1   ? 161.584 177.567 123.097 1.00 167.91 ? 18   ASN A CA  1
        ATOM 3     C C   . ASN A 1 1   ? 162.885 177.931 122.400 1.00 167.91 ? 18   ASN A C   1
        ATOM 4     O O   . ASN A 1 1   ? 163.204 177.388 121.344 1.00 167.91 ? 18   ASN A O   1
        ATOM 5     C CB  . ASN A 1 1   ? 161.063 178.766 123.890 1.00 167.91 ? 18   ASN A CB  1
        ATOM 6     N N   . ASP A 1 2   ? 163.646 178.843 123.006 1.00 156.80 ? 19   ASP A N   1
        ATOM 7     C CA  . ASP A 1 2   ? 164.896 179.346 122.441 1.00 156.80 ? 19   ASP A CA  1
        ATOM 8     C C   . ASP A 1 2   ? 166.019 179.103 123.443 1.00 156.80 ? 19   ASP A C   1
        ATOM 9     O O   . ASP A 1 2   ? 166.344 179.993 124.242 1.00 156.80 ? 19   ASP A O   1
        ATOM 10    C CB  . ASP A 1 2   ? 164.785 180.832 122.110 1.00 156.80 ? 19   ASP A CB  1
        ATOM 11    C CG  . ASP A 1 2   ? 163.728 181.117 121.068 1.00 156.80 ? 19   ASP A CG  1
        ATOM 12    O OD1 . ASP A 1 2   ? 163.633 180.352 120.088 1.00 156.80 ? 19   ASP A OD1 1
        ATOM 13    O OD2 . ASP A 1 2   ? 162.978 182.100 121.238 1.00 156.80 ? 19   ASP A OD2 1
        ATOM 14    N N   . PRO A 1 3   ? 166.641 177.917 123.429 1.00 147.62 ? 20   PRO A N   1
        ATOM 15    C CA  . PRO A 1 3   ? 167.701 177.635 124.411 1.00 147.62 ? 20   PRO A CA  1
        ATOM 16    C C   . PRO A 1 3   ? 168.987 178.399 124.159 1.00 147.62 ? 20   PRO A C   1
        ATOM 17    O O   . PRO A 1 3   ? 169.871 178.383 125.023 1.00 147.62 ? 20   PRO A O   1
        ATOM 18    C CB  . PRO A 1 3   ? 167.932 176.131 124.256 1.00 147.62 ? 20   PRO A CB  1
        ATOM 19    C CG  . PRO A 1 3   ? 166.666 175.609 123.703 1.00 147.62 ? 20   PRO A CG  1
        ATOM 20    C CD  . PRO A 1 3   ? 166.164 176.682 122.783 1.00 147.62 ? 20   PRO A CD  1
        ATOM 21    N N   . ALA A 1 4   ? 169.132 179.048 123.008 1.00 137.42 ? 21   ALA A N   1
        ATOM 22    C CA  . ALA A 1 4   ? 170.239 179.960 122.792 1.00 137.42 ? 21   ALA A CA  1
        ATOM 23    C C   . ALA A 1 4   ? 169.923 181.367 123.264 1.00 137.42 ? 21   ALA A C   1
        ATOM 24    O O   . ALA A 1 4   ? 170.816 182.221 123.264 1.00 137.42 ? 21   ALA A O   1
        ATOM 25    C CB  . ALA A 1 4   ? 170.623 179.995 121.312 1.00 137.42 ? 21   ALA A CB  1
        ATOM 6817  N N   . PHE B 2 1 ? 209.205 229.398 157.280 1.00 201.42 ? 1079 PHE B N   1
        ATOM 6818  C CA  . PHE B 2 1 ? 209.135 228.904 155.908 1.00 201.42 ? 1079 PHE B CA  1
        ATOM 6819  C C   . PHE B 2 1 ? 210.333 229.321 155.074 1.00 201.42 ? 1079 PHE B C   1
        ATOM 6820  O O   . PHE B 2 1 ? 211.134 230.155 155.490 1.00 201.42 ? 1079 PHE B O   1
        ATOM 6821  C CB  . PHE B 2 1 ? 207.855 229.387 155.225 1.00 201.42 ? 1079 PHE B CB  1
        ATOM 6822  C CG  . PHE B 2 1 ? 206.660 228.517 155.487 1.00 201.42 ? 1079 PHE B CG  1
        ATOM 6823  C CD1 . PHE B 2 1 ? 206.755 227.404 156.306 1.00 201.42 ? 1079 PHE B CD1 1
        ATOM 6824  C CD2 . PHE B 2 1 ? 205.442 228.810 154.905 1.00 201.42 ? 1079 PHE B CD2 1
        ATOM 6825  C CE1 . PHE B 2 1 ? 205.660 226.607 156.540 1.00 201.42 ? 1079 PHE B CE1 1
        ATOM 6826  C CE2 . PHE B 2 1 ? 204.349 228.015 155.128 1.00 201.42 ? 1079 PHE B CE2 1
        ATOM 6827  C CZ  . PHE B 2 1 ? 204.461 226.904 155.940 1.00 201.42 ? 1079 PHE B CZ  1
        ATOM 6828  N N   . TRP B 2 2 ? 210.456 228.732 153.891 1.00 199.36 ? 1080 TRP B N   1
        ATOM 6829  C CA  . TRP B 2 2 ? 211.515 229.114 152.957 1.00 199.36 ? 1080 TRP B CA  1
        ATOM 6830  C C   . TRP B 2 2 ? 210.986 230.156 151.979 1.00 199.36 ? 1080 TRP B C   1
        ATOM 6831  O O   . TRP B 2 2 ? 211.175 230.070 150.768 1.00 199.36 ? 1080 TRP B O   1
        ATOM 6832  C CB  . TRP B 2 2 ? 212.054 227.888 152.221 1.00 199.36 ? 1080 TRP B CB  1
        ATOM 6833  C CG  . TRP B 2 2 ? 212.754 226.844 153.077 1.00 199.36 ? 1080 TRP B CG  1
        ATOM 6834  C CD1 . TRP B 2 2 ? 212.843 226.809 154.438 1.00 199.36 ? 1080 TRP B CD1 1
        ATOM 6835  C CD2 . TRP B 2 2 ? 213.449 225.681 152.607 1.00 199.36 ? 1080 TRP B CD2 1
        ATOM 6836  N NE1 . TRP B 2 2 ? 213.547 225.703 154.845 1.00 199.36 ? 1080 TRP B NE1 1
        ATOM 6837  C CE2 . TRP B 2 2 ? 213.933 224.994 153.738 1.00 199.36 ? 1080 TRP B CE2 1
        ATOM 6838  C CE3 . TRP B 2 2 ? 213.717 225.161 151.338 1.00 199.36 ? 1080 TRP B CE3 1
        ATOM 6839  C CZ2 . TRP B 2 2 ? 214.664 223.812 153.639 1.00 199.36 ? 1080 TRP B CZ2 1
        ATOM 6840  C CZ3 . TRP B 2 2 ? 214.439 223.983 151.241 1.00 199.36 ? 1080 TRP B CZ3 1
        ATOM 6841  C CH2 . TRP B 2 2 ? 214.905 223.324 152.385 1.00 199.36 ? 1080 TRP B CH2 1
        ATOM 6842  N N   . GLY B 2 3 ? 210.350 231.188 152.527 1.00 202.02 ? 1081 GLY B N   1
        ATOM 6843  C CA  . GLY B 2 3 ? 209.680 232.197 151.747 1.00 202.02 ? 1081 GLY B CA  1
        ATOM 6844  C C   . GLY B 2 3 ? 210.610 233.296 151.287 1.00 202.02 ? 1081 GLY B C   1
        ATOM 6845  O O   . GLY B 2 3 ? 211.731 233.041 150.836 1.00 202.02 ? 1081 GLY B O   1
        ATOM 8992  N N   . ASP D 1 2    ? 155.940 184.063 164.115 1.00 63.23  ? 1213 ASP C N   1
        ATOM 8993  C CA  . ASP D 1 2    ? 156.815 184.198 162.956 1.00 63.23  ? 1213 ASP C CA  1
        ATOM 8998  N N   . PRO D 1 3    ? 157.660 186.520 163.031 1.00 74.86  ? 1214 PRO C N   1
        ATOM 9000  C C   . PRO D 1 3    ? 159.295 187.841 161.644 1.00 74.86  ? 1214 PRO C C   1
        ATOM 9001  O O   . PRO D 1 3    ? 160.223 188.477 162.151 1.00 74.86  ? 1214 PRO C O   1
        ATOM 9002  C CB  . PRO D 1 3    ? 157.966 188.792 163.561 1.00 74.86  ? 1214 PRO C CB  1
        ATOM 9003  C CG  . PRO D 1 3    ? 157.507 187.993 164.774 1.00 74.86  ? 1214 PRO C CG  1
        ATOM 9004  C CD  . PRO D 1 3    ? 157.962 186.611 164.460 1.00 74.86  ? 1214 PRO C CD  1
        #
        """


@pytest.fixture
def pdb_data() -> str:
    """Make a pdb string that will be parsed into the same structure as the cif data."""
    return """ATOM      1    N ASN A  18     161.767 176.410 123.966  1.00167.91           N
ATOM      2   CA ASN A  18     161.584 177.567 123.097  1.00167.91           C
ATOM      3    C ASN A  18     162.885 177.931 122.400  1.00167.91           C
ATOM      4    O ASN A  18     163.204 177.388 121.344  1.00167.91           O
ATOM      5   CB ASN A  18     161.063 178.766 123.890  1.00167.91           C
ATOM      6    N ASP A  19     163.646 178.843 123.006  1.00156.80           N
ATOM      7   CA ASP A  19     164.896 179.346 122.441  1.00156.80           C
ATOM      8    C ASP A  19     166.019 179.103 123.443  1.00156.80           C
ATOM      9    O ASP A  19     166.344 179.993 124.242  1.00156.80           O
ATOM     10   CB ASP A  19     164.785 180.832 122.110  1.00156.80           C
ATOM     11   CG ASP A  19     163.728 181.117 121.068  1.00156.80           C
ATOM     12  OD1 ASP A  19     163.633 180.352 120.088  1.00156.80           O
ATOM     13  OD2 ASP A  19     162.978 182.100 121.238  1.00156.80           O
ATOM     14    N PRO A  20     166.641 177.917 123.429  1.00147.62           N
ATOM     15   CA PRO A  20     167.701 177.635 124.411  1.00147.62           C
ATOM     16    C PRO A  20     168.987 178.399 124.159  1.00147.62           C
ATOM     17    O PRO A  20     169.871 178.383 125.023  1.00147.62           O
ATOM     18   CB PRO A  20     167.932 176.131 124.256  1.00147.62           C
ATOM     19   CG PRO A  20     166.666 175.609 123.703  1.00147.62           C
ATOM     20   CD PRO A  20     166.164 176.682 122.783  1.00147.62           C
ATOM     21    N ALA A  21     169.132 179.048 123.008  1.00137.42           N
ATOM     22   CA ALA A  21     170.239 179.960 122.792  1.00137.42           C
ATOM     23    C ALA A  21     169.923 181.367 123.264  1.00137.42           C
ATOM     24    O ALA A  21     170.816 182.221 123.264  1.00137.42           O
ATOM     25   CB ALA A  21     170.623 179.995 121.312  1.00137.42           C
ATOM   6817    N PHE B1079     209.205 229.398 157.280  1.00201.42           N
ATOM   6818   CA PHE B1079     209.135 228.904 155.908  1.00201.42           C
ATOM   6819    C PHE B1079     210.333 229.321 155.074  1.00201.42           C
ATOM   6820    O PHE B1079     211.134 230.155 155.490  1.00201.42           O
ATOM   6821   CB PHE B1079     207.855 229.387 155.225  1.00201.42           C
ATOM   6822   CG PHE B1079     206.660 228.517 155.487  1.00201.42           C
ATOM   6823  CD1 PHE B1079     206.755 227.404 156.306  1.00201.42           C
ATOM   6824  CD2 PHE B1079     205.442 228.810 154.905  1.00201.42           C
ATOM   6825  CE1 PHE B1079     205.660 226.607 156.540  1.00201.42           C
ATOM   6826  CE2 PHE B1079     204.349 228.015 155.128  1.00201.42           C
ATOM   6827   CZ PHE B1079     204.461 226.904 155.940  1.00201.42           C
ATOM   6828    N TRP B1080     210.456 228.732 153.891  1.00199.36           N
ATOM   6829   CA TRP B1080     211.515 229.114 152.957  1.00199.36           C
ATOM   6830    C TRP B1080     210.986 230.156 151.979  1.00199.36           C
ATOM   6831    O TRP B1080     211.175 230.070 150.768  1.00199.36           O
ATOM   6832   CB TRP B1080     212.054 227.888 152.221  1.00199.36           C
ATOM   6833   CG TRP B1080     212.754 226.844 153.077  1.00199.36           C
ATOM   6834  CD1 TRP B1080     212.843 226.809 154.438  1.00199.36           C
ATOM   6835  CD2 TRP B1080     213.449 225.681 152.607  1.00199.36           C
ATOM   6836  NE1 TRP B1080     213.547 225.703 154.845  1.00199.36           N
ATOM   6837  CE2 TRP B1080     213.933 224.994 153.738  1.00199.36           C
ATOM   6838  CE3 TRP B1080     213.717 225.161 151.338  1.00199.36           C
ATOM   6839  CZ2 TRP B1080     214.664 223.812 153.639  1.00199.36           C
ATOM   6840  CZ3 TRP B1080     214.439 223.983 151.241  1.00199.36           C
ATOM   6841  CH2 TRP B1080     214.905 223.324 152.385  1.00199.36           C
ATOM   6842    N GLY B1081     210.350 231.188 152.527  1.00202.02           N
ATOM   6843   CA GLY B1081     209.680 232.197 151.747  1.00202.02           C
ATOM   6844    C GLY B1081     210.610 233.296 151.287  1.00202.02           C
ATOM   6845    O GLY B1081     211.731 233.041 150.836  1.00202.02           O
ATOM   8992    N ASP D1213     155.940 184.063 164.115  1.00 63.23           N
ATOM   8993   CA ASP D1213     156.815 184.198 162.956  1.00 63.23           C
ATOM   8998    N PRO D1214     157.660 186.520 163.031  1.00 74.86           N
ATOM   9000    C PRO D1214     159.295 187.841 161.644  1.00 74.86           C
ATOM   9001    O PRO D1214     160.223 188.477 162.151  1.00 74.86           O
ATOM   9002   CB PRO D1214     157.966 188.792 163.561  1.00 74.86           C
ATOM   9003   CG PRO D1214     157.507 187.993 164.774  1.00 74.86           C
ATOM   9004   CD PRO D1214     157.962 186.611 164.460  1.00 74.86           C
"""


@pytest.fixture
def malformed_cif_data() -> str:
    """Make a malformed CIF data string, in which the sequence is not continuous."""
    return """
        data_TEST
        #
        _entry.id TEST
        #
        loop_
        _entity.id
        _entity.type
        1 polymer
        2 polymer
        3 polymer
        #
        loop_
        _entity_poly.entity_id
        _entity_poly.pdbx_seq_one_letter_code
        _entity_poly.type
        _entity_poly.pdbx_strand_id
        1 NDA 'polypeptide(L)' A,D
        2 FWG 'polypeptide(L)' B
        #
        loop_
        _entity_poly_seq.entity_id
        _entity_poly_seq.mon_id
        _entity_poly_seq.num
        1 ASN 1
        1 ASP 2
        1 ALA 3
        2 PHE 1
        2 TRP 2
        2 GLY 3
        #
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
        ATOM 1     N N   . ASN A 1 1   ? 161.767 176.410 123.966 1.00 167.91 ? 18   ASN A N   1
        ATOM 2     C CA  . ASN A 1 1   ? 161.584 177.567 123.097 1.00 167.91 ? 18   ASN A CA  1
        ATOM 3     C C   . ASN A 1 1   ? 162.885 177.931 122.400 1.00 167.91 ? 18   ASN A C   1
        ATOM 6     N N   . ASP A 1 2   ? 163.646 178.843 123.006 1.00 156.80 ? 19   ASP A N   1
        ATOM 7     C CA  . ASP A 1 2   ? 164.896 179.346 122.441 1.00 156.80 ? 19   ASP A CA  1
        ATOM 8     C C   . ASP A 1 2   ? 166.019 179.103 123.443 1.00 156.80 ? 19   ASP A C   1
        ATOM 9     O O   . ASP A 1 2   ? 166.344 179.993 124.242 1.00 156.80 ? 19   ASP A O   1
        ATOM 10    C CB  . ASP A 1 2   ? 164.785 180.832 122.110 1.00 156.80 ? 19   ASP A CB  1
        ATOM 11    C CG  . ASP A 1 2   ? 163.728 181.117 121.068 1.00 156.80 ? 19   ASP A CG  1
        ATOM 12    O OD1 . ASP A 1 2   ? 163.633 180.352 120.088 1.00 156.80 ? 19   ASP A OD1 1
        ATOM 13    O OD2 . ASP A 1 2   ? 162.978 182.100 121.238 1.00 156.80 ? 19   ASP A OD2 1
        ATOM 14    N N   . PRO A 1 3   ? 166.641 177.917 123.429 1.00 147.62 ? 20   PRO A N   1
        ATOM 15    C CA  . PRO A 1 3   ? 167.701 177.635 124.411 1.00 147.62 ? 20   PRO A CA  1
        ATOM 16    C C   . PRO A 1 3   ? 168.987 178.399 124.159 1.00 147.62 ? 20   PRO A C   1
        ATOM 17    O O   . PRO A 1 3   ? 169.871 178.383 125.023 1.00 147.62 ? 20   PRO A O   1
        ATOM 18    C CB  . PRO A 1 3   ? 167.932 176.131 124.256 1.00 147.62 ? 20   PRO A CB  1
        ATOM 19    C CG  . PRO A 1 3   ? 166.666 175.609 123.703 1.00 147.62 ? 20   PRO A CG  1
        ATOM 20    C CD  . PRO A 1 3   ? 166.164 176.682 122.783 1.00 147.62 ? 20   PRO A CD  1
        ATOM 21    N N   . ALA A 1 4   ? 169.132 179.048 123.008 1.00 137.42 ? 21   ALA A N   1
        ATOM 22    C CA  . ALA A 1 4   ? 170.239 179.960 122.792 1.00 137.42 ? 21   ALA A CA  1
        ATOM 23    C C   . ALA A 1 4   ? 169.923 181.367 123.264 1.00 137.42 ? 21   ALA A C   1
        ATOM 24    O O   . ALA A 1 4   ? 170.816 182.221 123.264 1.00 137.42 ? 21   ALA A O   1
        ATOM 25    C CB  . ALA A 1 4   ? 170.623 179.995 121.312 1.00 137.42 ? 21   ALA A CB  1
        ATOM 6817  N N   . PHE B 2 1 ? 209.205 229.398 157.280 1.00 201.42 ? 1079 PHE B N   1
        ATOM 6818  C CA  . PHE B 2 1 ? 209.135 228.904 155.908 1.00 201.42 ? 1079 PHE B CA  1
        ATOM 6819  C C   . PHE B 2 1 ? 210.333 229.321 155.074 1.00 201.42 ? 1079 PHE B C   1
        ATOM 6820  O O   . PHE B 2 1 ? 211.134 230.155 155.490 1.00 201.42 ? 1079 PHE B O   1
        ATOM 6821  C CB  . PHE B 2 1 ? 207.855 229.387 155.225 1.00 201.42 ? 1079 PHE B CB  1
        ATOM 6822  C CG  . PHE B 2 1 ? 206.660 228.517 155.487 1.00 201.42 ? 1079 PHE B CG  1
        ATOM 6823  C CD1 . PHE B 2 1 ? 206.755 227.404 156.306 1.00 201.42 ? 1079 PHE B CD1 1
        ATOM 6824  C CD2 . PHE B 2 1 ? 205.442 228.810 154.905 1.00 201.42 ? 1079 PHE B CD2 1
        ATOM 6825  C CE1 . PHE B 2 1 ? 205.660 226.607 156.540 1.00 201.42 ? 1079 PHE B CE1 1
        ATOM 6826  C CE2 . PHE B 2 1 ? 204.349 228.015 155.128 1.00 201.42 ? 1079 PHE B CE2 1
        ATOM 6827  C CZ  . PHE B 2 1 ? 204.461 226.904 155.940 1.00 201.42 ? 1079 PHE B CZ  1
        ATOM 6828  N N   . TRP B 2 2 ? 210.456 228.732 153.891 1.00 199.36 ? 1080 TRP B N   1
        ATOM 6829  C CA  . TRP B 2 2 ? 211.515 229.114 152.957 1.00 199.36 ? 1080 TRP B CA  1
        ATOM 6830  C C   . TRP B 2 2 ? 210.986 230.156 151.979 1.00 199.36 ? 1080 TRP B C   1
        ATOM 6831  O O   . TRP B 2 2 ? 211.175 230.070 150.768 1.00 199.36 ? 1080 TRP B O   1
        ATOM 6832  C CB  . TRP B 2 2 ? 212.054 227.888 152.221 1.00 199.36 ? 1080 TRP B CB  1
        ATOM 6833  C CG  . TRP B 2 2 ? 212.754 226.844 153.077 1.00 199.36 ? 1080 TRP B CG  1
        ATOM 6834  C CD1 . TRP B 2 2 ? 212.843 226.809 154.438 1.00 199.36 ? 1080 TRP B CD1 1
        ATOM 6835  C CD2 . TRP B 2 2 ? 213.449 225.681 152.607 1.00 199.36 ? 1080 TRP B CD2 1
        ATOM 6836  N NE1 . TRP B 2 2 ? 213.547 225.703 154.845 1.00 199.36 ? 1080 TRP B NE1 1
        ATOM 6837  C CE2 . TRP B 2 2 ? 213.933 224.994 153.738 1.00 199.36 ? 1080 TRP B CE2 1
        ATOM 6838  C CE3 . TRP B 2 2 ? 213.717 225.161 151.338 1.00 199.36 ? 1080 TRP B CE3 1
        ATOM 6839  C CZ2 . TRP B 2 2 ? 214.664 223.812 153.639 1.00 199.36 ? 1080 TRP B CZ2 1
        ATOM 6840  C CZ3 . TRP B 2 2 ? 214.439 223.983 151.241 1.00 199.36 ? 1080 TRP B CZ3 1
        ATOM 6841  C CH2 . TRP B 2 2 ? 214.905 223.324 152.385 1.00 199.36 ? 1080 TRP B CH2 1
        ATOM 6842  N N   . GLY B 2 3 ? 210.350 231.188 152.527 1.00 202.02 ? 1081 GLY B N   1
        ATOM 6843  C CA  . GLY B 2 3 ? 209.680 232.197 151.747 1.00 202.02 ? 1081 GLY B CA  1
        ATOM 6844  C C   . GLY B 2 3 ? 210.610 233.296 151.287 1.00 202.02 ? 1081 GLY B C   1
        ATOM 6845  O O   . GLY B 2 3 ? 211.731 233.041 150.836 1.00 202.02 ? 1081 GLY B O   1
        ATOM 8992  N N   . ASP D 1 2    ? 155.940 184.063 164.115 1.00 63.23  ? 1213 ASP C N   1
        ATOM 8993  C CA  . ASP D 1 2    ? 156.815 184.198 162.956 1.00 63.23  ? 1213 ASP C CA  1
        ATOM 8998  N N   . PRO D 1 3    ? 157.660 186.520 163.031 1.00 74.86  ? 1214 PRO C N   1
        ATOM 9000  C C   . PRO D 1 3    ? 159.295 187.841 161.644 1.00 74.86  ? 1214 PRO C C   1
        ATOM 9001  O O   . PRO D 1 3    ? 160.223 188.477 162.151 1.00 74.86  ? 1214 PRO C O   1
        ATOM 9002  C CB  . PRO D 1 3    ? 157.966 188.792 163.561 1.00 74.86  ? 1214 PRO C CB  1
        ATOM 9003  C CG  . PRO D 1 3    ? 157.507 187.993 164.774 1.00 74.86  ? 1214 PRO C CG  1
        ATOM 9004  C CD  . PRO D 1 3    ? 157.962 186.611 164.460 1.00 74.86  ? 1214 PRO C CD  1
        #
        """


@pytest.fixture
def expected_atom_names() -> dict[str, list[str]]:
    return {
        "ALA": ["N", "CA", "C", "O", "CB"],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        "GLY": ["N", "CA", "C", "O"],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
        "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    }


@pytest.fixture
def small_ccd(expected_atom_names: dict[str, list[str]]) -> dict[str, Mol]:
    """Make a small map from residue name to RDKit molecule."""
    from rdkit import Chem

    resis = ["ALA", "ASP", "ASN", "GLY", "PRO", "PHE", "TRP", "GLY", "VAL", "MET"]

    ccd = {}
    for residue_name in resis:
        # mol has trailing oxygen
        mol = Chem.MolFromSequence(AA_THREE_TO_ONE_MAP[residue_name])
        c_terminus_O_idx = -1
        for atom in reversed(mol.GetAtoms()):
            if atom.GetSymbol() == "O" and atom.GetDegree() == 1:
                c_terminus_O_idx = atom.GetIdx()
                break

        if c_terminus_O_idx == -1:
            raise ValueError

        # make modifiable atom
        rwmol = Chem.RWMol(mol)
        # remove trailing O
        rwmol.RemoveAtom(c_terminus_O_idx)
        # update the atom names
        for atom, name in zip(rwmol.GetAtoms(), expected_atom_names[residue_name]):
            atom.SetProp("name", name)
        # we don't care about coords here, so just add
        ccd[residue_name] = rwmol

    return ccd


### Gemmi fixtures ###
def create_residue(residue_name: str, auth_res_idx: int) -> gemmi.Residue:
    resi = gemmi.Residue()
    resi.seqid = gemmi.SeqId(auth_res_idx, " ")
    resi.entity_id = "1"
    resi.name = residue_name
    resi.entity_type = gemmi.EntityType.Polymer
    resi.subchain = "A"
    resi.segment = "A"
    return resi


def add_atom_to_resi(
    residue: gemmi.Residue, atom_name: str, atom_element: str, atom_coords: tuple[float, float, float]
) -> None:
    """Add an atom to a gemmi residue."""
    atom = gemmi.Atom()
    atom.name = atom_name
    atom.element = gemmi.Element(atom_element)
    atom.pos = gemmi.Position(atom_coords[0], atom_coords[1], atom_coords[2])
    atom.occ = 1.0
    atom.b_iso = 50.0 if atom_element == "O" else 100.0
    atom.altloc = "\0"

    residue.add_atom(atom)


@pytest.fixture
def gemmi_residues() -> list[gemmi.Residue | None]:
    """Make a list of gemmi residues."""
    ala_resi = create_residue("ALA", 1)

    atoms = [
        ("N", "N", (1.0, 2.0, 3.0)),
        ("CA", "C", (4.0, 5.0, 6.0)),
        ("C", "C", (7.0, 8.0, 9.0)),
        ("O", "O", (10.0, 11.0, 12.0)),
        ("CB", "C", (13.0, 14.0, 15.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(ala_resi, atom_name, atom_element, atom_coords)

    # make a residue with missing sidechain atoms
    asp_resi = create_residue("ASP", 2)
    atoms = [
        ("N", "N", (1.0, 2.0, 3.0)),
        ("CA", "C", (4.0, 5.0, 6.0)),
        ("C", "C", (7.0, 8.0, 9.0)),
        ("O", "O", (10.0, 11.0, 12.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(asp_resi, atom_name, atom_element, atom_coords)

    # make a residue missing backbone atoms but not CA
    asn_resi = create_residue("ASN", 3)

    atoms = [
        ("CA", "C", (4.0, 5.0, 6.0)),
        ("CB", "C", (13.0, 14.0, 15.0)),
        ("CG", "C", (0.0, 0.0, 0.0)),
        ("OD1", "O", (0.0, 0.0, 0.0)),
        ("ND2", "N", (0.0, 0.0, 0.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(asn_resi, atom_name, atom_element, atom_coords)

    # make a residue missing ca
    val_resi_no_ca = create_residue("VAL", 4)

    atoms = [
        ("N", "N", (1.0, 2.0, 3.0)),
        ("C", "C", (7.0, 8.0, 9.0)),
        ("O", "O", (10.0, 11.0, 12.0)),
        ("CB", "C", (13.0, 14.0, 15.0)),
        ("CG1", "C", (0.0, 0.0, 0.0)),
        ("CG2", "C", (0.0, 0.0, 0.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(val_resi_no_ca, atom_name, atom_element, atom_coords)

    # make atoms out of order
    out_of_order_resi = create_residue("GLY", 7)

    atoms = [
        ("CA", "C", (4.0, 5.0, 6.0)),
        ("N", "N", (1.0, 2.0, 3.0)),
        ("O", "O", (10.0, 11.0, 12.0)),
        ("C", "C", (7.0, 8.0, 9.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(out_of_order_resi, atom_name, atom_element, atom_coords)

    # make a MSE residue
    mse_resi = create_residue("MSE", 8)

    atoms = [
        ("N", "N", (1.0, 2.0, 3.0)),
        ("CA", "C", (4.0, 5.0, 6.0)),
        ("C", "C", (7.0, 8.0, 9.0)),
        ("O", "O", (10.0, 11.0, 12.0)),
        ("CB", "C", (13.0, 14.0, 15.0)),
        ("CG", "C", (0.0, 0.0, 0.0)),
        ("SE", "SE", (0.0, 0.0, 0.0)),
        ("CE", "C", (0.0, 0.0, 0.0)),
    ]
    for atom_name, atom_element, atom_coords in atoms:
        add_atom_to_resi(mse_resi, atom_name, atom_element, atom_coords)

    residues = [ala_resi, asp_resi, asn_resi, val_resi_no_ca, out_of_order_resi, mse_resi, None]
    return residues


### Structure.py fixtures ###
@pytest.fixture
def resi_names() -> list[str]:
    return ["ALA", "ASN", "TRP", "VAL", "PHE", "MET"]


@pytest.fixture
def chains_standard_coords_by_resi(expected_atom_names: dict[str, list[str]]) -> dict[str, tuple[float, float, float]]:
    return {
        "ALA": [(i * 0.01, 0.0, 0.0) for i in range(len(expected_atom_names["ALA"]))],
        "ASN": [(i * 0.1, 0.0, 0.0) for i in range(len(expected_atom_names["ASN"]))],
        "TRP": [(i * 1.0, 0.0, 0.0) for i in range(len(expected_atom_names["TRP"]))],
        "VAL": [(i * 10.0, 0.0, 0.0) for i in range(len(expected_atom_names["VAL"]))],
        "PHE": [(i * 100.0, 0.0, 0.0) for i in range(len(expected_atom_names["PHE"]))],
        "MET": [(i * 1000.0, 0.0, 0.0) for i in range(len(expected_atom_names["MET"]))],
    }


@pytest.fixture
def chains_dtype() -> np.ndarray:
    return np.array(
        [
            ("A", "U", 0, 20, 0, 0, 0, 13, 0, 2),
            ("B", "V", 1, 21, 1, 0, 13, 21, 2, 2),
            ("C", "W", 2, 22, 2, 0, 34, 19, 4, 2),
        ],
        dtype=CHAINS_DTYPE,
    )


@pytest.fixture
def residues_dtype() -> np.ndarray:
    return np.array(
        [
            ("ALA", TOKEN_IDS["ALA"], 0, 0, 1, 0, 5, 1, True, True, True, True),
            ("ASN", TOKEN_IDS["ASN"], 0, 1, 2, 5, 8, 6, True, True, True, True),
            ("TRP", TOKEN_IDS["TRP"], 1, 0, 2, 13, 14, 14, True, True, True, True),
            ("VAL", TOKEN_IDS["VAL"], 1, 1, 3, 27, 7, 28, True, True, True, True),
            ("PHE", TOKEN_IDS["PHE"], 2, 0, 4, 34, 11, 35, True, True, True, True),
            ("MET", TOKEN_IDS["MET"], 2, 1, 5, 45, 8, 46, True, True, False, False),
        ],
        dtype=RESIDUES_DTYPE,
    )


@pytest.fixture
def atoms_dtype(
    chains_standard_coords_by_resi: dict[str, tuple[float, float, float]], expected_atom_names: dict[str, list[str]]
) -> np.ndarray:

    return np.concatenate(
        [
            np.array(
                [
                    (
                        asym_id,  # asym_id
                        res_idx,  # res_idx
                        atom_name,  # atom_name
                        chains_standard_coords_by_resi[rname][j],  # coords
                        True if rname != "MET" else False,  # is_present
                        100.0 if atom_name != "N" else 50.0,  # bfactor
                        np.nan if atom_name != "N" else 100.0,  # plddt
                    )
                    for j, atom_name in enumerate(expected_atom_names[rname])
                ],
                dtype=ATOMS_DTYPE,
            )
            for asym_id, res_idx, rname in [
                (0, 0, "ALA"),
                (0, 1, "ASN"),
                (1, 0, "TRP"),
                (1, 1, "VAL"),
                (2, 0, "PHE"),
                (2, 1, "MET"),
            ]
        ]
    )


@pytest.fixture
def interfaces_dtype() -> tuple[np.ndarray, np.ndarray]:
    chains_interfaces = np.array(
        [
            (0, 1),
            (1, 2),
        ],
        dtype=CHAIN_INTERFACE_DTYPE,
    )

    residue_interfaces = np.array([(0, 1, 1, 0), (1, 2, 0, 0), (1, 2, 1, 0)], dtype=RESIDUE_INTERFACE_DTYPE)
    return chains_interfaces, residue_interfaces


@pytest.fixture
def structure_standard(
    chains_dtype: np.ndarray,
    residues_dtype: np.ndarray,
    atoms_dtype: np.ndarray,
    interfaces_dtype: tuple[np.ndarray, np.ndarray],
) -> Structure:

    chains_dtype = chains_dtype.copy()
    residues_dtype = residues_dtype.copy()
    atoms_dtype = atoms_dtype.copy()

    return Structure(
        chains=chains_dtype,
        residues=residues_dtype,
        atoms=atoms_dtype,
        interfaces=interfaces_dtype[0],
        residue_interfaces=interfaces_dtype[1],
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


@pytest.fixture
def structure_standard_no_interfaces(
    chains_dtype: np.ndarray,
    residues_dtype: np.ndarray,
    atoms_dtype: np.ndarray,
) -> Structure:
    chains_dtype = chains_dtype.copy()
    residues_dtype = residues_dtype.copy()
    atoms_dtype = atoms_dtype.copy()
    return Structure(
        chains=chains_dtype,
        residues=residues_dtype,
        atoms=atoms_dtype,
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


@pytest.fixture
def structure_chain_A_only(
    chains_dtype: np.ndarray,
    residues_dtype: np.ndarray,
    atoms_dtype: np.ndarray,
    interfaces_dtype: tuple[np.ndarray, np.ndarray],
) -> Structure:

    chains_dtype = chains_dtype.copy()
    residues_dtype = residues_dtype.copy()
    atoms_dtype = atoms_dtype.copy()

    residues_cropped = residues_dtype[:2]

    atom_start_idx = residues_cropped["atom_start_idx"][0]
    atom_end_idx = atom_start_idx + np.sum(residues_cropped["num_atoms"])
    atoms_cropped = atoms_dtype[atom_start_idx:atom_end_idx]

    # update atom start idx after assigning atoms_cropped
    residues_cropped["atom_start_idx"] = np.cumsum([0] + residues_cropped["num_atoms"].tolist()[:-1])
    residues_cropped["atom_center"] = residues_cropped["atom_start_idx"] + 1

    chains_cropped = chains_dtype[np.isin(chains_dtype["asym_id"], residues_cropped["asym_id"])]
    chains_cropped["num_residues"] = np.array([2])
    chains_cropped["res_start_idx"] = np.array([0])
    chains_cropped["num_atoms"] = np.array(
        [
            np.sum(residues_cropped[residues_cropped["asym_id"] == 0]["num_atoms"]),
        ]
    )
    chains_cropped["atom_start_idx"] = np.array([0])

    return Structure(
        residues=residues_cropped,
        atoms=atoms_cropped,
        chains=chains_cropped,
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
    )


@pytest.fixture
def structure_chain_B_C_only(
    chains_dtype: np.ndarray,
    residues_dtype: np.ndarray,
    atoms_dtype: np.ndarray,
    interfaces_dtype: tuple[np.ndarray, np.ndarray],
) -> Structure:
    chains_dtype = chains_dtype.copy()
    residues_dtype = residues_dtype.copy()
    atoms_dtype = atoms_dtype.copy()

    residues_cropped = residues_dtype[2:]

    atom_start_idx = residues_cropped["atom_start_idx"][0]
    atom_end_idx = atom_start_idx + np.sum(residues_cropped["num_atoms"])
    atoms_cropped = atoms_dtype[atom_start_idx:atom_end_idx]

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

    return Structure(
        residues=residues_cropped,
        atoms=atoms_cropped,
        chains=chains_cropped,
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
    )


@pytest.fixture
def structure_token_features_standard(
    chains_standard_coords_by_resi: dict[str, tuple[float, float, float]],
) -> StructureTokenFeatures:
    structure_token_features = StructureTokenFeatures(
        token_index=torch.arange(6),
        residue_index=torch.tensor([0, 1, 0, 1, 0, 1]),
        auth_residue_index=torch.tensor([1, 2, 2, 3, 4, 5]),
        asym_id=torch.tensor([0, 0, 1, 1, 2, 2]),
        auth_asym_id=torch.tensor([20, 20, 21, 21, 22, 22]),
        entity_id=torch.tensor([0, 0, 1, 1, 2, 2]),
        sym_id=torch.zeros(6),
        token_resolved_mask=torch.tensor([True, True, True, True, True, False]),
        token_pad_mask=torch.tensor([True, True, True, True, True, True]),
        coords=torch.tensor(
            [chains_standard_coords_by_resi[rname][1] for rname in ["ALA", "ASN", "TRP", "VAL", "PHE", "MET"]]
        ),
    )
    return structure_token_features


@pytest.fixture
def frame_structure_token_features_bb_standard(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> FrameStructureTokenFeatures:
    """Make a frame structure token features object for the standard backbone structure w/ atoms N, CA, C."""
    coords = torch.cat([torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]).unsqueeze(0)
    frames_idx = torch.arange(len(resi_names) * 3).reshape(-1, 3)
    atom_to_token = torch.repeat_interleave(torch.eye(len(resi_names)), repeats=3, dim=0)
    frame_resolved_mask = torch.tensor([True, True, True, True, True, False])
    atom_resolved_mask = torch.cat(
        [
            torch.ones((len(resi_names) - 1) * 3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        ]
    )
    atom_pad_mask = torch.ones(len(resi_names) * 3, dtype=torch.bool)
    frame_structure_token_features = FrameStructureTokenFeatures(
        token_index=torch.arange(6),
        residue_index=torch.tensor([0, 1, 0, 1, 0, 1]),
        auth_residue_index=torch.tensor([1, 2, 2, 3, 4, 5]),
        asym_id=torch.tensor([0, 0, 1, 1, 2, 2]),
        auth_asym_id=torch.tensor([20, 20, 21, 21, 22, 22]),
        entity_id=torch.tensor([0, 0, 1, 1, 2, 2]),
        sym_id=torch.zeros(6),
        token_resolved_mask=torch.tensor([True, True, True, True, True, False]),
        token_pad_mask=torch.tensor([True, True, True, True, True, True]),
        coords=coords,
        frames_idx=frames_idx,
        atom_to_token=atom_to_token,
        frame_resolved_mask=frame_resolved_mask,
        atom_resolved_mask=atom_resolved_mask,
        atom_pad_mask=atom_pad_mask,
    )
    return frame_structure_token_features


@pytest.fixture
def frame_structure_token_features_chain_A_only(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> FrameStructureTokenFeatures:
    resi_names = resi_names[:2]
    coords = torch.cat([torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]).unsqueeze(0)
    frames_idx = torch.arange(len(resi_names) * 3).reshape(-1, 3)
    atom_to_token = torch.repeat_interleave(torch.eye(len(resi_names)), repeats=3, dim=0)
    frame_resolved_mask = torch.tensor([True, True])
    atom_resolved_mask = torch.cat(
        [
            torch.ones((len(resi_names) - 1) * 3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        ]
    )
    atom_pad_mask = torch.ones(len(resi_names) * 3, dtype=torch.bool)
    frame_structure_token_features = FrameStructureTokenFeatures(
        token_index=torch.arange(2),
        residue_index=torch.tensor([0, 1]),
        auth_residue_index=torch.tensor([1, 2]),
        asym_id=torch.tensor([0, 0]),
        auth_asym_id=torch.tensor([20, 20]),
        entity_id=torch.tensor([0, 0]),
        sym_id=torch.zeros(2),
        token_resolved_mask=torch.tensor([True, True]),
        token_pad_mask=torch.tensor([True, True]),
        coords=coords,
        frames_idx=frames_idx,
        atom_to_token=atom_to_token,
        frame_resolved_mask=frame_resolved_mask,
        atom_resolved_mask=atom_resolved_mask,
        atom_pad_mask=atom_pad_mask,
    )
    return frame_structure_token_features


@pytest.fixture
def frame_structure_token_features_chain_B_C_only(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> FrameStructureTokenFeatures:
    resi_names = resi_names[2:]
    coords = torch.cat([torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]).unsqueeze(0)
    frames_idx = torch.arange(len(resi_names) * 3).reshape(-1, 3)
    atom_to_token = torch.repeat_interleave(torch.eye(len(resi_names)), repeats=3, dim=0)
    frame_resolved_mask = torch.tensor([True, True, True, False])
    atom_resolved_mask = torch.cat(
        [
            torch.ones((len(resi_names) - 1) * 3, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool),
        ]
    )
    atom_pad_mask = torch.ones(len(resi_names) * 3, dtype=torch.bool)
    frame_structure_token_features = FrameStructureTokenFeatures(
        token_index=torch.arange(len(resi_names)),
        residue_index=torch.tensor([0, 1, 0, 1]),
        auth_residue_index=torch.tensor([2, 3, 4, 5]),
        asym_id=torch.tensor([1, 1, 2, 2]),
        auth_asym_id=torch.tensor([21, 21, 22, 22]),
        entity_id=torch.tensor([0, 0, 1, 1]),
        sym_id=torch.zeros(len(resi_names)),
        token_resolved_mask=torch.tensor([True, True, True, False]),
        token_pad_mask=torch.tensor([True, True, True, True]),
        coords=coords,
        frames_idx=frames_idx,
        atom_to_token=atom_to_token,
        frame_resolved_mask=frame_resolved_mask,
        atom_resolved_mask=atom_resolved_mask,
        atom_pad_mask=atom_pad_mask,
    )
    return frame_structure_token_features


@pytest.fixture
def feature_dict_standard(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> dict[str, torch.Tensor]:

    feat_dict = {
        "token_index": torch.arange(6),
        "residue_index": torch.tensor([0, 1, 0, 1, 0, 1]),
        "auth_residue_index": torch.tensor([1, 2, 2, 3, 4, 5]),
        "original_residue_index": torch.tensor([0, 1, 0, 1, 0, 1]),
        "asym_id": torch.tensor([0, 0, 1, 1, 2, 2]),
        "auth_asym_id": torch.tensor([20, 20, 21, 21, 22, 22]),
        "entity_id": torch.tensor([0, 0, 1, 1, 2, 2]),
        "sym_id": torch.zeros(6),
        "token_resolved_mask": torch.tensor([True, True, True, True, True, False]),
        "token_pad_mask": torch.tensor([True, True, True, True, True, True]),
        "coords": torch.cat(
            [torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]
        ).unsqueeze(0),
        "frames_idx": torch.arange(6 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(6), repeats=3, dim=0),
        "frame_resolved_mask": torch.tensor([True, True, True, True, True, False]),
        "atom_resolved_mask": torch.cat(
            [
                torch.ones((5) * 3, dtype=torch.bool),
                torch.zeros(3, dtype=torch.bool),
            ]
        ),
        "atom_pad_mask": torch.ones(6 * 3, dtype=torch.bool),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 2
            + [PocketFeature.UNSELECTED, PocketFeature.POCKET, PocketFeature.UNSELECTED, PocketFeature.UNSELECTED]
        ),
        "inpaint_feature": torch.tensor([InpaintFeature.BINDER_INPAINT] * 2 + [InpaintFeature.TARGET_CONTEXT] * 4),
        "noised_mask": torch.tensor([True, True, False, False, False, False]),
        "binder_mask": torch.tensor([True, True, False, False, False, False]),
        "target_mask": torch.tensor([False, False, True, True, True, True]),
    }
    check_feature_dict_lengths(feat_dict)
    return feat_dict


@pytest.fixture
def feature_dict_chain_A_only(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> dict[str, torch.Tensor]:
    resi_names = resi_names[:2]
    feat_dict = {
        "token_index": torch.arange(2),
        "residue_index": torch.tensor([0, 1]),
        "auth_residue_index": torch.tensor([1, 2]),
        "original_residue_index": torch.tensor([0, 1]),
        "asym_id": torch.tensor(
            [
                0,
                0,
            ]
        ),
        "auth_asym_id": torch.tensor([20, 20]),
        "entity_id": torch.tensor([0, 0]),
        "sym_id": torch.zeros(2),
        "token_resolved_mask": torch.tensor([True, True]),
        "token_pad_mask": torch.tensor([True, True]),
        "coords": torch.cat(
            [torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]
        ).unsqueeze(0),
        "frames_idx": torch.arange(2 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(2), repeats=3, dim=0),
        "frame_resolved_mask": torch.tensor([True, True]),
        "atom_resolved_mask": torch.ones(2 * 3, dtype=torch.bool),
        "atom_pad_mask": torch.ones(2 * 3, dtype=torch.bool),
        "pocket_feature": torch.tensor([PocketFeature.BINDER] * 2),
        "inpaint_feature": torch.tensor([InpaintFeature.BINDER_INPAINT] * 2),
        "noised_mask": torch.tensor([True, True]),
        "binder_mask": torch.tensor([True, True]),
        "target_mask": torch.tensor([False, False]),
    }
    check_feature_dict_lengths(feat_dict)
    return feat_dict


@pytest.fixture
def feature_dict_chain_B_C_only(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> dict[str, torch.Tensor]:
    resi_names = resi_names[2:]
    feat_dict = {
        "token_index": torch.arange(),
        "residue_index": torch.tensor([0, 1, 0, 1, 0, 1]),
        "auth_residue_index": torch.tensor([2, 3, 4, 5]),
        "original_residue_index": torch.tensor([0, 1, 0, 1]),
        "asym_id": torch.tensor([1, 1, 2, 2]),
        "auth_asym_id": torch.tensor([21, 21, 22, 22]),
        "entity_id": torch.tensor([0, 0, 1, 1]),
        "sym_id": torch.zeros(4),
        "token_resolved_mask": torch.tensor([True, True, True, False]),
        "token_pad_mask": torch.tensor([True, True, True, True]),
        "coords": torch.cat(
            [torch.tensor(chains_standard_coords_by_resi[rname][:3]) for rname in resi_names]
        ).unsqueeze(0),
        "frames_idx": torch.arange(4 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(4), repeats=3, dim=0),
        "frame_resolved_mask": torch.tensor([True, True, True, False]),
        "atom_resolved_mask": torch.cat(
            [
                torch.ones((3) * 3, dtype=torch.bool),
                torch.zeros(3, dtype=torch.bool),
            ]
        ),
        "atom_pad_mask": torch.ones(4 * 3, dtype=torch.bool),
        "pocket_feature": torch.tensor(
            [PocketFeature.UNSELECTED, PocketFeature.POCKET, PocketFeature.UNSELECTED, PocketFeature.UNSELECTED]
        ),
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT] * 4),
        "noised_mask": torch.tensor([False, False, False, False]),
        "binder_mask": torch.tensor([False, False, False, False]),
        "target_mask": torch.tensor([True, True, True, True]),
    }
    check_feature_dict_lengths(feat_dict)
    return feat_dict


@pytest.fixture
def feature_dict_standard_inpaint(
    resi_names: list[str], chains_standard_coords_by_resi: dict[str, tuple[float, float, float]]
) -> dict[str, torch.Tensor]:
    """feature_dict_standard inpainted A2 --> 4 residues, C1-2 --> 5 residues
    - Total size: 6-1+4-2+5 = 12 residues.
    """
    BC = InpaintFeature.BINDER_CONTEXT
    TC = InpaintFeature.TARGET_CONTEXT
    BI = InpaintFeature.BINDER_INPAINT

    output_feature_dict = {
        "token_index": torch.arange(12),
        "asym_id": torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]),
        "entity_id": torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]),
        "sym_id": torch.zeros(12),
        "auth_asym_id": torch.tensor([20, 20, 20, 20, 20, 21, 21, 22, 22, 22, 22, 22]),
        "residue_index": torch.tensor([0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 4]),
        "auth_residue_index": torch.tensor([1, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4]),
        "original_residue_index": torch.tensor([0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]),
        "token_resolved_mask": torch.full((12,), True),
        "token_pad_mask": torch.full((12,), True),
        "frame_resolved_mask": torch.full((12,), True),
        "frames_idx": torch.arange(12 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(12), repeats=3, dim=0),
        "atom_resolved_mask": torch.full((12 * 3,), True),
        "atom_pad_mask": torch.full((12 * 3,), True),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 5 + [PocketFeature.UNSELECTED, PocketFeature.POCKET] + [PocketFeature.BINDER] * 5
        ),
        "inpaint_feature": torch.tensor([BC, BI, BI, BI, BI, TC, TC, BI, BI, BI, BI, BI]),
        "noised_mask": torch.tensor([True] * 5 + [False] * 2 + [True] * 5),
        "binder_mask": torch.tensor([True] * 5 + [False] * 2 + [True] * 5),
        "target_mask": torch.tensor([False] * 5 + [True] * 2 + [False] * 5),
    }
    # get inpaint region coordinates
    # A2 => resi position 1 0-indexed
    inpaint_region1_coords = torch.tensor(chains_standard_coords_by_resi[resi_names[1]][:3]).mean(dim=0)
    # C1-2 => resi positions 4 and 5, 0-indexed. Remove position 5 because it is unresolved
    inpaint_region2_coords = torch.tensor(chains_standard_coords_by_resi[resi_names[4]][:3]).mean(dim=0)
    coords_lst = (
        [torch.tensor(chains_standard_coords_by_resi[resi_names[0]][:3])]
        + [torch.tile(torch.tensor(inpaint_region1_coords), (4 * 3, 1))]
        + [torch.tensor(chains_standard_coords_by_resi[resname][:3]) for resname in resi_names[2:4]]
        + [torch.tile(torch.tensor(inpaint_region2_coords), (5 * 3, 1))]
    )

    output_feature_dict["coords"] = torch.cat(coords_lst).unsqueeze(0)

    check_feature_dict_lengths(output_feature_dict)

    return output_feature_dict


## Fixtures for COM tests


@pytest.fixture
def features_all_design() -> dict[str, torch.Tensor]:
    """Features with all design residues."""
    atom_to_token = torch.cat(
        [torch.cat([torch.eye(7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long)], dim=0), torch.zeros(8, 1)],
        dim=1,
    ).repeat_interleave(3, dim=0)
    feature_dict = {
        "asym_id": torch.zeros(8),
        "auth_asym_id": torch.ones(8) * 20,
        "entity_id": torch.zeros(8),
        "sym_id": torch.zeros(8),
        "residue_index": torch.arange(8, dtype=torch.long),
        "token_index": torch.arange(8, dtype=torch.long),
        "auth_residue_index": torch.arange(8, dtype=torch.long),
        # atom features
        # residues 0-5 and part of 6 (N, CA) resolved and have coordinates 5.5 -> 6 * 3 - 1 atoms resolved
        # residue 6 unresolved atom (C) and residue 7 is fully unresolved have coordinates 6.6 -> 3 + 1 atoms resolved
        # pad tokens have coordinates 0.0 -> 3 atoms with 0.0
        "coords": torch.cat([torch.ones(6 * 3 - 1, 3) * 5.5, torch.ones(3 + 1, 3) * 6.6, torch.zeros(3, 3)]).unsqueeze(
            0
        ),
        "atom_resolved_mask": torch.cat([torch.ones(6 * 3 - 1), torch.zeros(2 * 3 + 1)]),
        "atom_pad_mask": torch.cat([torch.ones(7 * 3), torch.zeros(1 * 3)]),
        # other token features
        "token_pad_mask": torch.cat([torch.ones(7), torch.zeros(1)]),
        "token_resolved_mask": torch.cat([torch.ones(6), torch.zeros(1), torch.zeros(1)]),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 8),
        "frame_resolved_mask": torch.cat([torch.ones(5), torch.zeros(3)]),
        "noised_mask": torch.cat([torch.ones(7, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]),
        "frames_idx": torch.cat([torch.arange(7 * 3).reshape(7, 3), torch.zeros(1, 3)]),
        "inpaint_feature": torch.tensor([InpaintFeature.BINDER_INPAINT] * 7 + [0]),
        "binder_mask": torch.tensor([True] * 7 + [False]),
        "target_mask": torch.zeros(8, dtype=torch.bool),
        # atom to token
        "atom_to_token": atom_to_token,
    }
    check_feature_dict_lengths(feature_dict)
    return feature_dict


@pytest.fixture
def features_all_inpaint_options() -> dict[str, torch.Tensor]:
    """Features with a mix of context and design residues.
    - all resolved
    - all inpaint options
    - coordinates are different for each inpaint option
    - noise is split across inpaint options
    - single pad token
    """
    atom_to_token = torch.cat(
        [
            torch.cat([torch.eye(8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long)], dim=0),
            torch.zeros(9, 1, dtype=torch.long),
        ],
        dim=1,
    ).repeat_interleave(3, dim=0)
    feature_dict = {
        "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.long),
        "auth_asym_id": torch.tensor([20, 20, 20, 20, 21, 21, 21, 21, 0], dtype=torch.long),
        "entity_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.long),
        "sym_id": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        "residue_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        "auth_residue_index": torch.tensor([5, 6, 7, 8, 2, 3, 4, 5, 0], dtype=torch.long),
        # atom features
        "coords": torch.cat(
            [
                torch.ones(2 * 3, 3) * 1.5,  # binder context
                torch.ones(2 * 3, 3) * 3.3,  # binder inpaint
                torch.ones(2 * 3, 3) * 4.4,  # target context
                torch.ones(2 * 3, 3) * 6.1,  # target inpaint
                torch.zeros(1 * 3, 3),  # pad
            ]
        ).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 8 * 3 + [False] * 1 * 3, dtype=torch.bool),
        "atom_pad_mask": torch.tensor([True] * 8 * 3 + [False] * 1 * 3, dtype=torch.bool),
        # other token features
        "token_pad_mask": torch.tensor([True] * 8 + [False], dtype=torch.bool),
        "token_resolved_mask": torch.tensor([True] * 8 + [False], dtype=torch.bool),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 4
            + [PocketFeature.UNSELECTED, PocketFeature.POCKET, PocketFeature.UNSELECTED, PocketFeature.POCKET]
            + [PocketFeature.UNSPECIFIED],
            dtype=torch.long,
        ),
        "frame_resolved_mask": torch.tensor([True] * 8 + [False]),
        "noised_mask": torch.tensor([True, False] * 4 + [False]),  # alternating to cover all inpaint options
        "frames_idx": torch.cat(
            [torch.arange(8 * 3, dtype=torch.long).reshape(8, 3), torch.zeros(1, 3, dtype=torch.long)]
        ),
        "inpaint_feature": torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 2
            + [InpaintFeature.TARGET_INPAINT] * 2
            + [0],
            dtype=torch.long,
        ),
        "binder_mask": torch.tensor([True] * 4 + [False] * 4 + [False], dtype=torch.bool),
        "target_mask": torch.tensor([False] * 4 + [True] * 4 + [False], dtype=torch.bool),
        # atom to token
        "atom_to_token": atom_to_token,
    }
    check_feature_dict_lengths(feature_dict)
    return feature_dict


## Fixtures for featurization utils tests


@pytest.fixture
def features_A() -> dict[str, torch.Tensor]:
    atom_to_token = torch.cat(
        [torch.cat([torch.eye(7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long)], dim=0), torch.zeros(8, 1)],
        dim=1,
    ).repeat_interleave(3, dim=0)
    return {
        "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "auth_asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "entity_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "sym_id": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 0], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        "auth_residue_index": torch.tensor([5, 6, 7, 8, 2, 3, 4, 0], dtype=torch.long),
        # atom features
        "coords": torch.cat([torch.zeros(4 * 3, 3), torch.ones(3 * 3, 3), torch.zeros(1 * 3, 3)]).unsqueeze(
            0
        ),  # 0's for asym_id 0, 1's for asym_id 1
        "atom_resolved_mask": torch.tensor([True] * 5 * 3 + [False] * 2 * 3 + [False] * 1 * 3),
        "atom_pad_mask": torch.tensor([True] * 7 * 3 + [False] * 1 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 7 + [False]),
        "token_resolved_mask": torch.tensor([True] * 5 + [False] * 2 + [False]),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 4 + [PocketFeature.POCKET] * 3 + [PocketFeature.UNSPECIFIED]
        ),
        "frame_resolved_mask": torch.tensor([True] * 5 + [False] * 2 + [False]),
        "noised_mask": torch.tensor([False] * 5 + [True] * 2 + [False]),
        "frames_idx": torch.cat([torch.arange(7 * 3).reshape(7, 3), torch.zeros(1, 3)]).to(torch.long),
        "inpaint_feature": torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 4 + [InpaintFeature.TARGET_CONTEXT] * 3 + [0]
        ),
        "binder_mask": torch.tensor([True] * 4 + [False] * 3 + [False]),
        "target_mask": torch.tensor([False] * 4 + [True] * 3 + [False]),
        # atom to token
        "atom_to_token": atom_to_token,
        "res_type": torch.tensor([1, 22, 3, 5, 6, 7, 8, 0]),
    }


@pytest.fixture
def features_B() -> dict[str, torch.Tensor]:
    return {
        "asym_id": torch.tensor([1, 1, 1]),
        "auth_asym_id": torch.tensor([21, 21, 21]),
        "entity_id": torch.tensor([1, 1, 1]),
        "sym_id": torch.tensor([0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "auth_residue_index": torch.tensor([13, 14, 15], dtype=torch.long),
        # atom features
        "coords": torch.ones(3 * 3, 3).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 3 * 3),
        "atom_pad_mask": torch.tensor([True] * 3 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 3),
        "token_resolved_mask": torch.tensor([True] * 3),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 3),
        "frame_resolved_mask": torch.tensor([True] * 3),
        "noised_mask": torch.tensor([False] * 3),
        "frames_idx": torch.arange(3 * 3).reshape(3, 3),
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT] * 3),
        "binder_mask": torch.tensor([False] * 3),
        "target_mask": torch.tensor([True] * 3),
        # atom to token
        "atom_to_token": torch.eye(3, dtype=torch.long).repeat_interleave(3, dim=0),
        "res_type": torch.tensor([6, 2, 8]),
    }


@pytest.fixture
def features_C() -> dict[str, torch.Tensor]:
    return {
        "asym_id": torch.tensor([1, 1, 1, 1]),
        "auth_asym_id": torch.tensor([21, 21, 21, 21]),
        "entity_id": torch.tensor([0, 0, 0, 0]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "auth_residue_index": torch.tensor([13, 14, 15, 16], dtype=torch.long),
        # atom features
        "coords": torch.ones(4 * 3, 3).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 4 * 3),
        "atom_pad_mask": torch.tensor([True] * 4 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 4),
        "token_resolved_mask": torch.tensor([True] * 4),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 4),
        "frame_resolved_mask": torch.tensor([True] * 4),
        "noised_mask": torch.tensor([False] * 4),
        "frames_idx": torch.arange(4 * 3).reshape(4, 3),
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT] * 4),
        "binder_mask": torch.tensor([False] * 4),
        "target_mask": torch.tensor([True] * 4),
        # atom to token
        "atom_to_token": torch.eye(4, dtype=torch.long).repeat_interleave(3, dim=0),
    }
