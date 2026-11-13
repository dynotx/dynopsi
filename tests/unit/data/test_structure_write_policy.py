"""PDB write-time refuse: unique [A-Z] chain IDs, and unique auth seqids when saving auth_res_idx."""

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from dynopsi.data.auth_chain_identity import OPAQUE_AUTH_ASYM_ID_START
from dynopsi.data.constants import TOKEN_IDS
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Structure,
)
from dynopsi.data.structure_io import parse_mmcif, parse_pdb, save_structures_to_file
from dynopsi.data.structure_write_policy import enforce_pdb_write_policy, persisted_auth_chain_ids
from dynopsi.data.utils.sequence_utils import SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP

INPAINT_REPEAT_AUTH_RES_IDX = (1, 2, 2, 2, 2, 2, 5, 6)
GAPPED_UNIQUE_AUTH_RES_IDX = (1, 2, 5, 6)


def _one_chain_structure(
    *,
    auth_chain_id: str,
    auth_asym_id: int,
    label_asym_id: int = 0,
    label_chain_id: str = "A",
    entity_id: int = 0,
    n_res: int = 2,
    auth_res_idx: Sequence[int] | None = None,
    residue_is_present: Sequence[bool] | None = None,
) -> Structure:
    if auth_res_idx is not None:
        n_res = len(auth_res_idx)
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
    if auth_res_idx is None:
        residues["auth_res_idx"] = np.arange(1, n_res + 1)
    else:
        residues["auth_res_idx"] = np.asarray(auth_res_idx, dtype=np.int32)
    residues["atom_start_idx"] = np.arange(n_res) * atoms_per_res
    residues["num_atoms"] = atoms_per_res
    residues["atom_center"] = residues["atom_start_idx"] + 1
    residues["is_standard"] = True
    if residue_is_present is None:
        residues["is_present"] = True
    else:
        residues["is_present"] = np.asarray(residue_is_present, dtype=bool)
    residues["ca_present"] = True
    residues["backbone_present"] = True

    chains = np.empty(1, dtype=CHAINS_DTYPE)
    chains["chain_id"] = label_chain_id
    chains["auth_chain_id"] = auth_chain_id
    chains["asym_id"] = label_asym_id
    chains["auth_asym_id"] = auth_asym_id
    chains["entity_id"] = entity_id
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


def _a7_structure() -> Structure:
    return _one_chain_structure(auth_chain_id="A-7", auth_asym_id=OPAQUE_AUTH_ASYM_ID_START)


def _letter_structure(
    auth_chain_id: str,
    *,
    label_asym_id: int = 0,
    entity_id: int = 0,
    auth_res_idx: Sequence[int] | None = None,
    residue_is_present: Sequence[bool] | None = None,
) -> Structure:
    return _one_chain_structure(
        auth_chain_id=auth_chain_id,
        auth_asym_id=SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[auth_chain_id],
        label_asym_id=label_asym_id,
        label_chain_id=auth_chain_id,
        entity_id=entity_id,
        auth_res_idx=auth_res_idx,
        residue_is_present=residue_is_present,
    )


def _inpaint_repeat_structure() -> Structure:
    return _letter_structure("A", auth_res_idx=INPAINT_REPEAT_AUTH_RES_IDX)


class TestEnforcePdbWritePolicy:
    def test_a7_raises_and_names_the_id(self) -> None:
        structure = _a7_structure()
        assert persisted_auth_chain_ids([structure]) == ("A-7",)
        with pytest.raises(ValueError, match=r"A-7") as exc_info:
            enforce_pdb_write_policy([structure])
        assert "never aliases" in str(exc_info.value)

    def test_duplicate_letter_strings_raise(self) -> None:
        duplicated = _letter_structure("V").merge(
            _letter_structure("V", label_asym_id=1, entity_id=1),
            mode="concatenate",
            reset_residue_index=False,
        )
        with pytest.raises(ValueError, match=r"repeated \['V'\]"):
            enforce_pdb_write_policy([duplicated])

    def test_same_letter_on_separate_structures_is_allowed(self) -> None:
        enforce_pdb_write_policy([_letter_structure("V"), _letter_structure("V")])

    def test_unique_letters_pass(self) -> None:
        enforce_pdb_write_policy([_letter_structure("V"), _letter_structure("W", label_asym_id=1, entity_id=1)])
        enforce_pdb_write_policy(
            [
                _letter_structure("V").merge(
                    _letter_structure("W", label_asym_id=1, entity_id=1),
                    mode="concatenate",
                    reset_residue_index=False,
                )
            ]
        )

    def test_mixed_a7_and_binder_letter_raises(self) -> None:
        mixed = _a7_structure().merge(
            _letter_structure("A", label_asym_id=1, entity_id=1),
            mode="concatenate",
            reset_residue_index=False,
        )
        with pytest.raises(ValueError, match=r"A-7"):
            enforce_pdb_write_policy([mixed])

    def test_auth_seqid_collisions_raise_only_for_auth_key(self) -> None:
        structure = _letter_structure("A", auth_res_idx=INPAINT_REPEAT_AUTH_RES_IDX)
        enforce_pdb_write_policy([structure])
        with pytest.raises(ValueError, match=r"auth_res_idx"):
            enforce_pdb_write_policy([structure], primary_residue_index_key="auth_res_idx")

    def test_same_auth_seqid_on_different_chains_is_allowed(self) -> None:
        structure = _letter_structure("A", auth_res_idx=(1, 2)).merge(
            _letter_structure("B", label_asym_id=1, entity_id=1, auth_res_idx=(1, 2)),
            mode="concatenate",
            reset_residue_index=False,
        )
        enforce_pdb_write_policy([structure], primary_residue_index_key="auth_res_idx")

    def test_absent_residue_does_not_count_as_written_seqid(self) -> None:
        structure = _letter_structure(
            "A",
            auth_res_idx=(1, 2, 2),
            residue_is_present=(True, True, False),
        )
        enforce_pdb_write_policy([structure], primary_residue_index_key="auth_res_idx")


class TestSaveStructuresPdbRefuse:
    def test_a7_pdb_raises_and_writes_no_file(self, tmp_path: Path) -> None:
        path = tmp_path / "designed.pdb"
        with pytest.raises(ValueError, match=r"A-7"):
            save_structures_to_file([_a7_structure()], path, primary_residue_index_key="res_idx")
        assert not path.exists()

    def test_duplicate_letter_pdb_raises_and_writes_no_file(self, tmp_path: Path) -> None:
        path = tmp_path / "dup_v.pdb"
        structure = _letter_structure("V").merge(
            _letter_structure("V", label_asym_id=1, entity_id=1),
            mode="concatenate",
            reset_residue_index=False,
        )
        with pytest.raises(ValueError, match=r"repeated \['V'\]"):
            save_structures_to_file([structure], path, primary_residue_index_key="res_idx")
        assert not path.exists()

    def test_same_letter_on_separate_structures_pdb_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "two_models.pdb"
        save_structures_to_file(
            [_letter_structure("V"), _letter_structure("V")],
            path,
            primary_residue_index_key="res_idx",
        )
        assert path.exists()

    def test_unique_letter_pdb_writes_without_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "designed.pdb"
        structure = _letter_structure("V").merge(
            _letter_structure("W", label_asym_id=1, entity_id=1),
            mode="concatenate",
            reset_residue_index=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_structures_to_file([structure], path, primary_residue_index_key="res_idx")
        dynopsi_warnings = [w for w in caught if "dynopsi" in Path(str(w.filename)).as_posix()]
        assert dynopsi_warnings == []
        assert path.exists()
        parsed = parse_pdb(path, compute_residue_interfaces=False)
        assert set(parsed.chains["auth_chain_id"]) == {"V", "W"}

    def test_ab_pdb_writes_without_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "ab.pdb"
        structure = _letter_structure("A").merge(
            _letter_structure("B", label_asym_id=1, entity_id=1),
            mode="concatenate",
            reset_residue_index=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            save_structures_to_file([structure], path, primary_residue_index_key="res_idx")
        dynopsi_warnings = [w for w in caught if "dynopsi" in Path(str(w.filename)).as_posix()]
        assert dynopsi_warnings == []
        parsed = parse_pdb(path, compute_residue_interfaces=False)
        assert set(parsed.chains["auth_chain_id"]) == {"A", "B"}

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif"])
    def test_a7_cif_write_succeeds(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"designed{suffix}"
        save_structures_to_file([_a7_structure()], path, primary_residue_index_key="res_idx")
        assert path.exists()
        parsed = parse_mmcif(path, compute_residue_interfaces=False)
        assert list(parsed.chains["auth_chain_id"]) == ["A-7"]


class TestSaveStructuresAuthSeqidUniqueness:
    def test_inpaint_repeat_pdb_auth_raises_and_writes_no_file(self, tmp_path: Path) -> None:
        path = tmp_path / "inpaint.pdb"
        with pytest.raises(ValueError, match=r"auth_res_idx"):
            save_structures_to_file(
                [_inpaint_repeat_structure()],
                path,
                primary_residue_index_key="auth_res_idx",
            )
        assert not path.exists()

    @pytest.mark.parametrize("suffix", [".cif", ".mmcif"])
    def test_inpaint_repeat_cif_auth_succeeds(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"inpaint{suffix}"
        save_structures_to_file(
            [_inpaint_repeat_structure()],
            path,
            primary_residue_index_key="auth_res_idx",
        )
        assert path.exists()

    def test_inpaint_repeat_pdb_res_idx_succeeds(self, tmp_path: Path) -> None:
        path = tmp_path / "inpaint.pdb"
        save_structures_to_file(
            [_inpaint_repeat_structure()],
            path,
            primary_residue_index_key="res_idx",
        )
        assert path.exists()
        parsed = parse_pdb(path, compute_residue_interfaces=False)
        assert list(parsed.chains["auth_chain_id"]) == ["A"]

    def test_unique_gapped_auth_seqids_pdb_writes(self, tmp_path: Path) -> None:
        path = tmp_path / "gapped.pdb"
        structure = _letter_structure("A", auth_res_idx=GAPPED_UNIQUE_AUTH_RES_IDX)
        save_structures_to_file([structure], path, primary_residue_index_key="auth_res_idx")
        assert path.exists()
        parsed = parse_pdb(path, compute_residue_interfaces=False)
        assert list(parsed.chains["auth_chain_id"]) == ["A"]
        assert list(parsed.residues["auth_res_idx"]) == list(GAPPED_UNIQUE_AUTH_RES_IDX)
