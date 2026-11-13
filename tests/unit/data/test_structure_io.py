import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest

import dynopsi.data.structure_io as structure_io
from dynopsi.data import constants
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
)
from dynopsi.data.structure_io import (
    parse_chain_to_residues,
    parse_mmcif,
    parse_pdb,
    parse_residue_to_atoms,
)

from .conftest import make_two_chain_structure


def _polymer_span_from_residues(residues: list[gemmi.Residue]) -> gemmi.ResidueSpan:
    """Build a gemmi polymer span from test residues."""
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")
    for residue in residues:
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    structure.assign_label_seq_id()
    return chain.get_polymer()


def _capture_parse_structure_from_gemmi_kwargs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, bool]]:
    captured_kwargs: list[dict[str, bool]] = []

    def fake_parse_structure_from_gemmi(
        structure: gemmi.Structure,
        allow_nonstandard_residues: bool = False,
        allow_mismatches: bool = False,
        use_assembly: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captured_kwargs.append(
            {
                "allow_nonstandard_residues": allow_nonstandard_residues,
                "allow_mismatches": allow_mismatches,
                "use_assembly": use_assembly,
            }
        )
        return (
            np.empty(0, dtype=CHAINS_DTYPE),
            np.empty(0, dtype=RESIDUES_DTYPE),
            np.empty(0, dtype=ATOMS_DTYPE),
        )

    monkeypatch.setattr(structure_io, "parse_structure_from_gemmi", fake_parse_structure_from_gemmi)
    return captured_kwargs


class TestParseFiles:
    """
    Tests for the parse_mmcif function.

    Use reference to define fields: https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/
    """

    expected_residue_attributes: dict[str, dict[str, list[Any]]] = {
        "asym_id": {"A": [0, 0, 0, 0], "D": [3, 3, 3, 3], "B": [1, 1, 1]},
        "is_present": {"A": [True, True, True, True], "D": [False, True, True, False], "B": [True, True, True]},
        "backbone_present": {"A": [True, True, True, True], "D": [False, False, False, False], "B": [True, True, True]},
        "ca_present": {"A": [True, True, True, True], "D": [False, True, False, False], "B": [True, True, True]},
        "res_name": {"A": ["ASN", "ASP", "PRO", "ALA"], "D": ["ASN", "ASP", "PRO", "ALA"], "B": ["PHE", "TRP", "GLY"]},
        "res_idx": {"A": [0, 1, 2, 3], "D": [0, 1, 2, 3], "B": [0, 1, 2]},
        "auth_res_idx": {"A": [18, 19, 20, 21], "D": [1, 1213, 1214, 1215], "B": [1079, 1080, 1081]},
        "is_standard": {"A": [True, True, True, True], "D": [True, True, True, True], "B": [True, True, True]},
        "atom_center_idx": {"A": [1, 1, 1, 1], "D": [1, 1, 1, 1], "B": [1, 1, 1]},
        "disto_center_idx": {"A": [4, 4, 4, 4], "D": [4, 4, 4, 4], "B": [4, 4, 1]},
    }

    expected_residue_attributes_pdb_no_seq_provided: dict[str, dict[str, list[Any]]] = {
        "asym_id": {"A": [0, 0, 0, 0], "D": [3, 3], "B": [1, 1, 1]},
        "is_present": {"A": [True, True, True, True], "D": [True, True], "B": [True, True, True]},
        "backbone_present": {"A": [True, True, True, True], "D": [False, False], "B": [True, True, True]},
        "ca_present": {"A": [True, True, True, True], "D": [True, False], "B": [True, True, True]},
        "res_name": {"A": ["ASN", "ASP", "PRO", "ALA"], "D": ["ASP", "PRO"], "B": ["PHE", "TRP", "GLY"]},
        "res_idx": {"A": [0, 1, 2, 3], "D": [0, 1], "B": [0, 1, 2]},
        "auth_res_idx": {"A": [18, 19, 20, 21], "D": [1213, 1214], "B": [1079, 1080, 1081]},
        "is_standard": {"A": [True, True, True, True], "D": [True, True], "B": [True, True, True]},
        "atom_center_idx": {"A": [1, 1, 1, 1], "D": [1, 1], "B": [1, 1, 1]},
        "disto_center_idx": {"A": [4, 4, 4, 4], "D": [4, 4], "B": [4, 4, 1]},
    }

    @pytest.mark.parametrize("file_type", [".cif", ".pdb"])
    def test_parse_structure_file_normal(
        self,
        file_type: str,
        cif_data: str,
        pdb_data: str,
        expected_atom_names: dict[str, list[str]],
    ) -> None:
        if file_type == ".cif":
            data_str = cif_data
            kwargs: dict[str, Any] = {
                "atomic_interface_threshold": 41.3
            }  # set arbitraty threshold so that only some residues are interfaces
        elif file_type == ".pdb":
            data_str = pdb_data
            kwargs = {
                "chain_id_to_sequence": {"A": "NDPA", "B": "FWG", "D": "NDPA"},
                "atomic_interface_threshold": 41.3,
            }
        else:
            raise ValueError(f"Invalid file type: {file_type}")

        with tempfile.NamedTemporaryFile(suffix=file_type, mode="w") as temp_file:
            temp_file.write(data_str)
            temp_file.flush()
            if file_type == ".cif":
                parsed_structure = parse_mmcif(temp_file.name, **kwargs)
            elif file_type == ".pdb":
                parsed_structure = parse_pdb(temp_file.name, **kwargs)
            else:
                raise ValueError(f"Invalid file type: {file_type}")

            chain_id_to_sequence = {
                "A": ["ASN", "ASP", "PRO", "ALA"],
                "B": ["PHE", "TRP", "GLY"],
                "D": ["ASN", "ASP", "PRO", "ALA"],
            }
            chain_id_to_auth_chain_id = {
                ".pdb": {
                    "A": "A",
                    "B": "B",
                    "D": "D",
                },
                ".cif": {
                    "A": "A",
                    "B": "B",
                    "D": "C",
                },
            }
            chain_id_to_asym_id = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3,
            }
            num_atoms_per_chain = {
                chain_id: sum(len(expected_atom_names[res_name]) for res_name in chain_id_to_sequence[chain_id])
                for chain_id in chain_id_to_sequence
            }
            expected_chains = np.array(
                [
                    (
                        "A",
                        chain_id_to_auth_chain_id[file_type]["A"],
                        0,
                        chain_id_to_asym_id[chain_id_to_auth_chain_id[file_type]["A"]],
                        0,
                        0,
                        0,
                        num_atoms_per_chain["A"],
                        0,
                        len(chain_id_to_sequence["A"]),
                    ),
                    (
                        "B",
                        chain_id_to_auth_chain_id[file_type]["B"],
                        1,
                        chain_id_to_asym_id[chain_id_to_auth_chain_id[file_type]["B"]],
                        1,
                        0,
                        num_atoms_per_chain["A"],
                        num_atoms_per_chain["B"],
                        len(chain_id_to_sequence["A"]),
                        len(chain_id_to_sequence["B"]),
                    ),
                    (
                        "D",
                        chain_id_to_auth_chain_id[file_type]["D"],
                        3,
                        chain_id_to_asym_id[chain_id_to_auth_chain_id[file_type]["D"]],
                        0,
                        1,
                        num_atoms_per_chain["A"] + num_atoms_per_chain["B"],
                        num_atoms_per_chain["D"],
                        len(chain_id_to_sequence["A"]) + len(chain_id_to_sequence["B"]),
                        len(chain_id_to_sequence["D"]),
                    ),
                ],
                dtype=CHAINS_DTYPE,
            )
            print("parsed structure chains")
            print(parsed_structure.chains)
            print("expected chains")
            print(expected_chains)
            assert np.all(parsed_structure.chains == expected_chains), (
                f"Expected chains {expected_chains}, got {parsed_structure.chains}"
            )

            # easier to build expected residues w/ a dict
            expected_residues = {
                key: np.concatenate([self.expected_residue_attributes[key][chain_id] for chain_id in ["A", "B", "D"]])
                for key in [
                    "res_name",
                    "asym_id",
                    "res_idx",
                    "auth_res_idx",
                    "is_standard",
                    "is_present",
                    "backbone_present",
                    "ca_present",
                ]
            }
            expected_residues["res_type"] = np.array(
                [constants.TOKEN_IDS[res_name] for res_name in expected_residues["res_name"]]
            )
            expected_residues["num_atoms"] = np.array(
                [len(expected_atom_names[res_name]) for res_name in expected_residues["res_name"]]
            )
            expected_residues["atom_start_idx"] = np.cumsum(
                np.concatenate([np.zeros(1), expected_residues["num_atoms"][:-1]])
            )
            expected_residues["atom_center"] = expected_residues["atom_start_idx"] + 1

            for attr in RESIDUES_DTYPE.names:
                actual_values = parsed_structure.residues[attr]
                expected_values = expected_residues[attr]
                assert np.all(actual_values == expected_values), (
                    f"Expected {attr} {expected_values}, got {actual_values}"
                )

        expected_interfaces = np.array(
            [
                (0, 3),
            ],
            dtype=CHAIN_INTERFACE_DTYPE,
        )

        assert np.all(parsed_structure.interfaces == expected_interfaces), (
            f"Expected interfaces {expected_interfaces}, got {parsed_structure.interfaces}"
        )

        expected_residue_interfaces = np.array(
            [
                (0, 3, 0, 1),
                (0, 3, 0, 2),
                (0, 3, 1, 1),
                (0, 3, 1, 2),
                (0, 3, 2, 1),
                (0, 3, 2, 2),
                (0, 3, 3, 2),
            ],
            dtype=RESIDUE_INTERFACE_DTYPE,
        )

        assert np.all(parsed_structure.residue_interfaces == expected_residue_interfaces), (
            f"Expected residue interfaces {expected_residue_interfaces}, got {parsed_structure.residue_interfaces}"
        )

    @pytest.mark.parametrize(("file_type", "fixture_name"), [(".cif", "cif_data"), (".pdb", "pdb_data")])
    def test_parse_rejects_insertion_codes(
        self,
        file_type: str,
        fixture_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        structure_data = request.getfixturevalue(fixture_name)
        if file_type == ".cif":
            structure_data = structure_data.replace("ASN A 1 1   ?", "ASN A 1 1   A")
            parse = parse_mmcif
        else:
            structure_data = structure_data.replace("ASN A  18 ", "ASN A  18A")
            parse = parse_pdb

        with tempfile.NamedTemporaryFile(suffix=file_type, mode="w") as temp_file:
            temp_file.write(structure_data)
            temp_file.flush()
            with pytest.raises(
                ValueError,
                match=r"Insertion-coded residue A18A is not supported",
            ):
                parse(temp_file.name, compute_residue_interfaces=False)

    def test_parse_mmcif_sequence_gap(self, malformed_cif_data: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(malformed_cif_data)
            temp_file.flush()
            with pytest.raises(ValueError):
                parse_mmcif(temp_file.name)

    def test_parse_mmcif_allows_mismatches(self, cif_data: str) -> None:
        mismatched_cif_data = cif_data.replace("1 NDPA 'polypeptide(L)' A,D", "1 DDPA 'polypeptide(L)' A,D")
        mismatched_cif_data = mismatched_cif_data.replace("1 ASN 1", "1 ASP 1", 1)

        with tempfile.NamedTemporaryFile(suffix=".cif", mode="w") as temp_file:
            temp_file.write(mismatched_cif_data)
            temp_file.flush()

            with pytest.raises(ValueError, match="Mismatch between expected and observed sequence"):
                parse_mmcif(temp_file.name)

            parsed_structure = parse_mmcif(temp_file.name, allow_mismatches=True)

        assert parsed_structure.residues["res_name"][0] == "ASN"

    @pytest.mark.parametrize(
        ("function_name", "file_type", "fixture_name"),
        [
            ("parse_mmcif", ".cif", "cif_data"),
            ("load_structure_from_cif", ".cif", "cif_data"),
            ("parse_pdb", ".pdb", "pdb_data"),
            ("load_structure_from_pdb", ".pdb", "pdb_data"),
        ],
    )
    def test_file_parse_wrappers_forward_gemmi_parse_flags(
        self,
        function_name: str,
        file_type: str,
        fixture_name: str,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_kwargs = _capture_parse_structure_from_gemmi_kwargs(monkeypatch)
        data = request.getfixturevalue(fixture_name)

        with tempfile.NamedTemporaryFile(suffix=file_type, mode="w") as temp_file:
            temp_file.write(data)
            temp_file.flush()
            getattr(structure_io, function_name)(
                temp_file.name,
                allow_nonstandard_residues=True,
                allow_mismatches=True,
                use_assembly=True,
            )

        assert captured_kwargs == [
            {
                "allow_nonstandard_residues": True,
                "allow_mismatches": True,
                "use_assembly": True,
            }
        ]

    def test_load_structure_from_rcsb_forwards_gemmi_parse_flags(
        self, cif_data: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs = _capture_parse_structure_from_gemmi_kwargs(monkeypatch)
        cif_path = tmp_path / "1abc.cif"
        cif_path.write_text(cif_data)

        def fake_download_mmcif_from_rcsb(record_id: str, target_dir: str | Path) -> str:
            assert record_id == "1ABC"
            return str(cif_path)

        monkeypatch.setattr(structure_io, "download_mmcif_from_rcsb", fake_download_mmcif_from_rcsb)

        structure_io.load_structure_from_rcsb(
            "1ABC",
            target_dir=tmp_path,
            allow_nonstandard_residues=True,
            allow_mismatches=True,
            use_assembly=True,
        )

        assert captured_kwargs == [
            {
                "allow_nonstandard_residues": True,
                "allow_mismatches": True,
                "use_assembly": True,
            }
        ]

    def test_load_structure_from_afdb_forwards_gemmi_parse_flags(
        self, cif_data: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs = _capture_parse_structure_from_gemmi_kwargs(monkeypatch)
        cif_path = tmp_path / "AF-Q9TEST-model_v6.cif"
        cif_path.write_text(cif_data)

        def fake_download_mmcif_from_afdb(uniprot_id: str, target_dir: str | Path) -> str:
            assert uniprot_id == "Q9TEST"
            return str(cif_path)

        monkeypatch.setattr(structure_io, "download_mmcif_from_afdb", fake_download_mmcif_from_afdb)

        structure_io.load_structure_from_afdb(
            "Q9TEST",
            target_dir=tmp_path,
            allow_nonstandard_residues=True,
            allow_mismatches=True,
        )

        assert captured_kwargs == [
            {
                "allow_nonstandard_residues": True,
                "allow_mismatches": True,
                "use_assembly": False,
            }
        ]

    def test_parse_pdb_allows_mismatches_with_chain_id_to_sequence(self, pdb_data: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as temp_file:
            temp_file.write(pdb_data)
            temp_file.flush()

            parsed_structure = parse_pdb(
                temp_file.name,
                chain_id_to_sequence={"A": "DDPA", "B": "FWG", "D": "DDPA"},
                allow_mismatches=True,
            )

        assert parsed_structure.residues["res_name"][0] == "ASN"

    def test_parse_pdb_allow_mismatches_keeps_a_chain_absent_from_the_map(self, pdb_data: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as temp_file:
            temp_file.write(pdb_data)
            temp_file.flush()

            parsed_structure = parse_pdb(
                temp_file.name,
                chain_id_to_sequence={"A": "DDPA"},
                allow_mismatches=True,
            )

        assert set(parsed_structure.chains["chain_id"]) == {"A", "B", "D"}
        assert parsed_structure.residues["res_name"][0] == "ASN"

    def test_parse_pdb_rejects_different_sequences_for_shared_entity(self, tmp_path: Path) -> None:
        path = tmp_path / "shared_entity.pdb"
        path.write_text(
            "ATOM      1  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2       5.000   0.000   0.000  1.00  0.00           C\n"
            "TER\n"
            "ATOM      3  CA  ALA B   1       1.000   5.000   0.000  1.00  0.00           C\n"
            "ATOM      4  CA  ALA B   2       5.000   5.000   0.000  1.00  0.00           C\n"
            "TER\n"
            "END\n"
        )

        with pytest.raises(ValueError, match="resolve to different sequences"):
            parse_pdb(
                path,
                chain_id_to_sequence={"A": "AA", "B": "AAG"},
                allow_mismatches=True,
                compute_residue_interfaces=False,
            )


class TestParseResidueToAtoms:
    expected_atom_coords_by_name: dict[str, tuple[float, float, float]] = defaultdict(lambda: (0.0, 0.0, 0.0))
    expected_atom_coords_by_name.update(
        {
            "N": (1.0, 2.0, 3.0),
            "CA": (4.0, 5.0, 6.0),
            "C": (7.0, 8.0, 9.0),
            "O": (10.0, 11.0, 12.0),
            "CB": (13.0, 14.0, 15.0),
        }
    )

    expected_presence_by_name = {
        "ALA": [True, True, True, True, True],
        "ASP": [True, True, True, True, False, False, False, False],  # no sidechains
        "ASN": [False, True, False, False, True, True, True, True],  # no backbone except CA
        "VAL": [True, False, True, True, True, True, True],
        "GLY": [True, True, True, True],
        "MET": [True, True, True, True, True, True, True, True],  # MSE is parsed as MET
        "PHE": [False] * 11,
    }

    @pytest.mark.parametrize("residue_idx", [0, 1, 2, 3, 4, 5, 6])
    def test_parse_residue_to_atoms(
        self,
        gemmi_residues: list[gemmi.Residue],
        expected_atom_names: dict[str, list[str]],
        residue_idx: int,
    ) -> None:
        if gemmi_residues[residue_idx] is None:
            residue_name = "PHE"
        else:
            residue_name = gemmi_residues[residue_idx].name
            if residue_name == "MSE":
                residue_name = "MET"
        residue_atoms = parse_residue_to_atoms(
            observed_residue=gemmi_residues[residue_idx],
            residue_name=residue_name,
            asym_id=0,
            res_idx=residue_idx,
        )

        expect_names = np.array(expected_atom_names[residue_name])
        assert len(residue_atoms) == len(expect_names), f"Expected {len(expect_names)} atoms, got {len(residue_atoms)}"
        actual_names = residue_atoms["atom_name"]
        assert np.all(actual_names == expect_names), f"Expected atom names {expect_names}, got {actual_names}"

        expected_presence = np.array(self.expected_presence_by_name[residue_name])
        actual_presence = residue_atoms["is_present"]
        assert np.all(actual_presence == expected_presence), (
            f"Expected atom is_present {expected_presence}, got {actual_presence}"
        )

        expected_coords = np.array(
            [
                self.expected_atom_coords_by_name[atom_name] if is_present else (0.0, 0.0, 0.0)
                for atom_name, is_present in zip(residue_atoms["atom_name"], expected_presence)
            ]
        )
        actual_coords = residue_atoms["coords"]
        assert np.all(actual_coords == expected_coords), f"Expected coords {expected_coords}, got {actual_coords}"

        expected_bfactor_lst = [50.0 if "O" in x else 100.0 for x in expect_names]
        # filter for presence
        expected_bfactor = np.array(
            [bfactor if is_present else 0.0 for bfactor, is_present in zip(expected_bfactor_lst, expected_presence)]
        )
        actual_bfactor = residue_atoms["bfactor"]
        assert np.all(actual_bfactor == expected_bfactor), (
            f"Expected atom bfactor {expected_bfactor}, got {actual_bfactor}"
        )

    def test_parse_residue_to_atoms_mse_without_se(self, gemmi_residues: list[gemmi.Residue]) -> None:
        mse_residue = gemmi_residues[5]
        mse_residue.remove_atom("SE", "\0", gemmi.Element("Se"))

        residue_atoms = parse_residue_to_atoms(
            observed_residue=mse_residue,
            residue_name="MET",
            asym_id=0,
            res_idx=0,
        )

        sd_atom = residue_atoms[residue_atoms["atom_name"] == "SD"][0]
        assert not sd_atom["is_present"]


class TestParseChainToResidues:
    def test_parse_chain_to_residues(
        self, gemmi_residues: list[gemmi.Residue], expected_atom_names: dict[str, list[str]]
    ) -> None:
        st = gemmi.Structure()
        model = gemmi.Model("1")

        chain = gemmi.Chain("A")
        ungapped_sequence = []
        for residue in gemmi_residues:
            if residue is not None:
                chain.add_residue(residue)
                ungapped_sequence.append(residue.name)
            else:
                # by design the last residue is None and also arbitrarily PHE
                ungapped_sequence.append("PHE")
        model.add_chain(chain)
        st.add_model(model)
        st.setup_entities()
        st.assign_label_seq_id()

        parsed_residues, parsed_atoms = parse_chain_to_residues(
            chain=chain.get_polymer(),
            ungapped_sequence=ungapped_sequence,
            asym_id=1,
            chain_atom_start_idx=0,
        )

        assert len(parsed_residues) == len(ungapped_sequence), (
            f"Expected {len(ungapped_sequence)} residues, got {len(parsed_residues)}"
        )

        expected_res_names = ["ALA", "ASP", "ASN", "VAL", "GLY", "MET", "PHE"]
        num_atoms = np.array([len(expected_atom_names[res_name]) for res_name in expected_res_names])
        atom_start_idx = np.concatenate([np.zeros(1), np.cumsum(num_atoms)[:-1]])
        expected_residue_attributes = {
            "res_name": np.array(expected_res_names),
            "res_type": np.array([constants.TOKEN_IDS[res_name] for res_name in expected_res_names]),
            "asym_id": np.ones(len(expected_res_names)),
            "res_idx": np.arange(len(expected_res_names)),
            "auth_res_idx": np.array([1, 2, 3, 4, 7, 8, 9]),
            "is_standard": np.array([True, True, True, True, True, False, True]),
            "num_atoms": num_atoms,
            "atom_start_idx": atom_start_idx,
            "atom_center": atom_start_idx + np.ones(len(expected_res_names)),
            "backbone_present": np.array([True, True, False, False, True, True, False]),
            "ca_present": np.array([True, True, True, False, True, True, False]),
        }

        for attribute, expected_values in expected_residue_attributes.items():
            actual_values = parsed_residues[attribute]
            assert np.all(actual_values == expected_values), (
                f"Expected {attribute} {expected_values}, got {actual_values}"
            )

        expected_atoms = np.concatenate([expected_atom_names[res_name] for res_name in parsed_residues["res_name"]])
        actual_atoms = parsed_atoms["atom_name"]
        assert np.all(actual_atoms == expected_atoms), f"Expected atoms {expected_atoms}, got {actual_atoms}"

    def test_parse_chain_to_residues_rejects_nonstandard_by_default(self, gemmi_residues: list[gemmi.Residue]) -> None:
        nonstandard_residue = gemmi_residues[0]
        nonstandard_residue.name = "SEP"
        chain = _polymer_span_from_residues([nonstandard_residue])

        parsed_residues, parsed_atoms = parse_chain_to_residues(
            chain=chain,
            ungapped_sequence=["SEP"],
            asym_id=0,
            chain_atom_start_idx=0,
        )

        assert len(parsed_residues) == 0
        assert len(parsed_atoms) == 0

    def test_parse_chain_to_residues_allows_nonstandard_as_unk(self, gemmi_residues: list[gemmi.Residue]) -> None:
        nonstandard_residue = gemmi_residues[0]
        nonstandard_residue.name = "SEP"
        chain = _polymer_span_from_residues([nonstandard_residue])

        parsed_residues, parsed_atoms = parse_chain_to_residues(
            chain=chain,
            ungapped_sequence=["SEP"],
            asym_id=0,
            chain_atom_start_idx=0,
            allow_nonstandard_residues=True,
        )

        assert parsed_residues["res_name"].tolist() == ["UNK"]
        assert parsed_residues["res_type"].tolist() == [constants.TOKEN_IDS["UNK"]]
        assert parsed_residues["is_standard"].tolist() == [False]
        assert parsed_atoms["atom_name"].tolist() == constants.REF_ATOM_NAMES["UNK"]

    def test_parse_chain_to_residues_mismatch_policy(self, gemmi_residues: list[gemmi.Residue]) -> None:
        chain = _polymer_span_from_residues([gemmi_residues[0]])

        with pytest.raises(ValueError, match="Mismatch between expected and observed sequence"):
            parse_chain_to_residues(
                chain=chain,
                ungapped_sequence=["ASP"],
                asym_id=0,
                chain_atom_start_idx=0,
            )

        parsed_residues, _ = parse_chain_to_residues(
            chain=chain,
            ungapped_sequence=["ASP"],
            asym_id=0,
            chain_atom_start_idx=0,
            allow_mismatches=True,
        )

        assert parsed_residues["res_name"].tolist() == ["ALA"]


def test_distinct_entity_cif_round_trip_keeps_residues(tmp_path: Path) -> None:
    source = make_two_chain_structure()
    residue_names = ["GLY", "SER", "ALA", "VAL"]
    source.residues["res_name"] = residue_names
    source.residues["res_type"] = [constants.TOKEN_IDS[name] for name in residue_names]

    path = tmp_path / "distinct_entities.cif"
    structure_io.save_structures_to_file([source], path, primary_residue_index_key="res_idx")
    loaded = structure_io.load_structure_from_cif(path)

    assert len(loaded.residues) == len(source.residues)
    assert list(loaded.residues["res_name"]) == list(source.residues["res_name"])
