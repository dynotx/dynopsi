from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

import dynopsi.data.featurization.primitives.reference_structure as rs
from dynopsi.data.constants import AtomFilter, DesignDesignation
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.structure import (
    Structure,
)
from dynopsi.data.tokenizer import FrameStructureTokenFeatures
from tests.unit.data.conftest import (
    assert_feature_dict_equality,
    assert_metadata_equality,
    assert_structure_equality,
)


class TestReferenceStructureFeaturizer:
    """Test the ReferenceStructureFeaturizer."""

    @pytest.mark.parametrize(
        ("design_designation", "expect_value_error"),
        [
            pytest.param(DesignDesignation.FIXED, False, id="fixed"),
            pytest.param(DesignDesignation.DESIGN, False, id="design"),
            pytest.param({DesignDesignation.FIXED: ["A"], DesignDesignation.DESIGN: ["B"]}, False, id="dict_design"),
            pytest.param(
                {DesignDesignation.FIXED: ["A"], DesignDesignation.DESIGN: ["A", "B"]}, True, id="dict_design_invalid"
            ),
        ],
    )
    def test_design_designation_configuration(
        self, design_designation: DesignDesignation | dict[DesignDesignation, list[str]], expect_value_error: bool
    ) -> None:
        """Make sure that chain ids are not used in multiple design designations"""
        featurizer_kwargs = {
            "name": "test_reference_structure_featurizer",
            "design_designation": design_designation,
            "rcsb_record_id": "7f7p",
            "atoms_to_include": AtomFilter.BB,
        }
        if expect_value_error:
            with pytest.raises(ValueError):
                primitives.ReferenceStructureFeaturizer(**featurizer_kwargs)
            return
        else:
            featurizer = primitives.ReferenceStructureFeaturizer(**featurizer_kwargs)
        if isinstance(design_designation, dict):
            assert featurizer.design_designations == list(design_designation.keys())
        else:
            assert featurizer.design_designations == [design_designation]
        assert featurizer.rcsb_record_id == "7f7p"
        assert featurizer.afdb_uniprot_id is None
        assert featurizer.structure_path is None

    @pytest.mark.parametrize(
        ("rcsb_record_id", "afdb_uniprot_id", "structure_path"),
        [
            ("7f7p", "P00533", "test.pdb"),
            ("7f7p", None, "test.pdb"),
            (None, "P00533", "test.pdb"),
            ("7f7p", "P00533", None),
        ],
        ids=[
            "all_three",
            "rcsb_record_id_and_structure_path",
            "afdb_uniprot_id_and_structure_path",
            "afdb_uniprot_id_and_rcsb_record_id",
        ],
    )
    def test_input_structure_validation(
        self, rcsb_record_id: str | None, afdb_uniprot_id: str | None, structure_path: str | None
    ) -> None:
        """Make sure that there is exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path"""
        with pytest.raises(ValueError):
            primitives.ReferenceStructureFeaturizer(
                name="test_reference_structure_featurizer",
                design_designation=DesignDesignation.FIXED,
                rcsb_record_id=rcsb_record_id,
                afdb_uniprot_id=afdb_uniprot_id,
                structure_path=structure_path,
                atoms_to_include=AtomFilter.BB,
            )

    @pytest.mark.parametrize(
        ("structure_path", "loader_name"),
        [
            (Path("test.PDB"), "load_structure_from_pdb"),
            (Path("test.CIF"), "load_structure_from_cif"),
        ],
    )
    def test_structure_path_suffix_ignores_case(
        self,
        frame_structure_token_features_bb_standard: FrameStructureTokenFeatures,
        structure_path: Path,
        loader_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load_pdb = Mock(return_value=None)
        load_cif = Mock(return_value=None)
        monkeypatch.setattr(rs, "load_structure_from_pdb", load_pdb)
        monkeypatch.setattr(rs, "load_structure_from_cif", load_cif)
        monkeypatch.setattr(
            rs.DynoPsiTokenizer, "tokenize", Mock(return_value=[frame_structure_token_features_bb_standard])
        )
        featurizer = primitives.ReferenceStructureFeaturizer(
            name="test_reference_structure_featurizer",
            design_designation=DesignDesignation.FIXED,
            structure_path=structure_path,
            atoms_to_include=AtomFilter.BB,
        )
        featurizer._get_reference_structure_feature_dict(atoms_to_include=AtomFilter.BB)
        loader = load_pdb if loader_name == "load_structure_from_pdb" else load_cif
        other = load_cif if loader is load_pdb else load_pdb
        loader.assert_called_once_with(structure_path)
        other.assert_not_called()

    @pytest.mark.parametrize(
        ("design_designation", "input_type", "expected_error", "expected_target_mask", "expected_binder_mask"),
        [
            pytest.param(
                DesignDesignation.FIXED,
                "rcsb_record_id",
                False,
                torch.ones(6, dtype=torch.bool),
                torch.zeros(6, dtype=torch.bool),
                id="fixed",
            ),
            pytest.param(
                DesignDesignation.DESIGN,
                "afdb_uniprot_id",
                False,
                torch.zeros(6, dtype=torch.bool),
                torch.ones(6, dtype=torch.bool),
                id="design",
            ),
            pytest.param(
                {DesignDesignation.FIXED: ["C"], DesignDesignation.DESIGN: ["B"]},
                "structure_path_pdb",
                False,
                torch.tensor([False, False, True, True]),
                torch.tensor([True, True, False, False]),
                id="dict_design_valid_subset",
            ),
            pytest.param(
                {DesignDesignation.FIXED: ["C", "A"], DesignDesignation.DESIGN: ["B"]},
                "structure_path_cif",
                False,
                torch.tensor([True, True, False, False, True, True]),
                torch.tensor([False, False, True, True, False, False]),
                id="dict_design_valid_full",
            ),
            pytest.param(
                {DesignDesignation.FIXED: ["A"], DesignDesignation.DESIGN: ["D"]},
                "structure_path_pdb",
                True,
                None,
                None,
                id="dict_design_invalid_wrong_asym_id",
            ),
            pytest.param(
                {DesignDesignation.FIXED: ["A", "D"], DesignDesignation.DESIGN: ["B"]},
                "structure_path_pdb",
                True,
                None,
                None,
                id="dict_design_invalid_wrong_asym_id2",
            ),
        ],
    )
    def test_build_reference_structure_feature_dict(
        self,
        frame_structure_token_features_bb_standard: FrameStructureTokenFeatures,
        input_type: str,
        design_designation: DesignDesignation | dict[DesignDesignation, list[str]],
        expected_error: bool,
        expected_target_mask: torch.Tensor,
        expected_binder_mask: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Make sure that the reference structure feature dict is built correctly"""
        monkeypatch.setattr(rs, "load_structure_from_rcsb", Mock(return_value=None))
        monkeypatch.setattr(rs, "load_structure_from_afdb", Mock(return_value=None))
        monkeypatch.setattr(rs, "load_structure_from_pdb", Mock(return_value=None))
        monkeypatch.setattr(rs, "load_structure_from_cif", Mock(return_value=None))
        monkeypatch.setattr(
            rs.DynoPsiTokenizer, "tokenize", Mock(return_value=[frame_structure_token_features_bb_standard])
        )

        if input_type == "rcsb_record_id":
            structure_kwargs = {"rcsb_record_id": "7f7p"}
        elif input_type == "afdb_uniprot_id":
            structure_kwargs = {"afdb_uniprot_id": "P00533"}
        elif input_type == "structure_path_pdb":
            structure_kwargs = {"structure_path": "test.pdb"}
        elif input_type == "structure_path_cif":
            structure_kwargs = {"structure_path": "test.cif"}

        featurizer = primitives.ReferenceStructureFeaturizer(
            name="test_reference_structure_featurizer",
            design_designation=design_designation,
            atoms_to_include=AtomFilter.BB,
            **structure_kwargs,
        )
        if expected_error:
            with pytest.raises(ValueError):
                featurizer._get_reference_structure_feature_dict(atoms_to_include=AtomFilter.BB)
        else:
            featurizer._get_reference_structure_feature_dict(atoms_to_include=AtomFilter.BB)
            assert torch.allclose(
                featurizer._reference_structure_feature_dict_cache["target_mask"], expected_target_mask
            )
            assert torch.allclose(
                featurizer._reference_structure_feature_dict_cache["binder_mask"], expected_binder_mask
            )

    @pytest.mark.parametrize("verbose_metadata", [True, False])
    def test_reference_structure_featurizer_empty_input(
        self,
        frame_structure_token_features_bb_standard: FrameStructureTokenFeatures,
        structure_standard_no_interfaces: Structure,
        feature_dict_standard: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
        verbose_metadata: bool,
    ) -> None:
        """Make sure that the reference structure feature dict is added correctly when the input is empty"""
        monkeypatch.setattr(rs, "load_structure_from_rcsb", Mock(return_value=structure_standard_no_interfaces))
        monkeypatch.setattr(
            rs.DynoPsiTokenizer, "tokenize", Mock(return_value=[frame_structure_token_features_bb_standard])
        )

        featurizer = primitives.ReferenceStructureFeaturizer(
            name="test_reference_structure_featurizer",
            rcsb_record_id="7f7p",  # dummy record id to avoid validation error
            design_designation={DesignDesignation.FIXED: ["B", "C"], DesignDesignation.DESIGN: ["A"]},
            atoms_to_include=AtomFilter.BB,
            verbose_metadata=verbose_metadata,
        )
        out_feats = next(featurizer([Features(feature_dict={}, atoms_to_include=AtomFilter.BB)]))

        # update the expected
        feature_dict_standard = {
            k: v.clone()
            for k, v in feature_dict_standard.items()
            if k not in ["inpaint_feature", "noised_mask", "pocket_feature", "original_residue_index"]
        }
        assert_feature_dict_equality(observed=out_feats.feature_dict, expected=feature_dict_standard)

        assert out_feats.featurization_steps == [("reference_structure", "test_reference_structure_featurizer")]
        expected_metadata = {
            "test_reference_structure_featurizer": {
                "type": "reference_structure",
                "name": "test_reference_structure_featurizer",
                "index_type": "original_residue_index",
                "verbose_metadata": verbose_metadata,
                "atoms_to_include": AtomFilter.BB,
                "rcsb_record_id": "7f7p",
                "afdb_uniprot_id": None,
                "structure_path": None,
                "design_designation": {DesignDesignation.FIXED: ["B", "C"], DesignDesignation.DESIGN: ["A"]},
                "ref_asym_id_map": None,
            }
        }
        if verbose_metadata:
            # check structure
            expected_metadata["test_reference_structure_featurizer"]["structure"] = structure_standard_no_interfaces

        assert_metadata_equality(observed=out_feats.featurization_metadata, expected=expected_metadata)

    @pytest.mark.parametrize("index_type", ["original_residue_index", "auth_residue_index"])
    @pytest.mark.parametrize("verbose_metadata", [True, False])
    def test_reference_structure_featurizer_append_to_existing_chains(
        self,
        feature_dict_chain_A_only: dict[str, torch.Tensor],
        structure_chain_A_only: Structure,
        frame_structure_token_features_chain_B_C_only: FrameStructureTokenFeatures,
        structure_chain_B_C_only: Structure,
        structure_standard_no_interfaces: Structure,
        feature_dict_standard: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
        index_type: str,
        verbose_metadata: bool,
    ) -> None:
        """Make sure that the reference structure feature dict is appended to existing chains correctly"""
        monkeypatch.setattr(rs, "load_structure_from_rcsb", Mock(return_value=structure_chain_B_C_only))
        monkeypatch.setattr(
            rs.DynoPsiTokenizer, "tokenize", Mock(return_value=[frame_structure_token_features_chain_B_C_only])
        )

        structure_standard_no_interfaces = deepcopy(structure_standard_no_interfaces)
        if index_type == "original_residue_index":
            design_designation = {DesignDesignation.FIXED: ["B", "C"]}
        else:
            design_designation = {DesignDesignation.FIXED: ["V", "W"]}
        featurizer = primitives.ReferenceStructureFeaturizer(
            name="test_reference_structure_featurizer",
            rcsb_record_id="7f7p",  # dummy record id to avoid validation error
            design_designation=design_designation,
            atoms_to_include=AtomFilter.BB,
            index_type=index_type,
            verbose_metadata=verbose_metadata,
        )
        # check that a value error is raised if the input features are not compatible with the reference structure
        with pytest.raises(ValueError):
            next(featurizer([Features(feature_dict=feature_dict_chain_A_only, atoms_to_include=AtomFilter.BB)]))

        # remove features that should not be present in structure
        input_feat_dict = {
            k: v.clone()
            for k, v in feature_dict_chain_A_only.items()
            if k not in ["inpaint_feature", "noised_mask", "pocket_feature", "original_residue_index"]
        }
        out_feats = next(
            featurizer(
                [
                    Features(
                        feature_dict=input_feat_dict,
                        atoms_to_include=AtomFilter.BB,
                        input_reference_structure=structure_chain_A_only,
                    )
                ]
            )
        )
        # output structure should have a slight change in entity_id
        # structure_standard_no_interfaces.chains["entity_id"] = np.array([0, 2, 3])
        print(out_feats.input_reference_structure.chains)
        print(structure_standard_no_interfaces.chains)
        assert_structure_equality(
            observed=out_feats.input_reference_structure, expected=structure_standard_no_interfaces
        )

        expected_feature_dict = {
            k: v.clone()
            for k, v in feature_dict_standard.items()
            if k not in ["inpaint_feature", "noised_mask", "pocket_feature", "original_residue_index"]
        }
        # expected_feature_dict["entity_id"] = torch.tensor([0, 2, 3])
        assert_feature_dict_equality(observed=out_feats.feature_dict, expected=expected_feature_dict)

        assert out_feats.featurization_steps == [("reference_structure", "test_reference_structure_featurizer")]
        expected_metadata = {
            "test_reference_structure_featurizer": {
                "type": "reference_structure",
                "name": "test_reference_structure_featurizer",
                "index_type": index_type,
                "verbose_metadata": verbose_metadata,
                "atoms_to_include": AtomFilter.BB,
                "rcsb_record_id": "7f7p",
                "afdb_uniprot_id": None,
                "structure_path": None,
                "design_designation": design_designation,
                "ref_asym_id_map": {1: 1, 2: 2},
            }
        }
        if verbose_metadata:
            expected_metadata["test_reference_structure_featurizer"]["structure"] = structure_chain_B_C_only

        assert_metadata_equality(observed=out_feats.featurization_metadata, expected=expected_metadata)
