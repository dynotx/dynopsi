import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import from_numpy

from dynopsi.data.constants import AtomFilter, DesignTask, InpaintFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.utils import (
    concat_features,
    crop_features,
    make_empty_chain,
    strip_chain_padding,
)
from dynopsi.data.structure import Structure
from dynopsi.data.utils import randomly_rotate
from tests.unit.data.conftest import assert_structure_equality, unique_letter_auth_table


@pytest.fixture
def coordinates_chain_A() -> torch.Tensor:
    """Make a chain with all Alanines"""
    torch.manual_seed(42)
    return torch.randn(4 * 5, 3) * 10


@pytest.fixture
def coordinates_chain_B() -> torch.Tensor:
    """Make a chain with all Alanines, different length than A"""
    torch.manual_seed(11)
    return torch.randn(5 * 5, 3) * 10


def subset_coordinates_to_atomfilter(coordinates: torch.Tensor, atoms_to_include: AtomFilter) -> torch.Tensor:
    """Subset the coordinates to the atoms in the atom filter"""
    if atoms_to_include == AtomFilter.BB:
        return coordinates.reshape(-1, 5, *coordinates.shape[1:])[:, :3].reshape(-1, *coordinates.shape[1:])
    elif atoms_to_include == AtomFilter.CA:
        return coordinates.reshape(-1, 5, *coordinates.shape[1:])[:, 1:2].reshape(-1, *coordinates.shape[1:])
    else:
        raise ValueError(f"Unsupported atoms_to_include: {atoms_to_include}")


def feature_dict_from_coordinates(
    asym_ids: list[int], coordinates: list[torch.Tensor], atoms_to_include: AtomFilter, use_coordinates: bool = True
) -> dict[str, torch.Tensor]:
    """Make a feature dict from a list of asym ids and corresponding coordinates
    if use_coordinates is True, then the specified coordinates will be used; otherwise zeros will be used
    """
    feat_dicts = []
    for asym_id, coords in zip(asym_ids, coordinates):
        nres = coords.shape[0] // 5
        feat_dict = make_empty_chain(
            asym_id=asym_id,
            auth_asym_id=asym_id,
            seq_length=nres,
            atoms_to_include=atoms_to_include,
            res_type="ALA",
        )
        coords_features = subset_coordinates_to_atomfilter(coords, atoms_to_include)
        if use_coordinates:
            feat_dict["coords"] = coords_features.unsqueeze(0)  # (1, n_atoms_features, 3),
        feat_dicts.append(feat_dict)
    return concat_features(feat_dicts, make_asym_ids_unique=True, make_entity_ids_unique=True)


def _features(*, feature_dict: dict[str, torch.Tensor], **kwargs) -> Features:
    kwargs.setdefault(
        "auth_chain_id_table",
        unique_letter_auth_table(feature_dict["auth_asym_id"].tolist()),
    )
    return Features(feature_dict=feature_dict, **kwargs)


class TestStructureOperations:
    def test_design_coordinates_to_structure_defaults_to_raw_noised_context(self) -> None:
        parameter = inspect.signature(Features.design_coordinates_to_structure).parameters[
            "use_reference_for_noised_context"
        ]
        assert parameter.default is False

    def test_to_structure_constructs_both_residue_index_columns(self) -> None:
        feature_dict = make_empty_chain(
            seq_length=2,
            atoms_to_include=AtomFilter.BB,
            res_type="ALA",
        )
        feature_dict["residue_index"] = torch.tensor([5, 7])
        feature_dict["auth_residue_index"] = torch.tensor([81, 90])

        structure = _features(feature_dict=feature_dict, atoms_to_include=AtomFilter.BB).to_structure()

        np.testing.assert_array_equal(structure.residues["res_idx"], [5, 7])
        np.testing.assert_array_equal(structure.residues["auth_res_idx"], [81, 90])

    def test_structure_align_to_feature_dict(self, features_A: dict[str, torch.Tensor]):
        features = _features(feature_dict=strip_chain_padding(features_A), atoms_to_include=AtomFilter.BB)
        struct = features.to_structure()

        # apply a random rotation and translation to the features
        features_rotated = features.safe_copy()
        rotation_90_degrees = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(torch.float32)
        translation_vector = torch.tensor([1, 2, 3]).to(torch.float32)
        features_rotated["coords"] = (
            torch.matmul(features_rotated["coords"][0], rotation_90_degrees) + translation_vector
        ).unsqueeze(0)

        # align the structure
        struct_rotated = struct.align_to_feature_dict(features_rotated, atoms_to_include=AtomFilter.BB)

        # expected coords
        non_oxygen_pos = np.arange(struct.atoms.shape[0])[np.tile([True, True, True, False], struct.residues.shape[0])]
        print("non_oxygen_pos shape: ", non_oxygen_pos.shape)
        expected_coords = (
            torch.matmul(from_numpy(struct.atoms["coords"].copy()), rotation_90_degrees) + translation_vector
        )
        assert np.allclose(struct_rotated.atoms["coords"], expected_coords.numpy())


class TestDesignCoordinatesToStructure:
    def _create_coords_stack(self, coordinates: torch.Tensor, atoms_to_include: AtomFilter) -> torch.Tensor:
        """Create a stack of coordinates 3 deep, rotated and translated. Last element of the batch is the original coordinates.
        coordinates: (n_atoms, 3) -> (batch, n_atoms_features, 3)
        """
        coords_features = subset_coordinates_to_atomfilter(coordinates, atoms_to_include)
        # create 2 random rotations and translations
        rotation_90_degrees = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(torch.float32)
        translation_vector = torch.tensor([1, 2, 3]).to(torch.float32)
        coords_features_rotated1 = torch.matmul(coords_features, rotation_90_degrees) + translation_vector
        coords_features_rotated2 = torch.matmul(coords_features_rotated1, rotation_90_degrees) + translation_vector
        return torch.stack([coords_features_rotated1, coords_features_rotated2, coords_features], dim=0)

    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_no_reference_structure(self, coordinates_chain_A: torch.Tensor, atoms_to_include: AtomFilter):
        coords_stack = self._create_coords_stack(coordinates_chain_A, atoms_to_include)
        # make the features
        # use zero coordinates to make sure the features don't influence the alignment
        features_zero_coords = _features(
            feature_dict=feature_dict_from_coordinates(
                [0], [coordinates_chain_A], atoms_to_include, use_coordinates=False
            ),
            atoms_to_include=atoms_to_include,
        )

        # design the coordinates to structure
        structures = features_zero_coords.design_coordinates_to_structure(
            coords_stack, use_reference_for_noised_context=False
        )

        # create a structure from the original coordinates
        struct = _features(
            feature_dict=feature_dict_from_coordinates(
                [0], [coordinates_chain_A], atoms_to_include, use_coordinates=True
            ),
            atoms_to_include=atoms_to_include,
        ).to_structure()

        # be slightly more lenient with the float tolerance for the aligned structures
        # because of the oxygen atoms
        for aligned_struct in structures:
            assert_structure_equality(aligned_struct, struct, float_tolerance=1e-5)

    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_single_chain_inpainted(self, coordinates_chain_A: torch.Tensor, atoms_to_include: AtomFilter):
        coords_stack = self._create_coords_stack(coordinates_chain_A, atoms_to_include)
        # make the features
        # - feature dict has inpainted regions
        feature_dict_inpainted = feature_dict_from_coordinates(
            [0], [coordinates_chain_A], atoms_to_include, use_coordinates=True
        )

        feature_dict_inpainted["inpaint_feature"] = torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2 + [InpaintFeature.BINDER_INPAINT] * 2
        )
        feature_dict_inpainted["noised_mask"] = torch.tensor([False] * 2 + [True] * 2)
        # - get the structure for all residues
        struct_all = _features(feature_dict=feature_dict_inpainted, atoms_to_include=atoms_to_include).to_structure()

        # - get the structure for context residues
        struct_context = _features(
            feature_dict=crop_features(
                feature_dict_inpainted, token_mask=~feature_dict_inpainted["noised_mask"].clone()
            ),
            atoms_to_include=atoms_to_include,
        ).to_structure()
        # set the feature dict inpainted coords to 0s
        feature_dict_inpainted["coords"] = torch.zeros_like(feature_dict_inpainted["coords"])
        # get the features
        features_inpainted = _features(
            feature_dict=feature_dict_inpainted,
            atoms_to_include=atoms_to_include,
            input_reference_structure=struct_context,
        )
        # now align the coordinates to the features
        coords_aligned = features_inpainted.design_coordinates_to_structure(
            coords_stack, use_reference_for_noised_context=False
        )

        for aligned_struct in coords_aligned:
            assert_structure_equality(aligned_struct, struct_all)

    @pytest.mark.parametrize("design_task", [DesignTask.MULTIMER_TARGET_DOCKING_INPAINT_BINDER])
    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_two_chain_inpainted(
        self,
        coordinates_chain_A: torch.Tensor,
        coordinates_chain_B: torch.Tensor,
        atoms_to_include: AtomFilter,
        design_task: DesignTask,
    ):
        """
        Setup:
        - Chain A is inpainted as in test_single_chain_inpainted, Chain B is not inpainted.
        - The input reference structure(s) are concatenated arbitrarily, and the inter-chain distances are meaningless. In this test, these
          coordinates are rotated and translated differently for each chain and this is set as the input_reference_structure.
        - We check to make sure that the final alignment is the same as the coords stack.
        """
        chains_A_and_B_coords = torch.cat([coordinates_chain_A, coordinates_chain_B], dim=0)
        chains_A_and_B_coords_stack = self._create_coords_stack(chains_A_and_B_coords, atoms_to_include)

        inpaint_feature = torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 5
        )
        noised_mask = torch.tensor(
            [design_task.get_noise_bool(feature_value.item()) for feature_value in inpaint_feature], dtype=torch.bool
        )

        # now rotate and translate the coordinates for chain B
        # make the features
        # - feature dict has inpainted regions
        feature_dict_inpainted = feature_dict_from_coordinates(
            [0, 1], [coordinates_chain_A, coordinates_chain_B], atoms_to_include, use_coordinates=True
        )

        feature_dict_inpainted["inpaint_feature"] = inpaint_feature
        feature_dict_inpainted["noised_mask"] = noised_mask
        # - get the expected structure for all residues
        expected_struct_all = _features(
            feature_dict=feature_dict_inpainted, atoms_to_include=atoms_to_include
        ).to_structure()

        # now create a input_reference_structure where the chains are randomly rotated and translated with respect to each other
        chain_B_coords_rotated = randomly_rotate(coordinates_chain_B.unsqueeze(0)).squeeze(0) + torch.randn(3)
        chains_A_and_B_coords_stack_rotated = torch.cat([coordinates_chain_A, chain_B_coords_rotated], dim=0)
        # assert that these coordinates are different from the original coordinates
        assert not torch.allclose(chains_A_and_B_coords_stack_rotated, chains_A_and_B_coords, atol=1)

        feature_dict_inpainted_rotated = feature_dict_from_coordinates(
            [0, 1], [coordinates_chain_A, chain_B_coords_rotated], atoms_to_include, use_coordinates=True
        )
        feature_dict_inpainted_rotated["inpaint_feature"] = inpaint_feature
        feature_dict_inpainted_rotated["noised_mask"] = noised_mask
        # - get the structure for context residues
        context_mask = (
            torch.isin(
                feature_dict_inpainted_rotated["inpaint_feature"],
                torch.tensor([InpaintFeature.BINDER_CONTEXT, InpaintFeature.TARGET_CONTEXT]),
            )
            & feature_dict_inpainted_rotated["token_pad_mask"].bool()
        )
        struct_context_rotated = _features(
            feature_dict=crop_features(feature_dict_inpainted_rotated, token_mask=context_mask),
            atoms_to_include=atoms_to_include,
        ).to_structure()

        # construct the features object with the input_reference_structure
        # set coordinates to 0
        feature_dict_inpainted["coords"] = torch.zeros_like(feature_dict_inpainted["coords"])
        features_inpainted_rotated = _features(
            feature_dict=feature_dict_inpainted,
            atoms_to_include=atoms_to_include,
            input_reference_structure=struct_context_rotated,
        )

        # now align the original coordinates and replace context regions with the input_reference_structure
        coords_aligned = features_inpainted_rotated.design_coordinates_to_structure(
            chains_A_and_B_coords_stack, use_reference_for_noised_context=False
        )
        for aligned_struct in coords_aligned:
            assert_structure_equality(
                aligned_struct, expected_struct_all, float_tolerance=1e-5, ignore_oxygen_atoms=True
            )

    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_noised_context_comes_from_design_coordinates(
        self,
        coordinates_chain_A: torch.Tensor,
        coordinates_chain_B: torch.Tensor,
        atoms_to_include: AtomFilter,
    ):
        """Verify noised context uses design coords and BINDER_CONTEXT is aligned from the reference structure.

        Setup:
        - Chain A: BINDER_CONTEXT (res 0-1, noised=False) + BINDER_INPAINT (res 2-3, noised=True)
        - Chain B: TARGET_CONTEXT (res 4-8, noised=True)

        Two behaviors are tested:
        1. Noised coordinates (BINDER_INPAINT and TARGET_CONTEXT) come from design coords, not the reference
           structure. Verified by making chain B reference coords a rigid rotation/translation of the
           original while chain B design coords are a non-rigid perturbation. If noised coordinates
           incorrectly use the reference structure, the output would not match the expected structure.
        2. Unnoised BINDER_CONTEXT is aligned from the reference structure to the design coord backbone.
           Verified by rotating chain A design coords relative to the reference structure coords (original
           chain A). If BINDER_CONTEXT incorrectly used the original feature dict coords rather than
           aligning the reference to the design coords, the N/CA/C positions would not match expected.
        """
        inpaint_feature = torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 5
        )
        noised_mask = torch.tensor([False, False, True, True, True, True, True, True, True], dtype=torch.bool)

        feature_dict = feature_dict_from_coordinates(
            [0, 1], [coordinates_chain_A, coordinates_chain_B], atoms_to_include, use_coordinates=True
        )
        feature_dict["inpaint_feature"] = inpaint_feature
        feature_dict["noised_mask"] = noised_mask

        torch.manual_seed(123)
        # Rotate chain A design coords so the reference structure (at original positions) must be
        # non-trivially aligned. If alignment is skipped or the feature dict coords are used instead,
        # the BINDER_CONTEXT atoms will land at the original (non-rotated) positions and the test fails.
        chain_a_design_coords = subset_coordinates_to_atomfilter(
            randomly_rotate(coordinates_chain_A.unsqueeze(0)).squeeze(0), atoms_to_include
        )
        # Non-rigid perturbation of chain B ensures noised TARGET_CONTEXT uses design coords.
        chain_b_design_coords = subset_coordinates_to_atomfilter(coordinates_chain_B, atoms_to_include) + (
            torch.randn_like(subset_coordinates_to_atomfilter(coordinates_chain_B, atoms_to_include)) * 5.0
        )
        design_coords = torch.cat([chain_a_design_coords, chain_b_design_coords], dim=0)
        coords_stack = design_coords.unsqueeze(0)

        expected_feature_dict = {k: v.clone() for k, v in feature_dict.items()}
        expected_feature_dict["coords"] = design_coords.unsqueeze(0)
        expected_struct = _features(
            feature_dict=expected_feature_dict,
            atoms_to_include=atoms_to_include,
        ).to_structure()

        chain_b_reference_coords = randomly_rotate(coordinates_chain_B.unsqueeze(0)).squeeze(0) + torch.randn(3)
        reference_feature_dict = feature_dict_from_coordinates(
            [0, 1], [coordinates_chain_A, chain_b_reference_coords], atoms_to_include, use_coordinates=True
        )
        reference_feature_dict["inpaint_feature"] = inpaint_feature
        reference_feature_dict["noised_mask"] = noised_mask
        context_mask = (
            torch.isin(
                reference_feature_dict["inpaint_feature"],
                torch.tensor([InpaintFeature.BINDER_CONTEXT, InpaintFeature.TARGET_CONTEXT]),
            )
            & reference_feature_dict["token_pad_mask"].bool()
        )
        reference_context_struct = _features(
            feature_dict=crop_features(reference_feature_dict, token_mask=context_mask),
            atoms_to_include=atoms_to_include,
        ).to_structure()

        features = _features(
            feature_dict=feature_dict,
            atoms_to_include=atoms_to_include,
            input_reference_structure=reference_context_struct,
        )

        observed_struct = features.design_coordinates_to_structure(
            coords_stack, use_reference_for_noised_context=False
        )[0]

        assert_structure_equality(
            observed_struct,
            expected_struct,
            float_tolerance=1e-5,
            ignore_oxygen_atoms=True,
        )


class TestDesignCoordinatesToStructureNonBackboneAtoms:
    """Test that non-backbone atoms from the reference structure are carried through to BINDER_CONTEXT residues."""

    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_unnoised_binder_context_o_atoms_come_from_reference_structure(
        self,
        coordinates_chain_A: torch.Tensor,
        coordinates_chain_B: torch.Tensor,
        atoms_to_include: AtomFilter,
    ):
        """Verify O atoms for unnoised BINDER_CONTEXT residues come from the reference structure, not re-imputed.

        When AtomFilter.BB is used, to_structure computes O positions by imputing from N/CA/C geometry.
        The reference structure stored in Features.input_reference_structure is a full Structure object that
        includes O atoms alongside the backbone. design_coordinates_to_structure aligns the reference
        structure's backbone (N/CA/C) to the design coords, and all other atoms — including O — are
        carried along by the same rigid transform.

        This test verifies that behavior by setting the O atoms of BINDER_CONTEXT residues in the
        reference structure to a large sentinel value ([999, 999, 999]), then asserting that the output
        structure's BINDER_CONTEXT residues carry those sentinel O positions (not freshly imputed ones).
        BINDER_INPAINT residues are built from design coords via to_structure, so their O atoms should
        be imputed and NOT equal to the sentinel.
        """
        inpaint_feature = torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 5
        )
        noised_mask = torch.tensor([False, False, True, True, True, True, True, True, True], dtype=torch.bool)

        feature_dict = feature_dict_from_coordinates(
            [0, 1], [coordinates_chain_A, coordinates_chain_B], atoms_to_include, use_coordinates=True
        )
        feature_dict["inpaint_feature"] = inpaint_feature
        feature_dict["noised_mask"] = noised_mask

        # Build reference context structure (BINDER_CONTEXT + TARGET_CONTEXT) from the feature dict.
        # to_structure with AtomFilter.BB imputes O from N/CA/C, giving 4 atoms per residue.
        context_mask = (
            torch.isin(
                feature_dict["inpaint_feature"],
                torch.tensor([InpaintFeature.BINDER_CONTEXT, InpaintFeature.TARGET_CONTEXT]),
            )
            & feature_dict["token_pad_mask"].bool()
        )
        reference_context_struct = _features(
            feature_dict=crop_features(feature_dict, token_mask=context_mask),
            atoms_to_include=atoms_to_include,
        ).to_structure()

        # Overwrite the O atoms for BINDER_CONTEXT residues (chain A, asym_id=0) with a distinctive
        # sentinel. If design_coordinates_to_structure re-imputes O instead of using the reference
        # structure, the output O positions will not match and the test will fail.
        sentinel_o_pos = np.array([999.0, 999.0, 999.0], dtype=np.float32)
        binder_context_o_mask = (reference_context_struct.atoms["atom_name"] == "O") & (
            reference_context_struct.atoms["asym_id"] == 0
        )
        assert binder_context_o_mask.sum() == 2, "Expected 2 BINDER_CONTEXT O atoms (one per residue)"
        reference_context_struct.atoms["coords"][binder_context_o_mask] = sentinel_o_pos

        # Design coords are the same N/CA/C as the feature dict for chain A (identity alignment for
        # BINDER_CONTEXT), so the sentinel O positions pass through unchanged after alignment.
        chain_a_design_coords = subset_coordinates_to_atomfilter(coordinates_chain_A, atoms_to_include)
        torch.manual_seed(42)
        chain_b_design_coords = subset_coordinates_to_atomfilter(coordinates_chain_B, atoms_to_include) + (
            torch.randn_like(subset_coordinates_to_atomfilter(coordinates_chain_B, atoms_to_include)) * 5.0
        )
        design_coords = torch.cat([chain_a_design_coords, chain_b_design_coords], dim=0)
        coords_stack = design_coords.unsqueeze(0)

        features = _features(
            feature_dict=feature_dict,
            atoms_to_include=atoms_to_include,
            input_reference_structure=reference_context_struct,
        )

        observed_struct = features.design_coordinates_to_structure(
            coords_stack, use_reference_for_noised_context=False
        )[0]

        # BINDER_CONTEXT O atoms (chain A, res_idx 0 and 1) should equal the sentinel.
        chain_a = observed_struct.atoms["asym_id"] == 0
        o_atoms = observed_struct.atoms["atom_name"] == "O"
        binder_context_res = np.isin(observed_struct.atoms["res_idx"], [0, 1])
        binder_inpaint_res = np.isin(observed_struct.atoms["res_idx"], [2, 3])

        binder_context_o_coords = observed_struct.atoms["coords"][chain_a & o_atoms & binder_context_res]
        binder_inpaint_o_coords = observed_struct.atoms["coords"][chain_a & o_atoms & binder_inpaint_res]

        assert binder_context_o_coords.shape[0] == 2, "Expected 2 BINDER_CONTEXT O atoms in output"
        np.testing.assert_allclose(
            binder_context_o_coords,
            np.tile(sentinel_o_pos, (2, 1)),
            atol=5e-4,
            err_msg="BINDER_CONTEXT O atoms should come from the reference structure (sentinel [999,999,999]), not be re-imputed",
        )

        # BINDER_INPAINT O atoms are computed fresh via to_structure and should NOT be at the sentinel.
        assert binder_inpaint_o_coords.shape[0] == 2, "Expected 2 BINDER_INPAINT O atoms in output"
        assert not np.allclose(binder_inpaint_o_coords, sentinel_o_pos, atol=10.0), (
            "BINDER_INPAINT O atoms should be imputed from design coords, not from the reference sentinel"
        )


class TestSaveToNpz:
    @pytest.mark.parametrize("atoms_to_include", [AtomFilter.BB])
    def test_save_to_npz_uses_residue_index(
        self,
        coordinates_chain_A: torch.Tensor,
        atoms_to_include: AtomFilter,
        tmp_path: Path,
    ):
        """Verify save_to_npz stores sequential residue_index (0-based) in auth_res_idx, not auth_residue_index (1-based)."""
        feature_dict = feature_dict_from_coordinates([0], [coordinates_chain_A], atoms_to_include, use_coordinates=True)
        features = _features(feature_dict=feature_dict, atoms_to_include=atoms_to_include)

        n_residues = coordinates_chain_A.shape[0] // 5
        expected_sequential = np.arange(n_residues)

        npz_path = tmp_path / "test.npz"
        features.save_to_npz(str(npz_path))

        loaded = Structure.load_from_npz(str(npz_path))

        np.testing.assert_array_equal(
            loaded.residues["auth_res_idx"],
            expected_sequential,
            err_msg="auth_res_idx should equal sequential residue_index (0-based), not auth_residue_index (1-based)",
        )
