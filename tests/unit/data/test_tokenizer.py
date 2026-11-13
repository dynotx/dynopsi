import pytest
import torch

from dynopsi.data.constants import AtomFilter
from dynopsi.data.structure import Structure
from dynopsi.data.tokenizer import DynoPsiTokenizer, FrameStructureTokenFeatures


class TestDynoPsiTokenizer:
    """Class for testing DynoPsiTokenizer."""

    def test_tokenize_bb_only(
        self, structure_standard: Structure, frame_structure_token_features_bb_standard: FrameStructureTokenFeatures
    ) -> None:
        tokenizer = DynoPsiTokenizer(atom_filter=AtomFilter.BB)
        features = tokenizer.tokenize(structure_standard)
        assert len(features) == 2
        assert isinstance(features[0], FrameStructureTokenFeatures)
        expected_attrs = [
            "token_index",
            "residue_index",
            "auth_residue_index",
            "asym_id",
            "auth_asym_id",
            "entity_id",
            "sym_id",
            "token_resolved_mask",
            "token_pad_mask",
            "coords",
            "frames_idx",
            "atom_to_token",
            "frame_resolved_mask",
            "atom_resolved_mask",
            "atom_pad_mask",
        ]
        assert len(features[0].__dict__.keys()) == len(expected_attrs)
        for attr in expected_attrs:
            expected_vals = getattr(frame_structure_token_features_bb_standard, attr)
            actual_vals = getattr(features[0], attr)
            assert (actual_vals == expected_vals).all(), (
                f"{attr} does not match expected values: {actual_vals} != {expected_vals}"
            )

    def test_tokenize_full(self, structure_standard: Structure) -> None:
        tokenizer = DynoPsiTokenizer(atom_filter=AtomFilter.FULL)
        with pytest.raises(NotImplementedError):
            tokenizer.tokenize(structure_standard)

    def test_tokenize_ca_only(self, structure_standard: Structure) -> None:
        tokenizer = DynoPsiTokenizer(atom_filter=AtomFilter.CA)
        features = tokenizer.tokenize(structure_standard)
        assert len(features) == 2
        assert isinstance(features[0], FrameStructureTokenFeatures)

        structure_features = features[0]
        n_residues = len(structure_standard.residues)

        assert structure_features.coords.shape == (1, n_residues, 3)
        assert structure_features.frames_idx.shape == (n_residues, 1)
        assert structure_features.atom_to_token.shape == (n_residues, n_residues)
        expected_atom_to_token = torch.eye(n_residues, dtype=torch.long)
        assert torch.equal(structure_features.atom_to_token, expected_atom_to_token)
        expected_resolved_mask = structure_features.token_resolved_mask.bool()
        assert torch.equal(structure_features.frame_resolved_mask, expected_resolved_mask)
        assert torch.equal(structure_features.atom_resolved_mask, expected_resolved_mask)
