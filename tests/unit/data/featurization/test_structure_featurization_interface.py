import numpy as np
import torch
from torch import from_numpy

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization import Features
from dynopsi.data.featurization.utils import strip_chain_padding


class TestStructureOperations:
    def test_structure_align_to_feature_dict(self, features_A: dict[str, torch.Tensor]):
        features = Features(feature_dict=strip_chain_padding(features_A), atoms_to_include=AtomFilter.BB)
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
