import pytest
import torch

from dynopsi.data.constants import AtomFilter
from dynopsi.data.utils.structure_utils import center_coordinates, reshape_coords_for_model


class TestReshapeCoordsForModel:
    input_coords = torch.arange(90, dtype=torch.float32).reshape(5, 6, 3).unsqueeze(1)  # shape: (5, 1, 3*2, 3)
    frames_idx = torch.tile(torch.arange(6).reshape(1, 2, 3), (5, 1, 1))  # shape: (5, 2, 3)

    def test_bb_coords(self) -> None:
        coords = reshape_coords_for_model(self.input_coords, self.frames_idx, AtomFilter.BB)
        assert coords.shape == (5, 2, 9)
        assert torch.allclose(coords, torch.arange(90, dtype=torch.float32).reshape(5, 2, 9))

    def test_ca_coords(self) -> None:
        ca_frames_idx = self.frames_idx[:, :, :1]
        coords = reshape_coords_for_model(
            self.input_coords,
            ca_frames_idx,
            AtomFilter.CA,
        )
        assert coords.shape == (5, 2, 3)
        expected = torch.arange(90, dtype=torch.float32).reshape(5, 6, 3)[:, [0, 3], :]
        assert torch.allclose(coords, expected)

    @pytest.mark.parametrize("atoms", [AtomFilter.FULL])
    def test_invalid_atoms(self, atoms: AtomFilter) -> None:
        with pytest.raises(ValueError):
            reshape_coords_for_model(self.input_coords, self.frames_idx, atoms)


class TestCenterCoordinates:
    @pytest.mark.parametrize(
        "mask_positions",
        [torch.tensor([]), torch.arange(5), torch.arange(10)],
        ids=["no_masked_positions", "partial_masked_positions", "all_masked_positions"],
    )
    def test_center_coordinates(self, mask_positions: torch.Tensor) -> None:
        mask_positions = mask_positions.long()
        input_random_coords = torch.randn(1, 10, 3)
        center_mask = torch.zeros(input_random_coords.shape[1], dtype=torch.bool)
        center_mask[mask_positions] = True
        if not center_mask.any():
            print("center mask")
            print(center_mask)
            with pytest.raises(ValueError):
                center_coordinates(input_random_coords, center_mask)
        else:
            centered_coords = center_coordinates(input_random_coords, center_mask)
            expected_coords = input_random_coords - input_random_coords[:, mask_positions, :].mean(dim=1, keepdim=True)
            assert centered_coords.shape == (1, 10, 3)
            assert torch.allclose(centered_coords, expected_coords)
