from __future__ import annotations

import pytest
import torch

from dynopsi.data.utils.alignment_utils import slice_coords_by_frames_idx


class TestSliceCoordsByFramesIdx:
    def _make_coords(self, bs: int = 2, n_all: int = 30) -> torch.Tensor:
        """(bs, 1, n_all, 3) full-atom coord tensor."""
        return torch.randn(bs, 1, n_all, 3)

    def _make_frames_idx(self, bs: int = 2, seq_len: int = 8, n_all: int = 30) -> torch.Tensor:
        """(bs, seq_len, 3) index tensor — unique indices per residue."""
        idx = torch.zeros(bs, seq_len, 3, dtype=torch.long)
        for b in range(bs):
            for i in range(seq_len):
                # N, CA, C indices for residue i
                idx[b, i, 0] = i * 3
                idx[b, i, 1] = i * 3 + 1
                idx[b, i, 2] = i * 3 + 2
        return idx

    def test_ca_only_shape(self):
        bs, seq_len, n_all = 2, 8, 30
        coords = self._make_coords(bs, n_all)
        frames_idx = self._make_frames_idx(bs, seq_len, n_all)
        out = slice_coords_by_frames_idx(coords, frames_idx, num_atoms=1)
        assert out.shape == (bs, seq_len, 3)

    def test_backbone_shape(self):
        bs, seq_len, n_all = 2, 8, 30
        coords = self._make_coords(bs, n_all)
        frames_idx = self._make_frames_idx(bs, seq_len, n_all)
        out = slice_coords_by_frames_idx(coords, frames_idx, num_atoms=3)
        assert out.shape == (bs, seq_len, 9)

    def test_ca_extracts_middle_atom(self):
        bs, seq_len, n_all = 1, 4, 20
        coords = torch.zeros(bs, 1, n_all, 3)
        # Give CA atoms (index 1 of each triplet) a distinctive value
        for i in range(seq_len):
            coords[0, 0, i * 3 + 1, :] = float(i + 1)
        frames_idx = self._make_frames_idx(bs, seq_len, n_all)
        out = slice_coords_by_frames_idx(coords, frames_idx, num_atoms=1)
        for i in range(seq_len):
            assert torch.allclose(out[0, i], torch.tensor([float(i + 1)] * 3))

    def test_invalid_num_atoms_raises(self):
        coords = self._make_coords()
        frames_idx = self._make_frames_idx()
        with pytest.raises(ValueError, match="Unsupported num_atoms"):
            slice_coords_by_frames_idx(coords, frames_idx, num_atoms=2)
