"""Portions of this file are derived from the La Proteina project.

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the La Proteina-derived code were made by Dyno Therapeutics, Inc. (2026).
"""

from typing import Protocol

import torch
from torch import nn

from dynopsi.model.nn.primitives import (
    get_distogram,
    get_index_embedding,
    get_time_embedding,
)


class FeatureEncoder(Protocol):
    def get_output_dim(self) -> int: ...


class DistogramEncoder(nn.Module, FeatureEncoder):
    """Encodes pairwise distances into a binned representation.

    Pairwise distances are computed (and binned) in an all-to-all manner,
    i.e. both within atoms types and across atom types.
    This results in a distogram of shape [batch_size, N, N, num_atoms * num_atoms * num_bins]

    Parameters
    ----------
    num_distance_bins : int
        Number of bins to discretize the distances into
    num_atoms : int
        Number of atoms per residue in the protein
    bin_min : float, default=0
        Minimum distance value for binning
    bin_max : float, default=30
        Maximum distance value for binning
    """

    def __init__(self, num_distance_bins: int, num_atoms: int, bin_min: float = 0, bin_max: float = 30):
        super().__init__()
        self.num_distance_bins = num_distance_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_atoms = num_atoms

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        distogram = get_distogram(
            coords, self.num_distance_bins, self.bin_min, self.bin_max
        )  # [batch_size, N, N, num_atoms, num_atoms, num_bins]

        distogram = distogram.flatten(start_dim=-3)  # [batch_size, N, N, num_atoms * num_atoms * num_bins]
        return distogram

    def get_output_dim(self) -> int:
        return self.num_distance_bins * self.num_atoms**2


class PairCategoricalFeatureEmbedder(nn.Module, FeatureEncoder):
    """Embeds categorical feature vectors of shape [batch_size, seq_len] with values in {0, ..., num_classes - 1}
    into a matrix of dimension [batch_size, seq_len, seq_len, embed_dim].

    Parameters
    ----------
    num_classes: int
        Number of classes in the input feature alphabet.
        For example, the noised_mask has 2 classes: 0 for unnoised, 1 for noised.
    embed_dim: int
        Dimension of the output embedding

    Returns
    -------
    torch.Tensor
        Embedding of shape [batch_size, seq_len, seq_len, embed_dim]
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(self.num_classes**2, self.embed_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Matrix of unique indices for all ordered pairs {0, ..., num_classes * num_classes - 1}
        pair_ixs = features[:, :, None] * self.num_classes + (features[:, None, :])  # [batch_size, seq_len, seq_len]

        return self.embedder(pair_ixs)

    def get_output_dim(self) -> int:
        return self.embed_dim


class SeqIdxRelativePositionEncoder(nn.Module, FeatureEncoder):
    """Produces relative position (aka. sequence separation) embeddings

    Shape: [batch, seq_len, seq_len, embedding_dim].
    """

    def __init__(self, num_bins: int, symmetric: bool = False):
        super().__init__()
        # We use the strategy from Geffner et al., 2025, which
        # implies bins of size 1 that are symmetric around 0, plus special first and last bins
        # for values beyond the end limits. For this to be possible, the number of bins
        # (aka. embedding_dim) must be odd and >= 3.
        assert num_bins % 2 == 1 and num_bins >= 3, "Relative position embedding_dim must be odd and >= 3"
        self.num_bins = num_bins
        self.symmetric = symmetric

    def get_bins(self, symmetric: bool) -> torch.Tensor:
        if symmetric:
            return torch.linspace(0, self.num_bins - 1, self.num_bins - 1)
        else:
            return torch.linspace(-(self.num_bins / 2 - 1), self.num_bins / 2 - 1, self.num_bins - 1)

    def forward(self, indices: torch.Tensor, asym_id: torch.Tensor | None = None) -> torch.Tensor:
        # Indices shape [batch, seq_len]
        relative_ind = indices.unsqueeze(-1) - indices.unsqueeze(-2)  # [batch, seq_len, seq_len]
        if self.symmetric:
            relative_ind = torch.abs(relative_ind)

        bin_limits = self.get_bins(self.symmetric).to(device=indices.device)
        relative_ind_bins = torch.bucketize(relative_ind, bin_limits)  # [batch, seq_len, seq_len]

        # We incorporate the strategy from Boltz RelativePositionEncoder to set relative distance
        # to the maximum possible value when two residues are on different chains:
        # https://github.com/jwohlwend/boltz/blob/aaef5025c487911f868861ec2f82dfe68b280acf/src/boltz/model/modules/encodersv2.py#L49
        # Here we achieve this by manually adding those pairs to the max distance bin.
        if asym_id is not None:
            is_same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
            relative_ind_bins[~is_same_chain] = self.num_bins - 1
            if not self.symmetric:
                # In the symmetric case, the negative "max distance" bin is the 0th bin.
                relative_ind_bins[(~is_same_chain) & (relative_ind < 0)] = 0

        # Convert to one-hot encoding
        return torch.nn.functional.one_hot(
            relative_ind_bins, self.num_bins
        ).float()  # [batch, seq_len, seq_len, embedding_dim]

    def get_output_dim(self) -> int:
        return self.num_bins


class IndexEmbedder(nn.Module, FeatureEncoder):
    """Produces sine / cosine index embeddings

    Shape: [batch, seq_len, embedding_dim].
    """

    def __init__(self, embedding_dim: int, max_len: int = 2056):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def forward(self, indices: torch.Tensor) -> torch.Tensor:

        embedding = get_index_embedding(
            indices, embed_size=self.embedding_dim, max_len=self.max_len
        )  # [batch, seq_len, embedding_dim]
        return embedding

    def get_output_dim(self) -> int:
        return self.embedding_dim


class TimeEmbedder(nn.Module, FeatureEncoder):
    """Layer for building sinusoidal embeddings for a given vector of timesteps."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return get_time_embedding(t, self.embed_dim)

    def get_output_dim(self) -> int:
        return self.embed_dim


class ChainBreakEncoder(nn.Module, FeatureEncoder):
    """Creates a 1D binary sequence feature indicating whether each residue is followed by a chain break.

    Chain breaks are identified using 'asym_id' integer chain IDs.
    See Boltz dataset README and/or AF3 supplement, Table 5 for more details.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, asym_id: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            asym_id: per-residue integer IDs unique for each distinct chain; shape [batch, seq_len]
            mask: per-residue binary mask indicating which residues are resolved vs unresolved (or padding); shape [batch, seq_len]

        Returns
        -------
            Binary integer tensor of shape [batch, seq_len] indicating whether (1) or not (0) each residue is followed by a chain break.
            The final residue in the final chain is also considered a chain break (1).
        """
        if mask is not None:
            # Use mask to find the last resolved residue of each chain
            asym_ids_unique = torch.unique(asym_id.flatten())
            asym_mask_expanded = asym_id.unsqueeze(-1) == asym_ids_unique.reshape(
                1, 1, -1
            )  # [batch_size, seq_len, num_unique_asym_ids]
            asym_mask_expanded *= mask.unsqueeze(-1).bool()

            mask_sample_has_asym_id = asym_mask_expanded.any(
                dim=1
            )  # Whether asym_id is present in each sample [batch_size]

            # Largest index resolved index for each asym_id. Here the subtraction and flip
            # are to get the largest index given that all resolved positions are "tied" with
            # a value of 1. Without this we would get the smallest index.
            largest_resolved_idx = (mask.shape[1] - 1) - torch.argmax(
                torch.flip(asym_mask_expanded.int(), dims=[1]), dim=1
            )  # [batch_size, num_unique_asym_ids]

            # Batch indices for each position in largest_resolved_idx.
            batch_idxs = (
                torch.arange(largest_resolved_idx.shape[0], device=asym_id.device)
                .repeat_interleave(largest_resolved_idx.shape[1])
                .reshape(largest_resolved_idx.shape[0], largest_resolved_idx.shape[1])
            )  # [batch_size, num_unique_asym_ids]

            chain_breaks = torch.zeros_like(asym_id, device=asym_id.device, dtype=torch.long)

            # Set values of chain_breaks to 1 for positions in largest_resolved_idx for each asym_id and sample.
            # mask_sample_has_asym_id is used to remove positions in each sample that do not have the corresponding asym_id.
            chain_breaks[batch_idxs[mask_sample_has_asym_id], largest_resolved_idx[mask_sample_has_asym_id]] = 1
        else:
            asym_id_diff = torch.abs(asym_id[:, 1:] - asym_id[:, :-1])  # [batch_size, seq_len - 1]
            chain_breaks = torch.cat(
                [(asym_id_diff > 0) * 1, torch.zeros([asym_id.shape[0], 1], device=asym_id.device, dtype=torch.long)],
                dim=-1,
            )  # [batch_size, seq_len]
            chain_breaks[:, -1] = 1  # Assume end of last chain is final position
        return chain_breaks  # [batch_size, seq_len]

    def get_output_dim(self) -> int:
        return 1


class PolymerBreakEncoder(nn.Module, FeatureEncoder):
    """Creates a 1D binary polymer break feature indicating whether each residue is followed by any polymer break.

    Polymer breaks are identified using `residue_index` integers and `resolved_mask` values.
    A polymer break is defined as a residue with a residue index that is not equal to or 1 greater than the previous token's residue index.
    Residues at the beginning of or just after blocks of unresolved residues are also considered polymer breaks.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, residue_index: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            residue_index: per-residue integer IDs unique for each distinct chain; shape [batch, seq_len]
            mask: per-residue binary mask indicating which residues are resolved vs unresolved; shape [batch, seq_len]

        Returns
        -------
            Binary integer tensor of shape [batch, seq_len] indicating whether (1) or not (0) each residue is followed by a polymer break.
            The final residue in the final chain is also considered a polymer break (1).
        """
        max_idx = torch.max(residue_index)

        if mask is not None:
            idxs = torch.where(
                mask.bool(), residue_index.clone(), max_idx + 2
            )  # Set unresolved indices to a unique value
        else:
            idxs = residue_index

        diff = idxs[:, 1:] - idxs[:, :-1]  # [batch, seq_len - 1]

        breaks = torch.cat(
            [
                ((diff != 1) & (diff != 0)).to(torch.long),
                torch.zeros([idxs.shape[0], 1], device=idxs.device, dtype=torch.long),
            ],
            dim=-1,
        )  # [batch, seq_len]

        if mask is not None:
            # Use mask to find the end of last chain for setting final break
            final_nonpad_idx = (mask.shape[1] - 1) - torch.argmax(torch.flip(mask.int(), dims=[1]), dim=1)
            breaks[torch.arange(final_nonpad_idx.shape[0]), final_nonpad_idx] = 1
            breaks = breaks * mask.int()
        else:
            breaks[:, -1] = 1  # Assume end of last chain is final position
        return breaks  # [batch, seq_len]

    def get_output_dim(self) -> int:
        return 1
