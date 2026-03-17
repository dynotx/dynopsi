"""Portions of this file are derived from the La Proteina project.

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the La Proteina-derived code were made by Dyno Therapeutics, Inc. (2026).
"""

from typing import Any

import einops
import torch
import torch.nn as nn

from dynopsi.model.nn.primitives import LayerNorm, Linear


class OutputHead(nn.Module):
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("OutputHead is not implemented")


class DistogramHead(OutputHead):
    """Computes distogram logits.

    This produces distogram logits that are compatible with the output shape
    of DistogramEncoder. This is an all-to-all distogram
    (i.e. pairwise distances across both within and across different atom types) of shape
    [batch_size, N_res, N_res, num_atoms, num_atoms, num_bins] where num_atoms is the
    number of atoms per residue in the protein.

    Args:
        dim_pair: channel dimension of input pair representation
        no_bins: number of output distogram bins
        use_layer_norm: whether to use layer normalization on input pair representation
        num_atoms: number of atoms per residue
    """

    def __init__(self, dim_pair: int, num_atoms: int, no_bins: int, use_layer_norm: bool):
        super().__init__()
        self.dim_pair = dim_pair
        self.no_bins = no_bins
        self.num_atoms = num_atoms
        if use_layer_norm:
            self.layer_norm = LayerNorm(dim_pair)
        self.linear = Linear(self.dim_pair, self.no_bins * self.num_atoms**2, init="final")

    def _forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_rep: [*, N_res, N_res, dim_pair] pair embedding
        Returns:
            [*, N_res, N_res, num_atoms, num_atoms, no_bins] distogram logits,
            compatible with the output shape of get_distogram primitive
        """
        if hasattr(self, "layer_norm"):
            pair_rep = self.layer_norm(pair_rep)
        logits = self.linear(pair_rep)  # [*, N_res, N_res, no_bins * num_atoms * num_atoms]
        # Symmetrize -- while taking into account the transposition of the atom indices
        # must match the transposition of the residue indices
        logits = logits.reshape(*logits.shape[:-1], self.num_atoms, self.num_atoms, -1)
        logits = logits + logits.transpose(-4, -5).transpose(-2, -3)  # symmetrize
        return logits

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        pair_rep = batch["pair_rep"]
        out = self._forward(pair_rep)
        return {"distogram_prediction": out}


class SingleRepDecoderHead(OutputHead):
    """Implementation of linear single representation output head based on Figure 5d, Geffner et al. (2025).

    This head maps to an output dimension of [*, N, output_dim], which can then be reshaped to [*, N, num_atoms, 3] if needed.

    Attributes
    ----------
        dim_single : int
            Channel dimension of input single representation
        output_dim : int
            Dimension of the output single representation
        use_layer_norm : bool
            Whether to use layer normalization on input single representation
        num_atoms : int
            Number of atoms per residue
    """

    def __init__(self, dim_single: int, output_dim: int, use_layer_norm: bool):
        super().__init__()
        self.output_dim = output_dim
        if use_layer_norm:
            self.layer_norm = LayerNorm(dim_single)
        self.linear = Linear(dim_single, output_dim, bias=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        single_rep = batch["single_rep"]  # [*, N, dim_single]
        mask = batch["mask"]  # [*, N] binary mask
        if hasattr(self, "layer_norm"):
            single_rep = self.layer_norm(single_rep)
        single_rep = self.linear(single_rep)  # [*, N, output_dim]
        single_rep = single_rep * mask.unsqueeze(-1)
        return {"out_single_rep": single_rep}


class CoordsDecoderHead(SingleRepDecoderHead):
    """Linear 3D coordinates output head.

    Handles the conversion from [B, N, output_dim] to [B, N, num_atoms, 3].

    Attributes
    ----------
        dim_single : int
            Channel dimension of input single representation
        use_layer_norm : bool
            Whether to use layer normalization on input single representation
        num_atoms : int
            Number of atoms per residue
    """

    def __init__(self, num_atoms: int, dim_single: int, use_layer_norm: bool = False):
        super().__init__(dim_single, output_dim=3 * num_atoms, use_layer_norm=use_layer_norm)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = super().forward(batch)
        out_coords = einops.rearrange(out["out_single_rep"], "b n (a e) -> b n a e", e=3)  # [*, N, num_atoms, 3]

        output = {"out_coords": out_coords}

        return output


class DynoPsiOutputHead(OutputHead):
    """Combined 3D coordinate and optional distogram output head.

    Attributes
    ----------
        config_coords_decoder_head : dict
            Dictionary containing hyperparameters for the 3D coordinate output head
        config_distogram_head : dict (Optional)
            Dictionary containing hyperparameters for the distogram output head
    """

    def __init__(
        self,
        num_atoms: int,
        config_coords_decoder_head: dict[str, Any],
        config_distogram_head: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.coords_decoder_head = CoordsDecoderHead(
            dim_single=config_coords_decoder_head["dim_single"],
            num_atoms=num_atoms,
            use_layer_norm=config_coords_decoder_head["use_layer_norm"],
        )
        if config_distogram_head is not None:
            self.distogram_head = DistogramHead(
                dim_pair=config_distogram_head["dim_pair"],
                num_atoms=num_atoms,
                no_bins=config_distogram_head["no_bins"],
                use_layer_norm=config_distogram_head["use_layer_norm"],
            )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        coords_decoder_out = self.coords_decoder_head(batch)
        distogram_out = self.distogram_head(batch) if hasattr(self, "distogram_head") else {}
        return {**coords_decoder_out, **distogram_out}
