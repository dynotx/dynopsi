"""Portions of this file are derived from the La Proteina project.

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the La Proteina-derived code were made by Dyno Therapeutics, Inc. (2026).
"""

from typing import Any

import torch
import torch.nn as nn

from dynopsi.data import constants
from dynopsi.model.nn.featurization_layers import (
    ChainBreakEncoder,
    DistogramEncoder,
    FeatureEncoder,
    IndexEmbedder,
    PairCategoricalFeatureEmbedder,
    PolymerBreakEncoder,
    SeqIdxRelativePositionEncoder,
    TimeEmbedder,
)
from dynopsi.model.nn.layers import AdaptiveLayerNorm, Transition


def _per_chain_centroid_shift(
    coords: torch.Tensor,
    chain_ids: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Per-residue chain centroid of unpadded atoms.

    Parameters
    ----------
    coords : torch.Tensor, shape [batch, N, num_atoms, 3]
    chain_ids : torch.Tensor, shape [batch, N]
    mask : torch.Tensor or None, shape [batch, N]
        Residue-level valid mask. Pad residues are excluded from the mean.

    Returns
    -------
    torch.Tensor, shape [batch, N, 3]
        Centroid of each residue's chain. Empty / all-pad chains are zero.
    """
    batch, n_tokens, n_atoms, _ = coords.shape
    residue_valid = (
        torch.ones((batch, n_tokens), dtype=torch.bool, device=coords.device) if mask is None else mask.bool()
    )
    atom_valid = residue_valid.unsqueeze(-1).expand(batch, n_tokens, n_atoms)
    batch_ix = torch.arange(batch, device=coords.device).view(batch, 1, 1).expand(batch, n_tokens, n_atoms)
    pair = torch.stack((batch_ix, chain_ids.long().unsqueeze(-1).expand(batch, n_tokens, n_atoms)), dim=-1).reshape(
        -1, 2
    )  # [batch * N * num_atoms, 2]
    unique_pairs, inverse = torch.unique(pair, dim=0, return_inverse=True)
    weights = atom_valid.reshape(-1).to(dtype=coords.dtype)
    points = coords.reshape(-1, 3)
    n_keys = unique_pairs.shape[0]
    sums = points.new_zeros((n_keys, 3))  # [n_keys, 3]
    counts = points.new_zeros((n_keys, 1))  # [n_keys, 1]
    inverse_ex = inverse.unsqueeze(-1)
    sums.scatter_add_(0, inverse_ex.expand(-1, 3), points * weights.unsqueeze(-1))
    counts.scatter_add_(0, inverse_ex, weights.unsqueeze(-1))
    centroids = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    # Atoms of one residue share a chain id, so atom 0 indexes that residue's centroid.
    return centroids[inverse.reshape(batch, n_tokens, n_atoms)[:, :, 0]]  # [batch, N, 3]


class DynoPsiSingleRepEncoder(nn.Module, FeatureEncoder):
    """Single representation encoder.

    Follows Fig. 5a, Geffner et al. (2025).
    Single representation and additional features are processed with a linear layer,
    then concatenated with the backbone embedding,
    and then the concatenation is passed through another linear layer to map to the embedding dimension.

    Parameters
    ----------
        embed_dim: int
            Dimension of the embedding space for all features
        use_self_cond: bool
            Whether to use self-conditioning in the model
        num_atoms: int
            Number of backbone atoms used to represent each residue
        use_inpaint_dgram: bool
            Whether to use inpainting features in the single rep.
            Note: The name is a bit misleading (since there is no distogram in the single rep), but it has been kept for backwards compatibility.
        center_single_rep_per_chain: bool
            If True and ``chain_ids`` is provided, subtract each chain's unpadded
            centroid from ``coords`` and ``sc_coords`` before ``bb_dense``.
    """

    def __init__(
        self,
        embed_dim: int,
        use_self_cond: bool,
        num_atoms: int,
        index_embed_dim: int | None,
        use_inpaint_dgram: bool = False,
        center_single_rep_per_chain: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_self_cond = use_self_cond
        self.index_embed_dim = index_embed_dim or embed_dim
        self.use_inpaint_dgram = use_inpaint_dgram
        self.center_single_rep_per_chain = center_single_rep_per_chain

        assert embed_dim % num_atoms == 0, "embed_dim must be divisible by num_atoms"

        self.bb_dense = nn.Linear(3, embed_dim // num_atoms)
        combined_dim = embed_dim
        self.combined_dense = nn.Linear(combined_dim, embed_dim)
        self.index_encoder = IndexEmbedder(self.index_embed_dim)
        non_coord_feats_dim = self.index_embed_dim
        self.num_atoms = num_atoms

        if use_self_cond:
            self.x_sc_embedder = nn.Linear(3, embed_dim // num_atoms)
            non_coord_feats_dim += embed_dim
        if use_inpaint_dgram:
            self.inpaint_embedder = nn.Embedding(constants.n_inpaint_features, embed_dim)
            non_coord_feats_dim += embed_dim

        self.index_sc_dense = nn.Linear(non_coord_feats_dim, embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        indices: torch.Tensor,
        sc_coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        inpaint_features: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            coords: torch.Tensor, shape [batch_size, N, num_atoms, 3]
                Backbone coordinates of the protein
            indices: torch.Tensor, shape [batch_size, N]
                Residue indices
            sc_coords: torch.Tensor, shape [batch_size, N, num_atoms, 3] (optional)
                Self-condition backbone coordinates of the protein
            mask: torch.Tensor, shape [batch_size, N] (optional)
                Mask for the backbone coordinates
            inpaint_features: torch.Tensor, shape [batch_size, N] (optional)
                Inpainting features for the structure. If None, no inpainting features are used.
            chain_ids: torch.Tensor, shape [batch_size, N] (optional)
                Per-residue chain ids (``asym_id``). Centering runs when
                ``center_single_rep_per_chain`` is on and this tensor is provided.

        Returns
        -------
            torch.Tensor, shape [batch_size, N, embed_dim]
                Embedding of the protein
        """
        if self.center_single_rep_per_chain and chain_ids is not None:
            shift = _per_chain_centroid_shift(coords, chain_ids, mask)
            coords = coords - shift.unsqueeze(-2)
            if sc_coords is not None:
                sc_coords = sc_coords - shift.unsqueeze(-2)
        coords_emb = self.bb_dense(coords)  # [batch_size, N, num_atoms, embed_dim // num_atoms]
        coords_emb = coords_emb.flatten(start_dim=-2)  # [batch_size, N, embed_dim]
        if (self.use_inpaint_dgram) and (inpaint_features is None):
            raise ValueError("inpaint_features must be provided if use_inpaint_dgram is True.")

        non_coord_feat_list = []
        index_emb = self.index_encoder(indices)  # [batch_size, N, embed_dim]
        non_coord_feat_list.append(index_emb)

        if self.use_self_cond:
            sc_emb = self.add_sc(coords_emb, sc_coords)  # [batch_size, N, embed_dim]
            non_coord_feat_list.append(sc_emb)
        if self.use_inpaint_dgram and inpaint_features is not None:
            inpaint_emb = self.inpaint_embedder(inpaint_features)
            non_coord_feat_list.append(inpaint_emb)

        non_coord_feats = torch.cat(non_coord_feat_list, dim=-1)
        non_coord_feats = self.index_sc_dense(non_coord_feats)  # [batch_size, N, embed_dim]
        feats = coords_emb + non_coord_feats  # [batch_size, N, embed_dim]
        if mask is not None:
            feats = feats * mask.unsqueeze(-1)
        return feats  # [batch_size, N, embed_dim]

    def add_sc(self, coords_emb: torch.Tensor, sc_coords: torch.Tensor | None = None) -> torch.Tensor:
        sc_feats = torch.zeros_like(coords_emb)
        if sc_coords is not None:
            sc_feats = self.x_sc_embedder(sc_coords).flatten(start_dim=-2)
        return sc_feats

    def get_output_dim(self) -> int:
        return self.embed_dim


class ConditionalFeatureGenerator(nn.Module, FeatureEncoder):
    """Combines scalar, categorical, and per-residue features and embeds them into a common dimensional space.

    Approximately follows Fig. 5b, Geffner et al. (2025).

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding space for all features
    scalar_feats : list[str] | None, optional
        Names of scalar features with shape [B, 1]
    per_residue_scalar_feats : list[str] | None, optional
        Names of per-residue scalar features with shape [B, L, 1]
    categorical_feats : dict[str, int] | None, optional
        Dictionary mapping categorical feature names to their vocabulary sizes,
        features have shape [B, 1]
    per_residue_categorical_feats : dict[str, int] | None, optional
        Dictionary mapping per-residue categorical feature names to their
        vocabulary sizes, features have shape [B, L]
    num_time_embeddings: int = 1,
        Number of time embeddings to concatenate to the conditional features.

    Returns
    -------
    torch.Tensor
        Combined feature tensor with shape [B, L, C] where:
        - B is batch size
        - L is sequence length
        - C is (#all_feats + num_time_embeddings) * embed_dim

    Notes
    -----
    Expected batch dictionary keys and their shapes:
    - scalar_feats : [B, 1] with values like [.4, 1.2, 3.1, ...]
    - categorical_feats : [B, 1] with values like [0, 1, 2, ...]
    - per-residue scalar feats : [B, L, 1] with values like [[.4, 1.2], [.5, 1.3], ...]
    - per-residue categorical feats : [B, L] with values like [[0, 1], [1, 2], ...]
    - 't' : [B] or [B, 1] with values like [.1, .2, .3, ...]
    """

    def __init__(
        self,
        embed_dim: int,
        scalar_feats: list[str] | None,
        per_residue_scalar_feats: list[str] | None,
        categorical_feats: dict[str, int] | None,
        per_residue_categorical_feats: dict[str, int] | None,
        num_time_embeddings: int,
    ):
        super().__init__()

        self.scalar_feats = scalar_feats or []
        self.per_residue_scalar_feats = per_residue_scalar_feats or []

        self.categorical_feats = categorical_feats or {}
        self.per_residue_categorical_feats = per_residue_categorical_feats or {}

        self.embed_dim = embed_dim
        self.time_embedder = TimeEmbedder(embed_dim)
        self.num_time_embeddings = num_time_embeddings
        self.embedders = nn.ModuleDict(
            {
                **{name: nn.Linear(1, embed_dim, bias=False) for name in self.scalar_feats},
                **{name: nn.Linear(1, embed_dim, bias=False) for name in self.per_residue_scalar_feats},
                **{name: nn.Embedding(vocab, embed_dim) for name, vocab in self.categorical_feats.items()},
                **{name: nn.Embedding(vocab, embed_dim) for name, vocab in self.per_residue_categorical_feats.items()},
            }
        )

    def get_output_dim(self) -> int:
        return self.embed_dim * (len(self.embedders) + self.num_time_embeddings)

    def get_feature_names(self) -> list[str]:
        """Return a list of the names of all features specified in the constructor."""
        return sorted(list(self.embedders.keys()))

    def broadcast_feats(self, feats: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Broadcast features scalar features across the sequence length dimension.

        Parameters
        ----------
        feats : torch.Tensor
            Features to broadcast, shape [batch_size, embed_dim]
        seq_len : int
            Sequence length

        Returns
        -------
        torch.Tensor
            Broadcasted features, shape [batch_size, seq_len, embed_dim]
        """
        return feats.unsqueeze(-2).expand(-1, seq_len, -1)

    def forward(
        self,
        conditional_feats: dict[str, torch.Tensor],
        t: torch.Tensor,
        seq_len: int,
        mask: torch.Tensor | None = None,
        t_ss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditional_feats["t"] = t
        needed = (
            self.scalar_feats
            + self.per_residue_scalar_feats
            + list(self.categorical_feats)
            + list(self.per_residue_categorical_feats)
        )
        missing = [k for k in needed if k not in conditional_feats]
        if missing:
            raise KeyError(f"Batch missing keys: {missing}")

        features = []
        for key in self.get_feature_names():
            module = self.embedders[key]
            if key == "noised_mask":
                feat = module(conditional_feats[key].long())
            else:
                feat = module(conditional_feats[key])

            if (
                key in self.scalar_feats or key in self.categorical_feats
            ):  # Broadcast the sequence level features across the each residue: [batch_size, 1] -> [batch_size, seq_len, 1]
                feat = self.broadcast_feats(feat, seq_len)
            features.append(feat)

        features.append(self.broadcast_feats(self.time_embedder(conditional_feats["t"]), seq_len))
        if t_ss is not None:
            features.append(self.broadcast_feats(self.time_embedder(t_ss), seq_len))
        feats = torch.cat(features, dim=-1)
        if mask is not None:
            feats = feats * mask.unsqueeze(-1)
        return feats


class DynoPsiPairRepEncoder(nn.Module, FeatureEncoder):
    """Creates the pair representation in Fig. 5c, Geffner et al. (2025).

    This module concatenates the following features:
    - distogram
    - sequence separation
    - self-condition distogram, optionally
    - inpainting distogram, optionally (new)
    - inpainting features, optionally (new)
    - noised_mask, optionally (new)
    And then modulates them with the time embedding.

    Notes
    -----
    - `pair_inpaint_embed_dim` refers to the dimension for embedding the inpainting features, not the dimension of the inpainting distogram.
    """

    def __init__(
        self,
        time_dim: int,
        num_atoms: int,
        use_self_cond: bool,
        num_seq_sep_bins: int,
        num_distance_bins: int,
        num_distance_bins_sc: int,
        use_inpaint_dgram: bool = False,
        pair_noised_embed_dim: int = 0,
        pair_inpaint_embed_dim: int = 0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.use_self_cond = use_self_cond
        self.use_inpaint_dgram = use_inpaint_dgram
        self.pair_noised_embed_dim = pair_noised_embed_dim
        self.pair_inpaint_embed_dim = pair_inpaint_embed_dim

        if pair_noised_embed_dim > 0:
            # One dimension for each combination of noised/unnoised mask values
            self.noised_mask_embedder = PairCategoricalFeatureEmbedder(2, pair_noised_embed_dim)
        if pair_inpaint_embed_dim > 0:
            # One dimension for each combination of inpaint feature values
            self.inpaint_feature_embedder = PairCategoricalFeatureEmbedder(
                constants.n_inpaint_features, pair_inpaint_embed_dim
            )

        self.dgram = DistogramEncoder(num_distance_bins, num_atoms)
        if use_self_cond:
            self.dgram_sc = DistogramEncoder(num_distance_bins_sc, num_atoms)
        if self.use_inpaint_dgram:
            self.dgram_inpaint = DistogramEncoder(num_distance_bins, num_atoms)

        self.seq_idx_rel_pos_encoder = SeqIdxRelativePositionEncoder(num_seq_sep_bins)
        self.time_embedder = nn.Sequential(
            TimeEmbedder(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.LayerNorm(time_dim),
        )

        ada_ln_input_dim = (
            self.dgram.get_output_dim()
            + self.seq_idx_rel_pos_encoder.get_output_dim()
            + (self.dgram_sc.get_output_dim() if self.use_self_cond else 0)
            + (self.dgram_inpaint.get_output_dim() if self.use_inpaint_dgram else 0)
            + self.pair_noised_embed_dim
            + self.pair_inpaint_embed_dim
        )
        self.ada_ln = AdaptiveLayerNorm(dim=ada_ln_input_dim, dim_cond=time_dim)

    def add_sc(self, coords: torch.Tensor, sc_coords: torch.Tensor | None = None) -> torch.Tensor:
        if sc_coords is None:
            bs, seq_len = coords.shape[:2]
            return torch.zeros((bs, seq_len, seq_len, self.dgram_sc.get_output_dim()), device=coords.device)
        else:
            return self.dgram_sc(sc_coords)

    def add_inpaint(
        self,
        true_coords: torch.Tensor | None,
        inpaint_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Add the inpainting distograms to the feature list.

        Parameters
        ----------
        true_coords : torch.Tensor
            Coordinates of the true (x1) structure, shape [batch_size, # masked regions, seq_len, num_atoms, 3]
            Note: During inference, these will be the coordinates that come out of the data loader. So new chains will have random coordinates.
            If inpainting is used, all inpainted residues will have random coordinates instead of the true structure coordinates.
            All context residues in existing chains will have the true structure coordinates.

        inpaint_features : torch.Tensor
            Inpainting features, shape [batch_size, seq_len]
        """
        if true_coords is None:
            raise ValueError("true_coords must be provided if use_inpaint_dgram is True")
        if inpaint_features is None:
            raise ValueError("inpaint_features must be provided if use_inpaint_dgram is True")

        dgram = self.dgram_inpaint(true_coords)

        # Convert inpaint_feature to masks
        inpaint_masks = torch.stack(
            [
                (inpaint_features == constants.InpaintFeature.BINDER_CONTEXT.value),
                (inpaint_features == constants.InpaintFeature.TARGET_CONTEXT.value),
            ],
            dim=1,
        )  # [batch_size, 2, seq_len]

        # Compute outer product of each mask with itself and sum over the masks
        # This results in a mask that is true for all residue pairs that have either both binder context or both target context, and false otherwise.
        dgram_mask = (inpaint_masks.unsqueeze(-1) * inpaint_masks.unsqueeze(-2)).sum(dim=1).bool()
        return dgram * dgram_mask.unsqueeze(-1)  # [batch_size, seq_len, seq_len, num_distance_bins]

    def forward(
        self,
        coords: torch.Tensor,
        indices: torch.Tensor,
        t: torch.Tensor,
        sc_coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        true_coords: torch.Tensor | None = None,
        inpaint_features: torch.Tensor | None = None,
        noised_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feat_list = []
        feat_list.append(self.dgram(coords))
        feat_list.append(self.seq_idx_rel_pos_encoder(indices, chain_ids))
        if self.use_self_cond:
            feat_list.append(self.add_sc(coords, sc_coords))
        if self.use_inpaint_dgram:
            feat_list.append(self.add_inpaint(true_coords, inpaint_features))
        if self.pair_noised_embed_dim > 0:
            feat_list.append(self.noised_mask_embedder(noised_mask))
        if self.pair_inpaint_embed_dim > 0:
            feat_list.append(self.inpaint_feature_embedder(inpaint_features))

        feats = torch.cat(feat_list, dim=-1).to(coords)  # [batch_size, seq_len, seq_len, ada_ln_input_dim]
        t_encoding = self.time_embedder(t)  # [batch_size, time_dim]
        t_encoding = t_encoding.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, time_dim]
        feats_normed = self.ada_ln(feats, t_encoding)
        if mask is not None:
            pair_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1)  # [batch_size, seq_len, seq_len, 1]
            feats_normed = feats_normed * pair_mask

        return feats_normed

    def get_output_dim(self) -> int:
        return (
            self.dgram.get_output_dim()
            + self.seq_idx_rel_pos_encoder.get_output_dim()
            + (self.dgram_sc.get_output_dim() if self.use_self_cond else 0)
            + (self.dgram_inpaint.get_output_dim() if self.use_inpaint_dgram else 0)
            + self.pair_noised_embed_dim
            + self.pair_inpaint_embed_dim
        )


class DynoPsiSingleConditionalEmbedder(nn.Module):
    """Embedder class that combines the single and conditional representations.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding space for all features
    num_atoms : int
        Number of atoms to consider per residue.
    use_self_cond : bool
        Whether to use self-conditioning in the model
    scalar_feats : list[str] | None, optional
        Names of scalar features with shape [B, 1], by default None
    categorical_feats : dict[str, int] | None, optional
        Dictionary mapping categorical feature names to their vocabulary sizes,
        features have shape [B, 1], by default None
    per_residue_scalar_feats : list[str] | None, optional
        Names of per-residue scalar features with shape [B, L, 1], by default None
    per_residue_categorical_feats : dict[str, int] | None, optional
        Dictionary mapping per-residue categorical feature names to their
        vocabulary sizes, features have shape [B, L], by default None
    cond_embed_dim: int
        Dimension of the conditional feature embedding space; if not provided, embed_dim is used.
    index_embed_dim: int
        Dimension of the position index embedding space; if not provided, embed_dim is used.
    num_time_embeddings: int
        Number of time embeddings to concatenate to the conditional features.
    use_inpaint_dgram: bool
        Whether to use inpainting features in the single rep (and inpainting distogram in the pair rep if present).
        Note: The name is a bit misleading (since there is no distogram in the single rep), but it has been kept for backwards compatibility.
    use_pre_cond_feature_mlp1_layer_norm: bool
        Whether to apply a layer norm before the first conditional feature MLP transition block.
    use_pre_cond_feature_mlp2_layer_norm: bool
        Whether to apply a layer norm before the second conditional feature MLP transition block.
    mask_conditional_mlp_input: bool
        Whether to zero-out padding positions in the conditional features before the initial dense projection.
    mask_conditional_mlp_output: bool
        Whether to zero-out padding positions in the conditional features after the initial dense projection.
    center_single_rep_per_chain: bool
        If True and chain ids are provided, subtract each chain's unpadded
        centroid from ``coords`` and ``sc_coords`` before ``bb_dense``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_atoms: int,
        use_self_cond: bool,
        scalar_feats: list[str] | None,
        categorical_feats: dict[str, int] | None,
        per_residue_scalar_feats: list[str] | None,
        per_residue_categorical_feats: dict[str, int] | None,
        cond_embed_dim: int | None,
        index_embed_dim: int | None,
        num_time_embeddings: int,
        use_inpaint_dgram: bool = False,
        use_pre_cond_feature_mlp1_layer_norm: bool = False,
        use_pre_cond_feature_mlp2_layer_norm: bool = False,
        mask_conditional_mlp_input: bool = False,
        mask_conditional_mlp_output: bool = False,
        center_single_rep_per_chain: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_atoms = num_atoms
        self.use_self_cond = use_self_cond
        self.cond_embed_dim = cond_embed_dim or embed_dim
        self.index_embed_dim = index_embed_dim or embed_dim
        self.use_inpaint_dgram = use_inpaint_dgram
        self.use_pre_cond_feature_mlp1_layer_norm = use_pre_cond_feature_mlp1_layer_norm
        self.use_pre_cond_feature_mlp2_layer_norm = use_pre_cond_feature_mlp2_layer_norm
        self.mask_conditional_mlp_input = mask_conditional_mlp_input
        self.mask_conditional_mlp_output = mask_conditional_mlp_output

        self.single_rep_encoder = DynoPsiSingleRepEncoder(
            self.embed_dim,
            use_self_cond,
            num_atoms=num_atoms,
            index_embed_dim=index_embed_dim,
            use_inpaint_dgram=use_inpaint_dgram,
            center_single_rep_per_chain=center_single_rep_per_chain,
        )

        self.chain_break_encoder = ChainBreakEncoder()
        self.polymer_break_encoder = PolymerBreakEncoder()

        self.conditional_feature_generator = ConditionalFeatureGenerator(
            self.cond_embed_dim,
            scalar_feats=scalar_feats,
            categorical_feats=categorical_feats,
            per_residue_scalar_feats=per_residue_scalar_feats,
            per_residue_categorical_feats=per_residue_categorical_feats,
            num_time_embeddings=num_time_embeddings,
        )
        self.conditional_feature_dense = nn.Linear(
            self.conditional_feature_generator.get_output_dim(), self.cond_embed_dim, bias=False
        )
        self.conditional_feature_mlp_1 = Transition(
            dim=self.cond_embed_dim,
            expansion_factor=2,
            use_pre_layer_norm=use_pre_cond_feature_mlp1_layer_norm,
            mask_input=mask_conditional_mlp_input,
            mask_output=mask_conditional_mlp_output,
        )
        self.conditional_feature_mlp_2 = Transition(
            dim=self.cond_embed_dim,
            expansion_factor=2,
            use_pre_layer_norm=use_pre_cond_feature_mlp2_layer_norm,
            mask_input=mask_conditional_mlp_input,
            mask_output=mask_conditional_mlp_output,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        indices = batch["token_index"]
        t = batch["t"]
        sc_coords = batch.get("sc_coords", None)
        chain_ids = batch.get("asym_id", None)
        t_ss = batch.get("t_ss", None)
        mask = batch.get("mask", None)
        inpaint_features = batch.get("inpaint_feature", None)
        if chain_ids is not None and "chain_breaks" in self.conditional_feature_generator.get_feature_names():
            batch["chain_breaks"] = self.chain_break_encoder(chain_ids, mask)

        residue_indices = batch["residue_index"]
        batch["polymer_breaks"] = self.polymer_break_encoder(residue_indices, mask)

        single_rep = self.single_rep_encoder(
            coords,
            indices,
            sc_coords,
            mask=mask,
            inpaint_features=inpaint_features,
            chain_ids=chain_ids,
        )

        if mask is None:
            mask = torch.ones(single_rep.shape[:-1], device=single_rep.device).long()  # [batch_size, seq_len]
        single_mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        conditional_feats_embedding = self.conditional_feature_generator(
            batch, t, indices.shape[1], t_ss=t_ss, mask=mask
        )

        if self.mask_conditional_mlp_input:
            conditional_feature_dense_input = conditional_feats_embedding * single_mask
        else:
            conditional_feature_dense_input = conditional_feats_embedding
        conditional_feats_embedding = self.conditional_feature_dense(conditional_feature_dense_input)
        if self.mask_conditional_mlp_output:
            conditional_feats_embedding = conditional_feats_embedding * single_mask

        conditional_feats_embedding = self.conditional_feature_mlp_2(
            self.conditional_feature_mlp_1(conditional_feats_embedding, mask=single_mask), mask=single_mask
        )

        return {
            "single_rep": single_rep * single_mask,
            "conditional_feats": conditional_feats_embedding * single_mask,
            "mask": mask,
        }


class DynoPsiSingleConditionalPairEmbedder(DynoPsiSingleConditionalEmbedder):
    """Embedder class that combines the single and conditional representations with the pair representation.

    Parameters
    ----------
    time_dim : int
        Dimension of the time embedding space
    pair_embed_dim: int | None
        Dimension of the pair representation embedding space; if not provided, embed_dim is used.
    num_seq_sep_bins : int
        Number of bins for the relative position encoder
    num_distance_bins : int
        Number of distance bins for the distogram computed over noisy (xt) coordinates
    num_distance_bins_sc : int
        Number of distance bins for the distogram computed over self-conditioning (x1_pred) coordinates
    pair_noised_embed_dim: int
        Dimension of the noised mask embedding space in the pair rep. If 0, no noised mask is used.
    pair_inpaint_embed_dim: int
        Dimension of the inpaint feature embedding space in the pair rep. If 0, no inpaint features are used.
    """

    def __init__(
        self,
        time_dim: int,
        *parent_args: Any,
        pair_embed_dim: int | None,
        num_seq_sep_bins: int,
        num_distance_bins: int,
        num_distance_bins_sc: int,
        pair_noised_embed_dim: int = 0,
        pair_inpaint_embed_dim: int = 0,
        **parent_kwargs: Any,
    ):
        super().__init__(*parent_args, **parent_kwargs)

        if not self.use_inpaint_dgram and pair_inpaint_embed_dim > 0:
            raise ValueError("pair_inpaint_embed_dim must be 0 if use_inpaint_dgram is False")

        self.pair_embed_dim = pair_embed_dim or self.embed_dim
        self.pair_noised_embed_dim = pair_noised_embed_dim
        self.pair_inpaint_embed_dim = pair_inpaint_embed_dim

        self.pair_rep_encoder = DynoPsiPairRepEncoder(
            time_dim=time_dim,
            num_atoms=self.num_atoms,
            use_self_cond=self.use_self_cond,
            num_seq_sep_bins=num_seq_sep_bins,
            num_distance_bins=num_distance_bins,
            num_distance_bins_sc=num_distance_bins_sc,
            use_inpaint_dgram=self.use_inpaint_dgram,
            pair_noised_embed_dim=self.pair_noised_embed_dim,
            pair_inpaint_embed_dim=self.pair_inpaint_embed_dim,
        )
        self.pair_linear = nn.Linear(self.pair_rep_encoder.get_output_dim(), self.pair_embed_dim, bias=False)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        coords = batch["coords"]
        indices = batch["token_index"]
        t = batch["t"]
        sc_coords = batch.get("sc_coords", None)
        true_coords = batch.get("true_coords", None)
        inpaint_features = batch.get("inpaint_feature", None)
        chain_ids = batch.get("asym_id", None)
        noised_mask = batch.get("noised_mask", None)

        embedded_batch = super().forward(batch)

        mask = embedded_batch["mask"]
        pair_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(-1)  # [batch_size, seq_len, seq_len, 1]

        pair_rep = self.pair_rep_encoder(
            coords,
            indices,
            t,
            sc_coords,
            chain_ids=chain_ids,
            true_coords=true_coords,
            inpaint_features=inpaint_features,
            noised_mask=noised_mask,
        )

        pair_rep = self.pair_linear(pair_rep)

        return {**embedded_batch, "pair_rep": pair_rep * pair_mask}
