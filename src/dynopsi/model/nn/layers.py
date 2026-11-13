"""Portions of this file are derived from the OpenFold, La Proteina, and Flash IPA projects.

OpenFold
Copyright (c) 2024 AlQuraishi Laboratory
Licensed under the Apache License, Version 2.0.
Source: https://github.com/aqlaboratory/openfold

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Flash IPA
Copyright (c) 2025 anonymous
Licensed under the MIT License.
Source: https://github.com/flagshippioneering/flash_ipa

Modifications to the third-party derived code were made by Dyno Therapeutics, Inc. (2026).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial, partialmethod
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint as _checkpoint

from dynopsi.model.nn.primitives import (
    LayerNorm,
    Linear,
    SwiGLU,
)

# use_reentrant=False is required for DDP compatibility.
# determinism_check="none" suppresses a false-positive metadata mismatch error that
# arises from the bfloat16 autocast context interaction in LayerNorm on PyTorch 2.6+.
checkpoint = partial(_checkpoint, use_reentrant=False, determinism_check="none")


def _is_fp16_enabled() -> bool:
    """Check if FP16 precision is enabled.

    Implementation taken without modification from OpenFold precision_utils:
    https://github.com/aqlaboratory/openfold/blob/e938c184a291bf053af3b14c1e3e8bb29aee57e2/openfold/utils/precision_utils.py#L14C1-L23C24
    """
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    return fp16_enabled


max_neg_value = lambda x: torch.finfo(x.dtype).min  # noqa


class AdaptiveLayerNorm(torch.nn.Module):
    """AF3-style adaptive layer norm

    Based on proteinfoundation.nn.modules.adaptive_ln_scale.AdaptiveLayerNorm in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).

    Args:
        dim: dimension of primary input getting rescaled
        dim_cond: dimension of secondary input used to predict gamma/beta rescaling parameters
    """

    def __init__(self, *, dim: int, dim_cond: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_cond = torch.nn.LayerNorm(dim_cond, bias=False)
        self.to_gamma = torch.nn.Sequential(torch.nn.Linear(dim_cond, dim), torch.nn.Sigmoid())
        self.to_beta = torch.nn.Linear(dim_cond, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, dim]
        cond: torch.Tensor,  # [batch, seq_len, dim_cond]
        mask: torch.Tensor | None = None,  # [batch, seq_len]
    ) -> torch.Tensor:  # [batch, seq_len, dim]
        """
        Args:
            x: primary input to be rescaled, [batch, seq_len, dim]
            cond: secondary conditioning input, [batch, seq_len, dim_cond]
            mask: binary mask, [batch, seq_len].

        Returns
        -------
            Representation after adaptive layer norm, shape [batch, seq_len, dim].
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        output = normed * gamma + beta
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        return output


class AttentionPairBias(torch.nn.Module):
    """Masked attention with pair bias and gating.

    Adapted from protein_learning.network.modules.node_block.NodeAttention in DockGPT (https://github.com/MattMcPartlon/protein-docking).

    Parameters
    ----------
    node_dim : int
        dimension of node features; in our case, the feature dimension of the input single representation
    dim_head : int
        dimension of each attention head
    heads : int
        number of attention heads
    bias : bool
        whether to use bias in linear projection layers for query, key, and value
    dim_out : int | None (optional, default None)
        dimension of output features; by default, this is the same as node_dim and thus preserves the feature dimension of the input single representation
    attn_bias_mode : str (optional, default "none")
        which representation type to use to bias the self-attention weights, can be either "pair" or "none"
    dim_pair : int | None (optional, default None)
        dimension of pair features; has to be None if attn_bias_mode is "none"
    use_qk_layernorm : bool (optional, default True)
        whether to apply layer norm to query and key before attention
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int | None = None,
        attn_bias_mode: str = "none",
        dim_pair: int | None = None,
        use_qk_layernorm: bool = True,
        use_flash_attn: bool = False,
        flash_attn_dtype: str = "bf16",
    ):
        super().__init__()
        if use_flash_attn:
            assert dim_pair is None, "Cannot use flash attention with pair bias"
            flash_attn_dtype_dict = {"fp16": torch.float16, "bf16": torch.bfloat16}
            if flash_attn_dtype not in flash_attn_dtype_dict:
                raise ValueError(
                    f"Invalid option for flash_attn_dtype '{flash_attn_dtype}': must be one of {list(flash_attn_dtype_dict.keys())}."
                )
            self.flash_attn_dtype = flash_attn_dtype_dict[flash_attn_dtype]
        self.use_flash_attn = use_flash_attn
        self.attn_bias_mode = attn_bias_mode
        inner_dim = dim_head * heads
        dim_out = dim_out if dim_out is not None else node_dim
        self.node_dim, self.dim_pair = node_dim, dim_pair
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = torch.nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = torch.nn.Linear(node_dim, inner_dim)
        self.to_out_node = torch.nn.Linear(inner_dim, dim_out)
        self.node_norm = torch.nn.LayerNorm(node_dim)

        if attn_bias_mode != "none":
            if dim_pair is None:
                raise ValueError("`dim_pair` must be specified if `attn_bias_mode` is not 'none'")
            self.to_bias = torch.nn.Sequential(
                torch.nn.LayerNorm(dim_pair), torch.nn.Linear(dim_pair, heads, bias=False)
            )
        else:
            self.to_bias = None  # type: ignore

        self.use_qk_layernorm = use_qk_layernorm
        if self.use_qk_layernorm:
            self.q_layer_norm = torch.nn.LayerNorm(inner_dim)
            self.k_layer_norm = torch.nn.LayerNorm(inner_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        pair_feats: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-head scalar Attention Layer.

        Args:
            node_feats: scalar features
                Single representation of shape (batch, seq_len, node_dim)
            pair_feats: pair features
                Pair representation of shape (batch, seq_len, seq_len, pair_dim)
            mask: boolean single_mask of shape (batch, seq_len)
                Indicates existence/missingness of atoms in the single representation

        Returns
        -------
            updated node feats of shape (batch, seq_len, node_dim)
        """
        B, N, _ = node_feats.shape

        assert self.to_bias is not None or pair_feats is None

        if self.attn_bias_mode != "none" and pair_feats is None:
            raise ValueError("`pair_feats` is None when `attn_bias_mode` is *not* `none`.")

        node_feats, h = self.node_norm(node_feats), self.heads

        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        if self.use_qk_layernorm:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)
        g = self.to_g(node_feats)

        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...") if pair_feats is not None else 0
        )  # [B, heads, N, N]

        q, k, v, g = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g))

        if self.use_flash_attn:
            assert b == 0, "Cannot use flash attention with pair bias"
            attn_feats = self._flash_attn(q, k, v, mask)
        else:
            attn_feats = self._attn(q, k, v, b, mask)

        attn_feats = rearrange(torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h)

        return self.to_out_node(attn_feats)

    def _attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor | int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            q : torch.Tensor
                [B, H, N, Q], where Q is some feature dimension
            k : torch.Tensor
                [B, H, N, Q]
            v : torch.Tensor
                [B, H, N, V], where V is some feature dimension
            b : torch.Tensor | int (if int, it is equal to zero)
                [B, H, N, N]
            mask : torch.Tensor | None, default=None
                [B, N]

        Returns
        -------
            attn_res : torch.Tensor
            [B, H, N, V]
        """
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            # Pair mask logic is copied from OpenFold InvariantPointAttention
            mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # turn into a pair mask
            mask = rearrange(mask, "b i j -> b () i j")
            mask = mask.to(torch.bool)  # Ensure boolean type for masked_fill
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return torch.einsum("b h i j, b h j d -> b h i d", attn, v)

    def _flash_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Adapted from https://github.com/flagshippioneering/flash_ipa.

        Parameters
        ----------
            q : torch.Tensor
                [B, H, N, Q], where Q is some feature dimension
            k : torch.Tensor
                [B, H, N, Q]
            v : torch.Tensor
                [B, H, N, V], where V is some feature dimension
            mask : torch.Tensor | None, default=None
                [B, N]

        Returns
        -------
            attn_res : torch.Tensor
            [B, H, N, V]
        """
        if mask is None:
            mask = torch.ones((q.shape[0], q.shape[2]), device=q.device)

        q, k, v = map(lambda t: rearrange(t, "b h n d -> b n h d"), (q, k, v))  # rearrange tokens and heads

        qkv = torch.cat([q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)], dim=2)
        (
            qkv,
            indices,
            cu_seqlens,
            max_seqlen,
            _,
        ) = self._preprocess_flashattn_input(qkv, mask)

        if qkv.dtype != self.flash_attn_dtype:
            qkv = qkv.to(self.flash_attn_dtype)

        # importing here just so one doesn't have to have flash-attn installed if they don't want to use it
        from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore

        attn_res = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, softmax_scale=1)

        attn_res = self._postprocess_flashattn_output(
            attn_res,
            indices=indices,
            batch=q.shape[0],
            seqlen=q.shape[1],
        )

        attn_res = attn_res.float()  # cast back to float32

        attn_res = rearrange(attn_res, "b n h d -> b h n d")  # rearrange tokens and heads

        return attn_res

    def _preprocess_flashattn_input(
        self, qkv: torch.Tensor, attention_mask: torch.Tensor, unused_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | float, torch.Tensor]:
        """
        Process attention inputs (qkv: queries, keys, and values) to be used by flash-attention 2.
        Taken from https://github.com/flagshippioneering/flash_ipa.

        Arguments:
            qkv: (batch, seqlen, ...)
            attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
            unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
        Return:
            qkv: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
            indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
            cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
            max_seqlen_in_batch: int
            seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
        """
        all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
        seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
        used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return (
            rearrange(qkv, "b s ... -> (b s) ...")[indices],
            indices,
            cu_seqlens,
            max_seqlen_in_batch,
            used_seqlens_in_batch,
        )

    def _postprocess_flashattn_output(
        self, hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int
    ) -> torch.Tensor:
        """
        Process the output of flash-attention 2 to be a standard attention map.
        Taken from https://github.com/flagshippioneering/flash_ipa.

        Arguments:
            attn_res: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
            indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
            batch: int, batch size for the padded sequence.
            seqlen: int, maximum sequence length for the padded sequence.
        Return:
            attn_res: (batch, seqlen, ...)
        """
        dim = hidden_states.shape[1:]
        output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
        output[indices] = hidden_states
        return rearrange(output, "(b s) ... -> b s ...", b=batch)


class AdaptiveScale(torch.nn.Module):
    """Adaptively scales primary input based on secondary input.

    Based on proteinfoundation.nn.modules.adaptive_ln_scale.AdaptiveOutputScale in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    def __init__(self, *, dim: int, dim_cond: int):
        super().__init__()
        self.to_gamma = torch.nn.Sequential(Linear(dim_cond, dim, init="gating"), torch.nn.Sigmoid())

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: primary input to be rescaled, [batch, seq_len, dim]
            cond: secondary conditioning input, [batch, seq_len, dim_cond]
            mask: binary mask, [batch, seq_len]

        Returns
        -------
            Representation after adaptive rescaling, shape [batch, seq_len, dim].
        """
        gamma = self.to_gamma(cond)
        output = x * gamma
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        return output


class AdaptiveBiasedMultiHeadAttention(torch.nn.Module):
    """Adaptive multihead attention module with adaptive layer norm applied to the input and adaptive scaling applied to the output.

    Adapted from proteinfoundation.nn.modules.pair_bias_attn.MultiHeadBiasedAttentionADALN_MM in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).

    Parameters
    ----------
    dim_seq : int
        Dimensionality of the single_rep.
    dim_cond : int
        Dimensionality of the conditioning features.
    n_heads : int
        Number of heads for the self-attention operation.
    dim_head : int
        Dimensionality of each slef-attention head.
    attn_bias_mode : str (optional)
        The representation type with which the self-attention operation is biased, can be "pair" or "none"
    dim_pair : int | None (optional)
        Dimensionality of the pair representation.
    use_qkv_bias : bool (optional)
        Whether to use bias in the linear projection to construct queries, keys and values for the self-attention operation.
    use_qk_layernorm : bool (optional)
        Whether to apply a layernorm to queries and keys in the self-attention operation.
    """

    def __init__(
        self,
        dim_seq: int,
        dim_cond: int,
        n_heads: int,
        dim_head: int,
        attn_bias_mode: str = "none",
        dim_pair: int | None = None,
        use_qkv_bias: bool = True,
        use_qk_layernorm: bool = True,
        use_flash_attn: bool = False,
        flash_attn_dtype: str = "bf16",
    ):
        super().__init__()

        if (attn_bias_mode != "none") and (dim_pair is None):
            raise ValueError(f"Must specify `dim_pair` if attn_bias_mode is `{attn_bias_mode}")

        self.attn_bias_mode = attn_bias_mode
        self.adaptive_layernorm = AdaptiveLayerNorm(dim=dim_seq, dim_cond=dim_cond)
        self.attention = AttentionPairBias(
            dim_seq,
            dim_head,
            n_heads,
            bias=use_qkv_bias,
            dim_out=dim_seq,
            attn_bias_mode=attn_bias_mode,
            dim_pair=dim_pair,
            use_qk_layernorm=use_qk_layernorm,
            use_flash_attn=use_flash_attn,
            flash_attn_dtype=flash_attn_dtype,
        )
        self.adaptive_scale = AdaptiveScale(dim=dim_seq, dim_cond=dim_cond)

    def forward(
        self,
        seq_rep: torch.Tensor,
        seq_cond: torch.Tensor,
        pair_rep: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute and return updated sequence representation.

        Args:
            seq_rep: sequence representation input, [batch, seq_len, dim_seq]
            seq_cond: sequence conditioning input, [batch, seq_len, dim_cond]
            pair_rep: pair representation, [batch, seq_len, seq_len, dim_pair]
            mask: Binary mask, [batch, seq_len]
        """
        if (self.attn_bias_mode != "none") and (pair_rep is None):
            raise ValueError(
                "Inconsistent constructor and forward pass: expected `pair_rep` to be *not* None in the forward pass."
            )

        if (self.attn_bias_mode == "none") and (pair_rep is not None):
            raise ValueError(
                "Inconsistent constructor and forward pass: expected `pair_rep` to be None in the forward pass."
            )

        seq_rep = self.adaptive_layernorm(seq_rep, seq_cond, mask)
        seq_rep = self.attention(seq_rep, pair_feats=pair_rep, mask=mask)

        seq_rep = self.adaptive_scale(seq_rep, seq_cond)
        return seq_rep * mask.unsqueeze(-1)


class Transition(torch.nn.Module):
    """AF3-style transition layer.

    Adapted from proteinfoundation.nn.modules.seq_transition_af3.Transition in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).

    Args:
        dim: feature dimension of input
        expansion_factor: factor by which dim is multiplied to obtain hidden feature dimension
    """

    def __init__(
        self,
        *,
        dim: int,
        expansion_factor: int = 2,
        use_pre_layer_norm: bool = False,
        mask_input: bool = False,
        mask_output: bool = False,
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.use_pre_layer_norm = use_pre_layer_norm
        self.mask_input = mask_input
        self.mask_output = mask_output

        if self.use_pre_layer_norm:
            self.layer_norm = LayerNorm(dim)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_inner * 2, bias=False), SwiGLU(), torch.nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
        self,
        x: torch.Tensor,  # [..., dim]
        mask: torch.Tensor | None = None,  # [...] or [..., 1]
    ) -> torch.Tensor:  # [..., dim]
        if self.mask_input and mask is not None:
            x = x * mask

        if self.use_pre_layer_norm:
            x = self.layer_norm(x)

        x = self.ff(x)

        if self.mask_output and mask is not None:
            x = x * mask

        return x


class AdaptiveTransition(torch.nn.Module):
    """Adaptive layer norm followed by a transition layer and adaptive scaling.

    Adapted from proteinfoundation.nn.modules.seq_transition_af3.TransitionADALN in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    def __init__(self, *, dim: int, dim_cond: int, expansion_factor: int = 2):
        super().__init__()
        self.transition = Transition(dim=dim, expansion_factor=expansion_factor)
        self.adaptive_layernorm = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.adaptive_scale = AdaptiveScale(dim=dim, dim_cond=dim_cond)

    def forward(
        self,
        x: torch.Tensor,  # [..., dim]
        cond: torch.Tensor,  # [..., dim_cond]
        mask: torch.Tensor | None = None,  # [seq_len]
    ) -> torch.Tensor:  # [..., dim]
        x = self.adaptive_layernorm(x, cond, mask)
        x = self.transition(x)
        return self.adaptive_scale(x, cond, mask)


class AdaptiveAttentionTransitionBlock(torch.nn.Module):
    """Connects adaptive multihead attention and adaptive transition layers with residual connections.

    Adapted from proteinfoundation.nn.modules.seq_attention_transition_block.AttentionTransitionBlock in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).

    Parameters
    ----------
    dim_seq : int
        Dimensionality of the single_rep.
    dim_cond : int
        Dimensionality of the conditioning features.
    n_heads : int
        Number of heads for the self-attention operation.
    dim_head : int
        Dimensionality of each slef-attention head.
    attn_bias_mode : str (optional)
        The representation type with which the self-attention operation is biased, can be "pair" or "none"
    dim_pair : int | None (optional)
        Dimensionality of the pair representation.
    expansion_factor : int (optional)
        Factor by which dim is multiplied to obtain hidden feature dimension within the AdptiveTransition
    use_qkv_bias : bool (optional)
        Whether to use bias in the linear projection to construct queries, keys and values for the self-attention operation.
    use_qk_layernorm : bool (optional)
        Whether to apply a layernorm to queries and keys in the self-attention operation.
    use_flash_attn : bool (optional)
        Whether to use flash-attention 2.
    flash_attn_dtype : str (optional)
        The data-type to use for flash-attention 2.
    """

    def __init__(
        self,
        *,
        dim_single: int,
        dim_cond: int,
        n_heads: int,
        dim_head: int,
        attn_bias_mode: str = "none",
        dim_pair: int | None = None,
        expansion_factor: int = 2,
        use_qkv_bias: bool = True,
        use_qk_layernorm: bool = True,
        use_flash_attn: bool = False,
        flash_attn_dtype: str = "bf16",
    ):
        super().__init__()

        self.adaptive_attention = AdaptiveBiasedMultiHeadAttention(
            dim_seq=dim_single,
            dim_pair=dim_pair,
            dim_cond=dim_cond,
            n_heads=n_heads,
            dim_head=dim_head,
            attn_bias_mode=attn_bias_mode,
            use_qkv_bias=use_qkv_bias,
            use_qk_layernorm=use_qk_layernorm,
            use_flash_attn=use_flash_attn,
            flash_attn_dtype=flash_attn_dtype,
        )
        self.adaptive_transition = AdaptiveTransition(
            dim=dim_single, dim_cond=dim_cond, expansion_factor=expansion_factor
        )

    def forward(
        self,
        single_rep: torch.Tensor,
        seq_cond: torch.Tensor,
        pair_rep: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        single_rep_attention = self.adaptive_attention(
            single_rep,
            seq_cond,
            pair_rep,
            mask,
        )
        # Residual connection
        single_rep = single_rep + single_rep_attention

        single_rep_transition = self.adaptive_transition(single_rep, seq_cond, mask)
        # Residual connection
        single_rep = single_rep + single_rep_transition
        return single_rep * mask.unsqueeze(-1)


class PairUpdate(torch.nn.Module):
    """Propagate single rep update to the pair rep.

    Uses linear layers and, optionally, additional AF2-style triangle multiplicative layers.
    Adapted from proteinfoundation.nn.modules.pair_update.PairUpdate in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).

    Args:
        dim_single: feature dimension of input single representation
        dim_pair: feature dimension of pair representation
        use_triangle_mult: whether to use triangle multiplicative layers
        dim_triangle: dimension of triangle multiplicative layers
        transition_n: factor by which dim_pair is multiplied to obtain hidden channel in linear pair transition module
    """

    def __init__(
        self,
        *,
        dim_single: int,
        dim_pair: int,
        use_triangle_mult: bool,
        dim_triangle: int = 64,
        transition_n: int = 2,
    ):
        super().__init__()
        self.single_layer_norm = torch.nn.LayerNorm(dim_single)
        self.single_linear_1 = torch.nn.Linear(dim_single, dim_pair, bias=False)
        self.single_linear_2 = torch.nn.Linear(dim_single, dim_pair, bias=False)

        if use_triangle_mult:
            self.triangle_mult_out = TriangleMultiplicationOutgoing(c_z=dim_pair, c_hidden=dim_triangle)
            self.triangle_mult_in = TriangleMultiplicationIncoming(c_z=dim_pair, c_hidden=dim_triangle)

        self.pair_transition = PairTransition(c_z=dim_pair, n=transition_n)

    def forward(
        self,
        single_rep: torch.Tensor,
        pair_rep: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """
        Args:
            single_rep: single representation input, [batch, seq_len, dim_single]
            pair_rep: pair representation input, [batch, seq_len, seq_len, dim_pair]
            mask: binary mask, [batch, seq_len]

        Returns
        -------
            Updated pair representation, shape [batch, seq_len, seq_len, dim_pair].
        """
        if mask is not None:
            single_rep = single_rep * mask.unsqueeze(-1)
        single_rep = self.single_layer_norm(single_rep)
        single_rep_1 = self.single_linear_1(single_rep)
        single_rep_2 = self.single_linear_2(single_rep)
        # Outer sum
        pair_update = single_rep_1.unsqueeze(1) + single_rep_2.unsqueeze(2)
        pair_rep = pair_rep + pair_update
        # Pair mask logic is copied from OpenFold InvariantPointAttention
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        pair_rep = pair_rep * pair_mask.unsqueeze(-1)
        if hasattr(self, "triangle_mult_out"):
            pair_rep = pair_rep + checkpoint(self.triangle_mult_out, pair_rep, pair_mask)
            pair_rep = pair_rep * pair_mask.unsqueeze(-1)
            pair_rep = pair_rep + checkpoint(self.triangle_mult_in, pair_rep, pair_mask)
            pair_rep = pair_rep * pair_mask.unsqueeze(-1)
        pair_rep = pair_rep + checkpoint(self.pair_transition, pair_rep, pair_mask)
        return pair_rep * pair_mask.unsqueeze(-1)


class PairTransition(torch.nn.Module):
    """PairTransition from AF2

    Adapted from openfold.model.pair_transition.PairTransition in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    def __init__(self, c_z: int, n: int):
        """
        Args:
            c_z:
                Pair transition channel dimension.
            n:
                Factor by which c_z is multiplied to obtain hidden channel
                dimension.
        """
        super().__init__()

        self.c_z = c_z
        self.n = n

        self.layer_norm = LayerNorm(self.c_z)
        self.linear_1 = Linear(self.c_z, self.n * self.c_z, init="relu")
        self.relu = torch.nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_z, c_z, init="final")

    def _transition(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # [*, N_res, N_res, C_z]
        z = self.layer_norm(z)

        # [*, N_res, N_res, C_hidden]
        z = self.linear_1(z)
        z = self.relu(z)

        # [*, N_res, N_res, C_z]
        z = self.linear_2(z)

        z = z * mask

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding.

        Returns
        -------
            [*, N_res, N_res, C_z] pair embedding update.
        """
        # DISCREPANCY: DeepMind forgets to apply the mask in this module.
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # [*, N_res, N_res, 1]
        mask = mask.unsqueeze(-1)
        z = self._transition(z=z, mask=mask)
        return z


def _permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]) -> torch.Tensor:
    """Permutes the last len(inds) dimensions of a tensor."""
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


class BaseTriangleMultiplicativeUpdate(torch.nn.Module, ABC):
    """Base class for AF2-style triangle multiplicative updates.

    Adapted from openfold.model.triangle_multiplicative_update.TriangleMultiplicativeUpdate in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    @abstractmethod
    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool):
        """
        Args:
            c_z:
                Input channel dimension.
            c_hidden:
                Hidden channel dimension.
            _outgoing:
                Whether triangle update is outgoing or incoming.
        """
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = torch.nn.Sigmoid()

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: int | None = None
    ) -> torch.Tensor:
        if self._outgoing:
            a = _permute_final_dims(a, (2, 0, 1))
            b = _permute_final_dims(b, (2, 1, 0))
        else:
            a = _permute_final_dims(a, (2, 1, 0))
            b = _permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            p = torch.matmul(a, b)

        return _permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor.
            mask:
                [*, N_res, N_res] input mask.

        Returns
        -------
            [*, N_res, N_res, C_z] output tensor.
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """Base class for AF2-style triangle multiplicative updates.

    Adapted from openfold.model.triangle_multiplicative_update.TriangleMultiplicativeUpdate in La-Proteina
    (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    def __init__(self, c_z: int, c_hidden: int, _outgoing: bool = True):
        """
        Args:
            c_z:
                Input channel dimension.
            c_hidden:
                Hidden channel dimension.
            _outgoing:
                Whether triangle update is outgoing or incoming.
        """
        super().__init__(c_z=c_z, c_hidden=c_hidden, _outgoing=_outgoing)

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_chunk_size: int | None = None,
        with_add: bool = True,
    ) -> torch.Tensor:
        """Forward method for inference only.

        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).

        Returns
        -------
            A reference to the overwritten z

        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair: torch.Tensor, mask: torch.Tensor, a: bool = True) -> torch.Tensor:
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            pair = self.layer_norm_in(pair)
            p = linear_g(pair)
            p.sigmoid_()
            p *= linear_p(pair)
            p *= mask
            p = _permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(
            pair: torch.Tensor, mask: torch.Tensor, a: bool = True, chunked: bool = True
        ) -> torch.Tensor:
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                assert isinstance(inplace_chunk_size, int)  # Added for mypy
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.bias.shape[-1]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = pair[..., i : i + inplace_chunk_size, :, :]
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(z, mask, True, chunked=True)

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t: torch.Tensor) -> list[slice]:
                return [slice(None) for _ in t.shape]

            def slice_tensor(t: torch.Tensor, start: int, end: int | None, dim: int) -> torch.Tensor:
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[s]

            def flip_z_cache_(z_cache: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[first_half_slicer] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[quadrant_3_slicer] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[z_cache_slicer])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n])]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(i_range + after_half, initial_offsets + after_half_offsets)
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[z_chunk_slicer] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(z_cache, z_cache_offset, z_cache_offset + offset, row_dim)

                b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
                del z_chunk_b

                x_chunk = torch.matmul(
                    a,
                    b_chunk,
                )
                x_chunk = _permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)

                # The g dimension (col_dim) is parallel to and ahead of the
                # overwrites in z. We can extract the g chunk normally.
                z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                g_chunk.sigmoid_()
                del z_chunk_g

                x_chunk *= g_chunk

                # Write the columns into z in-place
                z_slicer = empty_slicer(z)
                z_slicer[col_dim] = slice(i, i + offset)
                if with_add:
                    z[z_slicer] += x_chunk
                else:
                    z[z_slicer] = x_chunk
        else:
            b = compute_projection(z, mask, False, False)
            x = torch.matmul(a, b)
            x = self.layer_norm_out(x)
            x = self.linear_z(x)
            g = self.linear_g(z)
            g.sigmoid_()
            x *= g
            if with_add:
                z += x
            else:
                z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor.
            mask:
                [*, N_res, N_res] input mask.

        Returns
        -------
            [*, N_res, N_res, C_z] output tensor.
        """
        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        # Prevents overflow of torch.matmul in combine projections in
        # reduced-precision modes
        a_std = a.std()
        b_std = b.std()
        if _is_fp16_enabled() and a_std != 0.0 and b_std != 0.0:
            a = a / a.std()
            b = b / b.std()

        if _is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """AF2-style outgoing triangle multiplicative update layer."""

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """AF2-style incoming triangle multiplicative update layer."""

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
