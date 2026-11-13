"""Portions of this file are derived from the OpenFold and La Proteina projects.

OpenFold
Copyright (c) 2024 AlQuraishi Laboratory
Licensed under the Apache License, Version 2.0.
Source: https://github.com/aqlaboratory/openfold

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the OpenFold and La Proteina derived code were made by Dyno Therapeutics, Inc. (2026).
"""

import math
import typing

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm


@typing.no_type_check
def get_time_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 2000) -> torch.Tensor:
    """Build sinusoidal embeddings for a given vector of timesteps.

    Code from FrameFlow (https://github.com/microsoft/protein-frame-flow).

    Args:
        timesteps: vector of times (float) of shape [batch_size].
        embedding_dim: dimension of the embeddings.
        max_positions: also referred to as "base wavelength" for sinusoidal embedding;
            there are 2 * pi * max_positions unique positions / context length.
            The original implementation hardcodes this to 10000, but FrameFlow
            uses a default value of 2000.

    Returns
    -------
        Embedding of shape [batch_size, embedding_dim]
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_index_embedding(indices: torch.Tensor, embed_size: int, max_len: int = 2056) -> torch.Tensor:
    """Create sine/cosine positional embeddings from prespecified indices.

    Code from models.utils.get_index_embedding in FrameFlow (https://github.com/microsoft/protein-frame-flow).

    Args:
        indices: offsets of size [N_edges] or [batch, N_edges] of type integer
        embed_size: dimension of the embeddings to create
        max_len: maximum length.

    Returns
    -------
        positional embedding of shape [N, embed_size] or [batch, N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    assert len(indices.shape) == 1 or len(indices.shape) == 2, (
        f"indices must be of shape [N_edges] or [batch, N_edges], got {indices.shape}"
    )
    if len(indices.shape) == 1:
        K = K[None]
    elif len(indices.shape) == 2:
        K = K[None, None]
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K / embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K / embed_size))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


def get_distogram(
    coords: torch.Tensor,
    num_bins: int,
    bin_min: float,
    bin_max: float,
) -> torch.Tensor:
    """Compute a distogram representation of atomic coordinates.

    Note: Currently, pairwise distances are computed (and binned) in an
    all-to-all manner, i.e. both within and across atom types.
    This results in num_atoms * num_atoms pairwise distances per residue,
    resulting in an overall distogram shape of
    [batch_size, N, N, num_atoms, num_atoms, num_bins].

    Parameters
    ----------
    coords : torch.Tensor
        Atomic coordinates tensor of shape [batch_size, N, num_atoms, 3]
    num_bins : int
        Number of distance bins to discretize distances into
    bin_min : float
        Minimum distance value for binning
    bin_max : float
        Maximum distance value for binning

    Returns
    -------
    torch.Tensor
        One-hot encoded distance bins of shape [batch_size, N, N, num_atoms, num_atoms, num_bins]
    """
    # compute pair distances
    dists_vectors = (
        coords[:, :, None, :, None, :] - coords[:, None, :, None, :, :]
    )  # [batch_size, N, N, num_atoms, num_atoms, 3]
    distances = torch.norm(dists_vectors, dim=-1)  # [batch_size, N, N, num_atoms, num_atoms]

    bins = torch.linspace(bin_min, bin_max, num_bins - 1, device=coords.device)
    dists_bins_idx = torch.bucketize(distances, bins)  # [batch_size, N, N, num_atoms, num_atoms]

    dist_bins_onehots = torch.nn.functional.one_hot(
        dists_bins_idx, num_classes=num_bins
    )  # [batch_size, N, N, num_atoms, num_atoms, num_bins]

    return dist_bins_onehots


class SwiGLU(torch.nn.Module):
    """SwiGLU activation

    Code from proteinfoundation.nn.modules.swiglu.SwiGLU in La-Proteina (https://github.com/NVIDIA-Digital-Bio/la-proteina).
    """

    def forward(
        self,
        x: torch.Tensor,  # shape [..., dim]
    ) -> torch.Tensor:  # shape [..., dim/2]
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class LayerNorm(torch.nn.Module):
    """Layer normalization layer."""

    def __init__(self, c_in: int, eps: float = 1e-5):
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(c_in))
        self.bias = torch.nn.Parameter(torch.zeros(c_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.dtype
        if d is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                out = torch.nn.functional.layer_norm(
                    x, self.c_in, self.weight.to(dtype=d), self.bias.to(dtype=d), self.eps
                )
        else:
            out = torch.nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        return out


class Linear(torch.nn.Linear):
    """Linear layer with built-in nonstandard initializations.

    Code from openfold.model.primitives in OpenFold (https://github.com/aqlaboratory/openfold).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: typing.Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, self.weight, self.bias)


def _prod(nums: typing.Iterable[int]) -> int:
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> float:
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2  # type: ignore
    else:
        raise ValueError("Invalid fan option")

    return f


# The following utility functions are from OpenFold (https://github.com/aqlaboratory/openfold).
def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in") -> None:
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor) -> None:
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor) -> None:
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
