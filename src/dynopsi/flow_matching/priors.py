from typing import Protocol

import torch


class Sampleable(Protocol):
    """Distribution that emits samples.

    Handles masked sampling, which enables data-dependent coupling by optionally
    taking unmasked data and a mask to determine which positions should be
    sampled from the noise distribution and which should be retained from the
    unmasked data.
    """

    def sample(
        self,
        num_samples: int,
        seq_len: int | None = None,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate `num_samples` samples, retaining values from the unmasked data where the mask is 0.

        Parameters
        ----------
        num_samples : int
            The desired number of samples.
        seq_len : int or None, optional
            Length of the sequence to sample, by default None.
            If None, the sequence length is inferred from the shape of the unmasked or masked data.
        unmasked : torch.Tensor or None, optional
            The unmasked data with shape self.shape.
        mask : torch.Tensor or None, optional
            The mask with shape self.shape.
            - 1 if the position should be sampled from the noise distribution
            - 0 if it should be retained from the unmasked data
        prior_mean : torch.Tensor or None, optional
            The prior mean with shape (batch_size, ...).
            If None, the prior mean is assumed to be 0.

        Returns
        -------
        torch.Tensor
            Samples with shape (batch_size, ...).
        """
        raise NotImplementedError


class IsotropicGaussian(torch.nn.Module, Sampleable):
    """`Sampleable` wrapper around `torch.randn`.

    An isotropic Gaussian distribution that can be sampled, with support
    for masked sampling to enable data-dependent coupling.
    """

    def __init__(self, std: float, sample_shape: list[int] | None):
        """Initialize the isotropic Gaussian distribution.

        Parameters
        ----------
        std : float, optional
            Standard deviation of the Gaussian distribution, by default 1.0.
        sample_shape : list[int] or None, optional
            Shape of the samples to generate, by default None.
            This is the dimension of the samples to generate, not including batch size or sequence length.
            If None, the shape of the samples is inferred from the shape of the unmasked/masked data.
        """
        super().__init__()
        self.sample_shape = sample_shape
        self.std = std

    def sample(
        self,
        num_samples: int,
        seq_len: int | None = None,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples from the isotropic Gaussian distribution.

        Parameters
        ----------
        num_samples : int
            The desired number of samples.
        seq_len : int or None, optional
            Length of the sequence to sample, by default None.
            If None, the sequence length is inferred from the shape of the unmasked or masked data.
        unmasked : torch.Tensor or None, optional
            The unmasked data with shape (num_samples, *self.sample_shape).
        mask : torch.Tensor or None, optional
            The mask with shape (num_samples, *self.sample_shape) and dtype bool.
            - 1 if the position should be sampled from the noise
            - 0 if it should be retained from the unmasked data
        prior_mean : torch.Tensor or None, optional
            The prior mean with shape (num_samples, *self.sample_shape).
            If None, the prior mean is assumed to be 0 for all positions.

        Returns
        -------
        torch.Tensor
            Samples with shape (num_samples, *self.sample_shape).

        Raises
        ------
        ValueError
            If only one of `unmasked` or `mask` is provided.
        """
        if (unmasked is None) != (mask is None):
            raise ValueError("Must provide both `unmasked` and `mask` or neither")

        device = None if unmasked is None else unmasked.device
        if unmasked is not None and mask is not None:
            sample_shape = list(unmasked.shape[1:])
        elif seq_len is not None and self.sample_shape is not None:
            sample_shape = [seq_len, *self.sample_shape]  # (L, d)
        else:
            raise ValueError("Must provide either `unmasked`, `mask`, or `sample_shape` and `seq_len`")

        noise = self.std * torch.randn(num_samples, *sample_shape, device=device)
        if prior_mean is not None:
            noise = noise + prior_mean.to(noise.device)
        if mask is not None:
            assert unmasked is not None  # enforced by the check at line 110
            mask = mask.to(torch.int)
            return mask * noise + (1 - mask) * unmasked
        return noise
