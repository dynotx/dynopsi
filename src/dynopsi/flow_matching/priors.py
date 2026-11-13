from typing import Protocol

import torch
from torch.distributions import Dirichlet


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
        generator: torch.Generator | None = None,
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
        generator : torch.Generator or None, optional
            Optional RNG for reproducible sampling. If None, draws from the global state.

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
        std : float
            Standard deviation of the Gaussian distribution.
        sample_shape : list[int] or None
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
        generator: torch.Generator | None = None,
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
        generator : torch.Generator or None, optional
            Optional RNG for reproducible sampling. If None, draws from the global state.

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

        # Only pass `generator` when set. `torch.randn(..., generator=...)` selects the
        # randn.generator overload, which can't trace a symbolic size under torch.compile
        # ("SymIntArrayRef expected to contain only concrete integers").
        if generator is None:
            base = torch.randn(num_samples, *sample_shape, device=device)
        else:
            base = torch.randn(num_samples, *sample_shape, device=device, generator=generator)
        noise = self.std * base
        if prior_mean is not None:
            noise = noise + prior_mean.to(noise.device)
        if mask is not None and unmasked is not None:
            mask = mask.to(torch.int)
            return mask * noise + (1 - mask) * unmasked
        return noise


class HarmonicPrior(torch.nn.Module, Sampleable):
    """Backbone coordinate prior with harmonic correlations between successive atoms.

    Samples from q(x) ∝ exp(-(3/2d²) Σ ||x_{i+1} - x_i||²), where d is roughly
    the average bond length between successive atoms (Å).

    Parameters
    ----------
    num_atoms : int
        Atoms per residue (1 for CA-only, 3 for backbone N/CA/C).
    d : float
        Average distance between consecutive atoms (Å).
    eps : float
        Unused; kept for API compatibility.
    """

    def __init__(self, num_atoms: int, d: float, eps: float = 0.0) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.d = d

    def sample(  # type: ignore[override]
        self,
        num_samples: int,
        seq_len: int,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample harmonic backbone coordinates.

        Parameters
        ----------
        num_samples : int
        seq_len : int
        unmasked : torch.Tensor | None
            Shape (num_samples, seq_len, num_atoms * 3); retained at mask=0 positions.
        mask : torch.Tensor | None
            Shape (num_samples, seq_len, num_atoms * 3).
        prior_mean : torch.Tensor | None
            Accepted because every prior is called with ``prior_mean``. Not applied:
            samples stay centered at the origin.

        Returns
        -------
        torch.Tensor
            Shape (num_samples, seq_len, num_atoms * 3).
        """
        if (unmasked is None) != (mask is None):
            raise ValueError("Must provide both `unmasked` and `mask` or neither")

        total_atoms = seq_len * self.num_atoms
        J = torch.zeros(total_atoms, total_atoms)
        alpha = 3 / (self.d**2)
        for i in range(total_atoms - 1):
            J[i, i] += alpha
            J[i + 1, i + 1] += alpha
            J[i, i + 1] = J[i + 1, i] = -alpha
        D, P = torch.linalg.eigh(J)
        D_inv = 1 / D
        D_inv[0] = 0
        sqrt_D_inv = torch.sqrt(D_inv)[:, None]

        device = unmasked.device if unmasked is not None else None
        P = P.to(device)
        sqrt_D_inv = sqrt_D_inv.to(device)
        if generator is None:
            z = torch.randn(num_samples, total_atoms, 3, device=device)
        else:
            z = torch.randn(num_samples, total_atoms, 3, device=device, generator=generator)
        samples = (P @ (sqrt_D_inv * z)).view(num_samples, seq_len, self.num_atoms * 3)

        if mask is not None and unmasked is not None:
            mask = mask.to(torch.int)
            return mask * samples + (1 - mask) * unmasked
        return samples


class DirichletPrior(torch.nn.Module, Sampleable):
    """Dirichlet distribution over a categorical simplex.

    Parameters
    ----------
    alphabet_size : int
        Number of categories (e.g. amino-acid alphabet size).
    alpha : float or list[float]
        Concentration parameter(s). A scalar sets uniform concentration;
        a list of length alphabet_size sets per-category concentration.
    """

    def __init__(self, alphabet_size: int, alpha: float | list[float]) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        if isinstance(alpha, float):
            self.alpha = alpha * torch.ones(alphabet_size)
        else:
            if len(alpha) != alphabet_size:
                raise ValueError(f"alpha must be a float or list of length {alphabet_size}")
            self.alpha = torch.tensor(alpha)
        self.dirichlet = Dirichlet(self.alpha)

    def sample(
        self,
        num_samples: int,
        seq_len: int | None = None,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample from the Dirichlet distribution.

        Parameters
        ----------
        num_samples : int
        seq_len : int
            Required; must be provided.

        Returns
        -------
        torch.Tensor
            Shape (num_samples, seq_len, alphabet_size).
        """
        if (unmasked is None) != (mask is None):
            raise ValueError("Must provide both `unmasked` and `mask` or neither")
        if seq_len is None:
            raise ValueError("seq_len is required for DirichletPrior")

        device = None if unmasked is None else unmasked.device
        samples = self.dirichlet.sample((num_samples, seq_len)).to(device)
        if mask is not None and unmasked is not None:
            mask = mask.int()
            return mask * samples + (1 - mask) * unmasked
        return samples


class UniformSpherePrior(torch.nn.Module, Sampleable):
    """Samples points uniformly on the positive orthant of the unit sphere.

    Parameters
    ----------
    d : int
        Dimension of the sphere.
    """

    def __init__(self, d: int):
        super().__init__()
        import geoopt  # type: ignore[import-not-found]  # optional; not a base dependency

        self.d = d
        self.sphere = geoopt.Sphere()

    def sample(
        self,
        num_samples: int,
        seq_len: int | None = None,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample uniformly on the positive orthant of the unit sphere."""
        if (unmasked is None) != (mask is None):
            raise ValueError("Must provide both `unmasked` and `mask` or neither")

        device = None if unmasked is None else unmasked.device
        samples = self.sphere.random_uniform(num_samples, seq_len, self.d).abs().to(device)
        if mask is not None and unmasked is not None:
            mask = mask.to(torch.int)
            return mask * samples + (1 - mask) * unmasked
        return samples


class ProjectDirichletToSphere(DirichletPrior):
    """Projects a Dirichlet distribution to the positive orthant of the unit sphere.

    Parameters
    ----------
    d : int
        Dimension of the sphere.
    """

    def __init__(self, d: int, alpha: float | list[float]):
        super().__init__(d, alpha)

    def sample(
        self,
        num_samples: int,
        seq_len: int | None = None,
        unmasked: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        prior_mean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Project Dirichlet samples onto the positive orthant of the unit sphere."""
        samples = super().sample(num_samples, seq_len, unmasked, mask)
        return torch.sqrt(samples)
