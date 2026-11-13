from typing import Protocol

import torch


class Interpolant(Protocol):
    """Protocol for arbitrary interpolation between two tensors used in flow matching and diffusion models.

    An `Interpolant` must:
    1. Compute `xt` as a function of `t`, `x0`, and `x1`.
        In the standard flow matching context, `x0` is a sample from a prior distribution and `x1` is a sample from the data distribution.
        `xt` is the point at time `t` on the trajectory of a conditional flow model `psi_t(x | x0, x1)` where `x0` is the initial condition.
    2. Compute the conditional vector field `u_t(x|x0, x1)` of the conditional flow model `psi_t(x | x0, x1)`.
        By definition, `u_t(x|x0, x1) = d/dt psi_t(x|x0, x1)`.
    3. Apply the conditional vector field `u_t(x|x0, x1)` to `xt` with a step size of `dt` to obtain `x(t+dt)`.
        This is useful to convert between a loss parameterized by `u_t(x|x0, x1)` and a loss parameterized by `x1`.
    """

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the interpolated point `xt` as a function of `t`, `x0`, and `x1`.

        `xt` is the point at time `t` on the trajectory of a conditional flow model `psi_t(x | x0, x1)`.

        `psi_t(x | x0, x1)` is deterministic because it is conditioned on samples from both the prior and data distributions.

        Parameters
        ----------
        x0 : torch.Tensor
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Interpolated point, shape (batch_size, L, n_atoms, 3) for atom coordinates.
        """
        raise NotImplementedError

    def conditional_vector_field(
        self,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """Evaluate the conditional vector field u_t(x | x0, x1) = d/dt psi_t(x | x0, x1) at time `t`.

        `psi_t(x | x0, x1)` is a deterministic function of samples from both the prior and data distributions.
        Since `xt` is a deterministic function of `x0` and `x1` computed via the `interpolate` method,
        `u_t(x | x0, x1)` = `u_t(x | xt, x1)`.

        By default, the second form is used here, even though it marginally complicates the form of `u_t`, because
        it simplifies the conversion of flow matching model predictions between vector fields and clean data points.
        However, at training time `x0` may be optionally provided to use the first form for additional numerical stability wrt t.

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).
        x0 : torch.Tensor | None
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
            If provided, `x0` and `x1` are used to compute `u_t(x | x0, x1)`.
            If None (default), `u_t(x | xt, x1)` is computed using `xt` and `x1`.
        eps: float | None
            Optional small float value used to control numerical stability.

        Returns
        -------
        torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        """
        raise NotImplementedError

    def apply_conditional_vector_field(self, xt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Apply the conditional vector field u_t(x|x0, x1) to `xt` with a step size of `dt` to obtain `x(t+dt)`.

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        ut : torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        dt : torch.Tensor
            Step size, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Point on the trajectory at time `t+dt`, shape (batch_size, L, d).
        """
        raise NotImplementedError

    def calc_clean_data_loss_weight(self, time_samples: torch.Tensor) -> torch.Tensor:
        """Calculate the time-dependent weight required when using clean data prediction loss.

        The weight is derived from letting u = dx_t/dt and plugging the result into the
        standard CFM loss calculated on conditional vector fields.

        It is typically equal to gamma'(t) / (1-gamma(t))**2 where gamma(t) is the interpolant schedule.

        Parameters
        ----------
        time_samples : torch.Tensor
            Time samples, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Time-dependent weight, shape (batch_size, 1, 1).
        """
        raise NotImplementedError


class InterpolantSchedule:
    """Base class for interpolant schedule for flow matching.

    In Euclidean flow matching, the interpolant is defined as:
            xt = alpha(t) * x1 + beta(t) * x0
    where alpha(t) and beta(t) define the noise schedule and have the following boundary conditions:
        alpha(0) = 0, alpha(1) = 1
        beta(0) = 1, beta(1) = 0

    Here we simplify this to beta(t) = 1 - alpha(t) and calling the interpolant schedule returns alpha(t).
    """

    def __init__(self) -> None:
        """Validate boundary conditions at initialization."""
        if not self.validate_boundary_conditions():
            raise ValueError("InterpolantSchedule boundary conditions not satisfied")

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the interpolant schedule at time `t` and return alpha(t).

        Parameters
        ----------
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Interpolant weight, shape (batch_size, 1, 1).
        """
        raise NotImplementedError

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time derivative of the interpolant schedule at time `t`.

        Parameters
        ----------
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Time derivative of the interpolant weight, shape (batch_size, 1, 1).
        """
        raise NotImplementedError

    def validate_boundary_conditions(self, atol: float = 1e-6) -> bool:
        """Validate that alpha(t) satisfies the boundary conditions: alpha(0) = 0, alpha(1) = 1.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance for the comparison, by default 1e-6.

        Returns
        -------
        bool
            True if boundary conditions are satisfied, False otherwise.
        """
        t0 = torch.zeros(1, 1)
        t1 = torch.ones(1, 1)

        alpha_0 = self(t0)
        alpha_1 = self(t1)

        condition_0 = torch.allclose(alpha_0, torch.zeros_like(alpha_0), atol=atol)
        condition_1 = torch.allclose(alpha_1, torch.ones_like(alpha_1), atol=atol)

        return condition_0 and condition_1


class LinearInterpolantSchedule(InterpolantSchedule):
    """Linear interpolant schedule, alpha(t) = t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the interpolant schedule at time `t`."""
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the time derivative of the interpolant schedule at time `t`."""
        return torch.ones_like(t)


class EuclideanInterpolant(Interpolant):
    def __init__(self, interpolant_schedule: InterpolantSchedule):
        """Initialize the Euclidean interpolant.

        Simple linear interpolation that sets alpha(t) = t and beta(t) = 1-t.

        Parameters
        ----------
        alpha : Alpha
            Alpha schedule defining the weight on the prior distribution sample.
        beta : Beta
            Beta schedule defining the weight on the data distribution sample.
        """
        self.interpolant_schedule = interpolant_schedule

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform Euclidean interpolation between x0 and x1.

        Parameters
        ----------
        x0 : torch.Tensor
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Interpolated point, shape (batch_size, L, d).
        """
        return self.interpolant_schedule(t) * x1 + (1 - self.interpolant_schedule(t)) * x0

    def conditional_vector_field(
        self,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor | None = None,
        eps: float | None = 1e-1,
    ) -> torch.Tensor:
        """Evaluate the conditional vector field u_t(xt|x1).

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).
        x0 : torch.Tensor | None
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
            If provided, x0 and x1 are used to compute the u_t (equation 22 in Example 11) for
            additional numerical stability wrt t.
            If None (default), u_t is computed using xt and x1 (equation 21 in Example 11).
        eps : float, optional
            Small value to avoid division by zero, by default 1e-1.

        Returns
        -------
        torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        """
        alpha_t = self.interpolant_schedule(t)  # (batch_size, 1, 1)
        beta_t = 1 - alpha_t  # (batch_size, 1, 1)
        dt_alpha_t = self.interpolant_schedule.dt(t)  # (batch_size, 1, 1)
        dt_beta_t = -dt_alpha_t  # (batch_size, 1, 1)

        if x0 is not None:
            # Use x0 and x1 to compute u_t
            flow = dt_alpha_t * x1 + dt_beta_t * x0
        else:
            # Use xt and x1 to compute u_t
            if eps is not None:
                beta_t = torch.clamp(
                    beta_t, min=eps, max=1
                )  # with eps=1e-1 this is equivalent to FrameFlow's max_t clamping with t=0.9
            flow = (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * x1 + dt_beta_t / beta_t * xt

        return flow

    def calc_clean_data_loss_weight(self, time_samples: torch.Tensor) -> torch.Tensor:
        """Calculate the time-dependent weight required when using clean data prediction loss.

        Parameters
        ----------
        time_samples : torch.Tensor
            Time samples, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Time-dependent weight, shape (batch_size, 1, 1).
        """
        dgamma_dt = self.interpolant_schedule.dt(time_samples)
        gamma_t = self.interpolant_schedule(time_samples)
        max_t = 0.9  # 0.9 is the value used in FrameFlow
        return dgamma_dt / (1 - torch.clamp(gamma_t, min=0, max=max_t)) ** 2

    @staticmethod
    def apply_conditional_vector_field(xt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Apply the conditional vector field u_t(xt|x1) to `xt` with a step size of `dt` to obtain `x(t+dt)`.

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        ut : torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        dt : torch.Tensor
            Step size, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Point on the trajectory at time `t+dt`, shape (batch_size, L, d).
        """
        return xt + ut * dt


class FisherInterpolant(Interpolant):
    """Implement the interpolant for Fisher flow matching from Davis, et al. (2024).

    This is ultimately just Riemannian flow matching on the positive orthant of the unit sphere.
    """

    def __init__(self) -> None:
        """Initialize the Fisher interpolant."""
        import geoopt  # type: ignore[import-not-found]  # optional; not a base dependency

        self.sphere = geoopt.Sphere()

    def _exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on the sphere.

        _exp_map(x, v) corresponds to exp_x(v) in usual differential geometry notation.
        The exponential maps a tangent vector v at x to a point on the manifold that follwos
        the geodesic starting at x with velocity v.

        Parameters
        ----------
        x : torch.Tensor
            Point on the manifold, shape (batch_size, L, d).
        v : torch.Tensor
            Tangent vector, shape (batch_size, L, d).
        """
        return self.sphere.expmap(x, v)

    def _log_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log map on the sphere.

        _log_map(x, y) corresponds to log_x(y) in usual differential geometry notation.
        The log map maps a point y on the manifold to a tangent vector at x that is the velocity
        of the geodesic starting at x and ending at y.

        Parameters
        ----------
        x : torch.Tensor
            Point on the manifold, shape (batch_size, L, d).
        y : torch.Tensor
            Point on the manifold, shape (batch_size, L, d).
        """
        return self.sphere.logmap(x, y)

    def interpolate(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform interpolation along the geodesic between x0 and x1.

        The is done by calculating exp_x0(t * log_x0(x1)) and projecting onto the positive orthant.

        Parameters
        ----------
        x0 : torch.Tensor
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Interpolated point, shape (batch_size, L, d).
        """
        x_ = self._exp_map(x0, self._log_map(x0, x1) * t)
        return self.sphere.projx(x_).abs()  # project onto positive orthant

    def conditional_vector_field(
        self,
        xt: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor | None = None,
        eps: float | None = 1e-1,
    ) -> torch.Tensor:
        """Evaluate the conditional vector field u_t(xt|x1).

        In Davis, et al. (2024), the conditional vector field is defined as:
            u_t(x|x0, x1) = log_{xt}(x1) / (1-t)

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        x1 : torch.Tensor
            Clean samples from the data distribution, shape (batch_size, L, d).
        t : torch.Tensor
            Time, shape (batch_size, 1, 1).
        x0 : torch.Tensor | None
            Samples from the prior (noise) distribution, shape (batch_size, L, d).
        eps : float, optional
            Small value to avoid division by zero, by default 1e-1.
            Default matches the default clip value in FrameFlow for the time reweighting term to prevent exploding gradients.

        Returns
        -------
        torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        """
        if x0 is not None:
            v = self._log_map(x0, x1)
            return self.sphere.transp(x0, xt, v)
        else:
            denominator = 1 - t
            if eps is not None:
                denominator = denominator.clamp_min(min=eps)
            return self._log_map(xt, x1) / denominator

    def apply_conditional_vector_field(self, xt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Apply the conditional vector field u_t(xt|x1) to `xt` with a step size of `dt` to obtain `x(t+dt)`.

        Parameters
        ----------
        xt : torch.Tensor
            Interpolated states between prior and data samples, shape (batch_size, L, d).
        ut : torch.Tensor
            Conditional vector field, shape (batch_size, L, d).
        dt : torch.Tensor
            Step size, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Point on the trajectory at time `t+dt`, shape (batch_size, L, d).
        """
        x_ = self._exp_map(xt, ut * dt)
        return self.sphere.projx(x_).abs()

    def calc_clean_data_loss_weight(self, time_samples: torch.Tensor) -> torch.Tensor:
        """Calculate the time-dependent weight required when using clean data prediction loss.

        Parameters
        ----------
        time_samples : torch.Tensor
            Time samples, shape (batch_size, 1, 1).

        Returns
        -------
        torch.Tensor
            Time-dependent weight, shape (batch_size, 1, 1).
        """
        return 1 / (1 - time_samples) ** 2
