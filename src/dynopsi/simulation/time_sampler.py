"""Portions of this file are derived from the La Proteina project.

La Proteina
Copyright (c) 2025 NVIDIA
Licensed under the Apache License, Version 2.0.
Source: https://github.com/NVIDIA-Digital-Bio/la-proteina

Modifications to the La Proteina-derived code were made by Dyno Therapeutics, Inc. (2026).
"""

from collections.abc import Iterator
from typing import Annotated, Literal, Self

import torch
from pydantic import Field, field_validator, model_validator

from dynopsi.utils.strict_config import StrictConfig

TimeStepResult = tuple[float, float] | tuple[torch.Tensor, torch.Tensor]

TimesByModality = dict[str, torch.Tensor]  # shape (1, num_steps, 1, 1) or (batch_size, num_steps, 1, 1)
TimestepsByModality = dict[str, torch.Tensor]  # shape (1, num_steps, 1, 1) or (batch_size, num_steps, 1, 1)
TimeSchedule = tuple[TimesByModality, TimestepsByModality]


class TimeSampler(StrictConfig):
    """
    Base class for time samplers, provides the sequence of steps and the full schedule for all modalities.
    Subclasses must implement get_steps() and get_full_schedule().

    Attributes
    ----------
    num_steps : int
        Number of integration steps.
    modalities : tuple[str, ...]
        Tuple of modalities to sample.

    Subclasses must define a type field as a Literal["subclass_type"]
    """

    num_steps: int
    modalities: tuple[str, ...]

    def get_steps(self) -> Iterator[int] | range:
        """Return the sequence of step indices."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_full_schedule(self, device: torch.device) -> TimeSchedule:
        """Return a TimeSchedule with all timesteps and dt, shapes (1, num_steps, 1, 1) for each modality."""
        raise NotImplementedError("Subclasses must implement this method.")


class LinearTimeSampler(TimeSampler):
    """
    Linear time schedule over [start_time, end_time].

    Attributes
    ----------
    num_steps : int
        Number of integration steps.
    start_time : float
        Time at step 0.
    end_time : float
        End time (trajectory runs start_time to end_time).
    modalities : tuple[str, ...]
        Tuple of modalities to sample.
    """

    type: Literal["linear_time_sampler"] = "linear_time_sampler"
    num_steps: int
    start_time: float = 0.0
    end_time: float = 1.0
    modalities: tuple[str, ...] = ("coords",)

    @model_validator(mode="after")
    def validate_time_sampler_and_send_to_device(self) -> Self:
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be less than end time.")
        if len(self.modalities) == 0:
            raise ValueError("modalities must be non-empty.")
        if self.end_time > 1.0:
            raise ValueError("end_time must be <= 1.0.")
        if self.start_time < 0.0:
            raise ValueError("start_time must be >= 0.0.")
        return self

    def get_steps(self) -> range:
        """Return the sequence of step indices."""
        return range(self.num_steps)

    def get_full_schedule(self, device: torch.device) -> TimeSchedule:
        """Return a TimeSchedule with all timesteps and dt, shapes (1, num_steps, 1, 1) for each modality."""
        timesteps = (
            torch.linspace(self.start_time, self.end_time, self.num_steps + 1, device=device)[:-1]
            .unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        step_sizes = torch.full(timesteps.shape, (self.end_time - self.start_time) / self.num_steps, device=device)

        ts: TimesByModality = {modality: timesteps.clone() for modality in self.modalities}
        dts: TimestepsByModality = {modality: step_sizes.clone() for modality in self.modalities}
        return ts, dts


class LogTimeSampler(TimeSampler):
    """Steps follow: f(x) = (1 - 1e(-p*(x/n_steps))) / (1 - 1e(-p)) for p > 0.

    Note: This is the exponential time schedule in Geffner et al. 2025.
    """

    type: Literal["log_time_sampler"] = "log_time_sampler"
    num_steps: int
    p: float
    modalities: tuple[str, ...] = ("coords",)

    @field_validator("p", mode="after")
    @classmethod
    def check_p(cls, p: float) -> float:
        if p <= 0:
            raise ValueError("p must be positive.")
        return p

    @field_validator("num_steps", mode="after")
    @classmethod
    def check_num_steps(cls, num_steps: int) -> int:
        if num_steps < 1:
            raise ValueError(f"num_steps needs to be >= 1 but is {num_steps}")
        return num_steps

    def get_steps(self) -> range:
        """Return the sequence of step indices."""
        return range(self.num_steps)

    def get_full_schedule(self, device: torch.device) -> TimeSchedule:
        """Return a TimeSchedule with all timesteps and dt, shapes (1, num_steps, 1, 1) for each modality."""
        timesteps = 1.0 - torch.logspace(-self.p, 0, self.num_steps + 1, device=device).flip(0)
        timesteps = timesteps - torch.min(timesteps)
        timesteps = timesteps / torch.max(timesteps)

        step_sizes = timesteps[1:] - timesteps[:-1]

        # reshapae
        timesteps = timesteps[:-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        step_sizes = step_sizes.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        ts: TimesByModality = {modality: timesteps.clone() for modality in self.modalities}
        dts: TimestepsByModality = {modality: step_sizes.clone() for modality in self.modalities}

        return ts, dts


class QuadraticTimeSampler(TimeSampler):
    """Steps follow: f(x) = (x / num_steps) ** 2 for x in [0, num_steps].

    Timesteps are spaced quadratically, concentrating more steps near t=0.
    """

    type: Literal["quadratic_time_sampler"] = "quadratic_time_sampler"
    num_steps: int
    modalities: tuple[str, ...] = ("coords",)

    @field_validator("num_steps", mode="after")
    @classmethod
    def check_num_steps(cls, num_steps: int) -> int:
        if num_steps < 1:
            raise ValueError(f"num_steps needs to be >= 1 but is {num_steps}")
        return num_steps

    def get_steps(self) -> range:
        """Return the sequence of step indices."""
        return range(self.num_steps)

    def get_full_schedule(self, device: torch.device) -> TimeSchedule:
        """Return a TimeSchedule with all timesteps and dt, shapes (1, num_steps, 1, 1) for each modality."""
        timesteps = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device) ** 2

        step_sizes = timesteps[1:] - timesteps[:-1]

        timesteps = timesteps[:-1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        step_sizes = step_sizes.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        ts: TimesByModality = {modality: timesteps.clone() for modality in self.modalities}
        dts: TimestepsByModality = {modality: step_sizes.clone() for modality in self.modalities}

        return ts, dts


# Discriminative union for TimeSampler
# Allows for automatic serialization/deserialization of different time samplers classes.
TimeSamplerType = Annotated[LinearTimeSampler | LogTimeSampler | QuadraticTimeSampler, Field(discriminator="type")]
