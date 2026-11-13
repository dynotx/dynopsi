from typing import Annotated, Literal

import torch
from pydantic import Field

from dynopsi.utils.strict_config import StrictConfig


class SingleParameterFunction(StrictConfig):
    """
    Abstract base class for functions with a single parameter.
    Pydantic BaseModel enables serialization and deserialization of the function.

    Subclasses must define a type field and update the SingleParameterFunctionType
    discriminative union below.

    NOTE: Subclasses must define a type field as a Literal["subclass_type"]
    this is used for serialization/deserialization.
    """

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement __call__")


class ConstantFunction(SingleParameterFunction):
    """Constant function: y = constant."""

    type: Literal["constant"] = "constant"
    constant: float

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        return torch.full_like(parameter, self.constant, dtype=torch.float32, device=parameter.device)


class PowerLawFunction(SingleParameterFunction):
    """Power law function: y = a * x^b + c, clamped between min_val and max_val."""

    type: Literal["power_law"] = "power_law"
    a: float
    b: float
    c: float
    min_val: float
    max_val: float

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        res = self.a * parameter**self.b + self.c
        res = torch.clamp(res, self.min_val, self.max_val)
        return res


class InverseParameterFunction(SingleParameterFunction):
    """Inverse parameter function: y = max(min(1.0 / (eps + parameter), clamp_max), clamp_min)."""

    type: Literal["inverse_parameter"] = "inverse_parameter"
    eps: float
    clamp_max: float | None = None
    clamp_min: float | None = None

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        res = 1.0 / (self.eps + parameter)
        if self.clamp_max is not None or self.clamp_min is not None:
            return torch.clamp(res, min=self.clamp_min, max=self.clamp_max)
        else:
            return res


class PiecewisePowerFunction(SingleParameterFunction):
    """Ramp from 0 to max_value and back to 0 with independently controllable steepness on both sides of the ramp."""

    type: Literal["piecewise_power"] = "piecewise_power"
    y_max_value: float
    t_mode: float
    t_mode_radius: float = 0.05
    t_left_zero: float = 0.0
    t_right_zero: float = 1.0
    left_steepness: float = 1.0
    right_steepness: float = 1.0

    def __call__(self, parameter: torch.Tensor) -> torch.Tensor:
        p = parameter
        t_lz, t_rz = self.t_left_zero, self.t_right_zero
        t_ml = self.t_mode - self.t_mode_radius
        t_mr = self.t_mode + self.t_mode_radius

        left_ramp = self.y_max_value * ((p - t_lz) / (t_ml - t_lz)) ** self.left_steepness
        right_ramp = self.y_max_value * ((t_rz - p) / (t_rz - t_mr)) ** self.right_steepness

        in_domain = (p > t_lz) & (p < t_rz)
        left_mask = in_domain & (p < t_ml)
        right_mask = in_domain & (p > t_mr)
        plateau_mask = in_domain & (p >= t_ml) & (p <= t_mr)

        out = torch.zeros_like(p)
        out = torch.where(left_mask, left_ramp, out)
        out = torch.where(right_mask, right_ramp, out)
        out = torch.where(plateau_mask, torch.full_like(p, self.y_max_value), out)
        return out


# Discriminative union for SingleParameterFunction
# Allows for automatic serialization/deserialization of the function.
# See https://docs.pydantic.dev/latest/usage/types/#discriminated-unions.
SingleParameterFunctionType = Annotated[
    ConstantFunction | PowerLawFunction | InverseParameterFunction | PiecewisePowerFunction,
    Field(discriminator="type"),
]
