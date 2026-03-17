from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field


class SingleParameterFunction(BaseModel):
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


# Discriminative union for SingleParameterFunction
# Allows for automatic serialization/deserialization of the function.
# See https://docs.pydantic.dev/latest/usage/types/#discriminated-unions.
SingleParameterFunctionType = Annotated[
    ConstantFunction | PowerLawFunction | InverseParameterFunction, Field(discriminator="type")
]
