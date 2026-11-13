from collections.abc import Callable
from typing import Any, TypeVar, get_args, get_type_hints

from dynopsi.simulation.ops.op import Op

OPS_REGISTRY: dict[str, type[Op]] = {}

_OpT = TypeVar("_OpT", bound=Op)


def register_op(overwrite: bool = False) -> Callable[[type[_OpT]], type[_OpT]]:
    """Register a featurizer class with the registry."""

    def register_decorator(op: type[_OpT]) -> type[_OpT]:
        # obtain the type of the featurizer
        type_hints = get_type_hints(op)
        type_literal = get_args(type_hints["type"])
        if len(type_literal) > 1:
            raise ValueError(f"Op {op.__name__} has multiple types: {type_literal}")
        else:
            type_string = type_literal[0]

        if type_string in OPS_REGISTRY and not overwrite:
            raise ValueError(f"Op {type_string} already registered")
        elif type_string is None:
            raise ValueError(f"Op {op.__name__} has no type")
        elif not issubclass(op, Op):
            raise ValueError(f"Op {op.__name__} is not a subclass of Op")

        OPS_REGISTRY[type_string] = op
        return op

    return register_decorator


def get_op_from_type(op_type: str) -> type[Op]:
    return OPS_REGISTRY[op_type]


def get_op_from_config(config: dict[str, Any]) -> Op:
    op_cls = OPS_REGISTRY[config["type"]]
    return op_cls.model_validate(config)
