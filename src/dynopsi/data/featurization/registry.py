from collections.abc import Callable
from typing import Any, get_args, get_type_hints

from dynopsi.data.featurization.base import Featurizer

FEATURIZER_REGISTRY: dict[str, type[Featurizer]] = {}


def register_featurizer(overwrite: bool = False) -> Callable[[type[Featurizer]], type[Featurizer]]:
    """Register a featurizer class with the registry."""

    def register_decorator(featurizer: type[Featurizer]) -> type[Featurizer]:
        # obtain the type of the featurizer
        type_hints = get_type_hints(featurizer)
        type_literal = get_args(type_hints["type"])
        if len(type_literal) > 1:
            raise ValueError(f"Featurizer {featurizer.__name__} has multiple types: {type_literal}")
        else:
            type_string = type_literal[0]

        if type_string in FEATURIZER_REGISTRY and not overwrite:
            raise ValueError(f"Featurizer {type_string} already registered")
        elif type_string is None:
            raise ValueError(f"Featurizer {featurizer.__name__} has no type")
        elif not issubclass(featurizer, Featurizer):
            raise ValueError(f"Featurizer {featurizer.__name__} is not a subclass of Featurizer")

        FEATURIZER_REGISTRY[type_string] = featurizer
        return featurizer

    return register_decorator


def get_featurizer_from_type(featurizer_type: str) -> type[Featurizer]:
    return FEATURIZER_REGISTRY[featurizer_type]


def get_featurizer_from_config(config: dict[str, Any]) -> Featurizer:
    feat_cls = FEATURIZER_REGISTRY[config["type"]]
    return feat_cls.model_validate(config)
