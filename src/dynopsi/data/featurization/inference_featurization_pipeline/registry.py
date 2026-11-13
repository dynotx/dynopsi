from collections.abc import Callable
from typing import Any, TypeVar, get_args, get_type_hints

from dynopsi.data.featurization.inference_featurization_pipeline.inference_featurization_pipeline_base import (
    InferenceFeaturizationPipeline,
)

FEATURIZATION_PIPELINE_REGISTRY: dict[str, type[InferenceFeaturizationPipeline]] = {}

_PipelineT = TypeVar("_PipelineT", bound=InferenceFeaturizationPipeline)


def register_featurization_pipeline(overwrite: bool = False) -> Callable[[type[_PipelineT]], type[_PipelineT]]:
    def register_decorator(pipeline: type[_PipelineT]) -> type[_PipelineT]:
        type_hints = get_type_hints(pipeline)
        type_literal = get_args(type_hints["type"])
        if len(type_literal) > 1:
            raise ValueError(f"{pipeline.__name__} has multiple types: {type_literal}")
        type_string = type_literal[0]
        if type_string is None:
            raise ValueError(f"{pipeline.__name__} has no type")
        if type_string in FEATURIZATION_PIPELINE_REGISTRY and not overwrite:
            raise ValueError(f"Featurization pipeline {type_string} already registered")
        FEATURIZATION_PIPELINE_REGISTRY[type_string] = pipeline
        return pipeline

    return register_decorator


def get_featurization_pipeline_from_config(config: dict[str, Any]) -> InferenceFeaturizationPipeline:
    if config["type"] in FEATURIZATION_PIPELINE_REGISTRY:
        pipeline_cls = FEATURIZATION_PIPELINE_REGISTRY[config["type"]]
        return pipeline_cls.model_validate(config)
    else:
        raise ValueError(
            f"""Featurization pipeline of type {config["type"]} is not found in the registry.
            Please check that your pipeline contains the appropriate registry decorator."""
        )
