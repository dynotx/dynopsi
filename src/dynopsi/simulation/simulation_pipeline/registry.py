from collections.abc import Callable
from typing import Any, TypeVar, get_args, get_type_hints

from dynopsi.simulation import Simulation

SIMULATION_PIPELINE_REGISTRY: dict[str, type[Simulation]] = {}

_SimT = TypeVar("_SimT", bound=Simulation)


def register_simulation_pipeline(overwrite: bool = False) -> Callable[[type[_SimT]], type[_SimT]]:
    """Register a simulation pipeline class with the registry."""

    def register_decorator(simulation_pipeline: type[_SimT]) -> type[_SimT]:
        # obtain the type of the simulation pipeline
        type_hints = get_type_hints(simulation_pipeline)
        type_literal = get_args(type_hints["type"])
        if len(type_literal) > 1:
            raise ValueError(f"Simulation pipeline {simulation_pipeline.__name__} has multiple types: {type_literal}")
        else:
            type_string = type_literal[0]

        if type_string in SIMULATION_PIPELINE_REGISTRY and not overwrite:
            raise ValueError(f"Simulation pipeline {type_string} already registered")
        elif type_string is None:
            raise ValueError(f"Simulation pipeline {simulation_pipeline.__name__} has no type")
        elif not issubclass(simulation_pipeline, Simulation):
            raise ValueError(f"Simulation pipeline {simulation_pipeline.__name__} is not a subclass of Simulation")

        SIMULATION_PIPELINE_REGISTRY[type_string] = simulation_pipeline
        return simulation_pipeline

    return register_decorator


def get_simulation_pipeline_from_config(config: dict[str, Any]) -> Simulation:
    sim_cls = SIMULATION_PIPELINE_REGISTRY[config["type"]]
    return sim_cls.model_validate(config)
