from typing import Literal

import pytest

from dynopsi.simulation import Simulation
from dynopsi.simulation.simulation_pipeline.registry import (
    SIMULATION_PIPELINE_REGISTRY,
    get_simulation_pipeline_from_config,
    register_simulation_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_simulation_class(type_str: str) -> type[Simulation]:
    """Return a minimal Simulation subclass with the given type literal."""

    class _Sim(Simulation):
        type: Literal[type_str] = type_str  # type: ignore[valid-type]

        def model_post_init(self, __context):
            pass  # skip plan building

    _Sim.__name__ = f"_TestSim_{type_str}"
    return _Sim


# ---------------------------------------------------------------------------
# register_simulation_pipeline
# ---------------------------------------------------------------------------


class TestRegisterSimulationPipeline:
    _REGISTERED: list[str] = []  # track types registered in this class for teardown

    def teardown_method(self, _):
        for key in self._REGISTERED:
            SIMULATION_PIPELINE_REGISTRY.pop(key, None)
        self._REGISTERED.clear()

    def _register(self, type_str: str, **kwargs) -> type[Simulation]:
        cls = _make_minimal_simulation_class(type_str)
        register_simulation_pipeline(**kwargs)(cls)
        self._REGISTERED.append(type_str)
        return cls

    def test_registered_class_is_retrievable(self) -> None:
        cls = self._register("_sp_reg_new")
        assert SIMULATION_PIPELINE_REGISTRY["_sp_reg_new"] is cls

    def test_duplicate_raises_without_overwrite(self) -> None:
        self._register("_sp_reg_dup")
        cls2 = _make_minimal_simulation_class("_sp_reg_dup")
        with pytest.raises(ValueError, match="already registered"):
            register_simulation_pipeline()(cls2)

    def test_overwrite_replaces_registration(self) -> None:
        self._register("_sp_reg_overwrite")
        cls2 = _make_minimal_simulation_class("_sp_reg_overwrite")
        register_simulation_pipeline(overwrite=True)(cls2)
        assert SIMULATION_PIPELINE_REGISTRY["_sp_reg_overwrite"] is cls2

    def test_decorator_returns_class_unchanged(self) -> None:
        cls = _make_minimal_simulation_class("_sp_reg_identity")
        result = register_simulation_pipeline()(cls)
        self._REGISTERED.append("_sp_reg_identity")
        assert result is cls


# ---------------------------------------------------------------------------
# get_simulation_pipeline_from_config
# ---------------------------------------------------------------------------


class TestGetSimulationPipelineFromConfig:
    def test_unknown_type_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_simulation_pipeline_from_config({"type": "_sp_does_not_exist_xyz"})

    def test_known_types_are_registered(self) -> None:
        for key in ("ode_simulation", "sde_simulation"):
            assert key in SIMULATION_PIPELINE_REGISTRY, f"{key} not in registry"
