from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dynopsi.simulation.ops.op import Op


class ConcreteOp(Op):
    """Subclass of Op to test instantiation."""

    type: Literal["concrete_op"] = "concrete_op"


@pytest.fixture
def valid_steps() -> dict[str, list[int]]:
    return {"coords": [0, 1, 2]}


class TestOpBase:
    @pytest.fixture
    def store(self) -> MagicMock:
        return MagicMock()

    def test_instantiation_valid(self, valid_steps: dict[str, list[int]]) -> None:
        op = ConcreteOp(valid_steps=valid_steps, name="test_op")
        assert op.valid_steps == valid_steps

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {},
            {"valid_steps": "not_a_dict"},
        ],
        ids=["empty_kwargs", "valid_steps_not_a_dict"],
    )
    def test_instantiation_invalid(self, invalid_kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            ConcreteOp(**invalid_kwargs)

    def test_state_keys(self, valid_steps: dict[str, list[int]]) -> None:
        op = ConcreteOp(valid_steps=valid_steps, name="test_op")
        assert op.input_state_keys(0) == set(), "Input state keys should be empty"
        assert op.input_feature_keys() == set(), "Input feature keys should be empty"
        assert op.output_state_keys(0) == set(), "Output state keys should be empty"

    @pytest.mark.parametrize("step_num", [0, 1, 100])
    def test_state_keys_step_no_effect(self, step_num: int, valid_steps: dict[str, list[int]]) -> None:
        op = ConcreteOp(valid_steps=valid_steps, name="test_op")
        assert op.input_state_keys(step_num) == set(), "Input state keys should be empty"
        assert op.output_state_keys(step_num) == set(), "Output state keys should be empty"

    def test_model_configs(self, valid_steps: dict[str, list[int]]) -> None:
        op = ConcreteOp(valid_steps=valid_steps, name="test_op")
        assert op.model_configs() == [], "Model configs should be empty"

    def test_run_not_implemented(self, store: MagicMock, valid_steps: dict[str, list[int]]) -> None:
        op = ConcreteOp(valid_steps=valid_steps, name="test_op")
        with pytest.raises(NotImplementedError):
            op.run(store, 0)
