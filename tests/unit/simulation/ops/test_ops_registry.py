from typing import Literal

import pytest

from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.ops.registry import OPS_REGISTRY, get_op_from_config, get_op_from_type, register_op

# ---------------------------------------------------------------------------
# register_op
# ---------------------------------------------------------------------------


class TestRegisterOp:
    def test_registers_op_by_type_literal(self) -> None:
        @register_op()
        class _RegTestOp(Op):
            type: Literal["_reg_test_new"] = "_reg_test_new"

            def run(self, store, step_num) -> None:
                pass

        assert "_reg_test_new" in OPS_REGISTRY
        assert OPS_REGISTRY["_reg_test_new"] is _RegTestOp

    def test_raises_on_duplicate_registration(self) -> None:
        @register_op()
        class _DupOp1(Op):
            type: Literal["_reg_test_dup"] = "_reg_test_dup"

            def run(self, store, step_num) -> None:
                pass

        with pytest.raises(ValueError, match="already registered"):

            @register_op()
            class _DupOp2(Op):
                type: Literal["_reg_test_dup"] = "_reg_test_dup"

                def run(self, store, step_num) -> None:
                    pass

    def test_overwrite_replaces_existing(self) -> None:
        @register_op()
        class _OverA(Op):
            type: Literal["_reg_test_overwrite"] = "_reg_test_overwrite"

            def run(self, store, step_num) -> None:
                pass

        @register_op(overwrite=True)
        class _OverB(Op):
            type: Literal["_reg_test_overwrite"] = "_reg_test_overwrite"

            def run(self, store, step_num) -> None:
                pass

        assert OPS_REGISTRY["_reg_test_overwrite"] is _OverB


# ---------------------------------------------------------------------------
# get_op_from_type
# ---------------------------------------------------------------------------


class TestGetOpFromType:
    def test_returns_registered_class(self) -> None:
        from dynopsi.simulation.ops.ode_update import ODEStepUpdate

        assert get_op_from_type("ode_update") is ODEStepUpdate

    def test_raises_on_unknown_type(self) -> None:
        with pytest.raises(KeyError):
            get_op_from_type("nonexistent_op_type_xyz")


# ---------------------------------------------------------------------------
# get_op_from_config
# ---------------------------------------------------------------------------


class TestGetOpFromConfig:
    def test_deserializes_ode_update(self) -> None:
        from dynopsi.simulation.ops.ode_update import ODEStepUpdate

        op = get_op_from_config(
            {
                "type": "ode_update",
                "name": "test",
                "valid_steps": {"coords": [0, 1]},
                "in_x_loc_store": "x",
                "in_cvf_loc_store": "cvf",
                "out_x_loc_store": "x_out",
            }
        )
        assert isinstance(op, ODEStepUpdate)
        assert op.name == "test"

    def test_raises_on_unknown_type(self) -> None:
        with pytest.raises(KeyError):
            get_op_from_config({"type": "totally_unknown_op"})
