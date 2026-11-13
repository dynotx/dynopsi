from typing import Literal
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.sample_centers import (
    SAMPLE_CENTERS_OF_MASS_METADATA_KEY,
    narrow_per_sample_features,
)
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.build_initial_state import InitializeFeatures, InitializeState, InitializeStep
from dynopsi.simulation.observable import Observable, ObservableOutputConfig
from dynopsi.simulation.ops.infer import Infer, SelfConditioningMode
from dynopsi.simulation.ops.op import Op
from dynopsi.simulation.simulation import Simulation
from dynopsi.simulation.store import Store
from dynopsi.simulation.time_sampler import LinearTimeSampler
from dynopsi.simulation.utils import is_active_at

NUM_STEPS = 3


class DummyOp(Op):
    type: Literal["_op_dummy"] = "_op_dummy"

    in_feat_loc_store: str
    in_state_loc_store: str
    out_state_loc_store: str

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def input_state_keys(self, step_num) -> set[str]:
        if is_active_at(self.valid_steps, step_num):
            return {self.in_state_loc_store}
        return set()

    def output_state_keys(self, step_num) -> set[str]:
        if is_active_at(self.valid_steps, step_num):
            return {self.out_state_loc_store}
        return set()

    def run(self, store, step_num) -> None:
        pass


class DummyOpSameStep(DummyOp):
    def get_output_step(self, input_step_num: int) -> int:
        if is_active_at(self.valid_steps, input_step_num):
            return input_step_num
        return None


class DummyOpNextStep(DummyOp):
    def get_output_step(self, input_step_num: int) -> int:
        if is_active_at(self.valid_steps, input_step_num):
            return input_step_num + 1
        return None


class DummyInit(InitializeStep):
    type: Literal["_init_dummy"] = "_init_dummy"

    in_feat_loc_store: str
    out_feat_loc_store: str
    out_state_loc_store: str

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def output_feature_keys(self) -> set[str]:
        return {self.out_feat_loc_store}

    def output_state_keys(self, step_num) -> set[str]:
        if step_num == 0:
            return {self.out_state_loc_store}
        return set()

    def run(self, store) -> None:
        pass


def make_simulation(
    plan_init=None,
    plan=None,
    num_steps=NUM_STEPS,
    observable_output_configs=None,
) -> Simulation:
    return Simulation(
        name="test",
        plan_init=plan_init if plan_init is not None else [],
        plan=plan if plan is not None else [],
        time_sampler=LinearTimeSampler(num_steps=num_steps, modalities=["coords"]),
        observable_output_configs=observable_output_configs or {},
    )


def make_init_features(name="dummy_init", in_feat="raw_feat", out_feat="prepared_feat") -> InitializeFeatures:
    return InitializeFeatures(
        name=name,
        in_feat_loc_store=in_feat,
        out_feat_loc_store=out_feat,
    )


def make_init_state(name="dummy_init", in_feat="prepared_feat", out_state="prepared_state") -> InitializeState:
    return InitializeState.model_construct(
        name=name,
        in_feat_loc_store=in_feat,
        out_loc_store=out_state,
        inference_model_config=MagicMock(),
    )


def make_op(
    output_step_gap: int,
    in_feat: str = "prepared_feat",
    in_state: str = "prepared_state",
    out_state: str = "out_state",
    valid_steps=None,
) -> Op:
    """
    Make an op that performs a specific operation based on the output step gap.

    The easiest for testing is output_step_gap = 1 with in_state = out_state.
    """
    vs = valid_steps or {"coords": list(range(NUM_STEPS))}
    if output_step_gap == 0:
        return DummyOpSameStep(
            name="dummy_same_step",
            in_feat_loc_store=in_feat,
            in_state_loc_store=in_state,
            out_state_loc_store=out_state,
            valid_steps=vs,
        )
    elif output_step_gap == 1:
        return DummyOpNextStep(
            name="dummy_step_update",
            in_state_loc_store=in_state,
            in_feat_loc_store=in_feat,
            out_state_loc_store=out_state,
            valid_steps=vs,
        )
    else:
        raise ValueError("output_step_gap must be 0 or 1")


# ---------------------------------------------------------------------------
# Construction and model_validator
# ---------------------------------------------------------------------------


class TestSimulationConstruction:
    def test_empty_plans_valid(self) -> None:
        sim = make_simulation()
        assert sim.name == "test"

    def test_previous_step_infer_names_must_be_unique(self) -> None:
        cfg = DynoPsiModelConfig.model_construct(dynopsi_model=MagicMock())

        def make_infer(name: str, mode: SelfConditioningMode = "previous_step") -> Infer:
            return Infer(
                name=name,
                valid_steps={"coords": [0]},
                inference_model_config=cfg,
                in_x_loc_store="x",
                in_feat_loc_store="feat",
                out_loc_store="cvf",
                self_conditioning_mode=mode,
            )

        with pytest.raises(ValueError, match="distinct names"):
            make_simulation(plan=[make_infer("infer"), make_infer("infer")])
        make_simulation(plan=[make_infer("infer_a"), make_infer("infer_b")])
        make_simulation(plan=[make_infer("infer", "same_timestep"), make_infer("infer", "same_timestep")])

    def test_intermediate_observables_empty_for_basic_ops(self) -> None:
        sim = make_simulation(plan=[make_op(output_step_gap=1)])
        assert sim.intermediate_observables == {}

    def test_intermediate_keys_cannot_overlap_with_observable_output_configs(self) -> None:
        class OpWithIntermediate(Op):
            type: Literal["_op_intermediate_test"] = "_op_intermediate_test"

            def intermediate_observable_keys(self) -> set[str]:
                return {"shared_key"}

            def run(self, store, step_num) -> None:
                pass

        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        with pytest.raises(ValueError, match="cannot overlap"):
            make_simulation(
                plan=[OpWithIntermediate(name="op", valid_steps={"coords": [0]})],
                observable_output_configs={"shared_key": cfg},
            )

    def test_overlapping_intermediate_keys_across_ops_raises(self) -> None:
        class OpInterA(Op):
            type: Literal["_op_inter_a"] = "_op_inter_a"

            def intermediate_observable_keys(self) -> set[str]:
                return {"dup_key"}

            def run(self, store, step_num) -> None:
                pass

        class OpInterB(Op):
            type: Literal["_op_inter_b"] = "_op_inter_b"

            def intermediate_observable_keys(self) -> set[str]:
                return {"dup_key"}

            def run(self, store, step_num) -> None:
                pass

        with pytest.raises(ValueError, match="overlapping"):
            make_simulation(
                plan=[
                    OpInterA(name="a", valid_steps={"coords": [0]}),
                    OpInterB(name="b", valid_steps={"coords": [0]}),
                ]
            )


# ---------------------------------------------------------------------------
# expected_feature_keys
# ---------------------------------------------------------------------------


class TestExpectedFeatureKeys:
    def test_aggregates_from_plan_init(self) -> None:
        sim = make_simulation(plan_init=[make_init_features(in_feat="raw")])
        assert "raw" in sim.expected_feature_keys

    def test_aggregates_from_plan_ops(self) -> None:
        class OpWithFeature(Op):
            type: Literal["_op_with_feature"] = "_op_with_feature"

            def input_feature_keys(self) -> set[str]:
                return {"feat_a"}

            def run(self, store, step_num) -> None:
                pass

        sim = make_simulation(plan=[OpWithFeature(name="op", valid_steps={"coords": [0]})])
        assert "feat_a" in sim.expected_feature_keys

    def test_empty_when_no_ops(self) -> None:
        sim = make_simulation()
        assert sim.expected_feature_keys == set()


# ---------------------------------------------------------------------------
# setup_observable
# ---------------------------------------------------------------------------


class TestSetupObservable:
    def test_sets_total_batch_size(self) -> None:
        sim = make_simulation()
        sim.setup_observable(8)
        assert sim.observable.total_batch_size == 8

    def test_includes_observable_output_configs(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        sim = make_simulation(observable_output_configs={"X": cfg})
        sim.setup_observable(2)
        assert "X" in sim.observable.observable_output_configs

    def test_replaces_previous_observable(self) -> None:
        sim = make_simulation()
        sim.setup_observable(4)
        sim.setup_observable(7)
        assert sim.observable.total_batch_size == 7


# ---------------------------------------------------------------------------
# _validate
# ---------------------------------------------------------------------------


class TestValidate:
    """
    - Test features keys in the init step.
    - Test init step makes all the appropriate keys for subsequent ops.
    - Test that there are not state key collisions
    - Check final step observable output keys
    - Check 'any' step observable output keys
    """

    @pytest.mark.parametrize("features_key", ["expected_key", "unexpected_key"])
    def test_features_keys_and_init_step(self, features_key: str) -> None:
        sim = make_simulation(plan_init=[make_init_features(in_feat="expected_key")])
        if features_key == "expected_key":
            sim._validate({features_key: MagicMock()})  # should not raise
            # also check empty
            with pytest.raises(KeyError):
                sim._validate({})
        else:
            with pytest.raises(KeyError):
                sim._validate({features_key: MagicMock()})

    @pytest.mark.parametrize("processed_features_key", ["expected_key", "unexpected_key"])
    def test_features_keys_main_plan(self, processed_features_key: str) -> None:
        """Plan init creates a features key that is used in the main plan."""
        sim = make_simulation(
            plan_init=[
                make_init_features(in_feat="raw", out_feat="expected_key"),
                make_init_state(in_feat="expected_key"),
            ],
            plan=[
                make_op(
                    output_step_gap=1,
                    in_feat=processed_features_key,
                    in_state="prepared_state",
                    out_state="prepared_state",
                )
            ],
        )
        if processed_features_key == "expected_key":
            sim._validate({"raw": MagicMock()})  # should not raise
        else:
            with pytest.raises(KeyError):
                sim._validate({"raw": MagicMock()})

    @pytest.mark.parametrize("state_key", ["expected_state", "unexpected_state"])
    def test_state_keys_main_plan(self, state_key: str) -> None:
        """Plan init creates a state key that is used in the main plan."""
        sim = make_simulation(
            plan_init=[make_init_features(in_feat="raw"), make_init_state(out_state="expected_state")],
            plan=[make_op(output_step_gap=1, in_state=state_key, out_state=state_key)],
        )
        if state_key == "expected_state":
            sim._validate({"raw": MagicMock()})  # should not raise
        else:
            with pytest.raises(KeyError):
                sim._validate({"raw": MagicMock()})

    def test_raises_feature_key_collision_between_init_steps(self) -> None:
        # Two init steps both write to the same features
        sim = make_simulation(plan_init=[make_init_features(in_feat="raw"), make_init_features(in_feat="raw")])
        with pytest.raises(KeyError, match="collision"):
            sim._validate({"raw": MagicMock()})

    def test_raises_feature_key_collision_init_step_self(self) -> None:
        # Single init step writes back to the same features
        sim = make_simulation(plan_init=[make_init_features(in_feat="raw", out_feat="raw")])
        with pytest.raises(KeyError, match="collision"):
            sim._validate({"raw": MagicMock()})

    def test_raises_state_key_collision_between_init_steps(self) -> None:
        # Two init steps both write to "x_out" at step 0.
        sim = make_simulation(
            plan_init=[
                make_init_state(in_feat="raw"),
                make_init_state(in_feat="raw"),
            ]
        )
        with pytest.raises(KeyError, match="collision"):
            sim._validate({"raw": MagicMock()})

    @pytest.mark.parametrize("output_step_gap", [0, 1])
    def test_raises_state_key_collision_between_ops(self, output_step_gap: int) -> None:
        # Two ops both write to "x_out"
        sim = make_simulation(
            plan_init=[make_init_state(in_feat="raw", out_state="prepared_state")],
            plan=[
                make_op(output_step_gap=output_step_gap, in_feat="raw", in_state="prepared_state", out_state="x_out"),
                make_op(output_step_gap=output_step_gap, in_feat="raw", in_state="prepared_state", out_state="x_out"),
            ],
        )
        with pytest.raises(KeyError, match="collision"):
            sim._validate({"raw": MagicMock()})

    def test_raises_state_key_collision_self(self) -> None:
        # op in plan collides with itself
        sim = make_simulation(
            plan_init=[make_init_state(in_feat="raw", out_state="prepared_state")],
            plan=[
                make_op(output_step_gap=0, in_feat="raw", in_state="prepared_state", out_state="prepared_state"),
            ],
        )
        with pytest.raises(KeyError, match="collision"):
            sim._validate({"raw": MagicMock()})

    def test_plan_does_not_produce_intermediate_states(self) -> None:
        # No op writes to the next step, so plan fails
        sim = make_simulation(
            plan_init=[make_init_state(in_feat="raw")],
            plan=[
                make_op(output_step_gap=0, in_feat="raw"),
            ],
        )
        with pytest.raises(KeyError, match="missing|not found"):
            sim._validate({"raw": MagicMock()})

    def test_raises_when_any_observable_key_not_produced_by_plan(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        sim = make_simulation(
            plan_init=[make_init_state(in_feat="raw", out_state="prepared_features")],
            plan=[],
            observable_output_configs={"ghost_key": cfg},
        )
        with pytest.raises(KeyError):
            sim._validate({"raw": MagicMock()})

    def test_raises_when_final_observable_key_not_at_final_step(self) -> None:
        # ODEStepUpdate writes at step_num+1, so at the last step it writes to "x_out".
        # If a "final" observable asks for "ghost", it should fail.
        cfg = ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": ("npz",)})
        sim = make_simulation(
            plan_init=[make_init_state(in_feat="raw", out_state="x_out")],
            plan=[make_op(output_step_gap=1, in_state="x_out", out_state="x_out")],
            observable_output_configs={"ghost_key": cfg},
        )
        with pytest.raises(KeyError):
            sim._validate({})


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


class TestRunBatch:
    """
    Tests for Simulation.run_batch().

    Ops and init steps have their run() methods patched so tests focus on
    the orchestration logic (call counts, ordering, state clearing, observable
    recording) rather than the ops themselves.

    Note that the patching occurs on the class level so total call count is
    assessed instead of individual. This is because pydantic doesn't allow
    patching at the instance level.
    """

    device: torch.device = torch.device("cpu")
    batch_slice: slice = slice(0, 2)
    batch_size: int = 2

    def _make_features(self) -> dict[str, Features]:
        x = torch.ones(self.batch_size, 4, 3)
        return {"feat": Features(feature_dict={"coords": x})}

    # ------------------------------------------------------------------
    # plan_init ordering and call count
    # ------------------------------------------------------------------

    def test_plan_init_step_runs_exactly_once(self) -> None:
        init = make_init_features()
        sim = make_simulation(plan_init=[init])
        sim.setup_observable(self.batch_size)

        # Patch the class, not the instance: pydantic's __setattr__ only allows
        # setting model fields on instances, so instance-level patching raises
        # AttributeError. Class-level patching modifies the class __dict__ directly.
        with mock.patch.object(InitializeFeatures, "run") as mock_run:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        mock_run.assert_called_once()

    def test_run_batch_narrows_using_sample_offset(self) -> None:
        sim = make_simulation(plan_init=[make_init_features()])
        sim.setup_observable(self.batch_size)

        with (
            mock.patch.object(InitializeFeatures, "run"),
            mock.patch(
                "dynopsi.simulation.simulation.narrow_per_sample_features",
                wraps=narrow_per_sample_features,
            ) as narrow,
        ):
            sim.run_batch(self._make_features(), slice(0, 2), self.device, sample_offset=3)

        assert narrow.call_args.args[1] == slice(3, 5)

    def test_multiple_init_steps_each_run_once(self) -> None:
        init1 = make_init_features(name="init1", in_feat="raw_feat", out_feat="prepared_feat")
        init2 = make_init_features(name="init2", in_feat="raw_feat", out_feat="other_feat")
        sim = make_simulation(plan_init=[init1, init2])
        sim.setup_observable(self.batch_size)

        # Both inits are InitializeFeatures instances; one class-level patch covers
        # both, so we assert the combined call count equals the number of init steps.
        with mock.patch.object(InitializeFeatures, "run") as mock_run:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        assert mock_run.call_count == 2

    def test_init_runs_before_plan_ops(self) -> None:
        call_order: list[str] = []
        init = make_init_features()
        op = make_op(output_step_gap=1)
        sim = make_simulation(plan_init=[init], plan=[op])
        sim.setup_observable(self.batch_size)

        with (
            mock.patch.object(InitializeFeatures, "run", side_effect=lambda store: call_order.append("init")),
            mock.patch.object(DummyOpNextStep, "run", side_effect=lambda store, step_num: call_order.append("op")),
        ):
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        assert call_order[0] == "init"
        assert all(item == "op" for item in call_order[1:])

    # ------------------------------------------------------------------
    # plan op call counts and step numbers
    # ------------------------------------------------------------------

    def test_plan_op_runs_once_per_step(self) -> None:
        op = make_op(output_step_gap=1)
        sim = make_simulation(plan=[op])
        sim.setup_observable(self.batch_size)

        with mock.patch.object(DummyOpNextStep, "run") as mock_run:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        assert mock_run.call_count == NUM_STEPS

    def test_multiple_ops_each_run_once_per_step(self) -> None:
        # Two DummyOpNextStep instances share the same class-level patch, so we
        # verify the combined call count equals num_ops * NUM_STEPS.
        op1 = make_op(output_step_gap=1, out_state="state_a")
        op2 = make_op(output_step_gap=1, in_state="state_a", out_state="state_b")
        sim = make_simulation(plan=[op1, op2])
        sim.setup_observable(self.batch_size)

        with mock.patch.object(DummyOpNextStep, "run") as mock_run:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        assert mock_run.call_count == 2 * NUM_STEPS

    def test_op_receives_correct_step_nums(self) -> None:
        op = make_op(output_step_gap=1)
        sim = make_simulation(plan=[op])
        sim.setup_observable(self.batch_size)

        with mock.patch.object(DummyOpNextStep, "run") as mock_run:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        # Class-level patching: MagicMock is not a descriptor, so op.run(store, step_num)
        # calls MagicMock(store, step_num) — args[0]=store, args[1]=step_num.
        called_step_nums = [call_args.args[1] for call_args in mock_run.call_args_list]
        assert called_step_nums == list(sim.time_sampler.get_steps())

    # ------------------------------------------------------------------
    # clear_states_each_step
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("clear_states", [True, False])
    def test_clear_states_called_according_to_flag(self, clear_states: bool) -> None:
        sim = Simulation(
            name="test",
            plan_init=[],
            plan=[],
            time_sampler=LinearTimeSampler(num_steps=NUM_STEPS, modalities=["coords"]),
            clear_states_each_step=clear_states,
        )
        sim.setup_observable(self.batch_size)

        with mock.patch.object(Store, "clear_states") as mock_clear:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        if clear_states:
            assert mock_clear.call_count == NUM_STEPS
            called_keep_steps = [c.kwargs["keep_step_num"] for c in mock_clear.call_args_list]
            expected_steps = [s + 1 for s in sim.time_sampler.get_steps()]
            assert called_keep_steps == expected_steps
        else:
            mock_clear.assert_not_called()

    # ------------------------------------------------------------------
    # observable final-timestep recording
    # ------------------------------------------------------------------

    def test_final_timestep_recorded_for_final_observable_keys(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
        sim = make_simulation(observable_output_configs={"x_out": cfg})
        sim.setup_observable(self.batch_size)

        with mock.patch.object(Store, "record_final_timestep") as mock_record:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        last_step = list(sim.time_sampler.get_steps())[-1]
        mock_record.assert_called_once_with("x_out", last_step + 1)

    def test_final_timestep_not_recorded_for_any_only_configs(self) -> None:
        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        sim = make_simulation(observable_output_configs={"x_out": cfg})
        sim.setup_observable(self.batch_size)

        with mock.patch.object(Store, "record_final_timestep") as mock_record:
            sim.run_batch(self._make_features(), self.batch_slice, self.device)

        mock_record.assert_not_called()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    """
    Tests for Simulation.run().

    run_batch is patched throughout to isolate orchestration logic from the
    simulation loop itself (tested separately in TestRunBatch).
    setup_models is left un-patched where plans are empty (it is a no-op),
    but patched when we need to assert it was called.
    """

    cpu = torch.device("cpu")

    def _make_sim(
        self,
        validate_before_run: bool = False,
        observable_output_configs: dict | None = None,
    ) -> Simulation:
        return Simulation(
            name="test",
            plan_init=[],
            plan=[],
            time_sampler=LinearTimeSampler(num_steps=NUM_STEPS, modalities=["coords"]),
            validate_before_run=validate_before_run,
            observable_output_configs=observable_output_configs or {},
        )

    def _make_features(self) -> dict[str, Features]:
        """Features with atoms_to_include set so the pre-run check in run() passes."""
        x = torch.ones(2, 4, 3)
        return {"feat": Features(feature_dict={"coords": x}, atoms_to_include=AtomFilter.FULL)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_validate_called_when_enabled(self) -> None:
        sim = self._make_sim(validate_before_run=True)
        features = self._make_features()

        with mock.patch.object(Simulation, "_validate") as mock_validate, mock.patch.object(Simulation, "run_batch"):
            sim.run(features, output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

        mock_validate.assert_called_once_with(features)

    def test_validate_not_called_when_disabled(self) -> None:
        sim = self._make_sim(validate_before_run=False)

        with mock.patch.object(Simulation, "_validate") as mock_validate, mock.patch.object(Simulation, "run_batch"):
            sim.run(self._make_features(), output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

        mock_validate.assert_not_called()

    # ------------------------------------------------------------------
    # Feature pre-flight check
    # ------------------------------------------------------------------

    def test_raises_when_features_missing_atom_filter(self) -> None:
        sim = self._make_sim()
        features = {"feat": Features(feature_dict={"coords": torch.ones(2, 4, 3)})}  # atoms_to_include=None

        with pytest.raises(ValueError, match="atoms_to_include"):
            sim.run(features, output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

    def test_raises_when_sample_centers_shorter_than_num_samples(self) -> None:
        sim = self._make_sim()
        features = self._make_features()
        features["feat"].featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = [(0.0, 0.0, 0.0)]
        with pytest.raises(ValueError, match="sample_centers_of_mass has 1 rows"):
            sim.run(features, output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    def test_raises_when_devices_none_and_no_cuda(self) -> None:
        sim = self._make_sim()

        with mock.patch("dynopsi.simulation.simulation.torch.cuda.device_count", return_value=0):
            with pytest.raises(ValueError, match="No devices"):
                sim.run(self._make_features(), output_dir=None, num_samples=2, batch_size=2, device=None)

    def test_setup_models_called_with_device(self) -> None:
        sim = self._make_sim()

        with mock.patch.object(Simulation, "setup_models") as mock_setup, mock.patch.object(Simulation, "run_batch"):
            sim.run(self._make_features(), output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

        mock_setup.assert_called_once_with(self.cpu)

    # ------------------------------------------------------------------
    # run_batch dispatch — single device
    # ------------------------------------------------------------------

    def test_run_batch_called_once_per_batch(self) -> None:
        # num_samples=4, batch_size=2, save_every_n_batches=1:
        # observable_size=2, two outer iterations, one batch each → 2 total calls.
        sim = self._make_sim()

        with mock.patch.object(Simulation, "run_batch") as mock_rb:
            sim.run(self._make_features(), output_dir=None, num_samples=4, batch_size=2, device=self.cpu)

        assert mock_rb.call_count == 2

    def test_run_batch_called_with_correct_batch_slices(self) -> None:
        # observable_size=2, batch_size=2 → each outer iteration has one batch: slice(0, 2).
        sim = self._make_sim()

        with mock.patch.object(Simulation, "run_batch") as mock_rb:
            sim.run(self._make_features(), output_dir=None, num_samples=4, batch_size=2, device=self.cpu)

        slices = [call.args[1] for call in mock_rb.call_args_list]
        offsets = [call.kwargs["sample_offset"] for call in mock_rb.call_args_list]
        assert slices == [slice(0, 2), slice(0, 2)]
        assert offsets == [0, 2]

    def test_multiple_batches_per_save_chunk(self) -> None:
        # save_every_n_batches=2: observable_size=4, num_samples=4 → 1 outer iteration,
        # 2 batches within it: slice(0,2) and slice(2,4).
        sim = self._make_sim()

        with mock.patch.object(Simulation, "run_batch") as mock_rb:
            sim.run(
                self._make_features(),
                output_dir=None,
                num_samples=4,
                batch_size=2,
                save_every_n_batches=2,
                device=self.cpu,
            )

        assert mock_rb.call_count == 2
        slices = [call.args[1] for call in mock_rb.call_args_list]
        assert slices == [slice(0, 2), slice(2, 4)]

    def test_final_batch_truncated_to_remaining_samples(self) -> None:
        # num_samples=3, batch_size=2: first batch slice(0,2), second slice(0,1) (only 1 left).
        sim = self._make_sim()

        with mock.patch.object(Simulation, "run_batch") as mock_rb:
            sim.run(
                self._make_features(),
                output_dir=None,
                num_samples=3,
                batch_size=2,
                save_every_n_batches=2,
                device=self.cpu,
            )

        slices = [call.args[1] for call in mock_rb.call_args_list]
        assert slices == [slice(0, 2), slice(2, 3)]

    def test_multiple_devices_request_raises(self) -> None:
        sim = self._make_sim()
        with pytest.raises(TypeError):
            sim.run(
                self._make_features(),
                output_dir=None,
                num_samples=4,
                batch_size=2,
                devices=[self.cpu, self.cpu],
            )

    # ------------------------------------------------------------------
    # Observable saving
    # ------------------------------------------------------------------

    def test_observable_save_called_when_output_dir_provided(self) -> None:
        sim = self._make_sim()

        with (
            mock.patch.object(Simulation, "run_batch"),
            mock.patch.object(Observable, "save", return_value=[]) as mock_save,
        ):
            sim.run(self._make_features(), output_dir="/tmp/out", num_samples=2, batch_size=2, device=self.cpu)

        mock_save.assert_called_once()

    def test_observable_save_not_called_when_output_dir_none(self) -> None:
        sim = self._make_sim()

        with (
            mock.patch.object(Simulation, "run_batch"),
            mock.patch.object(Observable, "save", return_value=[]) as mock_save,
        ):
            sim.run(self._make_features(), output_dir=None, num_samples=2, batch_size=2, device=self.cpu)

        mock_save.assert_not_called()

    def test_save_receives_correct_start_index_per_chunk(self) -> None:
        # Two outer iterations (observable_size=2, num_samples=4):
        # first save at start_index=0, second at start_index=2.
        sim = self._make_sim()

        with (
            mock.patch.object(Simulation, "run_batch"),
            mock.patch.object(Observable, "save", return_value=[]) as mock_save,
        ):
            sim.run(self._make_features(), output_dir="/tmp/out", num_samples=4, batch_size=2, device=self.cpu)

        start_indices = [call.kwargs["start_index"] for call in mock_save.call_args_list]
        assert start_indices == [0, 2]

    def test_output_files_accumulated_across_chunks(self) -> None:
        """Files returned by each save call are concatenated into the final return value."""
        sim = self._make_sim()
        dummy_file = MagicMock()

        with (
            mock.patch.object(Simulation, "run_batch"),
            mock.patch.object(Observable, "save", return_value=[dummy_file]),
        ):
            result = sim.run(
                self._make_features(),
                output_dir="/tmp/out",
                num_samples=4,
                batch_size=2,
                device=self.cpu,
            )

        # 2 outer iterations → 2 save calls → [dummy_file, dummy_file]
        assert result == [dummy_file, dummy_file]

    def test_returns_empty_list_when_no_output_dir(self) -> None:
        sim = self._make_sim()

        with mock.patch.object(Simulation, "run_batch"):
            result = sim.run(
                self._make_features(),
                output_dir=None,
                num_samples=2,
                batch_size=2,
                device=self.cpu,
            )

        assert result == []
