import pytest
import torch

from dynopsi.data.featurization.base import Features
from dynopsi.simulation.observable import Observable, ObservableOutputConfig
from dynopsi.simulation.store import Store
from dynopsi.simulation.time_sampler import LinearTimeSampler, TimeSchedule

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_time_schedule(
    num_steps: int = 5,
    modalities: tuple[str, ...] = ("coords",),
    batch_size: int = 1,
) -> TimeSchedule:
    """Build a minimal valid TimeSchedule with shape (batch_size, num_steps, 1, 1)."""
    sampler = LinearTimeSampler(num_steps=num_steps, modalities=modalities)
    ts, dts = sampler.get_full_schedule(torch.device("cpu"))
    if batch_size > 1:
        return {m: ts[m].expand(batch_size, -1, -1, -1) for m in modalities}, {
            m: dts[m].expand(batch_size, -1, -1, -1) for m in modalities
        }
    return ts, dts


def make_features(batch_size: int = 2, seq_len: int = 4) -> Features:
    return Features(feature_dict={"coords": torch.zeros(batch_size, seq_len, 3)})


def make_observable(keys_any: list[str] = (), keys_final: list[str] = ()) -> Observable:
    configs: dict[str, ObservableOutputConfig] = {}
    for k in keys_any:
        configs[k] = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
    for k in keys_final:
        configs[k] = ObservableOutputConfig(state_type="any", outputs={"final": ("npz",)})
    return Observable(observable_output_configs=configs)


def make_store(
    features: dict[str, Features] | None = None,
    time_schedule: TimeSchedule | None = None,
    observable: Observable | None = None,
    batch_slice: slice | None = None,
) -> Store:
    return Store(
        features=features or {"feat": make_features()},
        time_schedule=time_schedule or make_time_schedule(),
        observable=observable,
        batch_slice=batch_slice,
    )


def make_state(batch_size: int = 2, seq_len: int = 4) -> dict[str, torch.Tensor]:
    return {"coords": torch.ones(batch_size, seq_len, 9)}


# ---------------------------------------------------------------------------
# Store construction
# ---------------------------------------------------------------------------


class TestStoreConstruction:
    def test_valid_no_observable(self) -> None:
        store = make_store()
        assert store.states == {}
        assert store.observable is None
        assert store.sample_offset == 0

    def test_prior_sample_index_chunk_local_plus_offset(self) -> None:
        store = make_store(batch_slice=slice(0, 5))
        store.sample_offset = 10
        assert store.prior_sample_index() == 10

    def test_prior_sample_index_uses_batch_start(self) -> None:
        store = make_store(batch_slice=slice(5, 10))
        assert store.prior_sample_index() == 5

    def test_prior_sample_index_requires_batch_slice(self) -> None:
        store = make_store()
        with pytest.raises(ValueError, match="batch_slice"):
            store.prior_sample_index()

    def test_valid_with_observable(self) -> None:
        obs = make_observable(keys_any=["X"])
        store = make_store(observable=obs)
        assert store.observable is obs

    def test_states_defaults_to_empty(self) -> None:
        # states has init=False so it is excluded from the __init__ signature,
        # but Pydantic v2 still accepts it as a keyword arg and sets it.
        # The meaningful guarantee is just that a normal construction starts empty.
        store = make_store()
        assert store.states == {}

    # --- time_schedule shape validation ---

    @pytest.mark.parametrize(
        "bad_shape",
        [
            (5,),  # 1-D: missing batch / spatial dims
            (1, 5, 2, 1),  # spatial dim != 1
            (1, 5, 1, 2),  # trailing dim != 1
        ],
        ids=["1d", "spatial_dim_wrong", "trailing_dim_wrong"],
    )
    def test_invalid_time_schedule_ts_shape(self, bad_shape: tuple) -> None:
        bad_ts = {"coords": torch.zeros(*bad_shape)}
        bad_dts = {"coords": torch.zeros(1, 5, 1, 1)}
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_store(time_schedule=(bad_ts, bad_dts))

    def test_invalid_time_schedule_dts_shape(self) -> None:
        good_ts = {"coords": torch.zeros(1, 5, 1, 1)}
        bad_dts = {"coords": torch.zeros(1, 5, 2, 1)}
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_store(time_schedule=(good_ts, bad_dts))

    # --- observable / feature key overlap validation ---

    def test_observable_keys_any_overlap_with_features_raises(self) -> None:
        # "feat" is a feature key; listing it in keys_to_record_any_timestep must fail
        obs = make_observable(keys_any=["feat"])
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_store(features={"feat": make_features()}, observable=obs)

    def test_observable_keys_final_overlap_with_features_raises(self) -> None:
        obs = make_observable(keys_final=["feat"])
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_store(features={"feat": make_features()}, observable=obs)

    def test_observable_state_keys_disjoint_from_features_ok(self) -> None:
        obs = make_observable(keys_any=["X"], keys_final=["Y"])
        store = make_store(features={"feat": make_features()}, observable=obs)
        assert store.observable is obs


# ---------------------------------------------------------------------------
# get_time / get_timestep_size
# ---------------------------------------------------------------------------


class TestGetTimeAndTimestepSize:
    @pytest.mark.parametrize(
        "num_steps, batch_size_in, batch_size_out, error",
        [(5, 1, 4, False), (5, 4, 4, False), (5, 2, 5, True)],
        ids=["broadcast_1_to_4", "match_4_to_4", "mismatch_2_to_5"],
    )
    def test_get_time_and_timestep_batches(
        self, num_steps: int, batch_size_in: int, batch_size_out: int, error: bool
    ) -> None:
        store = make_store(time_schedule=make_time_schedule(num_steps=num_steps, batch_size=batch_size_in))

        if not error:
            t = store.get_time(step_num=0, modality="coords", batch_size=batch_size_out)
            assert t.shape == (batch_size_out, 1, 1), "Time tensor shape mismatch"
            dt = store.get_timestep_size(step_num=0, modality="coords", batch_size=batch_size_out)
            assert dt.shape == (batch_size_out, 1, 1), "Time step tensor shape mismatch"
        else:
            with pytest.raises(ValueError):
                store.get_time(step_num=0, modality="coords", batch_size=batch_size_out)
            with pytest.raises(ValueError):
                store.get_timestep_size(step_num=0, modality="coords", batch_size=batch_size_out)

    def test_get_time_values_correct(self) -> None:
        schedule = make_time_schedule(num_steps=4, modalities=("coords",), batch_size=1)
        store = make_store(time_schedule=schedule)
        expected = schedule[0]["coords"][0, 2, 0, 0]
        t = store.get_time(step_num=2, modality="coords", batch_size=1)
        assert t[0, 0, 0] == expected


# ---------------------------------------------------------------------------
# get_features / put_features
# ---------------------------------------------------------------------------


class TestGetAndPutFeatures:
    def test_get_existing_key(self) -> None:
        feat = make_features()
        store = make_store(features={"f": feat})
        assert store.get_features("f") is feat

    def test_get_missing_key_raises(self) -> None:
        store = make_store()
        with pytest.raises(KeyError, match="Unknown feature key"):
            store.get_features("nonexistent")

    def test_put_new_key(self) -> None:
        store = make_store()
        new_feat = make_features()
        store.put_features("new_f", new_feat)
        assert store.get_features("new_f") is new_feat

    def test_put_existing_key_raises(self) -> None:
        store = make_store(features={"f": make_features()})
        with pytest.raises(ValueError, match="already in features"):
            store.put_features("f", make_features())


# ---------------------------------------------------------------------------
# put / get_state
# ---------------------------------------------------------------------------


class TestPutAndGetState:
    def test_put_and_get_roundtrip(self) -> None:
        store = make_store()
        state = make_state()
        store.put("X", state, step_num=0)
        result = store.get_state("X", step_num=0)
        assert torch.equal(result["coords"], state["coords"])

    def test_put_duplicate_step_raises(self) -> None:
        store = make_store()
        store.put("X", make_state(), step_num=0)
        with pytest.raises(ValueError, match="already exists at step"):
            store.put("X", make_state(), step_num=0)

    def test_put_same_key_different_steps(self) -> None:
        store = make_store()
        store.put("X", make_state(), step_num=0)
        store.put("X", make_state(), step_num=1)
        assert 0 in store.states["X"] and 1 in store.states["X"]

    def test_put_feature_key_raises(self) -> None:
        # feature keys and state keys must be disjoint
        store = make_store(features={"feat": make_features()})
        with pytest.raises(ValueError, match="feature key"):
            store.put("feat", make_state(), step_num=0)

    def test_get_missing_key_raises(self) -> None:
        store = make_store()
        with pytest.raises(KeyError, match="Unknown store key"):
            store.get_state("X", step_num=0)

    def test_get_missing_step_raises(self) -> None:
        store = make_store()
        store.put("X", make_state(), step_num=0)
        with pytest.raises(KeyError, match="No state for key"):
            store.get_state("X", step_num=99)

    def test_put_records_to_observable_when_key_tracked(self) -> None:
        obs = make_observable(keys_any=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=0)
        assert obs.get_observable_step("X", 0) is not None

    def test_put_does_not_record_when_key_not_tracked(self) -> None:
        obs = make_observable(keys_any=["Y"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=0)
        # "X" is not in keys_to_record_any_timestep — observable should stay empty for it
        assert obs.get_observable_step("X", 0) is None

    def test_put_does_not_record_when_no_observable(self) -> None:
        store = make_store(observable=None)
        store.put("X", make_state(), step_num=0)
        # no error, no observable to check — just verify state was stored
        assert "X" in store.states

    def test_put_snapshot_is_detached_copy(self) -> None:
        # The observable snapshot must not share storage with the original tensor
        obs = make_observable(keys_any=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        original = make_state(batch_size=2)
        store.put("X", original, step_num=0)
        original["coords"][0] = 999.0
        recorded = obs.get_observable_step("X", 0)
        assert recorded is not None
        assert recorded["coords"][0].max().item() != 999.0


# ---------------------------------------------------------------------------
# record_final_timestep
# ---------------------------------------------------------------------------


class TestRecordFinalTimestep:
    def test_no_observable_is_noop(self) -> None:
        store = make_store(observable=None)
        store.put("X", make_state(), step_num=4)
        # should not raise even though there's no observable
        store.record_final_timestep("X", step_num=4)

    def test_already_in_observable_is_noop(self) -> None:
        obs = make_observable(keys_any=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=4)
        # put() already recorded step 4; calling record_final_timestep should not raise or duplicate
        store.record_final_timestep("X", step_num=4)
        assert obs.get_observable_step("X", 4) is not None

    def test_records_from_states_when_not_in_observable(self) -> None:
        # keys_final keys are NOT auto-recorded by put(); record_final_timestep does it explicitly
        obs = make_observable(keys_final=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=4)
        assert obs.get_observable_step("X", 4) is None  # not recorded yet
        store.record_final_timestep("X", step_num=4)
        assert obs.get_observable_step("X", 4) is not None

    def test_final_only_fills_later_batch_slice(self) -> None:
        obs = make_observable(keys_final=["X"])
        obs.total_batch_size = 4
        first = make_store(observable=obs, batch_slice=slice(0, 2))
        first.put("X", {"coords": torch.full((2, 4, 9), 1.0)}, step_num=4)
        first.record_final_timestep("X", step_num=4)
        second = make_store(observable=obs, batch_slice=slice(2, 4))
        second.put("X", {"coords": torch.full((2, 4, 9), 7.0)}, step_num=4)
        second.record_final_timestep("X", step_num=4)
        recorded = obs.get_observable_step("X", 4)
        assert recorded is not None
        assert torch.equal(recorded["coords"][:2], torch.full((2, 4, 9), 1.0))
        assert torch.equal(recorded["coords"][2:], torch.full((2, 4, 9), 7.0))

    def test_missing_key_raises(self) -> None:
        obs = make_observable(keys_final=["X"])
        store = make_store(observable=obs)
        with pytest.raises(KeyError):
            store.record_final_timestep("X", step_num=0)

    def test_missing_step_raises(self) -> None:
        obs = make_observable(keys_final=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=3)
        with pytest.raises(KeyError):
            store.record_final_timestep("X", step_num=99)


# ---------------------------------------------------------------------------
# clear_states
# ---------------------------------------------------------------------------


class TestClearStates:
    def test_clears_all_other_steps(self) -> None:
        store = make_store()
        for step in range(5):
            store.put("X", make_state(), step_num=step)
        store.clear_states(keep_step_num=3)
        assert list(store.states["X"].keys()) == [3]

    def test_keeps_correct_tensor(self) -> None:
        store = make_store()
        kept_state = {"coords": torch.full((2, 4, 3), 42.0)}
        store.put("X", make_state(), step_num=0)
        store.put("X", kept_state, step_num=1)
        store.clear_states(keep_step_num=1)
        assert torch.equal(store.states["X"][1]["coords"], kept_state["coords"])

    def test_key_without_kept_step_is_deleted(self) -> None:
        # "Y" only has step 0; keep_step_num=1 removes "Y" entirely
        store = make_store()
        store.put("X", make_state(), step_num=0)
        store.put("X", make_state(), step_num=1)
        store.put("Y", make_state(), step_num=0)
        store.clear_states(keep_step_num=1)
        assert "Y" not in store.states

    def test_empty_states_is_noop(self) -> None:
        store = make_store()
        store.clear_states(keep_step_num=0)
        assert store.states == {}

    def test_preserves_previous_sc(self) -> None:
        store = make_store()
        kept = {"coords": torch.ones(2, 4, 3)}
        store.previous_sc["infer"] = kept
        store.put("X", make_state(), step_num=0)
        store.put("X", make_state(), step_num=1)
        store.clear_states(keep_step_num=1)
        assert store.previous_sc["infer"] is kept
        assert list(store.states["X"].keys()) == [1]


# ---------------------------------------------------------------------------
# _infer_batch_size
# ---------------------------------------------------------------------------


class TestInferBatchSize:
    def test_infers_from_features(self) -> None:
        feat = Features(feature_dict={"coords": torch.zeros(3, 4, 3)})
        store = make_store(features={"f": feat})
        # verify that batch size is derived from the features, not state
        store.put("X", {"coords": torch.zeros(4, 20, 3)}, step_num=0)
        assert store._infer_batch_size() == 3

    @pytest.mark.parametrize("batch_size", [1, 20])
    def test_infers_from_states_when_no_feature_tensors(self, batch_size: int) -> None:
        # Features with an empty feature_dict — batch size must come from states
        store = make_store(features={"f": Features(feature_dict={})})
        store.put("X", {"coords": torch.zeros(batch_size, 4, 3)}, step_num=0)
        # state tensor has dim>1, batch_size comes from shape[1] per _infer_batch_size logic
        assert store._infer_batch_size() == batch_size

    def test_raises_when_no_tensors(self) -> None:
        store = make_store(features={"f": Features(feature_dict={})})
        with pytest.raises(ValueError, match="Cannot infer batch size"):
            store._infer_batch_size()


# ---------------------------------------------------------------------------
# to_device
# ---------------------------------------------------------------------------


class TestToDevice:
    def test_feature_dict_is_replaced(self) -> None:
        # to_device rebuilds feat.feature_dict via a dict comprehension, so the
        # dict object is replaced (not mutated in place).
        feat = make_features()
        store = make_store(features={"f": feat})
        original_dict_id = id(feat.feature_dict)
        store.to_device(torch.device("cpu"))
        assert id(feat.feature_dict) != original_dict_id

    def test_feature_tensors_accessible_after_move(self) -> None:
        store = make_store(features={"f": make_features(batch_size=2)})
        store.to_device(torch.device("cpu"))
        tensors = store.get_features("f").feature_dict
        assert "coords" in tensors
        assert tensors["coords"].shape == (2, 4, 3)

    def test_state_tensors_moved_to_device(self) -> None:
        store = make_store()
        state = {"coords": torch.zeros(2, 4, 3)}
        store.put("X", state, step_num=0)
        store.to_device(torch.device("meta"))
        assert store.states["X"][0]["coords"].device == torch.device("meta")

    def test_observable_not_touched(self) -> None:
        # Docstring says observable is excluded from device movement.
        obs = make_observable(keys_any=["X"])
        store = make_store(observable=obs, batch_slice=slice(0, 2))
        store.put("X", make_state(batch_size=2), step_num=0)
        obs_dict_id = id(obs.observable_outputs)
        store.to_device(torch.device("cpu"))
        assert id(obs.observable_outputs) == obs_dict_id
