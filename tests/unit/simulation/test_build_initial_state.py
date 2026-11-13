from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.base import Features
from dynopsi.simulation.build_initial_state import InitializeFeatures, InitializeState, InitializeStep
from dynopsi.simulation.store import Store
from tests.unit.data.conftest import assert_feature_dict_equality


class ConcreteStep(InitializeStep):
    type: Literal["_concrete_step"] = "_concrete_step"

    def run(self, store) -> None:
        pass


def make_features(include_atom_filter: bool, seq_len: int, batch_size: int | None = None) -> Features:
    bb_n_atoms = 3
    original = torch.ones(seq_len)
    n_noised = seq_len // 2
    n_non_noised = seq_len - n_noised
    noised = torch.tensor([True] * n_noised + [False] * n_non_noised)
    feature_dict: dict = {
        "frame_resolved_mask": original.clone(),
        "token_index": torch.arange(seq_len),
        "coords": torch.zeros(1, seq_len * bb_n_atoms, 3),
        "frames_idx": torch.arange(seq_len * bb_n_atoms, dtype=torch.long).reshape(-1, 3),
        "prior_mean": torch.ones(1, seq_len * bb_n_atoms, 3),
        "noised_mask": noised,
    }
    if batch_size is not None:
        for k, v in feature_dict.items():
            if isinstance(v, torch.Tensor):
                feature_dict[k] = torch.stack([v] * batch_size)
    if include_atom_filter:
        features = Features(feature_dict=feature_dict, atoms_to_include=AtomFilter.BB)
    else:
        features = Features(feature_dict=feature_dict)
    return features


def make_store(
    features: dict[str, Features],
    batch_size: int,
    *,
    batch_start: int = 0,
    sample_offset: int = 0,
) -> Store:
    dummy_time_schedule = ({"coords": torch.zeros(1, 10, 1, 1)}, {"coords": torch.zeros(1, 10, 1, 1)})
    return Store(
        features=features,
        time_schedule=dummy_time_schedule,
        batch_slice=slice(batch_start, batch_start + batch_size),
        sample_offset=sample_offset,
    )


# ---------------------------------------------------------------------------
# InitializeStep base
# ---------------------------------------------------------------------------


class TestInitializeStep:
    def make(self) -> ConcreteStep:
        return ConcreteStep(name="step")

    def test_input_feature_keys_empty(self) -> None:
        assert self.make().input_feature_keys() == set()

    def test_output_feature_keys_empty(self) -> None:
        assert self.make().output_feature_keys() == set()

    def test_output_state_keys_empty(self) -> None:
        assert self.make().output_state_keys(0) == set()

    def test_intermediate_observable_keys_empty(self) -> None:
        assert self.make().intermediate_observable_keys() == set()

    def test_model_configs_empty(self) -> None:
        assert self.make().model_configs() == []


# ---------------------------------------------------------------------------
# InitializeFeatures
# ---------------------------------------------------------------------------


class TestInitializeFeatures:
    """
    Test the InitializeFeatures step.

    - The first chunk of tests verifies basic input/outputs for the InitializeStep class.
    - Tests with the prefix test_run test the run() method. For InitializeFeatures, the steps
    are:
    1. Get the features from the store
    2. Expand the features to batch size --> we need to check that the original feature tensors are
    not modified by this operation.
    3. Reshape coordinates 'true_coords'
    4. Put back in store
    """

    in_key = "input_features"
    out_key = "processed_features"

    @pytest.fixture
    def step(self) -> InitializeFeatures:
        return InitializeFeatures(name="init", in_feat_loc_store=self.in_key, out_feat_loc_store=self.out_key)

    def test_input_feature_keys(self, step: InitializeFeatures) -> None:
        assert step.input_feature_keys() == {self.in_key}

    def test_output_feature_keys(self, step: InitializeFeatures) -> None:
        assert step.output_feature_keys() == {self.out_key}

    def test_output_state_keys_empty(self, step: InitializeFeatures) -> None:
        assert step.output_state_keys(0) == set()

    def test_model_configs_empty(self, step: InitializeFeatures) -> None:
        assert step.model_configs() == []

    def test_type_discriminator(self) -> None:
        step = InitializeFeatures.model_validate(
            {
                "type": "initialize_features",
                "name": "init",
                "in_feat_loc_store": self.in_key,
                "out_feat_loc_store": self.out_key,
            }
        )
        assert isinstance(step, InitializeFeatures)

    def test_run(self, step: InitializeFeatures) -> None:
        batch_size = 3
        seq_len = 4
        features = make_features(include_atom_filter=True, seq_len=seq_len)
        features_copy = features.safe_copy()
        store = make_store(features={self.in_key: features}, batch_size=batch_size)

        step.run(store)

        # check size of outputs --> everything is the appropriate batch size
        expected_features = {
            "frame_resolved_mask": torch.ones(batch_size, seq_len),
            "token_index": torch.tile(torch.arange(seq_len), (1, batch_size)).reshape(-1, seq_len),
            "coords": torch.zeros(batch_size, 1, seq_len * 3, 3),
            "frames_idx": torch.cat(
                [torch.arange(seq_len * 3, dtype=torch.long).reshape(-1, 3).unsqueeze(0) for i in range(batch_size)]
            ),
            "true_coords": torch.zeros(batch_size, seq_len, 3, 3),
            "prior_mean": torch.ones(batch_size, 1, seq_len * 3, 3),
            "noised_mask": torch.stack(
                [
                    torch.tensor([True] * (seq_len // 2) + [False] * (seq_len - seq_len // 2), dtype=torch.bool)
                    for _ in range(batch_size)
                ]
            ),
        }

        assert_feature_dict_equality(store.get_features(self.out_key).feature_dict, expected_features)

        # check that input features are still the same shape
        assert_feature_dict_equality(store.get_features(self.in_key).feature_dict, features_copy)

    def test_run_missing_atomfilter(self, step: InitializeFeatures) -> None:
        features = make_features(include_atom_filter=False, seq_len=4)
        store = make_store(features={self.in_key: features}, batch_size=3)
        with pytest.raises(AssertionError):
            step.run(store)


# ---------------------------------------------------------------------------
# InitializeState
# ---------------------------------------------------------------------------


class TestInitializeState:
    in_key = "feat"
    out_key = "X"

    @pytest.fixture
    def step(self, inference_model_config: MagicMock) -> InitializeState:
        return InitializeState(
            name="init_state",
            inference_model_config=inference_model_config,
            in_feat_loc_store=self.in_key,
            out_loc_store=self.out_key,
            seed=42,
        )

    def test_model_configs(self, step: InitializeState, inference_model_config: MagicMock) -> None:
        assert step.model_configs() == [inference_model_config]

    def test_input_feature_keys(self, step: InitializeState) -> None:
        assert step.input_feature_keys() == {self.in_key}

    def test_output_feature_keys(self, step: InitializeState) -> None:
        assert step.output_feature_keys() == set()

    def test_input_state_keys(self, step: InitializeState) -> None:
        assert step.input_state_keys() == set()

    def test_output_state_keys(self, step: InitializeState) -> None:
        assert step.output_state_keys(step_num=0) == {self.out_key}
        assert step.output_state_keys(step_num=1) == set()

    def test_run(self, step: InitializeState) -> None:
        features = make_features(include_atom_filter=True, seq_len=4, batch_size=3)

        store = make_store(features={self.in_key: features}, batch_size=3)

        step.run(store)

        # check the output state
        step_output = store.get_state(self.out_key, step_num=0)
        # expect the first half to be noised
        n_noised = features["noised_mask"].shape[1] // 2
        assert (step_output["coords"][:, :n_noised] != 0.0).all(), (
            "Noised coordinates should have non-zero interpolants"
        )
        assert (step_output["coords"][:, n_noised:] == 0.0).all(), "Unnoised coordinates should be zero"

        # check that only the desired state has been populated in the store
        assert store.states.keys() == {self.out_key}
        assert store.states[self.out_key].keys() == {0}
        assert store.states[self.out_key][0].keys() == {"coords"}

    def _capture_prior_seed(self, step: InitializeState) -> list[int]:
        captured: list[int] = []
        prior = step.inference_model_config.dynopsi_model.priors["coords"]
        original = prior.sample

        def wrapped(*args, generator=None, **kwargs):
            if generator is not None:
                captured.append(int(generator.initial_seed()))
            return original(*args, generator=generator, **kwargs)

        patch.object(prior, "sample", wrapped).start()
        return captured

    def test_prior_seed_offsets_by_sample_index(self, step: InitializeState) -> None:
        captured = self._capture_prior_seed(step)
        features = make_features(include_atom_filter=True, seq_len=4, batch_size=3)
        store = make_store(features={self.in_key: features}, batch_size=3, sample_offset=5)
        step.run(store)
        assert captured == [42 + 5]

    def test_chunk_local_slice_plus_offset_matches_global_slice(self, step: InitializeState) -> None:
        captured = self._capture_prior_seed(step)
        features = make_features(include_atom_filter=True, seq_len=4, batch_size=3)
        chunk_local = make_store(features={self.in_key: features}, batch_size=3, batch_start=0, sample_offset=5)
        step.run(chunk_local)
        global_slice = make_store(features={self.in_key: features}, batch_size=3, batch_start=5, sample_offset=0)
        step.run(global_slice)
        assert captured == [47, 47]

    def test_later_chunk_does_not_replay_first_batch_priors(self, step: InitializeState) -> None:
        features = make_features(include_atom_filter=True, seq_len=4, batch_size=3)
        first = make_store(features={self.in_key: features}, batch_size=3, sample_offset=0)
        second = make_store(features={self.in_key: features}, batch_size=3, sample_offset=3)
        step.run(first)
        a = first.get_state(self.out_key, step_num=0)["coords"].clone()
        step.run(second)
        b = second.get_state(self.out_key, step_num=0)["coords"]
        assert not torch.equal(a, b)
