from unittest import mock

import pytest
import torch

from dynopsi.simulation.utils import (
    all_steps_for_modalities,
    infer_batch_size,
    is_active_at,
    valid_modalities_at,
)

# ---------------------------------------------------------------------------
# all_steps_for_modalities
# ---------------------------------------------------------------------------


class TestAllStepsForModalities:
    def test_single_modality_maps_to_range(self) -> None:
        result = all_steps_for_modalities(["coords"], num_steps=4)
        assert result == {"coords": [0, 1, 2, 3]}

    def test_multiple_modalities_all_get_same_range(self) -> None:
        result = all_steps_for_modalities(["coords", "torsion"], num_steps=3)
        assert result == {"coords": [0, 1, 2], "torsion": [0, 1, 2]}

    def test_zero_steps_produces_empty_lists(self) -> None:
        result = all_steps_for_modalities(["coords"], num_steps=0)
        assert result == {"coords": []}

    def test_empty_modalities_produces_empty_dict(self) -> None:
        result = all_steps_for_modalities([], num_steps=5)
        assert result == {}

    def test_single_step(self) -> None:
        result = all_steps_for_modalities(["coords"], num_steps=1)
        assert result == {"coords": [0]}

    def test_keys_match_modality_names(self) -> None:
        modalities = ["a", "b", "c"]
        result = all_steps_for_modalities(modalities, num_steps=2)
        assert set(result.keys()) == set(modalities)


# ---------------------------------------------------------------------------
# valid_modalities_at
# ---------------------------------------------------------------------------


class TestValidModalitiesAt:
    def test_returns_active_modality(self) -> None:
        valid_steps = {"coords": [0, 1, 2]}
        assert valid_modalities_at(valid_steps, step_num=1) == ["coords"]

    def test_returns_empty_when_no_modality_active(self) -> None:
        valid_steps = {"coords": [0, 1]}
        assert valid_modalities_at(valid_steps, step_num=5) == []

    def test_returns_all_active_modalities(self) -> None:
        valid_steps = {"coords": [0, 1], "torsion": [0, 1]}
        result = valid_modalities_at(valid_steps, step_num=0)
        assert set(result) == {"coords", "torsion"}

    def test_returns_only_modalities_active_at_step(self) -> None:
        valid_steps = {"coords": [0, 1, 2], "torsion": [2, 3]}
        result = valid_modalities_at(valid_steps, step_num=1)
        assert result == ["coords"]

    def test_empty_valid_steps(self) -> None:
        assert valid_modalities_at({}, step_num=0) == []

    def test_modality_with_empty_step_list_never_active(self) -> None:
        valid_steps = {"coords": [], "torsion": [0]}
        result = valid_modalities_at(valid_steps, step_num=0)
        assert result == ["torsion"]

    def test_step_zero_boundary(self) -> None:
        valid_steps = {"coords": [0]}
        assert valid_modalities_at(valid_steps, step_num=0) == ["coords"]
        assert valid_modalities_at(valid_steps, step_num=1) == []


# ---------------------------------------------------------------------------
# is_active_at
# ---------------------------------------------------------------------------


class TestIsActiveAt:
    def test_true_when_any_modality_active(self) -> None:
        valid_steps = {"coords": [1, 2], "torsion": [3, 4]}
        assert is_active_at(valid_steps, step_num=1) is True

    def test_false_when_no_modality_active(self) -> None:
        valid_steps = {"coords": [0, 1]}
        assert is_active_at(valid_steps, step_num=5) is False

    def test_true_when_only_one_of_many_modalities_active(self) -> None:
        valid_steps = {"coords": [0, 1], "torsion": [5, 6]}
        assert is_active_at(valid_steps, step_num=1) is True
        assert is_active_at(valid_steps, step_num=5) is True
        assert is_active_at(valid_steps, step_num=3) is False

    def test_false_for_empty_valid_steps(self) -> None:
        assert is_active_at({}, step_num=0) is False

    def test_false_when_modality_step_list_empty(self) -> None:
        valid_steps = {"coords": []}
        assert is_active_at(valid_steps, step_num=0) is False

    @pytest.mark.parametrize("step_num, expected", [(0, True), (1, True), (2, True), (3, False)])
    def test_parametrized_active_and_inactive(self, step_num: int, expected: bool) -> None:
        valid_steps = {"coords": [0, 1, 2]}
        assert is_active_at(valid_steps, step_num=step_num) is expected


# ---------------------------------------------------------------------------
# infer_batch_size
# ---------------------------------------------------------------------------


class TestInferBatchSize:
    def test_cpu_device_returns_one(self) -> None:
        assert infer_batch_size(torch.device("cpu")) == 1

    def test_high_memory_gpu_returns_five(self) -> None:
        mock_props = mock.MagicMock()
        mock_props.total_memory = 80 * int(1e9)  # 80 GB
        with mock.patch("dynopsi.simulation.utils.torch.cuda.get_device_properties", return_value=mock_props):
            result = infer_batch_size(torch.device("cuda:0"))
        assert result == 5

    def test_mid_memory_gpu_returns_two(self) -> None:
        mock_props = mock.MagicMock()
        mock_props.total_memory = 40 * int(1e9)  # 40 GB
        with mock.patch("dynopsi.simulation.utils.torch.cuda.get_device_properties", return_value=mock_props):
            result = infer_batch_size(torch.device("cuda:0"))
        assert result == 2

    def test_low_memory_gpu_returns_one(self) -> None:
        mock_props = mock.MagicMock()
        mock_props.total_memory = 24 * int(1e9)  # 24 GB (e.g. RTX 3090)
        with mock.patch("dynopsi.simulation.utils.torch.cuda.get_device_properties", return_value=mock_props):
            result = infer_batch_size(torch.device("cuda:0"))
        assert result == 1
