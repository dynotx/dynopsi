import logging
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ValidationError

from dynopsi.data.utils.sequence_utils import (
    ChainResidueRepr,
    expand_chain_position_str,
    get_asym_id_from_chain_id,
    get_chain_id_from_asym_id,
    get_contiguous_intervals,
    update_ids,
)


class TestGetAsymIdFromChainId:
    @pytest.mark.parametrize(
        ("chain_id", "expected_asym_id"),
        [
            ("A", 0),
            ("B", 1),
            ("Z", 25),
            ("AA", 52),
            ("AB", 53),
            ("AZ", 77),
            ("BA", 104),
            ("c", 28),
            ("aa", 1430),
            ("A1", 0 + 1),
            ("B2", 1 + 2),
            ("Z12", 25 + 12),
            ("AA1", 52 + 1),
            ("AB2", 53 + 2),
            ("BA10", 104 + 10),
        ],
    )
    def test_valid_chain_id(self, chain_id: str, expected_asym_id: int) -> None:
        assert get_asym_id_from_chain_id(chain_id) == expected_asym_id

    @pytest.mark.parametrize("chain_id", ["1", "20", "26"])
    def test_warning_chain_id_int(self, chain_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """Special case where the chain id is an integer, we return it as is and log a warning."""
        with caplog.at_level(logging.WARNING):
            get_asym_id_from_chain_id(chain_id)
        assert f"Unexpected author chain id: {chain_id}, returning as is." in caplog.text

    @pytest.mark.parametrize("chain_id", ["*", "A1A", "AA1A"])
    def test_warning_chain_id_invalid(self, chain_id: str, caplog: pytest.LogCaptureFixture) -> None:
        """When chain ID is invalid, we don't want to break execution, but we want to log a warning."""
        with caplog.at_level(logging.WARNING):
            get_asym_id_from_chain_id(chain_id)
        assert f"Unexpected author chain id: {chain_id}, returning -1." in caplog.text


class TestGetChainIdFromAsymId:
    @pytest.mark.parametrize(
        ("asym_id", "expected_chain_id"),
        [
            (0, "A"),
            (1, "B"),
            (25, "Z"),
            (52, "AA"),
            (53, "AB"),
            (78, "Aa"),
            (77, "AZ"),
            (104, "BA"),
            (1430, "aa"),
        ],
    )
    def test_valid_asym_id(self, asym_id: int, expected_chain_id: str) -> None:
        assert get_chain_id_from_asym_id(asym_id) == expected_chain_id

    def test_invalid_negative_asym_id(self) -> None:
        with pytest.raises(ValueError, match="asym_id must be non-negative"):
            get_chain_id_from_asym_id(-1)


class TestChainResidueRepr:
    def test_chain_ids(self) -> None:
        crr = ChainResidueRepr(
            chain_id_to_residue_index_map={
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )
        assert crr.chain_ids == {"A", "B"}

    def test_asym_id_to_residue_index_map(self) -> None:
        crr = ChainResidueRepr(
            chain_id_to_residue_index_map={
                "A": [1, 2, 3],
                "B": None,
                "AA": [4],
            }
        )
        assert crr.asym_id_to_residue_index_map_1index == {
            0: [1, 2, 3],
            1: None,
            52: [4],
        }
        assert crr.asym_id_to_residue_index_map_0index == {
            0: [0, 1, 2],
            1: None,
            52: [3],
        }

    @pytest.mark.parametrize(
        ["input_repr", "is_valid"],
        [
            ("A1,A3-4,B", True),
            ({"chain_id_to_residue_index_map": {"A": [1, 3, 4], "B": None}}, True),
            (1, False),
            (None, False),
        ],
        ids=["valid_string", "valid_dict", "invalid_int", "invalid_none"],
    )
    def test_coerce(self, input_repr: Any, is_valid: bool) -> None:
        if is_valid:
            crr = ChainResidueRepr._coerce(input_repr)
            if isinstance(input_repr, str):
                assert crr.chain_id_to_residue_index_map == {
                    "A": [1, 3, 4],
                    "B": None,
                }
            elif isinstance(input_repr, dict):
                assert crr == input_repr
        else:
            with pytest.raises(ValueError):
                ChainResidueRepr._coerce(input_repr)

    @pytest.mark.parametrize(
        ["input_repr", "is_valid"],
        [
            ("A1,A3-4,B", True),
            ({"chain_id_to_residue_index_map": {"A": [1, 3, 4], "B": None}}, True),
            (20, False),
            (None, False),
        ],
        ids=["valid_string", "valid_dict", "invalid_int", "invalid_none"],
    )
    def test_pydantic_integration_from_string(self, input_repr: Any, is_valid: bool) -> None:
        """Test that _coerce is correctly wired."""

        class Wrapper(BaseModel):
            input_data: ChainResidueRepr

        if is_valid:
            result = Wrapper(input_data=input_repr)
            assert isinstance(result.input_data, ChainResidueRepr)
        else:
            with pytest.raises(ValidationError):
                Wrapper(input_data=input_repr)

    def test_from_string(self) -> None:
        crr = ChainResidueRepr.from_string("A1,A3-4,B")
        assert crr.chain_id_to_residue_index_map == {
            "A": [1, 3, 4],
            "B": None,
        }

    def test_from_string_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            ChainResidueRepr.from_string("A1,A3-4,A")

    def test_from_asym_id_mask(self) -> None:
        crr = ChainResidueRepr.from_asym_id_mask(
            mask=torch.tensor([True, True, True, True, False, False, False, False]),
            asym_id=torch.tensor([0, 0, 0, 0, 1, 1, 0, 0]),
        )
        assert crr.chain_id_to_residue_index_map == {
            "A": [1, 2, 3, 4],
        }

        crr = ChainResidueRepr.from_asym_id_mask(
            mask=torch.tensor([True, True, True, True, False, False, False, False]),
            asym_id=torch.tensor([52, 52, 52, 52, 1, 1, 0, 0]),
        )
        assert crr.chain_id_to_residue_index_map == {
            "AA": [1, 2, 3, 4],
        }

    def test_to_mask(self) -> None:
        crr = ChainResidueRepr.from_string("A1,A3-4,B")
        mask = crr.to_mask(
            asym_id=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            residue_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 0]),
            pad_mask=torch.tensor([True, True, True, True, True, True, True, True, True, True, False, False]),
        )
        assert (
            mask == torch.tensor([True, True, True, True, True, False, True, True, False, False, False, False])
        ).all()

        crr = ChainResidueRepr.from_string("AA1,AA3")
        mask = crr.to_mask(
            asym_id=torch.tensor([52, 52, 52, 52, 1, 1, 0, 0]),
            residue_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            pad_mask=torch.tensor([True, True, True, True, True, True, True, True]),
        )
        assert (mask == torch.tensor([True, False, True, False, False, False, False, False])).all()

        # test an invalid mask
        with pytest.raises(ValueError):
            crr.to_mask(
                asym_id=torch.tensor([1, 1, 1, 1, 0, 0]),
                residue_index=torch.tensor([0, 1, 2, 3, 0, 1]),
                pad_mask=torch.tensor([True, True, True, True, True, True]),
            )

        # invalid asym id
        with pytest.raises(ValueError):
            crr.to_mask(
                asym_id=torch.tensor([-1, -1, -1, -1, 1, 1, 0, 0]),
                residue_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                pad_mask=torch.tensor([True, True, True, True, True, True, True, True]),
            )

    @pytest.mark.parametrize(
        "index_type, stored",
        [
            ("original_residue_index", [0, 0]),
            ("residue_index", [0, 0]),
            ("auth_residue_index", [1, 1]),
        ],
    )
    def test_to_mask_rejects_missing_index_hidden_by_duplicate_stored_indices(
        self, index_type: str, stored: list[int]
    ) -> None:
        crr = ChainResidueRepr.from_string("A1,A2")
        with pytest.raises(ValueError, match="residue indices"):
            crr.to_mask(
                asym_id=torch.tensor([0, 0]),
                residue_index=torch.tensor(stored),
                pad_mask=torch.tensor([True, True]),
                index_type=index_type,
            )

    def test_is_contiguous(self) -> None:
        crr = ChainResidueRepr.from_string("A")
        assert crr.is_contiguous is True

        crr = ChainResidueRepr.from_string("A5-10")
        assert crr.is_contiguous is True

        crr = ChainResidueRepr.from_string("A1,A3-4,B")
        assert crr.is_contiguous is False

        crr = ChainResidueRepr.from_string("B,A")
        assert crr.is_contiguous is False

        crr = ChainResidueRepr.from_string("A2,A5-10")
        assert crr.is_contiguous is False

    @pytest.mark.parametrize(
        ("lhs", "rhs", "expected"),
        [
            ("A1-3", "B1-3", False),
            ("A1-3", "A4-6", False),
            ("A1-3", "A3-5", True),
            ("A1-3,B10", "A4-6,B10-12", True),
            ("A1-3,B10", "A4-6,B11-12", False),
            ("A", "B1-3", False),
            ("A", "A1-3", True),
            ("A", "A", True),
            ("A", "B", False),
            ("A5-6", "A", True),
        ],
    )
    def test_overlaps(self, lhs: str, rhs: str, expected: bool) -> None:
        lhs_repr = ChainResidueRepr.from_string(lhs)
        rhs_repr = ChainResidueRepr.from_string(rhs)

        assert lhs_repr.overlaps(rhs_repr) is expected
        assert rhs_repr.overlaps(lhs_repr) is expected

    def test_to_string(self) -> None:
        crr = ChainResidueRepr.from_string("A1,A3-4,B")
        assert crr.to_string == "A1,A3,A4,B"


class TestExpandChainPositionStr:
    def test_valid_chain_position_str(self) -> None:
        assert expand_chain_position_str("A1") == [("A", 1)]
        assert expand_chain_position_str("A1-3") == [("A", 1), ("A", 2), ("A", 3)]
        assert expand_chain_position_str("A1,A3-4") == [("A", 1), ("A", 3), ("A", 4)]
        assert expand_chain_position_str("B1,A3-4") == [("A", 3), ("A", 4), ("B", 1)]
        assert expand_chain_position_str("A1,A2-5,A20,B15-20") == [
            ("A", 1),
            ("A", 2),
            ("A", 3),
            ("A", 4),
            ("A", 5),
            ("A", 20),
            ("B", 15),
            ("B", 16),
            ("B", 17),
            ("B", 18),
            ("B", 19),
            ("B", 20),
        ]
        assert expand_chain_position_str("AA") == [("AA", None)]
        assert expand_chain_position_str("A,B") == [("A", None), ("B", None)]
        assert expand_chain_position_str("A15-20,A18-22") == [
            ("A", 15),
            ("A", 16),
            ("A", 17),
            ("A", 18),
            ("A", 19),
            ("A", 20),
            ("A", 21),
            ("A", 22),
        ]
        assert expand_chain_position_str("A1, A2") == [("A", 1), ("A", 2)]

    def test_a1_5_is_residue_range_on_unique_letter_a(self) -> None:
        assert expand_chain_position_str("A1-5") == [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5)]
        crr = ChainResidueRepr.from_string("A1-5")
        assert crr.chain_id_to_residue_index_map == {"A": [1, 2, 3, 4, 5]}
        original_mask = crr.to_mask(
            asym_id=torch.zeros(8, dtype=torch.long),
            residue_index=torch.arange(8),
            pad_mask=torch.ones(8, dtype=torch.bool),
            index_type="original_residue_index",
        )
        assert original_mask.tolist() == [True, True, True, True, True, False, False, False]
        auth_mask = crr.to_mask(
            asym_id=torch.zeros(8, dtype=torch.long),
            residue_index=torch.arange(1, 9),
            pad_mask=torch.ones(8, dtype=torch.bool),
            index_type="auth_residue_index",
        )
        assert auth_mask.tolist() == [True, True, True, True, True, False, False, False]

    def test_compact_a_hyphen_7_is_not_a_chain_id(self) -> None:
        with pytest.raises(ValueError):
            expand_chain_position_str("A-7")
        with pytest.raises(ValueError):
            ChainResidueRepr.from_string("A-7")

    def test_auth_id_keyed_dict_coerces_to_chain_residue_repr(self) -> None:
        residue = ChainResidueRepr.model_validate({"A-7": [10]})
        assert residue.chain_id_to_residue_index_map == {"A-7": [10]}
        whole_chain = ChainResidueRepr.model_validate({"A-7": None})
        assert whole_chain.chain_id_to_residue_index_map == {"A-7": None}
        nested = ChainResidueRepr.model_validate({"chain_id_to_residue_index_map": {"A-7": [10]}})
        assert nested.chain_id_to_residue_index_map == {"A-7": [10]}

    def test_invalid_chain_position_str(self) -> None:
        with pytest.raises(ValueError):
            expand_chain_position_str("A2-B5")
        with pytest.raises(ValueError):
            expand_chain_position_str("A5-2")
        with pytest.raises(ValueError):
            expand_chain_position_str("1")
        with pytest.raises(ValueError):
            expand_chain_position_str("1-5")
        with pytest.raises(ValueError):
            expand_chain_position_str("*")
        with pytest.raises(ValueError):
            expand_chain_position_str("B1B")
        with pytest.raises(ValueError):
            expand_chain_position_str("B15-20,B")
        with pytest.raises(ValueError):
            expand_chain_position_str("A,A")


class TestUpdateIds:
    def test_empty_input(self) -> None:
        with pytest.raises(ValueError):
            update_ids([])

    @pytest.mark.parametrize("tensor_or_array", ["tensor", "array"])
    def test_single_tensor_or_array(self, tensor_or_array: str) -> None:
        if tensor_or_array == "tensor":
            input_tensor = torch.arange(3)
            result = update_ids([input_tensor])
            assert type(result) == type(input_tensor)
            assert (result == input_tensor).all()
        elif tensor_or_array == "array":
            input_array = np.arange(3)
            result = update_ids([input_array])
            assert type(result) == type(input_array)
            assert (result == input_array).all()

    @pytest.mark.parametrize(
        "tensor_or_array, input, output",
        [
            ("tensor", [[0, 1], [2]], [0, 1, 2]),
            ("tensor", [[0, 1], [0, 1]], [0, 1, 2, 3]),
            ("array", [[0, 1], [2]], [0, 1, 2]),
            ("array", [[0, 1], [0, 1]], [0, 1, 2, 3]),
        ],
        ids=["tensor-nonoverlapping", "tensor-overlapping", "array-nonoverlapping", "array-overlapping"],
    )
    def test_multiple_array_input(self, tensor_or_array: str, input: list[list[int]], output: list[int]) -> None:

        actual_input: list[torch.Tensor] | list[np.ndarray[Any, Any]]
        expected_result: torch.Tensor | np.ndarray[Any, Any]
        if tensor_or_array == "tensor":
            actual_input = [torch.tensor(arr) for arr in input]
            expected_result = torch.tensor(output)
        elif tensor_or_array == "array":
            actual_input = [np.array(arr) for arr in input]
            expected_result = np.array(output)

        result = update_ids(actual_input)
        assert type(result) == type(expected_result)
        assert (result == expected_result).all()


class TestGetContiguousIntervals:
    def test_empty_input(self) -> None:
        interval_ids = get_contiguous_intervals(
            asym_ids=torch.tensor([], dtype=torch.long),
            residue_indices=torch.tensor([], dtype=torch.long),
        )
        assert torch.equal(interval_ids, torch.tensor([], dtype=torch.long))

    def test_get_contiguous_intervals_splits_residue_gaps_and_chain_changes(self) -> None:
        interval_ids = get_contiguous_intervals(
            asym_ids=torch.tensor([1, 1, 1, 1, 2, 2, 3], dtype=torch.long),
            residue_indices=torch.tensor([1, 2, 4, 5, 6, 7, 1], dtype=torch.long),
        )

        assert torch.equal(interval_ids, torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.long))
