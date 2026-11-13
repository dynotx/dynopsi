from copy import deepcopy

import pytest
import torch

from dynopsi.data.constants import AtomFilter, PocketFeature
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.featurization.primitives.inpainting import InpaintTokenIndexInfo
from dynopsi.data.utils import ChainResidueRepr
from tests.unit.data.conftest import assert_feature_dict_equality


def _inpaint_featurizer(
    positions: tuple[str, ...] = ("A1-3",),
    lengths: tuple[int, ...] = (3,),
) -> primitives.InpaintFeaturizer:
    return primitives.InpaintFeaturizer(
        inpainting_positions=positions,
        inpainting_lengths=lengths,
        inpaint_region_centers_of_mass=tuple(None for _ in positions),
        atoms_to_include=AtomFilter.CA,
    )


def _token_features(asym_id: list[int], residue_index: list[int]) -> Features:
    return Features(
        feature_dict={
            "asym_id": torch.tensor(asym_id, dtype=torch.long),
            "original_residue_index": torch.tensor(residue_index, dtype=torch.long),
            "token_pad_mask": torch.ones(len(asym_id)),
        }
    )


def _target_segment_features(*, pocket: list[int], asym_id: list[int] | None = None) -> dict[str, torch.Tensor]:
    n = len(pocket)
    return {
        "asym_id": torch.tensor(asym_id if asym_id is not None else [0] * n, dtype=torch.long),
        "auth_asym_id": torch.zeros(n, dtype=torch.long),
        "entity_id": torch.zeros(n, dtype=torch.long),
        "sym_id": torch.zeros(n, dtype=torch.long),
        "pocket_feature": torch.tensor(pocket, dtype=torch.long),
        "binder_mask": torch.zeros(n, dtype=torch.bool),
        "target_mask": torch.ones(n, dtype=torch.bool),
    }


class TestInpaintFeaturizerModelValidation:
    @pytest.mark.parametrize(
        ["inpainting_positions", "inpainting_lengths", "inpaint_region_centers_of_mass"],
        [
            (("A2-5",), (4, 5), (None, None)),  # one region but two lengths
            (("A2-5",), (4,), (None, None)),  # one region but two centers of mass
            (("A2-5", "B6-10"), (3,), ((1, 1, 1), (2, 2, 2))),  # two regions but one length
            (("A2-5", "B6-10"), (3, 4), ((1, 1, 1),)),  # two regions but one center of mass
            ((""), (4,), (None,)),  # empty region
            (("A2-5", "B6-10"), (3, 4), ((1,), (2, 2, 2))),  # invalid center of mass
        ],
    )
    def test_num_regions_invalid(
        self,
        inpainting_positions: tuple[str, ...],
        inpainting_lengths: tuple[int, ...],
        inpaint_region_centers_of_mass: tuple[tuple[float, float, float] | None, ...],
    ) -> None:
        chain_reprs = [ChainResidueRepr.from_string(pos) for pos in inpainting_positions]
        with pytest.raises(ValueError):
            primitives.InpaintFeaturizer(
                inpainting_positions=chain_reprs,
                inpainting_lengths=inpainting_lengths,
                inpaint_region_centers_of_mass=inpaint_region_centers_of_mass,
            )


class TestInpaintFeaturizer:
    @pytest.mark.parametrize("index_type", ["original_residue_index", "auth_residue_index"])
    @pytest.mark.parametrize("verbose_metadata", [True, False])
    @pytest.mark.parametrize(
        "inpaint_region_centers_of_mass",
        [(None, None), ((5, 5, 5), (6, 6, 6)), ((5, 5, 5), None)],
        ids=["none", "com1", "partial_com"],
    )
    def test_inpainting_featurizer(
        self,
        feature_dict_standard: dict[str, torch.Tensor],
        feature_dict_standard_inpaint: dict[str, torch.Tensor],
        index_type: str,
        verbose_metadata: bool,
        inpaint_region_centers_of_mass: tuple[tuple[float, float, float] | None, ...],
    ) -> None:
        if index_type == "original_residue_index":
            inpainting_positions = ("C1-2", "A2")
            inpainting_lengths = (5, 4)
            # flip the inpaint region centers of mass
            inpaint_com = (inpaint_region_centers_of_mass[1], inpaint_region_centers_of_mass[0])
        else:
            inpainting_positions = ("U2", "W4-5")
            inpainting_lengths = (4, 5)
            inpaint_com = inpaint_region_centers_of_mass

        featurizer = primitives.InpaintFeaturizer(
            inpainting_positions=inpainting_positions,
            inpainting_lengths=inpainting_lengths,
            inpaint_region_centers_of_mass=inpaint_com,
            atoms_to_include=AtomFilter.BB,
            index_type=index_type,
        )

        input_feature_dict = {
            key: value.clone()
            for key, value in feature_dict_standard.items()
            if key not in ["prior_mean", "inpaint_feature", "noised_mask"]
        }
        # change input target mask and binder mask
        input_feature_dict["target_mask"] = torch.tensor([False] * 2 + [True] * 2 + [False] * 2)
        input_feature_dict["binder_mask"] = torch.tensor([True] * 2 + [False] * 2 + [True] * 2)
        input_feature_dict["pocket_feature"] = torch.tensor(
            [PocketFeature.BINDER] * 2 + [PocketFeature.UNSELECTED, PocketFeature.POCKET] + [PocketFeature.BINDER] * 2
        )

        expected_feature_dict = deepcopy(
            {
                key: value.clone()
                for key, value in feature_dict_standard_inpaint.items()
                if key not in ["prior_mean", "noised_mask"]
            }
        )
        # update expected coords based on com
        region1, region2 = inpaint_region_centers_of_mass
        if region1 is not None:
            expected_feature_dict["coords"][:, 3:15, :] = region1[0]
        if region2 is not None:
            expected_feature_dict["coords"][:, 21:36, :] = region2[0]

        input_features = Features(feature_dict=input_feature_dict)
        out_feats = next(featurizer([input_features]))
        assert_feature_dict_equality(observed=out_feats.feature_dict, expected=expected_feature_dict)

    @pytest.mark.parametrize("input_pos_str", ["A1-2,B1-2", "A1,A5"])
    def test_discontinuous_inpaint_positions(self, input_pos_str: str) -> None:
        chain_reprs = ChainResidueRepr.from_string(input_pos_str)
        with pytest.raises(ValueError):
            primitives.InpaintFeaturizer(
                inpainting_positions=(chain_reprs,),
                inpainting_lengths=(4),
                inpaint_region_centers_of_mass=(None),
                atoms_to_include=AtomFilter.BB,
                index_type="original_residue_index",
            )

    def test_varying_pocket_feature_copies_slice(self) -> None:
        pocket = [int(PocketFeature.UNSELECTED), int(PocketFeature.POCKET), int(PocketFeature.UNSPECIFIED)]
        segment = InpaintTokenIndexInfo(start_idx=0, end_idx=3, feature_length=3, center_of_mass=torch.zeros(3))
        out = _inpaint_featurizer()._make_inpaint_feature_segment(_target_segment_features(pocket=pocket), segment)
        assert out["pocket_feature"].tolist() == pocket

    def test_varying_asym_id_raises(self) -> None:
        segment = InpaintTokenIndexInfo(start_idx=0, end_idx=3, feature_length=3, center_of_mass=torch.zeros(3))
        features = _target_segment_features(pocket=[int(PocketFeature.POCKET)] * 3, asym_id=[0, 0, 1])
        with pytest.raises(ValueError, match="asym_id"):
            _inpaint_featurizer()._make_inpaint_feature_segment(features, segment)

    def test_contiguous_inpaint_mask_returns_start_and_end(self) -> None:
        features = _token_features([1, 0, 0, 0], [0, 0, 1, 2])
        start, end = _inpaint_featurizer()._segment_token_bounds(features, ChainResidueRepr.from_string("A1-3"))
        assert (start, end) == (1, 4)

    def test_holey_inpaint_mask_raises(self) -> None:
        features = _token_features([0, 1, 0, 0], [0, 0, 1, 2])
        with pytest.raises(ValueError):
            _inpaint_featurizer()._segment_token_bounds(features, ChainResidueRepr.from_string("A1-3"))

    def test_empty_inpaint_mask_raises(self) -> None:
        features = _token_features([0, 0, 0], [0, 1, 2])
        with pytest.raises(ValueError):
            _inpaint_featurizer(positions=("B",), lengths=(1,))._extract_inpaint_regions(features)
