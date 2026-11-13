from typing import Any
from unittest.mock import patch

import pytest
import torch

from dynopsi.data.auth_chain_identity import OPAQUE_AUTH_ASYM_ID_START
from dynopsi.data.constants import (
    AtomFilter,
    DSSPLoopFeature,
    HomooligomerPartnerFeature,
    InpaintFeature,
    PocketFeature,
)
from dynopsi.data.featurization.utils import (
    TOKEN_KEYS_FLAT,
    concat_features,
    concat_features_with_auth_tables,
    crop_features,
    make_empty_chain,
    pad_features,
    strip_chain_padding,
)


class TestConditioningFeatureKeys:
    def test_registry_keys_are_token_keys(self) -> None:
        assert "chain_is_loopy" in TOKEN_KEYS_FLAT
        assert "has_homooligomer_partner" in TOKEN_KEYS_FLAT

    def test_pad_features_supports_chain_is_loopy(self) -> None:
        features = {
            "token_index": torch.arange(2),
            "chain_is_loopy": torch.tensor([DSSPLoopFeature.NON_LOOP, DSSPLoopFeature.LOOP_OR_OTHER]),
        }

        output = pad_features(features, max_tokens=4, atoms_to_include=AtomFilter.CA)

        assert output["chain_is_loopy"].tolist() == [
            int(DSSPLoopFeature.NON_LOOP),
            int(DSSPLoopFeature.LOOP_OR_OTHER),
            int(DSSPLoopFeature.MASK_OR_MISSING),
            int(DSSPLoopFeature.MASK_OR_MISSING),
        ]

    def test_pad_features_supports_has_homooligomer_partner(self) -> None:
        features = {
            "token_index": torch.arange(2),
            "has_homooligomer_partner": torch.tensor(
                [HomooligomerPartnerFeature.NO_PARTNER, HomooligomerPartnerFeature.HAS_PARTNER]
            ),
        }

        output = pad_features(features, max_tokens=4, atoms_to_include=AtomFilter.CA)

        assert output["has_homooligomer_partner"].tolist() == [
            int(HomooligomerPartnerFeature.NO_PARTNER),
            int(HomooligomerPartnerFeature.HAS_PARTNER),
            int(HomooligomerPartnerFeature.MASK_OR_MISSING),
            int(HomooligomerPartnerFeature.MASK_OR_MISSING),
        ]

    def test_pad_features_rejects_unknown_key(self) -> None:
        features = {"token_index": torch.arange(2), "not_a_registered_feature": torch.zeros(2)}

        with pytest.raises(ValueError, match="Unsupported key"):
            pad_features(features, max_tokens=4, atoms_to_include=AtomFilter.CA)


class TestStripChainPadding:
    def test_strip_chain_padding(self, features_A: dict[str, torch.Tensor]) -> None:
        features_A = {key: value.clone() for key, value in features_A.items()}
        original_features_A = {key: value.clone() for key, value in features_A.items()}
        features = strip_chain_padding(features_A)
        # check that the original features are not modified
        for k in features_A.keys():
            assert (features_A[k] == original_features_A[k]).all()

        for k, v in features.items():
            if k == "coords":
                expected_val = features_A["coords"][:, : 7 * 3, :]
            elif k == "atom_to_token":
                expected_val = features_A["atom_to_token"][: 7 * 3, :7]
            elif k in ["atom_resolved_mask", "atom_pad_mask"]:
                expected_val = features_A[k][: 7 * 3]
            else:
                expected_val = features_A[k][:7]

            assert torch.allclose(v, expected_val), f"Expected {expected_val} for {k}, got {v}"


class TestMakeEmptyChain:
    def test_make_empty_chain(self) -> None:
        features = make_empty_chain(
            seq_length=10,
            atoms_to_include=AtomFilter.BB,
            res_type="UNK",
            asym_id=1,
            auth_asym_id=21,
            entity_id=0,
            sym_id=2,
            pocket_feature=PocketFeature.BINDER,
            inpaint_feature=InpaintFeature.TARGET_INPAINT,
            coords_center_of_mass=torch.tensor([1.0, 2.0, 3.0]),
            keys_include=[
                "token_index",
                "residue_index",
                "asym_id",
                "auth_asym_id",
                "entity_id",
                "sym_id",
                "auth_residue_index",
                "coords",
                "atom_resolved_mask",
                "atom_pad_mask",
                "token_pad_mask",
                "token_resolved_mask",
                "pocket_feature",
                "frame_resolved_mask",
                "noised_mask",
                "frames_idx",
                "inpaint_feature",
                "atom_to_token",
            ],
        )
        expected_features = {
            "token_index": torch.arange(10, dtype=torch.long),
            "residue_index": torch.arange(10, dtype=torch.long),
            "auth_residue_index": torch.arange(1, 11, dtype=torch.long),
            "asym_id": torch.ones(10, dtype=torch.long),
            "auth_asym_id": torch.full((10,), 21, dtype=torch.long),
            "entity_id": torch.zeros(10, dtype=torch.long),
            "sym_id": torch.ones(10, dtype=torch.long) * 2,
            "coords": torch.cat([torch.ones(1, 30, 1), torch.ones(1, 30, 1) * 2, torch.ones(1, 30, 1) * 3], dim=2),
            "atom_resolved_mask": torch.tensor([True] * 10 * 3),
            "atom_pad_mask": torch.tensor([True] * 10 * 3),
            "token_pad_mask": torch.tensor([True] * 10),
            "token_resolved_mask": torch.tensor([True] * 10),
            "pocket_feature": torch.full((10,), PocketFeature.BINDER, dtype=torch.long),
            "frame_resolved_mask": torch.tensor([True] * 10),
            "noised_mask": torch.ones(10, dtype=torch.bool),
            "frames_idx": torch.arange(10 * 3).reshape(10, 3),
            "inpaint_feature": torch.full((10,), InpaintFeature.TARGET_INPAINT, dtype=torch.long),
            "atom_to_token": torch.eye(10, dtype=torch.long).repeat_interleave(3, dim=0),
        }
        for k, v in expected_features.items():
            assert (features[k] == v).all(), f"Expected {v} for {k}, got {features[k]}"

        # check the filter
        assert "binder_mask" not in features.keys()
        assert "target_mask" not in features.keys()

        import dynopsi.data.featurization.utils as utils

        def fake_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ones(*args, **kwargs)

        with patch.object(utils.torch, "randn", side_effect=fake_randn):
            features = make_empty_chain(
                seq_length=10,
                atoms_to_include=AtomFilter.BB,
            )
            assert (features["coords"] == torch.ones(1, 30, 3, dtype=torch.float32)).all()

    def test_make_empty_chain_ca(self) -> None:
        features = make_empty_chain(
            seq_length=10,
            atoms_to_include=AtomFilter.CA,
            coords_center_of_mass=torch.tensor([0.0, 0.0, 0.0]),
        )
        expected_features = {
            "asym_id": torch.zeros(10, dtype=torch.long),
            "auth_asym_id": torch.zeros(10, dtype=torch.long),
            "entity_id": torch.zeros(10, dtype=torch.long),
            "sym_id": torch.zeros(10, dtype=torch.long),
            "token_index": torch.arange(10, dtype=torch.long),
            "residue_index": torch.arange(10, dtype=torch.long),
            "auth_residue_index": torch.arange(1, 11, dtype=torch.long),
            "coords": torch.zeros(1, 10, 3, dtype=torch.float32),
            "atom_resolved_mask": torch.ones(10, dtype=torch.bool),
            "atom_pad_mask": torch.ones(10, dtype=torch.bool),
            "token_pad_mask": torch.ones(10, dtype=torch.bool),
            "token_resolved_mask": torch.ones(10, dtype=torch.bool),
            "pocket_feature": torch.full((10,), PocketFeature.UNSPECIFIED, dtype=torch.long),
            "frame_resolved_mask": torch.ones(10, dtype=torch.bool),
            "noised_mask": torch.ones(10, dtype=torch.bool),
            "frames_idx": torch.arange(10 * 1).reshape(10, 1),
            "inpaint_feature": torch.full((10,), InpaintFeature.BINDER_INPAINT, dtype=torch.long),
            "binder_mask": torch.ones(10, dtype=torch.bool),
            "target_mask": torch.zeros(10, dtype=torch.bool),
            "atom_to_token": torch.eye(10, dtype=torch.long),
        }
        for k, v in expected_features.items():
            assert (features[k] == v).all(), f"Expected {v} for {k}, got {features[k]}"


class TestConcatFeatures:
    """
    Conditions
    - make_asym_ids_unique = True, make_entity_ids_unique = True: update both asym_ids and entity_ids, sym_ids are set to 0
    - make_asym_ids_unique = True, make_entity_ids_unique = False: update asym_ids, entity_ids are not updated, sym_ids are updated within each entity_id
    - make_asym_ids_unique = False, make_entity_ids_unique = True: This is not allowed, should raise an error
    - make_asym_ids_unique = False, make_entity_ids_unique = False: This is only allowed if for matching asym_ids, entity_ids are the same. Otherwise, should raise an error
    """

    @pytest.mark.parametrize("make_asym_ids_unique", [True, False])
    @pytest.mark.parametrize("make_entity_ids_unique", [True, False])
    def test_make_asym_id_and_entity_id_unique(
        self,
        features_A: dict[str, torch.Tensor],
        features_B: dict[str, torch.Tensor],
        make_asym_ids_unique: bool,
        make_entity_ids_unique: bool,
    ) -> None:
        features_A = {key: value.clone() for key, value in features_A.items()}
        features_B = {key: value.clone() for key, value in features_B.items()}
        original_features_A = {key: value.clone() for key, value in features_A.items()}
        original_features_B = {key: value.clone() for key, value in features_B.items()}
        # a value error should be raised if asym_ids are the same but entity_ids are now different
        if not make_asym_ids_unique and make_entity_ids_unique:
            with pytest.raises(ValueError):
                concat_features(
                    [features_A, features_B],
                    make_asym_ids_unique=make_asym_ids_unique,
                    make_entity_ids_unique=make_entity_ids_unique,
                )
        else:
            features = concat_features(
                [features_A, features_B],
                make_asym_ids_unique=make_asym_ids_unique,
                make_entity_ids_unique=make_entity_ids_unique,
            )
            # check that the original features are not modified
            for k in features_A.keys():
                assert (features_A[k] == original_features_A[k]).all()
            for k in features_B.keys():
                assert (features_B[k] == original_features_B[k]).all()

            # test flag-specific values
            if make_asym_ids_unique and make_entity_ids_unique:
                assert (features["asym_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)).all()
                assert (features["entity_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)).all()
                assert (features["sym_id"] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)).all()
                assert (
                    features["residue_index"] == torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2], dtype=torch.long)
                ).all()
            elif make_asym_ids_unique and not make_entity_ids_unique:
                assert (features["asym_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)).all()
                assert (features["entity_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)).all()
                assert (features["sym_id"] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)).all()
                assert (
                    features["residue_index"] == torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2], dtype=torch.long)
                ).all()
            elif not make_asym_ids_unique and not make_entity_ids_unique:
                assert (features["asym_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)).all()
                assert (features["entity_id"] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)).all()
                assert (features["sym_id"] == torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)).all()
                assert (
                    features["residue_index"] == torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], dtype=torch.long)
                ).all()

            # updated feature values that are the same regardless of the make_asym_ids_unique and make_entity_ids_unique flags
            assert (features["token_index"] == torch.arange(10, dtype=torch.long)).all()
            assert (features["frames_idx"] == torch.arange(10 * 3).reshape(10, 3)).all()
            assert (features["atom_to_token"] == torch.eye(10, dtype=torch.long).repeat_interleave(3, dim=0)).all()

            # test the rest
            for k in [
                "auth_residue_index",
                "token_pad_mask",
                "token_resolved_mask",
                "pocket_feature",
                "frame_resolved_mask",
                "noised_mask",
                "binder_mask",
                "target_mask",
            ]:
                expected_val = torch.cat([features_A[k][:7], features_B[k]])
                assert (features[k] == expected_val).all(), f"Expected {expected_val} for {k}, got {features[k]}"
            for k in ["atom_pad_mask", "atom_resolved_mask"]:
                expected_val = torch.cat([features_A[k][: 7 * 3], features_B[k]])
                assert (features[k] == expected_val).all(), f"Expected {expected_val} for {k}, got {features[k]}"
            for k in ["coords"]:
                expected_val = torch.cat([features_A[k][:, : 7 * 3, :], features_B[k]], dim=1)
                assert (features[k] == expected_val).all(), f"Expected {expected_val} for {k}, got {features[k]}"

    def test_false_entity_id_and_asym_id_not_unique_invalid(
        self, features_A: dict[str, torch.Tensor], features_C: dict[str, torch.Tensor]
    ) -> None:
        # entity ids are different but asym_ids are the same
        with pytest.raises(ValueError):
            concat_features(
                [
                    {key: value.clone() for key, value in features_A.items()},
                    {key: value.clone() for key, value in features_C.items()},
                ],
                make_asym_ids_unique=False,
                make_entity_ids_unique=False,
            )


class TestConcatFeaturesWithAuthTables:
    def test_colliding_opaque_a7_tables_stay_distinct(self) -> None:
        first = make_empty_chain(
            seq_length=2,
            atoms_to_include=AtomFilter.BB,
            res_type="ALA",
            asym_id=0,
            auth_asym_id=OPAQUE_AUTH_ASYM_ID_START,
        )
        second = make_empty_chain(
            seq_length=2,
            atoms_to_include=AtomFilter.BB,
            res_type="ALA",
            asym_id=0,
            auth_asym_id=OPAQUE_AUTH_ASYM_ID_START,
        )
        tables = [
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        ]
        out, merged = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        auth_ints = {int(i) for i in out["auth_asym_id"].unique().tolist()}
        assert auth_ints == {OPAQUE_AUTH_ASYM_ID_START, OPAQUE_AUTH_ASYM_ID_START + 1}
        assert merged == {OPAQUE_AUTH_ASYM_ID_START: "A-7", OPAQUE_AUTH_ASYM_ID_START + 1: "A-7"}
        assert tables == [
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
            {OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        ]

    def test_forwards_reset_residue_index(self) -> None:
        first = make_empty_chain(
            seq_length=2,
            atoms_to_include=AtomFilter.BB,
            res_type="ALA",
            asym_id=0,
            auth_asym_id=0,
        )
        second = make_empty_chain(
            seq_length=2,
            atoms_to_include=AtomFilter.BB,
            res_type="ALA",
            asym_id=1,
            auth_asym_id=1,
        )
        first["residue_index"] = torch.tensor([5, 6], dtype=torch.long)
        second["residue_index"] = torch.tensor([10, 11], dtype=torch.long)
        tables = [{0: "A"}, {1: "B"}]
        reset_out, _ = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
        )
        keep_out, _ = concat_features_with_auth_tables(
            [first, second],
            tables,
            make_asym_ids_unique=True,
            make_entity_ids_unique=True,
            reset_residue_index=False,
        )
        assert reset_out["residue_index"].tolist() == [0, 1, 0, 1]
        assert keep_out["residue_index"].tolist() == [5, 6, 10, 11]


class TestCropFeatures:
    @pytest.mark.parametrize("reset_residue_index", [True, False])
    def test_crop_features(self, features_A: dict[str, torch.Tensor], reset_residue_index: bool) -> None:
        features_A = {key: value.clone() for key, value in features_A.items()}
        features = crop_features(
            features_A,
            token_mask=torch.tensor([False] + [True] * 3 + [False] * 4),
            reset_residue_index=reset_residue_index,
        )
        # check that this isn't cropping in-place
        assert features_A["coords"].shape != features["coords"].shape

        if reset_residue_index:
            assert (features["residue_index"] == torch.tensor([0, 1, 2], dtype=torch.long)).all()
        else:
            assert (features["residue_index"] == torch.tensor([1, 2, 3], dtype=torch.long)).all()

        assert (features["token_index"] == torch.tensor([0, 1, 2], dtype=torch.long)).all()
        assert (features["asym_id"] == torch.tensor([0, 0, 0], dtype=torch.long)).all()
        assert (features["entity_id"] == torch.tensor([0, 0, 0], dtype=torch.long)).all()
        assert (features["sym_id"] == torch.tensor([0, 0, 0], dtype=torch.long)).all()
        assert (features["auth_residue_index"] == torch.tensor([6, 7, 8], dtype=torch.long)).all()
        assert (features["coords"] == torch.zeros(3 * 3, 3).unsqueeze(0)).all()
        assert (features["atom_resolved_mask"] == torch.tensor([True] * 3 * 3, dtype=torch.bool)).all()
        assert (features["atom_pad_mask"] == torch.tensor([True] * 3 * 3, dtype=torch.bool)).all()
        assert (features["token_pad_mask"] == torch.tensor([True] * 3, dtype=torch.bool)).all()
        assert (features["token_resolved_mask"] == torch.tensor([True] * 3, dtype=torch.bool)).all()
        assert (features["atom_to_token"] == torch.eye(3, dtype=torch.long).repeat_interleave(3, dim=0)).all()

    def test_crop_features_empty(self, features_A: dict[str, torch.Tensor]) -> None:
        features = crop_features(
            {key: value.clone() for key, value in features_A.items()},
            token_mask=torch.tensor([False] * 8),
            reset_residue_index=True,
        )
        for k, v in features.items():
            if k == "coords":
                assert v.shape[1] == 0
            else:
                assert v.shape[0] == 0

    def test_crop_features_mask_size_mismatch(self, features_A: dict[str, torch.Tensor]) -> None:
        with pytest.raises(ValueError):
            crop_features(
                {key: value.clone() for key, value in features_A.items()},
                token_mask=torch.tensor([True] * 6),
                reset_residue_index=True,
            )

        with pytest.raises(ValueError):
            crop_features(
                {key: value.clone() for key, value in features_A.items()},
                token_mask=torch.tensor([True] * 10),
                reset_residue_index=True,
            )
