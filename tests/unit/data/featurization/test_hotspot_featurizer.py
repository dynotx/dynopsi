import pytest
import torch

from dynopsi.data.auth_chain_identity import OPAQUE_AUTH_ASYM_ID_START
from dynopsi.data.constants import PocketFeature
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.structure import ChainResidueRepr
from tests.unit.data.conftest import assert_feature_dict_equality, assert_metadata_equality


class TestHotspotFeaturizer:
    @pytest.mark.parametrize(
        "hotspot_positions", ["B2,C1", ChainResidueRepr.from_string("B2,C1")], ids=["str", "ChainResidueRepr"]
    )
    def test_hotspot_featurizer_position_input(self, hotspot_positions: str | ChainResidueRepr) -> None:

        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions=hotspot_positions,
            index_type="original_residue_index",
        )
        assert isinstance(featurizer.hotspot_positions, ChainResidueRepr)
        assert featurizer.hotspot_positions.get_asym_id_to_index_map("original_residue_index") == {
            1: [1],
            2: [0],
        }

    def test_hotspot_featurizer_input_pocket_feature(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions="B2",
            index_type="original_residue_index",
        )
        feats = Features(feature_dict=feature_dict_standard)
        with pytest.raises(ValueError):
            next(featurizer([feats]))

    @pytest.mark.parametrize("index_type", ["original_residue_index", "auth_residue_index"])
    @pytest.mark.parametrize("verbose_metadata", [True, False])
    def test_hotspot_featurizer(
        self, index_type: str, verbose_metadata: bool, feature_dict_standard: dict[str, torch.Tensor]
    ) -> None:
        if index_type == "original_residue_index":
            hotspot_positions = "B2"
        else:
            hotspot_positions = "V3"
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions=hotspot_positions,
            index_type=index_type,
            verbose_metadata=verbose_metadata,
        )
        feats = Features(feature_dict={k: v.clone() for k, v in feature_dict_standard.items() if k != "pocket_feature"})
        out_feats = next(featurizer([feats]))

        assert_feature_dict_equality(out_feats.feature_dict, feature_dict_standard)

        assert out_feats.featurization_steps == [("hotspot", "test_hotspot_featurizer")]
        if index_type == "original_residue_index":
            chain_id_to_residue_map = {
                "B": [2],
            }
        else:
            chain_id_to_residue_map = {
                "V": [3],
            }
        expected_metadata = {
            "test_hotspot_featurizer": {
                "name": "test_hotspot_featurizer",
                "type": "hotspot",
                "hotspot_positions": {
                    "chain_id_to_residue_index_map": chain_id_to_residue_map,
                },
                "index_type": index_type,
                "verbose_metadata": verbose_metadata,
            }
        }
        assert_metadata_equality(out_feats.featurization_metadata, expected_metadata)

    def test_hotspot_featurizer_invalid_hotspot_positions(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions="A1",
            index_type="original_residue_index",
        )
        with pytest.raises(ValueError):
            next(featurizer([Features(feature_dict=feature_dict_standard)]))

    def test_hotspot_featurizer_missing_auth_residue_index(
        self, feature_dict_standard: dict[str, torch.Tensor]
    ) -> None:
        new_features = {
            k: v.clone() for k, v in feature_dict_standard.items() if k not in ["pocket_feature", "auth_residue_index"]
        }
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions="B3",
            index_type="auth_residue_index",
        )
        with pytest.raises(ValueError):
            next(featurizer([Features(feature_dict=new_features)]))

    def test_hotspot_featurizer_outside_of_structure(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions="V3,V10",
            index_type="auth_residue_index",
        )
        with pytest.raises(ValueError):
            next(
                featurizer(
                    [
                        Features(
                            feature_dict={
                                k: v.clone() for k, v in feature_dict_standard.items() if k != "pocket_feature"
                            }
                        )
                    ]
                )
            )

    def test_auth_hotspot_unique_letter_chain_spec(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions="V3",
            index_type="auth_residue_index",
        )
        out_feats = next(
            featurizer(
                [
                    Features(
                        feature_dict={k: v.clone() for k, v in feature_dict_standard.items() if k != "pocket_feature"}
                    )
                ]
            )
        )
        assert out_feats["pocket_feature"][3] == feature_dict_standard["pocket_feature"][3]

    def test_auth_hotspot_dict_a7_selects_opaque_chain(self) -> None:
        n_res = 12
        feature_dict = {
            "token_index": torch.arange(n_res),
            "residue_index": torch.arange(n_res),
            "auth_residue_index": torch.arange(1, n_res + 1),
            "original_residue_index": torch.arange(n_res),
            "asym_id": torch.zeros(n_res, dtype=torch.long),
            "auth_asym_id": torch.full((n_res,), OPAQUE_AUTH_ASYM_ID_START, dtype=torch.long),
            "token_pad_mask": torch.ones(n_res, dtype=torch.bool),
            "binder_mask": torch.zeros(n_res, dtype=torch.bool),
            "target_mask": torch.ones(n_res, dtype=torch.bool),
        }
        feats = Features(
            feature_dict=feature_dict,
            auth_chain_id_table={OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions={"A-7": [10]},
            index_type="auth_residue_index",
        )
        out = next(featurizer([feats]))
        expected = torch.full((n_res,), int(PocketFeature.UNSELECTED))
        expected[9] = int(PocketFeature.POCKET)
        assert torch.equal(out["pocket_feature"], expected)

    def test_auth_hotspot_dict_a7_whole_chain(self) -> None:
        n_res = 12
        feature_dict = {
            "token_index": torch.arange(n_res),
            "residue_index": torch.arange(n_res),
            "auth_residue_index": torch.arange(1, n_res + 1),
            "original_residue_index": torch.arange(n_res),
            "asym_id": torch.zeros(n_res, dtype=torch.long),
            "auth_asym_id": torch.full((n_res,), OPAQUE_AUTH_ASYM_ID_START, dtype=torch.long),
            "token_pad_mask": torch.ones(n_res, dtype=torch.bool),
            "binder_mask": torch.zeros(n_res, dtype=torch.bool),
            "target_mask": torch.ones(n_res, dtype=torch.bool),
        }
        feats = Features(
            feature_dict=feature_dict,
            auth_chain_id_table={OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        featurizer = primitives.HotspotFeaturizer(
            name="test_hotspot_featurizer",
            hotspot_positions={"A-7": None},
            index_type="auth_residue_index",
        )
        out = next(featurizer([feats]))
        assert torch.equal(out["pocket_feature"], torch.full((n_res,), int(PocketFeature.POCKET)))
