from copy import deepcopy

import pytest
import torch

from dynopsi.data.constants import AminoAcidSequenceFeature, AtomFilter, InpaintFeature, PocketFeature
from dynopsi.data.featurization import primitives
from dynopsi.data.featurization.base import Features
from tests.unit.data.conftest import assert_feature_dict_equality, assert_metadata_equality


class TestNewChainsFeaturizer:
    def test_invalid_configurations(self) -> None:
        # invalid becuase same entity id but different lengths
        with pytest.raises(ValueError):
            primitives.NewChainsFeaturizer(
                name="test_new_chains_featurizer",
                lengths=(3, 2),
                entity_ids=(0, 0),
                estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                ),
            )

    @pytest.mark.parametrize(
        "entity_ids, expected_sym_ids, expected_entity_ids",
        [
            ((), (0, 0, 0, 0), (0, 1, 2, 3)),
            ((1, 1, 2, 2), (0, 1, 0, 1), (1, 1, 2, 2)),
        ],
    )
    def test_id_configuration(self, entity_ids, expected_sym_ids, expected_entity_ids) -> None:
        featurizer = primitives.NewChainsFeaturizer(
            name="test_new_chains_featurizer",
            lengths=(3, 3, 2, 2),
            entity_ids=entity_ids,
            estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                centers_of_mass=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            ),
            atoms_to_include=AtomFilter.BB,
        )
        assert featurizer.sym_ids == expected_sym_ids
        assert featurizer.entity_ids == expected_entity_ids

    def test_invalid_input_lengths(self) -> None:
        with pytest.raises(ValueError):
            primitives.NewChainsFeaturizer(
                name="test_new_chains_featurizer",
                lengths=(1, 2, 3),
                entity_ids=(1, 2),
                estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=((0.0, 0.0, 0.0) for i in range(2))
                ),
                atoms_to_include=AtomFilter.BB,
            )
        with pytest.raises(ValueError):
            primitives.NewChainsFeaturizer(
                name="test_new_chains_featurizer",
                lengths=(1, 2),
                entity_ids=(1, 2),
                estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=((0.0, 0.0, 0.0) for i in range(3))
                ),
                atoms_to_include=AtomFilter.BB,
            )

    def test_new_chains_at_origin(self) -> None:
        featurizer = primitives.NewChainsFeaturizer(
            name="test_new_chains_featurizer",
            lengths=(3, 3, 2),
            entity_ids=(1, 1, 2),
            estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                centers_of_mass=((0.0, 0.0, 0.0) for i in range(3))
            ),
            atoms_to_include=AtomFilter.BB,
        )
        feats = Features()
        out_feats = next(featurizer([feats]))

        expected_features = {
            "asym_id": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
            "auth_asym_id": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
            "entity_id": torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.long),
            "sym_id": torch.tensor([0, 0, 0, 1, 1, 1, 0, 0], dtype=torch.long),
            "token_index": torch.arange(8, dtype=torch.long),
            "residue_index": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long),
            "original_residue_index": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long),
            "auth_residue_index": torch.tensor([1, 2, 3, 1, 2, 3, 1, 2], dtype=torch.long),
            "coords": torch.cat(
                [
                    torch.zeros(3 * 3, 3),
                    torch.zeros(3 * 3, 3),
                    torch.zeros(2 * 3, 3),
                ]
            ).unsqueeze(0),
            "atom_resolved_mask": torch.tensor([True] * 8 * 3, dtype=torch.bool),
            "token_resolved_mask": torch.tensor([True] * 8, dtype=torch.bool),
            "frame_resolved_mask": torch.tensor([True] * 8, dtype=torch.bool),
            "atom_pad_mask": torch.tensor([True] * 8 * 3, dtype=torch.bool),
            "token_pad_mask": torch.tensor([True] * 8, dtype=torch.bool),
            "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 8, dtype=torch.long),
            "inpaint_feature": torch.tensor([InpaintFeature.BINDER_INPAINT] * 8, dtype=torch.long),
            "binder_mask": torch.tensor([True] * 8, dtype=torch.bool),
            "target_mask": torch.tensor([False] * 8, dtype=torch.bool),
            "frames_idx": torch.arange(8 * 3).reshape(-1, 3),
            "atom_to_token": torch.eye(8, dtype=torch.long).repeat_interleave(3, dim=0),
            "res_type": torch.full((8,), AminoAcidSequenceFeature.UNK.value, dtype=torch.long),
        }

        assert_feature_dict_equality(out_feats.feature_dict, expected_features)

        assert out_feats.featurization_steps == [("new_chains", "test_new_chains_featurizer")]
        expected_metadata = {
            "test_new_chains_featurizer": {
                "name": "test_new_chains_featurizer",
                "type": "new_chains",
                "verbose_metadata": False,
                "lengths": (3, 3, 2),
                "entity_ids": (1, 1, 2),
                "estimate_centers_of_mass_function": {
                    "name": "",
                    "type": "predefined_center_of_mass",
                    "centers_of_mass": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                },
                "atoms_to_include": AtomFilter.BB,
            }
        }
        assert_metadata_equality(out_feats.featurization_metadata, expected_metadata)

    def test_new_chains_invalid_atom_specification(self, features_all_inpaint_options: dict[str, torch.Tensor]) -> None:
        """Try to add new chains with CA or FULL atoms to existing chains with BB atoms"""
        with pytest.raises(ValueError):
            featurizer = primitives.NewChainsFeaturizer(
                name="test_new_chains_featurizer",
                lengths=(50,),
                estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=((0.0, 0.0, 0.0),)
                ),
                atoms_to_include=AtomFilter.CA,
            )
            next(
                featurizer(
                    [Features(feature_dict=deepcopy(features_all_inpaint_options), atoms_to_include=AtomFilter.BB)]
                )
            )
        with pytest.raises(ValueError):
            featurizer = primitives.NewChainsFeaturizer(
                name="test_new_chains_featurizer",
                lengths=(50,),
                estimate_centers_of_mass_function=primitives.PredefinedCenterOfMassEstimator(
                    centers_of_mass=((0.0, 0.0, 0.0),)
                ),
                atoms_to_include=AtomFilter.FULL,
            )
            next(
                featurizer(
                    [Features(feature_dict=deepcopy(features_all_inpaint_options), atoms_to_include=AtomFilter.BB)]
                )
            )
