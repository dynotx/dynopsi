from copy import deepcopy
from typing import Any

import pytest
import torch

import dynopsi.data.featurization.primitives.center_of_mass as com
from dynopsi.data.featurization import primitives
from dynopsi.data.featurization.base import Features
from dynopsi.utils.function_configs import ConstantFunction


class TestCenterOfMassFeaturizer:
    dummy_prior_means_config_chain_and_noise_separated = primitives.PriorMeansConfig(
        name="test_prior_means_config",
        prior_centering_strategy="chain_and_noise_separated",
        prior_mean_offset_function=ConstantFunction(constant=0.1),
    )
    dummy_prior_means_config_origin = primitives.PriorMeansConfig(
        name="test_prior_means_config",
        prior_centering_strategy="origin",
        prior_mean_offset_function=ConstantFunction(constant=0.1),
    )

    dummy_data_augmentation_config = primitives.DataAugmentationConfig(
        name="test_data_augmentation_config",
        centering_strategy="design_chains",
        random_global_translation_std=0.0,
        random_global_rotation=False,
    )

    def test_data_augmentation_config(self) -> None:

        cfg = primitives.DataAugmentationConfig(
            name="test_data_augmentation_config",
            centering_strategy="design_chains",
            random_global_translation_std=0.0,
            random_global_rotation=False,
        )

        expect_cfg = {
            "name": "test_data_augmentation_config",
            "centering_strategy": "design_chains",
            "random_global_translation_std": 0.0,
            "random_global_rotation": False,
        }
        assert cfg.model_dump() == expect_cfg

        with pytest.raises(ValueError):
            primitives.DataAugmentationConfig(
                centering_strategy="invalid",
                random_global_translation_std=0.1,
                random_global_rotation=True,
            )
        with pytest.raises(ValueError):
            primitives.DataAugmentationConfig(
                centering_strategy="design_chains",
                random_global_translation_std=-0.1,
                random_global_rotation=True,
            )

    @pytest.mark.parametrize("prior_centering_strategy", ["chain_and_noise_separated", "origin", "invalid"])
    def test_prior_means_config_offset_function(
        self,
        prior_centering_strategy: str,
    ) -> None:

        if prior_centering_strategy == "invalid":
            with pytest.raises(ValueError):
                primitives.PriorMeansConfig(
                    name="test_prior_means_config",
                    prior_centering_strategy=prior_centering_strategy,
                    prior_mean_offset_function=ConstantFunction(constant=0.1),
                )
        else:
            cfg = primitives.PriorMeansConfig(
                name="test_prior_means_config",
                prior_centering_strategy=prior_centering_strategy,
                prior_mean_offset_function=ConstantFunction(constant=0.1),
            )
            assert cfg.model_dump() == {
                "name": "test_prior_means_config",
                "prior_centering_strategy": prior_centering_strategy,
                "prior_mean_offset_function": {"type": "constant", "constant": 0.1},
            }

    @pytest.mark.parametrize("data_centering_strategy", ["design_chains"])
    def test_center_coordinates_multimer(
        self, data_centering_strategy: str, feature_dict_standard: dict[str, torch.Tensor]
    ) -> None:
        featurizer = primitives.CenterOfMassFeaturizer(
            data_augmentation_config=primitives.DataAugmentationConfig(
                centering_strategy=data_centering_strategy,
                random_global_translation_std=0.1,
                random_global_rotation=True,
            ),
            prior_means_config=self.dummy_prior_means_config_chain_and_noise_separated,
        )
        feature_dict_input = {key: value.clone() for key, value in feature_dict_standard.items() if key != "prior_mean"}

        resolved_binder_chain_mask = (
            torch.repeat_interleave(feature_dict_standard["binder_mask"], repeats=3, dim=0)
            & feature_dict_standard["atom_resolved_mask"]
        )
        expected_center = feature_dict_standard["coords"][:, resolved_binder_chain_mask].mean(dim=1)
        expected_coords_centered = feature_dict_input["coords"] - expected_center

        coords_centered = featurizer._center_coordinates(feature_dict_input)
        assert torch.allclose(coords_centered, expected_coords_centered)

    @pytest.mark.parametrize("data_centering_strategy", ["design_chains"])
    def test_center_coordinates_no_resolved_atoms(
        self,
        data_centering_strategy: str,
        feature_dict_standard: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        featurizer = primitives.CenterOfMassFeaturizer(
            data_augmentation_config=primitives.DataAugmentationConfig(
                centering_strategy=data_centering_strategy,
                random_global_translation_std=0.1,
                random_global_rotation=True,
            ),
            prior_means_config=self.dummy_prior_means_config_chain_and_noise_separated,
        )
        feature_dict_input = {key: value.clone() for key, value in feature_dict_standard.items() if key != "prior_mean"}
        feature_dict_input["atom_resolved_mask"] = torch.zeros_like(
            feature_dict_input["atom_resolved_mask"], dtype=torch.bool
        )
        feature_dict_input["frame_resolved_mask"] = torch.zeros_like(
            feature_dict_input["frame_resolved_mask"], dtype=torch.bool
        )
        # center will be on frame resolved mask, which is all resolved atoms
        with pytest.raises(ValueError):
            featurizer._center_coordinates(feature_dict_input)

    @pytest.mark.parametrize("prior_centering_strategy", ["chain_and_noise_separated", "origin"])
    def test_compute_prior_means(
        self,
        prior_centering_strategy: str,
        features_all_inpaint_options: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        offset_std = 0.2
        featurizer = primitives.CenterOfMassFeaturizer(
            data_augmentation_config=self.dummy_data_augmentation_config,
            prior_means_config=primitives.PriorMeansConfig(
                name="test_prior_means_config",
                prior_centering_strategy=prior_centering_strategy,
                prior_mean_offset_function=ConstantFunction(constant=offset_std),
            ),
        )
        features_all_inpaint_options = deepcopy(features_all_inpaint_options)

        def mock_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ones(*args, **kwargs)

        monkeypatch.setattr(com.torch, "randn", mock_randn)
        prior_means = featurizer._compute_prior_means(features_all_inpaint_options)

        # com values come from features_all_inpaint_options in tests/unit/dynopsi/data/conftest.py
        # specifically, 1.5 -> binder context, 3.3 -> binder inpaint
        # 4.4 -> target context, 6.1 target inpaint
        # there are equal numbers of residues for each component of target and binder
        binder_prior_com = (1.5 + 3.3) / 2 + offset_std
        target_prior_com = (4.4 + 6.1) / 2 + offset_std
        expected_prior_means = {
            "origin": torch.cat([torch.ones(8 * 3, 3) * offset_std, torch.zeros(1 * 3, 3)]).unsqueeze(0),
            "chain_and_noise_separated": torch.cat(
                [
                    torch.ones(3, 3) * binder_prior_com,  # noised binder context
                    torch.zeros(3, 3),  # non-noised binder context
                    torch.ones(3, 3) * binder_prior_com,  # noised binder inpaint
                    torch.zeros(3, 3),  # non-noised binder inpaint (shouldn't exist, but just for testing purposes)
                    torch.ones(3, 3) * target_prior_com,  # noised target context
                    torch.zeros(3, 3),  # non-noised target context
                    torch.ones(3, 3) * target_prior_com,  # noised target inpaint
                    torch.zeros(3, 3),  # non-noised target inpaint (shouldn't exist, but just for testing purposes)
                    torch.zeros(3, 3),  # padding
                ]
            ).unsqueeze(0),
        }[prior_centering_strategy]
        assert torch.allclose(prior_means, expected_prior_means)

    @pytest.mark.parametrize("prior_centering_strategy", ["chain_and_noise_separated", "origin"])
    def test_compute_prior_means_with_unresolved_atoms(
        self,
        prior_centering_strategy: str,
        features_all_design: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Specifically tests that the prior means are computed and assigned correctly when there are unresolved atoms."""
        offset_std = 0.2
        featurizer = primitives.CenterOfMassFeaturizer(
            data_augmentation_config=self.dummy_data_augmentation_config,
            prior_means_config=primitives.PriorMeansConfig(
                name="test_prior_means_config",
                prior_centering_strategy=prior_centering_strategy,
                prior_mean_offset_function=ConstantFunction(constant=offset_std),
            ),
        )

        features_all_design = deepcopy(features_all_design)

        def mock_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ones(*args, **kwargs)

        monkeypatch.setattr(com.torch, "randn", mock_randn)
        prior_means = featurizer._compute_prior_means(features_all_design)
        expected_prior_means = {
            "origin": torch.cat([torch.ones(7 * 3, 3) * offset_std, torch.zeros(1 * 3, 3)]).unsqueeze(0),
            "chain_and_noise_separated": torch.cat(
                [torch.ones(7 * 3, 3) * (5.5 + offset_std), torch.zeros(1 * 3, 3)]
            ).unsqueeze(0),
        }[prior_centering_strategy]
        assert torch.allclose(prior_means, expected_prior_means)

    @pytest.mark.parametrize("data_centering_strategy", ["design_chains"])
    @pytest.mark.parametrize("prior_centering_strategy", ["chain_and_noise_separated", "origin"])
    def test_com_featurizer(
        self,
        data_centering_strategy: str,
        prior_centering_strategy: str,
        features_all_inpaint_options: dict[str, torch.Tensor],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data_translation_std = 0.1
        prior_mean_offset_std = 0.2
        featurizer = primitives.CenterOfMassFeaturizer(
            name="test_com_featurizer",
            data_augmentation_config=primitives.DataAugmentationConfig(
                name="test_data_augmentation_config",
                centering_strategy=data_centering_strategy,
                random_global_translation_std=data_translation_std,
                random_global_rotation=False,
            ),
            prior_means_config=primitives.PriorMeansConfig(
                name="test_prior_means_config",
                prior_centering_strategy=prior_centering_strategy,
                prior_mean_offset_function=ConstantFunction(constant=prior_mean_offset_std),
            ),
        )
        feats = Features(feature_dict=deepcopy(features_all_inpaint_options))

        # mock for prior means
        def mock_randn(*args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ones(*args, **kwargs)

        def mock_randn_like(*args: Any, **kwargs: Any) -> torch.Tensor:
            return torch.ones_like(*args, **kwargs)

        monkeypatch.setattr(com.torch, "randn", mock_randn)
        monkeypatch.setattr(com.torch, "randn_like", mock_randn_like)
        out_feats = next(featurizer([feats]))

        # expectation
        expected_data_center = {
            "fixed": torch.tensor([(1.5 + 4.4 + 6.1) / 3] * 3),  # all target and binder context residues
            "design_chains": torch.tensor(
                [(1.5 + 3.3 + 4.4 + 6.1) / 4] * 3
            ),  # all residues because binder and target are both inpainted
        }[data_centering_strategy]

        expected_coords = features_all_inpaint_options["coords"] - expected_data_center + data_translation_std
        print("expected_coords:", expected_coords)

        # calc expected prior means for each of binder and target
        # apply the centering op first (coords - expected_data_center + data_translation_std)
        # then add the prior mean offset std
        binder_prior_com = (1.5 + 3.3) / 2 - expected_data_center + data_translation_std + prior_mean_offset_std
        target_prior_com = (4.4 + 6.1) / 2 - expected_data_center + data_translation_std + prior_mean_offset_std
        print("binder_prior_com:", binder_prior_com)
        print("target_prior_com:", target_prior_com)
        expected_prior_means = {
            "origin": torch.cat(
                [
                    torch.ones(8 * 3, 3) * (prior_mean_offset_std),
                    torch.zeros(1 * 3, 3),
                ]  # origin strategy for prior should be independent of data augmentation
            ).unsqueeze(0),
            "chain_and_noise_separated": torch.cat(
                [
                    torch.ones(3, 3) * binder_prior_com,  # noised binder context
                    torch.zeros(3, 3),  # non-noised binder context
                    torch.ones(3, 3) * binder_prior_com,  # noised binder inpaint
                    torch.zeros(3, 3),  # non-noised binder inpaint (shouldn't exist, but just for testing purposes)
                    torch.ones(3, 3) * target_prior_com,  # noised target context
                    torch.zeros(3, 3),  # non-noised target context
                    torch.ones(3, 3) * target_prior_com,  # noised target inpaint
                    torch.zeros(3, 3),  # non-noised target inpaint (shouldn't exist, but just for testing purposes)
                    torch.zeros(3, 3),  # padding
                ]
            ).unsqueeze(0),
        }[prior_centering_strategy]

        assert torch.allclose(out_feats.feature_dict["coords"], expected_coords)
        print("out_feats.feature_dict['prior_mean']:", out_feats.feature_dict["prior_mean"])
        print("expected_prior_means:", expected_prior_means)
        assert torch.allclose(out_feats.feature_dict["prior_mean"], expected_prior_means)

        assert out_feats.featurization_steps == [("center_of_mass", "test_com_featurizer")]
        print("out_feats.featurization_metadata:", out_feats.featurization_metadata)

        assert out_feats.featurization_metadata == {
            "test_com_featurizer": {
                "type": "center_of_mass",
                "name": "test_com_featurizer",
                "verbose_metadata": False,
                "data_augmentation_config": {
                    "name": "test_data_augmentation_config",
                    "centering_strategy": data_centering_strategy,
                    "random_global_translation_std": data_translation_std,
                    "random_global_rotation": False,
                },
                "prior_means_config": {
                    "name": "test_prior_means_config",
                    "prior_centering_strategy": prior_centering_strategy,
                    "prior_mean_offset_function": {
                        "type": "constant",
                        "constant": prior_mean_offset_std,
                    },
                },
            }
        }
