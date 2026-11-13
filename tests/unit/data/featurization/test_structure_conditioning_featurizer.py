import pytest
import torch

from dynopsi.data.conditioning import CategoricalCondition
from dynopsi.data.constants import (
    DSSPLoopFeature,
    HomooligomerPartnerFeature,
    InpaintFeature,
    SimplifiedDSSPFeature,
)
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.featurization.registry import get_featurizer_from_config
from dynopsi.data.utils import ChainResidueRepr


class TestChainIsLoopyConditioningFeaturizer:
    def test_labels_only_nonpadding_binder_inpaint_positions(
        self, feature_dict_standard: dict[str, torch.Tensor]
    ) -> None:
        feature_dict_standard["token_pad_mask"][1] = False
        input_features = Features(feature_dict=feature_dict_standard)
        featurizer = primitives.ChainIsLoopyConditioningFeaturizer(
            name="loopy_conditioning",
            design_is_loopy=DSSPLoopFeature.LOOP_OR_OTHER,
        )

        output_features = next(featurizer([input_features]))

        assert torch.equal(
            output_features["chain_is_loopy"],
            torch.tensor(
                [
                    DSSPLoopFeature.LOOP_OR_OTHER,
                    DSSPLoopFeature.MASK_OR_MISSING,
                    DSSPLoopFeature.MASK_OR_MISSING,
                    DSSPLoopFeature.MASK_OR_MISSING,
                    DSSPLoopFeature.MASK_OR_MISSING,
                    DSSPLoopFeature.MASK_OR_MISSING,
                ]
            ),
        )
        assert "chain_is_loopy" not in input_features
        assert output_features.featurization_steps == [("chain_is_loopy_conditioning", "loopy_conditioning")]
        assert output_features.featurization_metadata["loopy_conditioning"] == {
            "name": "loopy_conditioning",
            "verbose_metadata": False,
            "type": "chain_is_loopy_conditioning",
            "design_is_loopy": DSSPLoopFeature.LOOP_OR_OTHER,
        }

    def test_registry_lookup_returns_wrapper(self) -> None:
        featurizer = get_featurizer_from_config(
            {"type": "chain_is_loopy_conditioning", "name": "loopy", "design_is_loopy": DSSPLoopFeature.NON_LOOP}
        )
        assert isinstance(featurizer, primitives.ChainIsLoopyConditioningFeaturizer)


class TestConstantWriteConditioningFeaturizer:
    def test_registry_lookup_returns_generic_writer(self, register_categorical_condition) -> None:
        register_categorical_condition(
            CategoricalCondition(
                key="beta_test",
                vocab=DSSPLoopFeature,
                write_inpaint=(InpaintFeature.BINDER_INPAINT,),
                inference_type="constant_write_conditioning",
            )
        )
        featurizer = get_featurizer_from_config(
            {
                "type": "constant_write_conditioning",
                "name": "beta",
                "feature_key": "beta_test",
                "value": int(DSSPLoopFeature.NON_LOOP),
            }
        )
        assert isinstance(featurizer, primitives.ConstantWriteConditioningFeaturizer)
        assert featurizer.feature_key == "beta_test"

    def test_binder_or_target_scope_writes_both_inpaint_regions(
        self, register_categorical_condition, features_all_inpaint_options: dict[str, torch.Tensor]
    ) -> None:
        register_categorical_condition(
            CategoricalCondition(
                key="beta_test",
                vocab=DSSPLoopFeature,
                write_inpaint=(InpaintFeature.BINDER_INPAINT, InpaintFeature.TARGET_INPAINT),
                inference_type="constant_write_conditioning",
            ),
        )
        featurizer = primitives.ConstantWriteConditioningFeaturizer(
            name="beta", feature_key="beta_test", value=DSSPLoopFeature.NON_LOOP
        )

        output = next(featurizer([Features(feature_dict=features_all_inpaint_options)]))

        expected = torch.full((9,), int(DSSPLoopFeature.MASK_OR_MISSING), dtype=torch.long)
        expected[[2, 3, 6, 7]] = int(DSSPLoopFeature.NON_LOOP)
        assert torch.equal(output["beta_test"], expected)

    def test_monomer_guard_masks_single_chain(
        self, register_categorical_condition, feature_dict_chain_A_only: dict[str, torch.Tensor]
    ) -> None:
        register_categorical_condition(
            CategoricalCondition(
                key="beta_test",
                vocab=DSSPLoopFeature,
                write_inpaint=(InpaintFeature.BINDER_INPAINT,),
                inference_type="constant_write_conditioning",
                mask_if_nonpad_chains_le=1,
            ),
        )
        featurizer = primitives.ConstantWriteConditioningFeaturizer(
            name="beta", feature_key="beta_test", value=DSSPLoopFeature.NON_LOOP
        )

        output = next(featurizer([Features(feature_dict=feature_dict_chain_A_only)]))

        assert torch.equal(
            output["beta_test"],
            torch.full_like(output["beta_test"], int(DSSPLoopFeature.MASK_OR_MISSING)),
        )

    def test_monomer_guard_writes_when_multiple_chains(
        self, register_categorical_condition, features_all_inpaint_options: dict[str, torch.Tensor]
    ) -> None:
        register_categorical_condition(
            CategoricalCondition(
                key="beta_test",
                vocab=DSSPLoopFeature,
                write_inpaint=(InpaintFeature.BINDER_INPAINT,),
                inference_type="constant_write_conditioning",
                mask_if_nonpad_chains_le=1,
            ),
        )
        featurizer = primitives.ConstantWriteConditioningFeaturizer(
            name="beta", feature_key="beta_test", value=DSSPLoopFeature.NON_LOOP
        )

        output = next(featurizer([Features(feature_dict=features_all_inpaint_options)]))

        expected = torch.full((9,), int(DSSPLoopFeature.MASK_OR_MISSING), dtype=torch.long)
        expected[[2, 3]] = int(DSSPLoopFeature.NON_LOOP)
        assert torch.equal(output["beta_test"], expected)

    def test_rejects_unknown_feature_key(self) -> None:
        with pytest.raises(ValueError, match="Unknown design_conditioning key"):
            primitives.ConstantWriteConditioningFeaturizer(name="beta", feature_key="beta_test", value=1)

    def test_rejects_value_outside_vocab(self, register_categorical_condition) -> None:
        register_categorical_condition(
            CategoricalCondition(
                key="beta_test",
                vocab=DSSPLoopFeature,
                write_inpaint=(InpaintFeature.BINDER_INPAINT,),
                inference_type="constant_write_conditioning",
            )
        )
        with pytest.raises(ValueError, match="is not a valid"):
            primitives.ConstantWriteConditioningFeaturizer(name="beta", feature_key="beta_test", value=99)


class TestHomooligomerPartnerConstantWrite:
    def test_writes_binder_and_target_inpaint(self, features_all_inpaint_options: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.ConstantWriteConditioningFeaturizer(
            name="partner",
            feature_key="has_homooligomer_partner",
            value=HomooligomerPartnerFeature.NO_PARTNER,
        )

        output = next(featurizer([Features(feature_dict=features_all_inpaint_options)]))

        expected = torch.full((9,), int(HomooligomerPartnerFeature.MASK_OR_MISSING), dtype=torch.long)
        expected[[2, 3, 6, 7]] = int(HomooligomerPartnerFeature.NO_PARTNER)
        assert torch.equal(output["has_homooligomer_partner"], expected)
        assert output.featurization_steps == [("constant_write_conditioning", "partner")]

    def test_masks_monomer(self, feature_dict_chain_A_only: dict[str, torch.Tensor]) -> None:
        featurizer = primitives.ConstantWriteConditioningFeaturizer(
            name="partner",
            feature_key="has_homooligomer_partner",
            value=HomooligomerPartnerFeature.NO_PARTNER,
        )

        output = next(featurizer([Features(feature_dict=feature_dict_chain_A_only)]))

        assert torch.equal(
            output["has_homooligomer_partner"],
            torch.full_like(output["has_homooligomer_partner"], int(HomooligomerPartnerFeature.MASK_OR_MISSING)),
        )

    def test_make_constant_write_returns_generic_writer(self) -> None:
        featurizer = primitives.make_constant_write(
            key="has_homooligomer_partner",
            name="no_partner",
            value=int(HomooligomerPartnerFeature.NO_PARTNER),
        )
        assert isinstance(featurizer, primitives.ConstantWriteConditioningFeaturizer)
        assert featurizer.feature_key == "has_homooligomer_partner"


class TestSecondaryStructureConditioningFeaturizer:
    @pytest.mark.parametrize(
        ("index_type", "design_secondary_structure", "expected"),
        [
            (
                "original_residue_index",
                {
                    SimplifiedDSSPFeature.HELIX: "B1",
                    SimplifiedDSSPFeature.STRAND: "C1",
                },
                [
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.HELIX,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.STRAND,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                ],
            ),
            (
                "auth_residue_index",
                {
                    SimplifiedDSSPFeature.HELIX: "V2",
                    SimplifiedDSSPFeature.STRAND: "W4",
                },
                [
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.HELIX,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                    SimplifiedDSSPFeature.STRAND,
                    SimplifiedDSSPFeature.MASK_OR_MISSING,
                ],
            ),
        ],
    )
    def test_assigns_secondary_structure_using_configured_index_type(
        self,
        feature_dict_standard: dict[str, torch.Tensor],
        index_type: str,
        design_secondary_structure: dict[SimplifiedDSSPFeature, str],
        expected: list[SimplifiedDSSPFeature],
    ) -> None:
        feature_dict_standard["inpaint_feature"][2] = InpaintFeature.BINDER_INPAINT
        feature_dict_standard["inpaint_feature"][4] = InpaintFeature.BINDER_INPAINT
        featurizer = primitives.SecondaryStructureConditioningFeaturizer(
            name="secondary_structure_conditioning",
            design_secondary_structure=design_secondary_structure,
            index_type=index_type,
        )

        output_features = next(featurizer([Features(feature_dict=feature_dict_standard)]))

        assert torch.equal(
            output_features["secondary_structure_feature"],
            torch.tensor(expected, dtype=torch.long),
        )
        assert output_features.featurization_steps == [
            ("sparse_secondary_structure_conditioning", "secondary_structure_conditioning")
        ]

    def test_rejects_overlapping_secondary_structure_definitions(self) -> None:
        with pytest.raises(ValueError, match="secondary structure definitions overlap"):
            primitives.SecondaryStructureConditioningFeaturizer(
                design_secondary_structure={
                    SimplifiedDSSPFeature.HELIX: ChainResidueRepr.from_string("B1-2"),
                    SimplifiedDSSPFeature.STRAND: ChainResidueRepr.from_string("B2"),
                },
                index_type="original_residue_index",
            )

    def test_rejects_secondary_structure_on_non_design_position(
        self, feature_dict_standard: dict[str, torch.Tensor]
    ) -> None:
        featurizer = primitives.SecondaryStructureConditioningFeaturizer(
            design_secondary_structure={SimplifiedDSSPFeature.HELIX: "B1"},
            index_type="original_residue_index",
        )

        with pytest.raises(ValueError, match="overlaps with non-designed positions"):
            next(featurizer([Features(feature_dict=feature_dict_standard)]))
