import pytest
from pydantic import BaseModel, ValidationError

from dynopsi.data.conditioning import (
    ConditioningCheckpointError,
    DesignConditioning,
    assert_registered_conditioning_matches,
)
from dynopsi.data.constants import DSSPLoopFeature, HomooligomerPartnerFeature


class _Model(BaseModel):
    design_conditioning: DesignConditioning = None


class TestDesignConditioning:
    @pytest.mark.parametrize("raw", [1, DSSPLoopFeature.NON_LOOP, "NON_LOOP"])
    def test_accepts_enum_int_and_name(self, raw: object) -> None:
        model = _Model(design_conditioning={"chain_is_loopy": {"not_loopy": raw}})
        assert model.design_conditioning == {"chain_is_loopy": {"not_loopy": DSSPLoopFeature.NON_LOOP}}

    def test_accepts_multi_key_inner_dict(self) -> None:
        model = _Model(
            design_conditioning={
                "chain_is_loopy": {
                    "not_loopy": DSSPLoopFeature.NON_LOOP,
                    "loopy": DSSPLoopFeature.LOOP_OR_OTHER,
                }
            }
        )
        assert len(model.design_conditioning["chain_is_loopy"]) == 2

    def test_accepts_none_and_empty(self) -> None:
        assert _Model().design_conditioning is None
        assert _Model(design_conditioning={}).design_conditioning == {}
        assert _Model(design_conditioning={"chain_is_loopy": {}}).design_conditioning == {"chain_is_loopy": {}}

    def test_accepts_registered_partner_key(self) -> None:
        model = _Model(
            design_conditioning={
                "has_homooligomer_partner": {"no_partner": HomooligomerPartnerFeature.NO_PARTNER},
            }
        )
        assert model.design_conditioning == {
            "has_homooligomer_partner": {"no_partner": HomooligomerPartnerFeature.NO_PARTNER},
        }

    @pytest.mark.parametrize("bad_key", ["target_sequence", "secondary_structure_feature", "unknown"])
    def test_rejects_unknown_outer_key(self, bad_key: str) -> None:
        with pytest.raises(ValidationError, match="Registered keys"):
            _Model(design_conditioning={bad_key: {"x": DSSPLoopFeature.NON_LOOP}})

    @pytest.mark.parametrize("bad_value", [True, "1", 1.5])
    def test_rejects_bool_numeric_string_and_float(self, bad_value: object) -> None:
        with pytest.raises(ValidationError, match="design_conditioning.chain_is_loopy") as exc_info:
            _Model(design_conditioning={"chain_is_loopy": {"x": bad_value}})
        assert "MASK_OR_MISSING, NON_LOOP, LOOP_OR_OTHER" in str(exc_info.value)


class TestAssertRegisteredConditioningMatches:
    def test_missing_partner_names_the_field_and_members(self) -> None:
        with pytest.raises(ConditioningCheckpointError) as exc_info:
            assert_registered_conditioning_matches(
                checkpoint_label="v1",
                declared_keys={"has_homooligomer_partner"},
                design_conditioning=None,
            )
        message = str(exc_info.value)
        assert (
            "design_conditioning.has_homooligomer_partner is declared by checkpoint "
            "'v1' and is missing from this config."
        ) in message
        assert "Allowed HomooligomerPartnerFeature members: MASK_OR_MISSING, NO_PARTNER, HAS_PARTNER." in message

    def test_extra_loop_key_names_the_field_and_members(self) -> None:
        with pytest.raises(ConditioningCheckpointError) as exc_info:
            assert_registered_conditioning_matches(
                checkpoint_label="v1",
                declared_keys=set(),
                design_conditioning={"chain_is_loopy": {"non_loopy": DSSPLoopFeature.NON_LOOP}},
            )
        message = str(exc_info.value)
        assert (
            "design_conditioning.chain_is_loopy is set in this config and is not declared by checkpoint 'v1'."
        ) in message
        assert "Allowed DSSPLoopFeature members: MASK_OR_MISSING, NON_LOOP, LOOP_OR_OTHER." in message

    def test_one_error_lists_every_mismatched_key(self) -> None:
        with pytest.raises(ConditioningCheckpointError) as exc_info:
            assert_registered_conditioning_matches(
                checkpoint_label="v1",
                declared_keys={"has_homooligomer_partner"},
                design_conditioning={"chain_is_loopy": {"non_loopy": DSSPLoopFeature.NON_LOOP}},
            )
        message = str(exc_info.value)
        assert "design_conditioning.has_homooligomer_partner" in message
        assert "design_conditioning.chain_is_loopy" in message
        assert type(exc_info.value) is ConditioningCheckpointError

    def test_equal_sets_pass(self) -> None:
        assert_registered_conditioning_matches(
            checkpoint_label="v1",
            declared_keys=set(),
            design_conditioning=None,
        )
        assert_registered_conditioning_matches(
            checkpoint_label="v1",
            declared_keys={"chain_is_loopy", "has_homooligomer_partner"},
            design_conditioning={
                "has_homooligomer_partner": {"no_partner": HomooligomerPartnerFeature.NO_PARTNER},
                "chain_is_loopy": {"non_loopy": DSSPLoopFeature.NON_LOOP},
            },
        )

    def test_empty_variant_map_is_a_missing_declared_key(self) -> None:
        with pytest.raises(ConditioningCheckpointError) as exc_info:
            assert_registered_conditioning_matches(
                checkpoint_label="v1",
                declared_keys={"chain_is_loopy"},
                design_conditioning={"chain_is_loopy": {}},
            )
        assert (
            "design_conditioning.chain_is_loopy is declared by checkpoint 'v1' and is missing from this config."
        ) in str(exc_info.value)

    def test_empty_variant_map_matches_when_the_checkpoint_omits_the_key(self) -> None:
        assert_registered_conditioning_matches(
            checkpoint_label="v1",
            declared_keys=set(),
            design_conditioning={"chain_is_loopy": {}},
        )
