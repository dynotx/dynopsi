import itertools
from collections.abc import Iterable, Iterator
from typing import Literal, Self

import torch
from pydantic import model_validator

from dynopsi.data.conditioning import _coerce_condition_value, get_categorical_condition
from dynopsi.data.constants import DSSPLoopFeature, InpaintFeature, SimplifiedDSSPFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.utils import ChainResidueRepr


class ConstantWriteConditioning(Featurizer):
    """Subclasses supply the registry key and the value; the write scope, pad fill, and
    inference guard are read from the :class:`CategoricalCondition` for that key.
    """

    def _condition_key(self) -> str:
        raise NotImplementedError

    def _condition_value(self) -> int:
        raise NotImplementedError

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        key = self._condition_key()
        condition = get_categorical_condition(key)
        mask_value = int(condition.vocab["MASK_OR_MISSING"])
        value = int(self._condition_value())

        for input_features in inputs:
            new_feats = input_features.safe_copy()
            inpaint_feature = new_feats["inpaint_feature"]
            pad_mask = new_feats["token_pad_mask"].bool()

            write_inpaint = torch.tensor(
                [int(feature) for feature in condition.write_inpaint],
                device=inpaint_feature.device,
                dtype=inpaint_feature.dtype,
            )
            design_mask = torch.isin(inpaint_feature, write_inpaint) & pad_mask
            write_value = value
            threshold = condition.mask_if_nonpad_chains_le
            if threshold is not None and int(new_feats["asym_id"][pad_mask].unique().numel()) <= threshold:
                write_value = mask_value

            feature = torch.full(
                (inpaint_feature.shape[0],),
                mask_value,
                dtype=torch.long,
                device=inpaint_feature.device,
            )
            feature[design_mask] = write_value
            new_feats[key] = feature
            self._update_features_metadata(new_feats)

            yield new_feats


@register_featurizer()
class ConstantWriteConditioningFeaturizer(ConstantWriteConditioning):
    """Write a registry categorical's constant value onto designed tokens."""

    type: Literal["constant_write_conditioning"] = "constant_write_conditioning"
    feature_key: str
    value: int

    @model_validator(mode="after")
    def validate_registered_condition(self) -> Self:
        condition = get_categorical_condition(self.feature_key)
        self.value = int(_coerce_condition_value(self.feature_key, condition.vocab, self.value))
        return self

    def _condition_key(self) -> str:
        return self.feature_key

    def _condition_value(self) -> int:
        return self.value


@register_featurizer()
class ChainIsLoopyConditioningFeaturizer(ConstantWriteConditioning):
    """Pins the YAML ``type`` ``chain_is_loopy_conditioning`` and field ``design_is_loopy``.

    Attributes
    ----------
    design_is_loopy: DSSPLoopFeature
        The value written at designed positions.
    """

    type: Literal["chain_is_loopy_conditioning"] = "chain_is_loopy_conditioning"
    design_is_loopy: DSSPLoopFeature

    def _condition_key(self) -> str:
        return "chain_is_loopy"

    def _condition_value(self) -> int:
        return int(self.design_is_loopy)


def make_constant_write(*, key: str, name: str, value: int) -> Featurizer:
    """Build the writer for a registry categorical, keeping loopy's legacy type."""
    condition = get_categorical_condition(key)
    if condition.inference_type == "chain_is_loopy_conditioning":
        return ChainIsLoopyConditioningFeaturizer(name=name, design_is_loopy=DSSPLoopFeature(value))
    return ConstantWriteConditioningFeaturizer(name=name, feature_key=key, value=int(value))


@register_featurizer()
class SecondaryStructureConditioningFeaturizer(Featurizer):
    """Featurizer that adds 'secondary_structure_feature' as a conditioning feature to the input features.


    Attributes
    ----------
    design_secondary_structure: dict[SimplifiedDSSPFeature, ChainResidueRepr]
        The design_secondary_structure feature to add to the input features at the designed positions.
    index_type: Literal["original_residue_index", "auth_residue_index"]
        The type of index to use for the residue indices.
    """

    type: Literal["sparse_secondary_structure_conditioning"] = "sparse_secondary_structure_conditioning"
    design_secondary_structure: dict[SimplifiedDSSPFeature, ChainResidueRepr]
    index_type: Literal["original_residue_index", "auth_residue_index"]

    @model_validator(mode="after")
    def validate_no_overlapping_ss_definitions(self) -> Self:
        """Validate that the secondary structure definitions do not overlap.

        Raises
        ------
        ValueError
            If the secondary structure definitions overlap.
        """
        for crr1, crr2 in itertools.combinations(self.design_secondary_structure.values(), 2):
            if crr1.overlaps(crr2):
                raise ValueError("The secondary structure definitions overlap.")
        return self

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Call the featurizer.

        Parameters
        ----------
        inputs: Iterable[Features]
            The input features to add the secondary_structure_feature to.
            Must have "inpaint_feature", "token_pad_mask", "asym_id", and "residue_index" features.

        Returns
        -------
        Iterator[Features]
            The output features with the secondary_structure_feature added.

        Raises
        ------
        ValueError
            If the "secondary_structure_feature" overlaps with non-designed positions.
        """
        for input_features in inputs:
            new_feats = input_features.safe_copy()
            design_mask = (new_feats["inpaint_feature"] == InpaintFeature.BINDER_INPAINT) & new_feats[
                "token_pad_mask"
            ].bool()

            secondary_structure_feature = torch.full(
                (new_feats["inpaint_feature"].shape[0],),
                int(SimplifiedDSSPFeature.MASK_OR_MISSING),
                dtype=torch.long,
                device=new_feats["inpaint_feature"].device,
            )
            for ss_feature, crr in self.design_secondary_structure.items():
                asym_id_key = "asym_id" if self.index_type == "original_residue_index" else "auth_asym_id"
                ss_mask = crr.to_mask(
                    asym_id=new_feats[asym_id_key],
                    residue_index=new_feats[self.index_type],
                    pad_mask=new_feats["token_pad_mask"],
                    index_type=self.index_type,
                    auth_chain_id_table=new_feats.resolved_auth_chain_id_table(),
                )
                if (ss_mask & ~design_mask).any():
                    raise ValueError("The secondary structure definition overlaps with non-designed positions.")
                secondary_structure_feature[ss_mask] = ss_feature
            new_feats["secondary_structure_feature"] = secondary_structure_feature

            self._update_features_metadata(new_feats)

            yield new_feats
