from collections.abc import Iterable, Iterator
from typing import Literal

import torch

from dynopsi.data.constants import DesignTask, InpaintFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.utils import ChainResidueRepr


@register_featurizer()
class NoiseMaskFeaturizer(Featurizer):
    """Featurizer that sets a noise mask in accordance with the design task.

    Assigns a per-residue `noised_mask` based on the per-residue
    `inpaint_feature` and the design task, which the set of feature
    values that are noised.

    If the input Features contains an `input_reference_structure` to track the
    true full-atom structure, this featurizer will also crop the reference structure to
    be just the non-noised positions.

    Parameters
    ----------
    design_task : DesignTask
        Design task used to determine which inpaint features are noised via
        `DesignTask.get_noise_bool`.

    Raises
    ------
    ValueError
        If the design task does not define a noise setting for an inpaint
        feature value present in the input.

    Notes
    -----
    The mapping from inpaint feature to noise behavior is defined per design
    task; see `dynopsi.data.constants` for more.
    """

    type: Literal["noise_mask"] = "noise_mask"
    design_task: DesignTask

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Add or update the noise mask of each input, optionally cropping corresponding reference structures.

        For each input, computes `noised_mask` from `inpaint_feature`
        and the design task, records provenance by updating features metadata,
        and crops `input_reference_structure` to non-noised, frame-resolved residues
        if present.

        Parameters
        ----------
        inputs : Iterable[Features]
            Features to process; must include `inpaint_feature` and
            `token_pad_mask`.

        Yields
        ------
        Features
            Copy of each input features with `noised_mask` set and reference structure
            cropped if applicable.

        Raises
        ------
        ValueError
            If the design task has no noise setting for an inpaint feature
            value in the input features.
        """
        for input_features in inputs:
            new_feats = input_features.safe_copy()
            inpaint_feature_values = new_feats["inpaint_feature"][new_feats["token_pad_mask"].bool()].unique()

            noised_mask = torch.zeros_like(new_feats["token_pad_mask"], dtype=torch.bool)
            for inpaint_feature_value in inpaint_feature_values:
                noise_bool = self.design_task.get_noise_bool(inpaint_feature_value.item())
                if noise_bool is None:
                    raise ValueError(
                        f"Design task {self.design_task} does not support inpaint feature {inpaint_feature_value}"
                    )
                set_mask = (new_feats["inpaint_feature"] == inpaint_feature_value) & new_feats["token_pad_mask"].bool()
                noised_mask[set_mask] = noise_bool
            new_feats["noised_mask"] = noised_mask

            more_metadata = {}
            # crop the reference structure to be just the context positions
            if new_feats.input_reference_structure is not None:
                context_mask = torch.isin(
                    new_feats["inpaint_feature"],
                    torch.tensor([InpaintFeature.BINDER_CONTEXT, InpaintFeature.TARGET_CONTEXT]),
                )
                context_repr = ChainResidueRepr.from_asym_id_mask(
                    context_mask & new_feats["token_pad_mask"].bool(),
                    asym_id=new_feats["asym_id"],
                    residue_index=new_feats["residue_index"],
                    index_type="residue_index",
                )
                # the feature dictionary doesn't change, so the residue index must be preserved
                structure = new_feats.input_reference_structure.crop(
                    residue_index_chain_spec=context_repr, index_type="residue_index", reset_residue_index=False
                )
                new_feats.input_reference_structure = structure
                if self.verbose_metadata:
                    more_metadata["structure"] = structure

            # perform metadata updates for feature steps and input reference structure
            self._update_features_metadata(new_feats, more_metadata=more_metadata)

            yield new_feats
