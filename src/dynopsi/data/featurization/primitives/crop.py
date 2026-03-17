from collections.abc import Iterable, Iterator
from typing import Literal

import torch
from pydantic import field_validator

from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.featurization.utils import crop_features
from dynopsi.data.utils import ChainResidueRepr


@register_featurizer()
class CropFeaturizer(Featurizer):
    """Featurizer that crops features to a subset of residues based on chain and residue indices.

    The crop is specified by `residue_index_chain_spec` (a `ChainResidueRepr`) and
    `index_type`, which controls how indices in are interpreted:
    - original_residue_index: `ChainResidueRepr` specification using original residue indices that
        are 1-indexed and restart at the first residue in each chain.
    - auth_residue_index: `ChainResidueRepr` specification using PDB author residue indices. These
        exactly match `auth_residue_indices` in the PDB file and so can be arbitrary, and ordering
        may not be preserved in this index system. It is recommended to verify against the PDB file
        to ensure correct ordering.

    For index definitions, see `data.featurization.README.md#definition-of-indices`.

    Attributes
    ----------
    residue_index_chain_spec : ChainResidueRepr
        Which chain(s) and residue indices to keep (e.g. from string `"A1-10,B"`).
    index_type : {"original_residue_index", "auth_residue_index"}
        Which index type to use when applying the crop.
    """

    type: Literal["crop"] = "crop"
    residue_index_chain_spec: ChainResidueRepr
    index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"

    @field_validator("residue_index_chain_spec", mode="before")
    @classmethod
    def _parse_residue_index_chain_spec(cls, v: str | ChainResidueRepr) -> ChainResidueRepr:
        if isinstance(v, str):
            return ChainResidueRepr.from_string(v)
        return v

    @property
    def asym_id_to_index_map(self) -> dict[int, torch.Tensor | None]:
        """Map from asym ID to residue indices in the crop, using the specified index type.

        Returns
        -------
        dict[int, torch.Tensor | None]
            Keys are asym IDs; values are 1D tensors of residue indices, or None for "whole chain".

        See Also
        --------
        ChainResidueRepr.get_asym_id_to_index_map : Builds the underlying asym ID to index list mapping.
        """
        result = {
            k: torch.tensor(v) if v is not None else None
            for k, v in self.residue_index_chain_spec.get_asym_id_to_index_map(self.index_type).items()
        }
        return result

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Crop each feature to the specified subset of residues.

        Builds a mask correponding to `residue_index_chain_spec` and `index_type`, then crops
        features and reference structure (if present). Residue indices in the output are reset to
        start at 0 and `original_residue_index` is preserved before cropping.

        Parameters
        ----------
        inputs : Iterable[Features]
            Features to crop.

        Yields
        ------
        Features
            Cropped features with updated metadata tracking the crop operation
            and optional updated reference structure.

        Raises
        ------
        ValueError
            If `index_type` is invalid (should not occur with valid config).
        ValueError
            If `index_type` is `"auth_residue_index"` and `auth_residue_index` is not
            present in the feature dict.

        Notes
        -----
        If `original_residue_index` is not in the features, it is assumed to be the same
        as the current `residue_index` and is added to the features.
        """
        for input_features in inputs:
            new_feats = input_features.safe_copy()

            # if original_residue_index is not present in the features, it is assumed that
            # it is still the same as the residue_index and is added to the features.
            # we need to add it because the crop could potentially change residue_index
            # and we want to preserve the original_residue_index for the next featurizer.
            if "original_residue_index" not in new_feats.keys():
                new_feats["original_residue_index"] = new_feats["residue_index"].clone()

            if self.index_type == "auth_residue_index":
                asym_id_type = "auth_asym_id"
                if "auth_residue_index" not in new_feats.keys():
                    raise ValueError("auth_residue_index is not present in the features")
            else:
                asym_id_type = "asym_id"

            crop_mask = self.residue_index_chain_spec.to_mask(
                asym_id=new_feats[asym_id_type],
                residue_index=new_feats[self.index_type],
                pad_mask=new_feats["token_pad_mask"],
                index_type=self.index_type,
            )
            token_mask = torch.zeros_like(new_feats["token_index"], dtype=torch.bool)
            token_mask[crop_mask] = True

            # reset the residue index to start at 0
            # this means that the original_residue_index will potentially diverge from the residue_index
            new_feats.feature_dict = crop_features(new_feats.feature_dict, token_mask, reset_residue_index=True)

            # update the metadata
            more_metadata = {}
            # update reference structure that has sidechains
            index_type: Literal["residue_index", "auth_residue_index"] = (
                "residue_index" if self.index_type == "original_residue_index" else "auth_residue_index"
            )

            if new_feats.input_reference_structure is not None:
                structure = new_feats.input_reference_structure.crop(
                    residue_index_chain_spec=self.residue_index_chain_spec, index_type=index_type
                )
                new_feats.input_reference_structure = structure
                if self.verbose_metadata:
                    more_metadata["structure"] = structure

            self._update_features_metadata(new_feats, more_metadata=more_metadata)

            yield new_feats
