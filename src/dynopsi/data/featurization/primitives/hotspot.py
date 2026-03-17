from collections.abc import Iterable, Iterator
from typing import Literal

import torch
from pydantic import field_validator

from dynopsi.data.constants import PocketFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.utils import ChainResidueRepr


@register_featurizer()
class HotspotFeaturizer(Featurizer):
    """Featurizer that adds hotspot information to input features.

    Attributes
    ----------
    type : Literal["hotspot"]
      Featurizer type used for serialization/deserialization.
    hotspot_positions : ChainResidueRepr
      Positions of hotspot residues.
      These will be assigned PocketFeature.POCKET in the output features.
       See `data.constants.PocketFeature` for details on the encoding.
    index_type : Literal["original_residue_index", "auth_residue_index"]
      Which index type to use for specifying hotspot positions, either
      the original 1-index residue indices (`original_residue_index`) or
      PDB author residue indices (`author_residue_index`).
      See data.featurization.README.md#definition-of-indices for details on the indices.
    """

    type: Literal["hotspot"] = "hotspot"
    hotspot_positions: ChainResidueRepr
    index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"

    @field_validator("hotspot_positions", mode="before")
    @classmethod
    def _parse_hotspot_positions(cls, v: str | ChainResidueRepr) -> ChainResidueRepr:
        if isinstance(v, str):
            return ChainResidueRepr.from_string(v)
        return v

    @property
    def asym_id_to_index_map(self) -> dict[int, torch.Tensor | None]:
        """Map from asym ID to residue indices in the crop, using the specified index type.

        Returns
        -------
        dict[int, torch.Tensor | None]
          Keys are asym IDs; values are 1D tensors of residue indices, or None for whole chain.

        See Also
        --------
        ChainResidueRepr.get_aym_id_to_index_map : Builds the underlying asym ID to index list mapping.
        """
        result = {
            k: torch.tensor(v) if v is not None else None
            for k, v in self.hotspot_positions.get_asym_id_to_index_map(self.index_type).items()
        }
        return result

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Add hotspot information to each input feature.

        Hotspot information is encoded as a tensor using the
        encoding detailed in data.constants.PocketFeature and added
        to the input features with the key `pocket_feature`.

        Parameters
        ----------
        inputs : Iterable[Features]
          Features to which to add hotspot information.

        Yields
        ------
        Features
          Input features with added `pocket_feature` and updated metadata.

        Raises
        ------
        ValueError
          If `index_type` is `"auth_residue_index"` and `auth_residue_index`is not
          present in the feature dict.
        ValueError
          If the hotspot specified would include binder residues.

        Notes
        -----
          If `original_residue_index` is not in the features, it is assumed to be the same
          as the current `residue_index` and is added to the features.
        """
        for input_features in inputs:
            new_feats = input_features.safe_copy()
            # potentially add original_residue_index to the features
            if "original_residue_index" not in new_feats.keys():
                new_feats["original_residue_index"] = new_feats["residue_index"].clone()

            # check that auth_residue_index is present in the features
            # if it is used to specify the hotspot positions
            if self.index_type == "auth_residue_index":
                asym_id_type = "auth_asym_id"
                if "auth_residue_index" not in new_feats.keys():
                    raise ValueError("auth_residue_index is not present in the features")
            else:
                asym_id_type = "asym_id"

            # use hotspot_mask to determine POCKET
            hotspot_mask = self.hotspot_positions.to_mask(
                asym_id=new_feats[asym_id_type],
                residue_index=new_feats[self.index_type],
                pad_mask=new_feats["token_pad_mask"],
                index_type=self.index_type,
            )
            # check that the hotspot_mask doesn't intersect with binder_mask
            if (hotspot_mask & new_feats["binder_mask"]).any():
                raise ValueError("The hotspot mask intersects with the binder mask")

            # build the pocket_feature tensor
            # 1. initialize pocket_feature to UNSPECIFIED
            pocket_feature = torch.zeros_like(new_feats["token_index"]) + PocketFeature.UNSPECIFIED

            # 2. use binder_mask and target_mask to determine BINDER and UNSELECTED, respectively
            pocket_feature[new_feats["binder_mask"]] = PocketFeature.BINDER
            pocket_feature[new_feats["target_mask"]] = PocketFeature.UNSELECTED

            # 3. set the pocket feature to POCKET for the hotspot positions
            pocket_feature[hotspot_mask] = PocketFeature.POCKET

            # update the features w/ pocket_feature
            new_feats["pocket_feature"] = pocket_feature

            # update the metadata
            self._update_features_metadata(new_feats)

            yield new_feats
