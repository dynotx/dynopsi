from collections.abc import Iterable, Iterator
from typing import Literal, NamedTuple

import torch
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from dynopsi.data.constants import (
    AtomFilter,
    InpaintFeature,
)
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.featurization.utils import (
    concat_features,
    crop_features,
    make_empty_chain,
)
from dynopsi.data.utils import ChainResidueRepr


class InpaintTokenIndexInfo(NamedTuple):
    """Stores information about the inpainting token indices and segment center of mass.

    Attributes
    ----------
    start_idx: int
        The start index of the inpaint segment, relative to the seq_length axis of the features, inclusive.
    end_idx: int
        The end index of the inpaint segment, relative to the seq_length axis of the features, inclusive.
    segment_length: int
        The length of the inpaint segment, in residues.
    center_of_mass: torch.Tensor | None # shape (3,)
        The center of mass of the inpainting region provided by the user or if None, indicates that the center of mass should be calculated from the atoms in the segment
    """

    start_idx: int
    end_idx: int
    feature_length: int
    center_of_mass: torch.Tensor | None  # shape (3,)


@register_featurizer()
class InpaintFeaturizer(Featurizer):
    """Designates inpainted and context residues, editing the length of inpainted segments according to the inpainting specifications.

    Noised inpainted residues will be inserted into (deleted from) an inpainted segment if the specified length of the redesigned segment
    is longer (shorter) than the input segment. Context residues cannot be inserted or deleted.

    Attributes
    ----------
    inpaint_segments: tuple[ChainResidueRepr, ...]
        The sequence of segments to inpaint.
        Each segment must consist of a contiguous set of residue indices from a single chain.
        A ChainResidueRepr can be provided as a string: e.g. 'A1-5,B10-15'.
    inpaint_lengths: tuple[int, ...]
        The desired lengths of the inpainting segments.
        Must be the same length as `inpaint_segments`.
        If the desired length is greater than the number of residues in the corresponding `inpaint_segment`,
        then noised residues will be inserted.
    inpaint_segments_centers_of_mass: tuple[tuple[float, float, float] | None, ...]
        Coordinates of the center of mass for each corresponding `inpaint_segment`.
        If None, the center of mass will be calculated as the average of the coordinates of the atoms in the segment.
    atoms_to_include: AtomFilter
    index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"
        The index type of the `inpaint_segments` specification. See data.featurization.README.md#definition-of-indices for details.
    """

    type: Literal["inpainting"] = "inpainting"
    inpainting_positions: tuple[ChainResidueRepr, ...] = Field(
        description="Each Inpainted segment can be specified as ChainResidueRepr or string. It will be automatically parsed from string if provided."
    )
    inpainting_lengths: tuple[int, ...]
    inpaint_region_centers_of_mass: tuple[tuple[float, float, float] | None, ...]
    atoms_to_include: AtomFilter
    index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"

    @field_validator("inpainting_positions", mode="before")
    @classmethod
    def _parse_inpainting_positions(cls, v: tuple[ChainResidueRepr | str, ...]) -> tuple[ChainResidueRepr, ...]:
        """
        Parse string representations of inpainting positions to ChainResidueRepr objects.
        Raises: ValueError if the representation is not contiguous.
        """
        parsed_positions = []
        for pos in v:
            if isinstance(pos, str):
                pos = ChainResidueRepr.from_string(pos)
            if not pos.is_contiguous:
                raise ValueError("inpainting positions must be contiguous segments")
            parsed_positions.append(pos)
        return tuple(parsed_positions)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.inpainting_positions) != len(self.inpainting_lengths):
            raise ValueError("inpainting_positions and inpainting_lengths must have the same length")
        if len(self.inpainting_positions) != len(self.inpaint_region_centers_of_mass):
            raise ValueError("inpainting_positions and inpaint_region_centers_of_mass must have the same length")

        return self

    def _extract_inpaint_regions(self, feature_dict: dict[str, torch.Tensor]) -> list[InpaintTokenIndexInfo]:
        """Translate inpainting configuration into a list of InpaintTokenIndexInfo objects

        The inpainting configuration consists of the `inpaint_positions`, `inpaint_lengths`, and `inpaint_segments_centers_of_mass`,
        and the `InpaintTokenIndexInfo` objects describe the inpainting segments with respect to the input features (feature_dict).

        Parameters
        ----------
        feature_dict: dict[str, torch.Tensor]
            Required keys: `asym_id`, `{self.index_type}`, `token_pad_mask`.

        Returns
        -------
        list[InpaintTokenIndexInfo]
            Describes the inpainting regions with respect to the input features.
        """
        inpaint_regions: list[InpaintTokenIndexInfo] = []
        for chain_repr, feature_length, inpaint_region_center_of_mass in zip(
            self.inpainting_positions, self.inpainting_lengths, self.inpaint_region_centers_of_mass
        ):
            # get a mask corresponding to the positions to be inpainted
            # whether this mask is computed based on the original_residue_index or auth_residue_index depends on the config
            if self.index_type == "auth_residue_index":
                asym_id_type = "auth_asym_id"
            else:
                asym_id_type = "asym_id"
            position_mask = chain_repr.to_mask(
                feature_dict[asym_id_type],
                feature_dict[self.index_type],
                feature_dict["token_pad_mask"],
                index_type=self.index_type,
            )

            inpaint_start_idx = int(torch.where(position_mask)[0].min().long().item())
            inpaint_end_idx = int(torch.where(position_mask)[0].max().long().item()) + 1

            inpaint_regions.append(
                InpaintTokenIndexInfo(
                    start_idx=inpaint_start_idx,
                    end_idx=inpaint_end_idx,
                    feature_length=feature_length,
                    center_of_mass=(
                        torch.tensor(inpaint_region_center_of_mass)
                        if inpaint_region_center_of_mass is not None
                        else None
                    ),
                )
            )

        # sort the inpaint regions by the start position with respect to the full features
        inpaint_regions.sort(key=lambda x: x.start_idx)

        return inpaint_regions

    def _make_inpaint_feature_segment(
        self, dict_features: dict[str, torch.Tensor], segment_params: InpaintTokenIndexInfo
    ) -> dict[str, torch.Tensor]:
        """Edit the length of the feature segment based on the segment length parameter and add the `inpaint_feature` key to the input.

        Parameters
        ----------
        dict_features: dict[str, torch.Tensor]
            The input features to modify.
            Required keys: `binder_mask`, `target_mask`, `entity_id`, `asym_id`, `sym_id`, `pocket_feature`
            If the segment parameters do not specify a center of mass, then `atom_to_token` and `coords` are also required keys.
        segment_params: InpaintTokenIndexInfo
            The parameters for the inpaint segment.

        Returns
        -------
        new_segment_features: dict[str, torch.Tensor]
            The dictionary of features for the new inpaint segment.
            The value of `inpaint_feature` will be determined based on whether the region is binder or target, using binder_mask and target_mask.
            The length of the new segment will be `segment_params.feature_length`.
            The coords will be set to the center of mass provided in the segment parameters, or the computed center of mass of the segment.
            `original_residue_index` and `auth_residue_index` will be set to the first residue index of the segment for all residues in the segment.
            `residue_index` and `token_index` will be consecutive integers starting from 0.
            All other keys will inherit their values from the input (see Raises for special cases and errors).

        Raises
        ------
        ValueError: An invalid inpainting region triggers an error if:
        - `binder_mask` and `target_mask` are both `False` for any residue, indicating a malformed input or no residues in the specified segment
        - the segment corresponds to a region for which the binder_mask and target_mask are both True, indicating a malformed input
        - the segment contains multiple unique values for the following keys: `asym_id`, `entity_id`, `sym_id`, `pocket_feature`, `binder_mask`, `target_mask`
          EXCEPT for the case where the segment `inpaint_feature` is `TARGET_INPAINT`, in which case `pocket_feature` can have multiple unique values as long as the segment length does not change relative to the input features.
        """
        segment_feature_values: dict[str, int] = {}

        start_idx = segment_params.start_idx
        end_idx = segment_params.end_idx
        segment_length = segment_params.feature_length

        # determine the inpaint feature type
        binder_mask = dict_features["binder_mask"][start_idx:end_idx]
        target_mask = dict_features["target_mask"][start_idx:end_idx]
        if binder_mask.any() and target_mask.any():
            raise ValueError("Inpainting positions cannot be both binder and target")
        if not binder_mask.any() and not target_mask.any():
            raise ValueError("Inpainting positions must be either binder or target")
        if binder_mask.any():
            segment_feature_values["inpaint_feature"] = InpaintFeature.BINDER_INPAINT
        else:
            segment_feature_values["inpaint_feature"] = InpaintFeature.TARGET_INPAINT

        # determine other values for the segment
        for k in ["asym_id", "auth_asym_id", "entity_id", "sym_id", "pocket_feature", "binder_mask", "target_mask"]:
            if dict_features[k][start_idx:end_idx].unique().shape[0] > 1:
                # specifically check the TARGET_INPAINT case, in which case a POCKET could be specified if the segment_length doesn't change
                if (end_idx - start_idx == segment_length) and (
                    segment_feature_values["inpaint_feature"] == InpaintFeature.TARGET_INPAINT
                ):
                    set_pocket_feature_exactly = True
                else:
                    raise ValueError(f"Inpaint feature {k} has multiple unique values in the segment")
            else:
                segment_feature_values[k] = int(dict_features[k][start_idx].item())
                if k == "pocket_feature":
                    set_pocket_feature_exactly = False

        # obtain the coords center of mass
        # if the inpaint region center of mass is specified, use it
        # otherwise, calculate the center of mass from the atoms in the segment
        if segment_params.center_of_mass is not None:
            coords_center_of_mass = segment_params.center_of_mass
        else:
            atom_mask = (dict_features["atom_to_token"][:, start_idx:end_idx].sum(dim=1) > 0) & dict_features[
                "atom_resolved_mask"
            ].bool()
            coords_center_of_mass = dict_features["coords"][0][atom_mask].mean(dim=0)

        # determine the keys to include
        keys_to_include = list(dict_features.keys()) + ["inpaint_feature"]

        # make an empty chain with the appropriate length and features
        new_segment_features = make_empty_chain(
            seq_length=segment_length,
            atoms_to_include=self.atoms_to_include,
            res_type="UNK",
            asym_id=segment_feature_values["asym_id"],
            auth_asym_id=segment_feature_values["auth_asym_id"],
            entity_id=segment_feature_values["entity_id"],
            sym_id=segment_feature_values["sym_id"],
            pocket_feature=segment_feature_values["pocket_feature"],
            inpaint_feature=segment_feature_values["inpaint_feature"],
            coords_center_of_mass=coords_center_of_mass,
            keys_include=keys_to_include,
        )

        # set the original_residue_index and auth_residue_index to match the start
        if "original_residue_index" in new_segment_features.keys():
            start_original_residue_index = int(dict_features["original_residue_index"][start_idx].item())
            new_segment_features["original_residue_index"] = torch.full(
                (segment_length,), start_original_residue_index, dtype=torch.long
            )
        if "auth_residue_index" in new_segment_features:
            start_auth_residue_index = int(dict_features["auth_residue_index"][start_idx].item())
            new_segment_features["auth_residue_index"] = torch.full(
                (segment_length,), start_auth_residue_index, dtype=torch.long
            )

        # set the pocket feature exactly if it was specified and the segment length didn't change
        if set_pocket_feature_exactly:
            new_segment_features["pocket_feature"] = dict_features["pocket_feature"][start_idx:end_idx]

        return new_segment_features

    def _make_context_feature_segment(
        self, dict_features: dict[str, torch.Tensor], segment_params: InpaintTokenIndexInfo
    ) -> dict[str, torch.Tensor]:
        """Add the `inpaint_feature` key to a context segment.

        This segment may contain multiple chains and span both target and binder segments.

        Arguments
        ---------
        dict_features: dict[str, torch.Tensor]
            The dictionary of features.
            Required keys: `asym_id`, `binder_mask`, `target_mask`
        segment_params: InpaintTokenIndexInfo
            The parameters for the context segment.

        Returns
        -------
        context_features: dict[str, torch.Tensor]
            The dictionary of features cropped to the context segment.
            Each token in the newly added `inpaint_feature` will take either the value `InpaintFeature.BINDER_CONTEXT` or `InpaintFeature.TARGET_CONTEXT`.
        """
        # extract the start and end indices from the segment parameters
        # ignore all other attributes
        start_idx = segment_params.start_idx
        end_idx = segment_params.end_idx

        # determine a mask of tokens to include in the context segment
        region_mask = torch.zeros_like(dict_features["token_pad_mask"], dtype=torch.bool)
        region_mask[start_idx:end_idx] = True
        token_mask = dict_features["token_pad_mask"].bool() & region_mask

        # crop the features to the context segment
        context_features = crop_features(dict_features, token_mask=token_mask, reset_residue_index=False)

        # determine the inpaint feature for each token in the context segment
        context_inpaint_feature = torch.zeros_like(context_features["asym_id"], dtype=torch.long)
        context_inpaint_feature[context_features["binder_mask"].bool()] = InpaintFeature.BINDER_CONTEXT
        context_inpaint_feature[context_features["target_mask"].bool()] = InpaintFeature.TARGET_CONTEXT
        context_features["inpaint_feature"] = context_inpaint_feature.to(torch.int32)

        return context_features

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Add `inpaint_feature` and `original_residue_index` (if necessary) keys, edits feature length according to inpainting specification.

        See _make_inpaint_feature_segment for details on how inpainted segment features are constructed.
        See _make_context_feature_segment for details on how context segment features are constructed.
        """
        for input_features in inputs:
            new_features = input_features.safe_copy()
            # verify that the features are not padded
            if ~new_features.feature_dict["token_pad_mask"].any():
                raise ValueError("Features are padded, but inpainting requires unpadded features")

            # add an original_residue_index key to the features if it doesn't exist
            # this is used to track the original residue index of the input features, because we expect the sequence length to change
            if "original_residue_index" not in new_features.keys():
                new_features.feature_dict["original_residue_index"] = new_features.feature_dict["residue_index"].clone()

            # check that auth_residue_index is present in the features
            # if it is used to specify the hotspot positions
            if self.index_type == "auth_residue_index":
                if "auth_residue_index" not in new_features.keys():
                    raise ValueError("auth_residue_index is not present in the features")

            inpaint_regions = self._extract_inpaint_regions(new_features.feature_dict)

            # initialize a list to store the features to concatenate
            features_to_concat = []

            # start at the 0th token position and iterate to the right, constructing context feature segments and inpaint feature segments
            context_start_idx = 0
            for inpaint_segment_params in inpaint_regions:
                context_end_idx = inpaint_segment_params.start_idx  # left-inclusive, right-exclusive
                # if there is context to the left of the features, construct it
                if context_end_idx > context_start_idx:
                    context_segment_params = InpaintTokenIndexInfo(
                        start_idx=context_start_idx,
                        end_idx=context_end_idx,
                        feature_length=context_end_idx - context_start_idx,
                        center_of_mass=None,
                    )
                    context_features = self._make_context_feature_segment(
                        new_features.feature_dict, context_segment_params
                    )
                    features_to_concat.append(context_features)

                # add the inpaint feature
                inpaint_features = self._make_inpaint_feature_segment(new_features.feature_dict, inpaint_segment_params)
                features_to_concat.append(inpaint_features)

                # update the context start index
                context_start_idx = inpaint_segment_params.end_idx

            # add the final context segment if there are any tokens to the right of the inpainting regions
            if context_start_idx < len(new_features.feature_dict["asym_id"]):
                end_pos = len(new_features.feature_dict["asym_id"])
                context_segment_params = InpaintTokenIndexInfo(
                    start_idx=context_start_idx,
                    end_idx=end_pos,
                    feature_length=end_pos - context_start_idx,
                    center_of_mass=None,
                )
                context_features = self._make_context_feature_segment(new_features.feature_dict, context_segment_params)
                features_to_concat.append(context_features)

            # concatenate all the features to get the final output features
            # don't update the asym_id or entity_id, because we are explicitly constructing them to be the same as their parent chains
            # do reset the residue_index because the sequence length can change and we don't want duplicates or missing residues
            new_features.feature_dict = concat_features(
                features_to_concat, make_asym_ids_unique=False, make_entity_ids_unique=False, reset_residue_index=True
            )

            self._update_features_metadata(new_features)

            yield new_features
