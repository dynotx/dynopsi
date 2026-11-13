from collections.abc import Iterable, Iterator
from typing import Literal, Self

import torch
from pydantic import Field, model_validator

from dynopsi.data.constants import InpaintFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.featurization.utils import map_residue_to_atom_mask
from dynopsi.data.utils import center_coordinates, randomly_rotate
from dynopsi.utils.function_configs import SingleParameterFunctionType
from dynopsi.utils.strict_config import StrictConfig


class DataAugmentationConfig(StrictConfig):
    """Configuration of parameters for data augmentation.

     Supports the following data augmentation options:
     - strategy for centering the frame of reference
     - random global translation
     - random global rotation

    Parameters
    ----------
    name : str, optional
        Identifier for this config.
    centering_strategy : {'design_chains', 'unnoised_residues'}
        How to define the frame of reference:
        - 'design_chains': Center on all residues of any chains
          that include designed (i.e. noised) residues.
          For example, for the binder design task this is equivalent to
          centering on binder residues.
        - 'unnoised_residues': Center on all unnoised resolved residues.
          For example, for the target docking task this is equivalent to
          centering on binder residues.
    random_global_translation_std : float, default=0.0
        Standard deviation of random global translation in Angstroms.
        Must be non-negative.
    random_global_rotation : bool, default=False
        Whether to apply random global rotation.

    Raises
    ------
    ValueError
        If `random_global_translation_std` is negative.
    ValueError
        If `centering_strategy` is not 'design_chains' or 'unnoised_residues'.
    """

    name: str = Field(default_factory=str)
    centering_strategy: Literal["design_chains", "unnoised_residues"]
    random_global_translation_std: float = 0.0
    random_global_rotation: bool = False

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.random_global_translation_std < 0:
            raise ValueError("random_global_translation_std must be non-negative")
        if self.centering_strategy not in ["design_chains", "unnoised_residues"]:
            raise ValueError("centering_strategy must be 'design_chains' or 'unnoised_residues'")
        return self


class PriorMeansConfig(StrictConfig):
    """Configuration of parameters for prior means (centering and offset).

    Parameters
    ----------
    name : str, optional
        Identifier for this config. Defaults to empty string.
    prior_centering_strategy : {'origin', 'chain_and_noise_separated'}
        How to center the prior means:
        - 'origin': Center prior on the origin.
        - 'chain_and_noise_separated' : Center prior for each chain
          separately using center of mass of noised residues.
    prior_mean_offset_function : SingleParameterFunctionType
        Config defining the function to use for the standard deviation
        (in Angstroms) of random global translation offset for
        prior means.

    See Also
    --------
    ConstantFunction : Constant offset standard deviation.
    PowerLawFunction : Power-law-based offset standard deviation.
    dynopsi.utils.function_configs : Module defining prior mean offset functions
        and their configuration keywords.
    """

    name: str = Field(default_factory=str)
    prior_centering_strategy: Literal["origin", "chain_and_noise_separated"]

    prior_mean_offset_function: SingleParameterFunctionType


@register_featurizer()
class CenterOfMassFeaturizer(Featurizer):
    """Featurizer that centers coordinates, applies data augmentation, and computes prior means.

    Centers coordinates and applies data augmentation
    (optional global translation and rotation)  and computes per-chain
    prior means, each according to configurable strategies
    (`data_augmentation_config` and `prior_means_config`, respectively).

    Parameters
    ----------
    data_augmentation_config : DataAugmentationConfig
        Config for centering strategy and random translation/rotation.
    prior_means_config : PriorMeansConfig
        Config for how prior means are centered and randomly offset.

    See Also
    --------
    DataAugmentationConfig : Centering and data augmentation parameters.
    PriorMeansConfig : Prior centering and offset parameters.
    """

    type: Literal["center_of_mass"] = "center_of_mass"

    data_augmentation_config: DataAugmentationConfig
    prior_means_config: PriorMeansConfig

    def _center_coordinates(self, feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Center the coordinates of the feature.

        Arguments:
        ----------
        feature_dict: dict[str, torch.Tensor], the feature dictionary. Must contain the following keys:
        - asym_id: torch.Tensor, the asym id, shape (num_tokens)
        - atom_pad_mask: torch.Tensor, the atom pad mask, shape (num_atoms)
        - atom_resolved_mask: torch.Tensor, the atom resolved mask, shape (num_atoms)
        - atom_to_token: torch.Tensor, the atom to token mapping, shape (num_atoms, num_tokens)
        - coords: torch.Tensor, the coordinates, shape (1, num_atoms, 3)
        - frame_resolved_mask: torch.Tensor, the frame resolved mask, shape (num_tokens)
        - inpaint_feature: torch.Tensor, the inpaint feature, shape (num_tokens)
        - token_pad_mask: torch.Tensor, the token pad mask, shape (num_tokens)

        Returns
        -------
            torch.Tensor: The centered coordinates, shape (1, num_atoms, 3)
            - If the data_augmentation_config centering_strategy is "fixed": coordinates are centered on
              all resolved target residues (TARGET_INPAINT, TARGET_CONTEXT) and all resolved binder context residues (BINDER_CONTEXT).
            - If the data_augmentation_config centering_strategy is "design_chains": coordinates are centered on
              all resolved residues of any chains that contain designed residues (TARGET_INPAINT, BINDER_INPAINT).

        Raises
        ------
        ValueError: If the data_augmentation_config centering_strategy is invalid.
        ValueError: If there are no resolved atoms to center on.
        """
        if self.data_augmentation_config.centering_strategy == "design_chains":
            # look for all residues of any chains being designed/inpainted
            designed_mask = (
                torch.isin(
                    feature_dict["inpaint_feature"],
                    torch.tensor([InpaintFeature.TARGET_INPAINT, InpaintFeature.BINDER_INPAINT]),
                )
                & feature_dict["frame_resolved_mask"].bool()
                & feature_dict["token_pad_mask"].bool()
            )

            if not designed_mask.any():
                # If there are no chains being designed (i.e. docking) center all residues' resolved atoms.
                residue_center_mask = feature_dict["frame_resolved_mask"].bool() & feature_dict["token_pad_mask"].bool()
            else:
                # center on the chains being designed
                residue_center_mask = torch.isin(feature_dict["asym_id"], feature_dict["asym_id"][designed_mask])
        elif self.data_augmentation_config.centering_strategy == "unnoised_residues":
            # center on all unnoised resolved residues
            if feature_dict["noised_mask"].bool().all():
                raise ValueError("No unnoised residues to center on")
            residue_center_mask = (
                ~feature_dict["noised_mask"].bool()
                & feature_dict["frame_resolved_mask"].bool()
                & feature_dict["token_pad_mask"].bool()
            )
        else:
            raise ValueError(f"Invalid centering strategy: {self.data_augmentation_config.centering_strategy}")

        # make sure that token padding is applied
        residue_center_mask = residue_center_mask & feature_dict["token_pad_mask"].bool()

        atom_mask = map_residue_to_atom_mask(
            residue_mask=residue_center_mask,
            atom_to_token=feature_dict["atom_to_token"],
            atom_mask=feature_dict["atom_resolved_mask"].bool() & feature_dict["atom_pad_mask"].bool(),
        )

        # if no atoms to center on, raise an error
        if not atom_mask.any():
            raise ValueError("No atoms to center on")

        # center the coordinates
        centered_coords = center_coordinates(feature_dict["coords"], atom_mask)
        return centered_coords

    def _augment_coordinates(self, feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply random global translation and rotation to coordinates.

        Parameters
        ----------
        feature_dict : dict[str, torch.Tensor]
            Feature dictionary containing `coords` of shape `(1, num_atoms, 3)`.

        Returns
        -------
        torch.Tensor
            Augmented coordinates, shape `(1, num_atoms, 3)`.
            Random global translation and rotation are defined in `data_augmentation_config`.
        """
        coords = feature_dict["coords"]  # shape (bs, num_atoms, 3)
        # 1. Translate
        bs = coords.shape[0]  # batch size
        rand_tsl = (
            torch.randn(bs, 1, 3) * self.data_augmentation_config.random_global_translation_std
        )  # shape (bs, 1, 3)
        coords += rand_tsl  # shape (bs, num_atoms, 3)
        # 2. Rotate
        if self.data_augmentation_config.random_global_rotation:
            coords = randomly_rotate(coords)
        return coords

    def _compute_prior_means(self, feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the prior means for the features.

            Per-position prior means are computed in two steps:
            1. Centering: Center of mass is computed for each chain, according to `prior_means_config.prior_centering_strategy`:
               - 'origin': center at origin (data-independent).
               - 'chain_and_noise_separated': center at center of mass of the chain's noised,
                 resolved atoms.
            2. Offset: Random global translation (i.e. offset) is added to center of masses with standard deviation according to
               `prior_means_config.prior_mean_offset_function` (see `dynopsi.utils.function_configs`).

        Parameters
        ----------
        feature_dict : dict[str, torch.Tensor]
            Feature dictionary. Must contain the following keys:
            - `coords`: 3D atom coordinates, shape (1, num_atoms, 3)
            - `asym_id`: per-residue asymmetric unit IDs, shape (num_tokens,)
            - `atom_pad_mask`:  atom-level boolean padding mask, shape (num_atoms,)
            - `atom_resolved_mask`: atom-level boolean resolved mask, shape (num_atoms,)
            - `atom_to_token`: one-hot mapping between atoms and residues, shape (num_atoms, num_tokens)
            - `noised_mask`: per-residue boolean noised mask, shape (num_tokens,)
            - `token_pad_mask`: per-residue boolean padding mask, (num_tokens,)

        Returns
        -------
        torch.Tensor
            Prior means for the input features, shape (1, num_atoms, 3).

        Raises
        ------
        ValueError
            If `prior_means_config.prior_centering_strategy` is not
            'origin' or 'chain_and_noise_separated'.
        ValueError
            If centering strategy is 'chain_and_noise_separated' and the
            chain's noised residues contain no resolved atoms.
        ValueError
            If the prior mean offset function returns a negative standard deviation.
        """
        prior_means = torch.zeros_like(feature_dict["coords"])
        asym_ids = feature_dict["asym_id"][feature_dict["token_pad_mask"].bool()].unique()
        for asym_id in asym_ids:
            if self.prior_means_config.prior_centering_strategy == "chain_and_noise_separated":
                # only consider noised residues
                residue_mask = (
                    (feature_dict["asym_id"] == asym_id)
                    & feature_dict["noised_mask"].bool()
                    & feature_dict["token_pad_mask"].bool()
                )
            elif self.prior_means_config.prior_centering_strategy == "origin":
                # just specify the residue mask
                residue_mask = (feature_dict["asym_id"] == asym_id) & feature_dict["token_pad_mask"].bool()
            else:
                raise ValueError(
                    f"Invalid prior centering strategy: {self.prior_means_config.prior_centering_strategy}"
                )

            # if residue_mask is all False, skip
            # this is relevant for target chains (i.e. no noised residues)
            if not residue_mask.any():
                continue

            # map residue mask to atom mask
            atom_to_set_mask = map_residue_to_atom_mask(
                residue_mask=residue_mask,
                atom_to_token=feature_dict["atom_to_token"],
                atom_mask=feature_dict["atom_pad_mask"].bool(),
            )
            if self.prior_means_config.prior_centering_strategy == "chain_and_noise_separated":
                atom_centering_mask = atom_to_set_mask & feature_dict["atom_resolved_mask"].bool()
                if atom_centering_mask.any():
                    center_of_mass = feature_dict["coords"][0, atom_centering_mask, :].mean(dim=0)
                else:
                    raise ValueError("No resolved atoms to compute prior means")

            elif self.prior_means_config.prior_centering_strategy == "origin":
                center_of_mass = torch.zeros(3)
            else:
                raise ValueError(
                    f"Invalid prior centering strategy: {self.prior_means_config.prior_centering_strategy}"
                )

            # apply the prior mean offset
            prior_mean_offset_function = self.prior_means_config.prior_mean_offset_function
            offset_std = prior_mean_offset_function(feature_dict["noised_mask"][residue_mask].sum())
            if offset_std < 0:
                raise ValueError(f"Offset standard deviation is negative: {offset_std}")
            # add the offset to the prior means
            prior_mean_center_of_mass = center_of_mass + torch.randn(3) * offset_std
            # set the prior means
            prior_means[0, atom_to_set_mask, :] = prior_mean_center_of_mass

        return prior_means

    def __call__(self, features: Iterable[Features]) -> Iterator[Features]:
        """Center and augment coordinates, compute prior means, and optionally update reference structure.

        For each input, applies the following steps:
        1. center coordinates,
        2. augment coordinates,
        3. compute prior means,
        4. update features metadata to track provenance, and then
        5. optionally update the input reference structure to align with the new coordinates from 1-2.

        Parameters
        ----------
        features : Iterable[Features]
            Features to process.

        Yields
        ------
        Features
            Copy of each input Features with:
            - `coords` replaced with centered and augmented coordinates
            - `prior_mean` added with computed prior means
            - `input_reference_structure` updated by alignment with updated `coords`, when present

        Raises
        ------
        ValueError
            If a feature has `input_reference_structure` set but
            `atoms_to_include` is None (alignment requires defined atoms).

        See Also
        --------
        _center_coordinates : Centering details.
        _augment_coordinates : Global translation/rotation details.
        _compute_prior_means : Prior means computation details.
        """
        for feature in features:
            new_features = feature.safe_copy()
            # center the coordinates
            new_features["coords"] = self._center_coordinates(new_features.feature_dict)
            # augment the coordinates
            new_features["coords"] = self._augment_coordinates(new_features.feature_dict)
            # compute the prior means
            new_features["prior_mean"] = self._compute_prior_means(new_features.feature_dict)

            more_metadata = {}
            # update the ref structure by aligning on coords
            if new_features.input_reference_structure is not None:
                if new_features.atoms_to_include is None:
                    raise ValueError("Cannot align structure to features without atoms defined for features.")
                structure = new_features.input_reference_structure.align_to_feature_dict(
                    feature_dict=new_features.feature_dict, atoms_to_include=new_features.atoms_to_include
                )
                new_features.input_reference_structure = structure
                if self.verbose_metadata:
                    more_metadata["structure"] = structure

            # update the metadata
            self._update_features_metadata(new_features, more_metadata=more_metadata)

            yield new_features
