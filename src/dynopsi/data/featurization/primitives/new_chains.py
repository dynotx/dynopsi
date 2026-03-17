import logging
from abc import abstractmethod
from collections.abc import Iterable, Iterator
from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from dynopsi.data.constants import AtomFilter, PocketFeature
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.featurization.utils import concat_features, make_empty_chain
from dynopsi.data.utils import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP

logger = logging.getLogger(__name__)


class NewChainsCentersOfMass(BaseModel):
    """Represents a function for returning the centers of mass for the new chains.

    Attribute
    ----------
    name: str = Field(default_factory=""), the name of the specific instance of the function. Used for
       metadata, logging, etc.

    NOTE: Subclasses must define a `type` class attribute as a `Literal["subclass_name"]`
    for Pydantic discriminated union serialization/deserialization.
    """

    name: str = Field(default_factory=str)

    @abstractmethod
    def check_num_chains_compatible(self, num_chains: int) -> bool:
        """Check if the number of chains is compatible with the function."""
        ...

    @abstractmethod
    def __call__(self, features: Features, new_chain_lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """Estimate the centers of mass for the new chains. Subclasses must implement this method."""
        ...


class BinderRgCenterOfMassEstimator(NewChainsCentersOfMass):
    """Estimates the center of mass of a binder chain based on the position of hotspots and surrounding target residues and the estimated binder radius of gyration.

    Attributes specific to this subclass parameterize a power law function to estimate the binder radius of gyration based on the binder length.

    Attributes
    ----------
     power_law_scale: float
     power_law_exponent: float
     power_law_offset: float
     power_law_min_val: float
     power_law_max_val: float
    """

    type: Literal["estimate_binder_center_of_mass_radius_of_gyration"] = (
        "estimate_binder_center_of_mass_radius_of_gyration"
    )
    name: str = "default_length_to_radius_of_gyration_function"

    power_law_scale: float = 2.11
    power_law_exponent: float = 0.37
    power_law_offset: float = 3.65
    power_law_min_val: float = 5
    power_law_max_val: float = 30

    def check_num_chains_compatible(self, num_chains: int) -> bool:
        """Check if the number of chains is compatible with the function."""
        return num_chains == 1

    def _estimate_distance_from_hotspot(self, new_chain_length: int) -> float:
        """Estimate the distance from the hotspot to the binder."""
        res = self.power_law_scale * new_chain_length**self.power_law_exponent + self.power_law_offset
        res = np.clip(res, self.power_law_min_val, self.power_law_max_val)
        return res

    def __call__(self, features: Features, new_chain_lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """
        Compute the center of mass of an idealized binder relative to the hotspot, based on the RFDiffusion 3 procedure.

        Steps:
        1. Extract CA coordinates
        2. Identify all target residues proximal to the hotspot. Proximal residues are defined as residues with a Ca-Ca distance less than 12 A to any hotspot, following RFDiffusion 3.
        3. Compute the mean position of the CA atoms of these target residues.
        4. Find the vector V from the hotspot-proximal target mean to the hotspot mean.
        5. Estimate the distance D (angstroms) from the hotspot to the binder. The goal is for this to be the average
           radius of gyration of the new chain, based on the number of residues in the new chain. We estimate this as a
           power law function of the number of residues in the new chain, with default parameters provided.
        5. Compute the center of mass of the binder as distance D on the vector V from the hotspot mean.

        Parameters
        ----------
        features: dict[str, Tensor]
            The features to compute the binder center of mass from the hotspot for. Must contain the following keys:
            - "coords": Tensor, shape [N, num_atoms, 3]: assumes Ca is the second atom for each residue.
            - "frame_resolved_mask": Tensor, shape  [N], True if the residue is resolved.
            - "pocket_feature": Tensor, shape [N], Defines both the target residues and the hotspot residues.
            - "inpaint_feature": Tensor, shape [N], Defines the binder and target residues.
        only_backbone_atoms: bool
            Whether the structure is a backbone only structure that contains N, CA, C atoms only (in that order).

        Returns
        -------
        binder_center_of_mass: tuple[torch.Tensor, ...]
            The center of mass of the binder relative to the hotspot, tuple of length 1.

        Raises
        ------
        ValueError: If the features don't have any hotspot residues or all pocket_feature residues are unspecified and
        inpaint_feature indicates that there are target residues.
        """
        if len(new_chain_lengths) != 1:
            raise ValueError("new_chain_lengths must be a list of length 1")

        # check for expected cases
        # pocket_feature unspecified
        if (features["pocket_feature"] == PocketFeature.UNSPECIFIED).all():
            raise ValueError("All of features['pocket_feature'] are UNSPECIFIED. This is not supported.")
        else:
            # check for no hotspot residues
            if not (features["pocket_feature"] == PocketFeature.POCKET).any():
                raise ValueError(
                    "No residues designated as `POCKET` in `features['pocket_feature']`. Cannot estimate the binder center of mass."
                )
            # check for no target residues other than pocket
            if not (features["pocket_feature"] == PocketFeature.UNSELECTED).any():
                raise ValueError(
                    "No residues designated as `UNSELECTED` in `features['pocket_feature']`. Cannot estimate the binder center of mass."
                )

        # ca atom position
        if features["frames_idx"].shape[1] == 3:
            coords_ca = features["coords"][0, 1::3, :]  # N x num_residues x 3
        elif features["frames_idx"].shape[1] == 1:
            coords_ca = features["coords"][0, :, :]  # N x num_residues x 3
        else:
            raise ValueError("CA position is ambiguous.")

        hotspot_mask = features["pocket_feature"] == PocketFeature.POCKET  # N x num_tokens
        target_non_hotspot_mask = features["pocket_feature"] == PocketFeature.UNSELECTED  # N x num_tokens
        resolved_mask = features["frame_resolved_mask"].bool()  # N x num_tokens

        coords_hotspot = coords_ca[hotspot_mask & resolved_mask]  # num_hotspot_residues x 3
        coords_target_non_hotspot = coords_ca[
            target_non_hotspot_mask & resolved_mask
        ]  # num_target_non_hotspot_residues x 3
        dist = torch.cdist(
            coords_hotspot, coords_target_non_hotspot
        )  # num_hotspot_residues x num_target_non_hotspot_residues

        ca_ca_interaction_dist = 12.0
        hotspot_proximal_mask = (dist < ca_ca_interaction_dist).any(dim=0)  # num_target_non_hotspot_residues
        target_center_of_mass = torch.cat([coords_target_non_hotspot[hotspot_proximal_mask], coords_hotspot]).mean(
            dim=0
        )  # 3

        hotspot_center_of_mass = coords_hotspot.mean(dim=0)  # 3

        # vector from target center of mass to hotspot center of mass
        vector = hotspot_center_of_mass - target_center_of_mass

        # get unit vector
        vector = vector / torch.norm(vector)

        # center of mass of the binder is D on the vector from the hotspot center of mass
        distance_from_hotspot = self._estimate_distance_from_hotspot(new_chain_lengths[0])
        binder_center_of_mass = hotspot_center_of_mass + distance_from_hotspot * vector

        return (binder_center_of_mass,)


class PredefinedCenterOfMassEstimator(NewChainsCentersOfMass):
    """
    Estimates the center of mass of the new chains at a predefined position.

    Attributes
    ----------
    name: str
    cfg_name: str
    max_new_chains: int | None

    Returns
    -------
    centers_of_mass: tuple[torch.Tensor, ...]
    The center of mass of the new chains at the origin, tuple of length len(new_chain_lengths).
    """

    type: Literal["predefined_center_of_mass"] = "predefined_center_of_mass"
    name: str = Field(default_factory=str)
    centers_of_mass: tuple[tuple[float, float, float], ...]

    def check_num_chains_compatible(self, num_chains: int) -> bool:
        """Check if the number of chains is compatible with the function."""
        return num_chains == len(self.centers_of_mass)

    def __call__(self, features: Features, new_chain_lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        """Estimate the center of mass of the new chains at the predefined positions."""
        if len(new_chain_lengths) != len(self.centers_of_mass):
            raise ValueError("new_chain_lengths must match the number of centers of mass")
        return tuple(torch.tensor(center_of_mass) for center_of_mass in self.centers_of_mass)


# Discriminative union for NewChainsCentersOfMassFunctionType
# Allows for automatic serialization/deserialization of the function.
NewChainsCentersOfMassFunctionType = Annotated[
    BinderRgCenterOfMassEstimator | PredefinedCenterOfMassEstimator, Field(discriminator="type")
]


@register_featurizer()
class NewChainsFeaturizer(Featurizer):
    """Adds new chains to the input features.

    The new chains will be denoised during sampling.

    Attributes
    ----------
    type: Literal["new_chains"]
        For serialization/deserialization.
    lengths: tuple[int, ...]
        Number of residues in each of the new chains.
    entity_ids: tuple[int, ...] = Field(default_factory=tuple)
        The entity ids of the new chains.
        The default is for every new chain to have a unique entity id. If entity_ids is provided, chains with the same entity id must have the same length.
    atoms_to_include: AtomFilter | None = None
        Atoms to include in the new chains.
    estimate_centers_of_mass_function: NewChainsCentersOfMassFunctionType
        Function to estimate the centers of mass for the new chains.
    """

    type: Literal["new_chains"] = "new_chains"
    lengths: tuple[int, ...]
    entity_ids: tuple[int, ...] = Field(default_factory=tuple)
    atoms_to_include: AtomFilter | None = None
    estimate_centers_of_mass_function: NewChainsCentersOfMassFunctionType

    @model_validator(mode="after")
    def set_or_validate_entity_ids_and_centers_of_mass(self) -> Self:
        if not self.entity_ids:
            # generate the default entity_ids
            self.entity_ids = tuple(range(len(self.lengths)))
        else:
            if len(self.entity_ids) != len(self.lengths):
                raise ValueError("Length of `entity_ids` must match the length of `lengths`.")
            unique_entity_ids = set(self.entity_ids)
            for entity_id in unique_entity_ids:
                lengths = [self.lengths[i] for i in range(len(self.lengths)) if self.entity_ids[i] == entity_id]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Entity id {entity_id} has multiple lengths: {lengths}")

        # make sure that the function is compatible with the number of new chains
        if not self.estimate_centers_of_mass_function.check_num_chains_compatible(len(self.lengths)):
            raise ValueError(
                f"The number of new chains to add is {len(self.lengths)}, but the function {self.estimate_centers_of_mass_function.name} is not compatible with this number of chains."
            )

        return self

    @property
    def sym_ids(self) -> tuple[int, ...]:
        """Return a tuple of sym_ids to use for the new chains."""
        if self.entity_ids is not None:
            sym_ids_lst = []
            entities_counter = {entity_id: 0 for entity_id in self.entity_ids}
            for entity_id in self.entity_ids:
                sym_ids_lst.append(entities_counter[entity_id])
                entities_counter[entity_id] += 1
            return tuple(sym_ids_lst)
        else:
            return tuple(0 for _ in range(len(self.lengths)))

    def get_new_asym_ids(self, num_new_chains: int, asym_ids_in_use: set[int]) -> tuple[int, ...]:
        """Get new asym_ids to use for the new chains."""
        valid_asym_ids = [x for x in ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP.keys() if x not in asym_ids_in_use]
        if len(valid_asym_ids) < num_new_chains:
            raise ValueError(
                f"Not enough valid asym ids to use for the new chains. Need {num_new_chains}, but only {len(valid_asym_ids)} are available."
            )
        return tuple(valid_asym_ids[:num_new_chains])

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Concatenate new chains to the input.

        The new chains will have the same features as the input, and the concatenation is on the sequence length dimension.
        See data.featurization.utils.make_empty_chain for more details on the possible keys that can be added.

        The pocket feature will be determined based on the pocket feature of the input features.
        - If input features are all UNSPECIFIED, the pocket feature of the new chains will be UNSPECIFIED.
        - If input features contain any UNSELECTED or POCKET residues, the pocket feature of the new chains will be BINDER.
        - If input features are empty, the pocket feature of the new chains will be UNSPECIFIED.
        """
        for input_features in inputs:
            new_feats = input_features.safe_copy()
            new_feats_keys = new_feats.feature_dict.keys()

            # if atoms_to_include is not provided, infer from the input features
            if self.atoms_to_include is None:
                if new_feats.atoms_to_include is not None:
                    atoms_to_include = new_feats.atoms_to_include
                else:
                    raise ValueError("atoms_to_include is not provided and cannot be guessed from the input features")
            # if atoms_to_include is provided, check that it matches the input features
            else:
                if new_feats.atoms_to_include is not None and new_feats.atoms_to_include != self.atoms_to_include:
                    raise ValueError(
                        "atoms_to_include is provided in the input features and does not match the atoms_to_include provided to the featurizer"
                    )
                else:
                    atoms_to_include = self.atoms_to_include

            # compute the centers of mass for the new chains
            centers_of_mass = self.estimate_centers_of_mass_function(new_feats, self.lengths)
            logger.info("New chains centers of mass: %s", centers_of_mass)

            # check the pocket feature values
            # if they exist and are all UNSPECIFIED, set the pocket feature to UNSPECIFIED
            # otherwise, set the pocket feature to BINDER
            if "pocket_feature" in new_feats_keys:
                pocket_feature = new_feats["pocket_feature"][new_feats["token_pad_mask"].bool()]
                if (pocket_feature == PocketFeature.UNSPECIFIED).all():
                    pocket_feature = PocketFeature.UNSPECIFIED
                else:
                    pocket_feature = PocketFeature.BINDER
            else:
                pocket_feature = PocketFeature.UNSPECIFIED

            new_chains = []

            # determine the new asym_ids to use for the new chains to avoid collisions with existing asym_ids
            # these are defined by (1) the pdb chain IDs and (2) the author chain IDs
            # for simplicity, the new chains will have the same asym_id and auth_asym_id
            asym_ids_in_use = set(
                new_feats.feature_dict["asym_id"].unique().tolist()
                + new_feats.feature_dict["auth_asym_id"].unique().tolist()
            )
            new_asym_ids = self.get_new_asym_ids(num_new_chains=len(self.lengths), asym_ids_in_use=asym_ids_in_use)

            for new_length, asym_id, sym_id, entity_id, com in zip(
                self.lengths, new_asym_ids, self.sym_ids, self.entity_ids, centers_of_mass
            ):
                new_chains.append(
                    make_empty_chain(
                        seq_length=new_length,
                        atoms_to_include=atoms_to_include,
                        coords_center_of_mass=com,
                        res_type="UNK",
                        asym_id=asym_id,
                        auth_asym_id=asym_id,
                        sym_id=sym_id,
                        entity_id=entity_id,
                        keys_include=list(new_feats_keys),
                        pocket_feature=pocket_feature,
                    )
                )

            concat_new_chains = concat_features(new_chains, make_asym_ids_unique=True, make_entity_ids_unique=False)

            if len(new_feats.keys()) == 0:
                new_feats.feature_dict = concat_new_chains
            else:
                new_feats.feature_dict = concat_features(
                    [new_feats.feature_dict, concat_new_chains], make_asym_ids_unique=True, make_entity_ids_unique=True
                )

            self._update_features_metadata(new_feats)

            yield new_feats
