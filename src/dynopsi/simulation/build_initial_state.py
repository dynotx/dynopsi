import logging
from abc import abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field

from dynopsi.data.constants import AtomFilter, PocketFeature
from dynopsi.data.utils import reshape_coords_for_model
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.store import Store

logger = logging.getLogger(__name__)


class InitializeStep(BaseModel):
    """Base class for an inference initialization step.

    Initialization steps take only features from the store as input and not
    any states.

    Subclasses must define a `type` field as a `Literal["subclass_type"]` for
    serialization/deserialization.

    Attributes
    ----------
    name : str
        Identifier for this step.
    """

    name: str

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return []

    def input_feature_keys(self) -> set[str]:
        return set()

    def output_feature_keys(self) -> set[str]:
        return set()

    def output_state_keys(self, step_num: int) -> set[str]:
        return set()

    @abstractmethod
    def run(self, store: Store) -> None: ...


class InitializeFeatures(InitializeStep):
    """Initialize features for simulation.

    Reads features from the store, adds the batch dimension and
    renames keys and reshapes coordinates for the model.
    Writes updated features to the store at `out_feat_loc_store`.

    Attributes
    ----------
    name : str
        Identifier for this step.
    in_feat_loc_store : str
        Store key to read input features from.
    out_feat_loc_store : str
        Store key to write prepared features to.
    """

    type: Literal["initialize_features"] = "initialize_features"

    name: str
    in_feat_loc_store: str
    out_feat_loc_store: str

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def output_feature_keys(self) -> set[str]:
        return {self.out_feat_loc_store}

    def run(self, store: Store) -> None:
        # get the features
        features = store.get_features(self.in_feat_loc_store).safe_copy()

        # expand to batch size
        batch_size = store.batch_slice.stop - store.batch_slice.start
        for key, value in features.feature_dict.items():
            if isinstance(value, torch.Tensor):
                features.feature_dict[key] = torch.stack([value] * batch_size)

        # model needs some specific keys that renamed from features
        features.feature_dict["resolved_mask"] = features.feature_dict["frame_resolved_mask"]
        features.feature_dict.pop("frame_resolved_mask")
        features.feature_dict["indices"] = features.feature_dict["token_index"]
        features.feature_dict.pop("token_index")

        assert isinstance(features.atoms_to_include, AtomFilter), (
            "Features must have atoms_to_include set as an AtomFilter."
        )

        reshaped_backbone_coords = reshape_coords_for_model(
            features.feature_dict["coords"], features.feature_dict["frames_idx"], features.atoms_to_include
        )  # shape: (batch_size, seq_len, 3, 3)

        seq_len = features.feature_dict["frames_idx"].shape[1]
        features.feature_dict["true_coords"] = reshaped_backbone_coords.reshape(
            batch_size, seq_len, -1, 3
        )  # shape: (batch_size, seq_len, 3, 3)

        store.put_features(self.out_feat_loc_store, features)


class InitializeUnconditionalPocketFeatures(InitializeStep):
    """Initialize the features for the simulation for unconditional pocket sampling.

    Specifically, this step reads the features from the store,
    sets the hotspots to unselected in the target, and writes the updated features to a new store key.

    Attributes
    ----------
    name : str
        Name of the operation.
    in_feat_loc_store : str
        Store key for input features.
    out_feat_loc_store : str
        Store key for output features, with the pocket set to unselected.
    """

    type: Literal["initialize_unconditional_pocket_features"] = "initialize_unconditional_pocket_features"

    name: str
    in_feat_loc_store: str
    out_feat_loc_store: str

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def output_feature_keys(self) -> set[str]:
        return {self.out_feat_loc_store}

    def run(self, store: Store) -> None:
        # get the features
        features = store.get_features(self.in_feat_loc_store).safe_copy()

        # set hotspots to unspecified for all residues
        pocket_feature = features.feature_dict["pocket_feature"]
        n_hotspots = (pocket_feature == PocketFeature.POCKET).sum().item()
        features.feature_dict["pocket_feature"] = torch.full_like(pocket_feature, PocketFeature.UNSPECIFIED)
        logger.info(f"Removing {n_hotspots} hotspot residues in unconditional features.")

        store.put_features(self.out_feat_loc_store, features)


class InitializeState(InitializeStep):
    """Initialize the first state (interpolant at t=0) for the simulation.

    Loads features from the store, samples from priors, and
    computes the initial interpolant state. Writes the state to
    the store at `out_loc_store`.

    Attributes
    ----------
    name : str
        Identifier for this step.
    inference_model_config : DynoPsiModelConfig
        Model config providing priors and interpolants.
    in_feat_loc_store : str
        Store key for input features.
    out_loc_store : str
        Store key to write the initial state to.
    """

    type: Literal["initialize_state"] = "initialize_state"

    name: str
    inference_model_config: DynoPsiModelConfig
    in_feat_loc_store: str
    out_loc_store: str

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return [self.inference_model_config]

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def run(self, store: Store) -> None:
        """Run the steps to initialize the first state for the simulation.

        Steps:
        1. Feature pre-processing:
            - Pad the features to the max_tokens required by the flow model.
            - Expand the features to the batch size (all elements of the batch will have the same immutable features).
            - Process the features as specified by the model.
            - Put the processed features in the store (in-place)
        2. Prior sampling:
            - Sample priors using the prior Sampler defined by the flow model
        3. Initial state computation:
            - Compute the initial state using the Interpolant defined by the flow model
        4. Update the store with the initial state.
        """
        # get the features
        features = store.get_features(self.in_feat_loc_store)
        batch_size = store.batch_slice.stop - store.batch_slice.start

        # sample prior and compute the interpolant
        priors = self.inference_model_config.dynopsi_model.priors
        interpolants = self.inference_model_config.dynopsi_model.interpolants
        assert isinstance(features.atoms_to_include, AtomFilter), (
            "Features must have atoms_to_include set as an AtomFilter."
        )
        prior_mean_coords = reshape_coords_for_model(
            features.feature_dict["prior_mean"], features.feature_dict["frames_idx"], features.atoms_to_include
        )
        unmasked_data_coords = reshape_coords_for_model(
            features.feature_dict["coords"], features.feature_dict["frames_idx"], features.atoms_to_include
        )

        # only the coordinate modality is supported currently
        prior_sample = (
            priors["coords"]
            .sample(
                batch_size,
                prior_mean=prior_mean_coords,
                unmasked=unmasked_data_coords,
                mask=features.feature_dict["noised_mask"].unsqueeze(-1),
            )
            .to(unmasked_data_coords.device)
        )

        start_time = store.get_time(step_num=0, modality="coords", batch_size=batch_size)
        interpolant = interpolants["coords"].interpolate(prior_sample, unmasked_data_coords, start_time)

        store.put(self.out_loc_store, {"coords": interpolant}, step_num=0)


InitializeStepType = Annotated[
    InitializeFeatures | InitializeUnconditionalPocketFeatures | InitializeState, Field(discriminator="type")
]
