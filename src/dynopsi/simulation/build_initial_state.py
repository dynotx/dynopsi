import logging
from abc import abstractmethod
from typing import Annotated, Literal

import torch
from pydantic import Field

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.sample_centers import (
    apply_random_global_translation,
    expand_features_to_batch,
)
from dynopsi.data.utils import reshape_coords_for_model
from dynopsi.model.model_config import DynoPsiModelConfig
from dynopsi.simulation.store import Store
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)


class InitializeStep(StrictConfig):
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

    def input_state_keys(self) -> set[str]:
        return set()

    def output_state_keys(self, step_num: int) -> set[str]:
        return set()

    def intermediate_observable_keys(self) -> set[str]:
        """
        Intermediate state keys that this op writes, step agnostic.

        This method provides a way to output a standard intermediate keys for
        observation and debugging without prior knowledge of the intermediate state keys,
        and makes it so that the user doesn't have to list all these keys in the observable output config.

        Returns
        -------
        set[str]
            Store keys for intermediate state(s) written by this step. Empty by default.
        """
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
        batch_slice = store.batch_slice
        if batch_slice is None:
            raise ValueError("InitializeFeatures requires store.batch_slice")
        batch_size = batch_slice.stop - batch_slice.start
        features = expand_features_to_batch(features, batch_size)

        apply_random_global_translation(features)

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
    seed: int | None = None

    def model_configs(self) -> list[DynoPsiModelConfig]:
        return [self.inference_model_config]

    def input_feature_keys(self) -> set[str]:
        return {self.in_feat_loc_store}

    def output_state_keys(self, step_num: int) -> set[str]:
        if step_num == 0:
            return {self.out_loc_store}
        return set()

    def run(self, store: Store) -> None:
        """Run the steps to initialize the first state for the simulation.

        Steps:
        1. Retrieve the features from the store.
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

        generator = None
        if self.seed is not None:
            sample_index = store.prior_sample_index()
            generator = torch.Generator(device=unmasked_data_coords.device)
            generator.manual_seed(self.seed + sample_index)
            logger.debug(
                "sampling prior with seed=%s sample_index=%s batch_size=%s",
                self.seed + sample_index,
                sample_index,
                batch_size,
            )

        # only the coordinate modality is supported currently
        prior_sample = (
            priors["coords"]
            .sample(
                batch_size,
                prior_mean=prior_mean_coords,
                unmasked=unmasked_data_coords,
                mask=features.feature_dict["noised_mask"].unsqueeze(-1),
                generator=generator,
            )
            .to(unmasked_data_coords.device)
        )

        start_time = store.get_time(step_num=0, modality="coords", batch_size=batch_size)
        interpolant = interpolants["coords"].interpolate(prior_sample, unmasked_data_coords, start_time)

        store.put(self.out_loc_store, {"coords": interpolant}, step_num=0)


InitializeStepType = Annotated[InitializeFeatures | InitializeState, Field(discriminator="type")]
