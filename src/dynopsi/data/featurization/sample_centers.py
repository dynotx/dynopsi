from __future__ import annotations

import torch

from dynopsi.data.featurization.base import Features

SAMPLE_CENTERS_OF_MASS_METADATA_KEY = "sample_centers_of_mass"
PER_SAMPLE_FEATURES_METADATA_KEY = "per_sample_center_features"
RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY = "random_global_translation_std"
_COM_MATCH_ATOL = 1e-5


def centers_match(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return all(abs(a - b) <= _COM_MATCH_ATOL for a, b in zip(left, right, strict=True))


def draw_prior_com_offsets(num_samples: int, num_prior_coms: int, std_angstrom: float, *, seed: int) -> torch.Tensor:
    """Draw per-sample, per-prior-COM offsets ``N(0, std_angstrom**2 I)`` of shape ``(num_samples, K, 3)``.

    ``K`` is ``num_prior_coms``: the number of prior COMs on the featurization product (the
    predefined binder centers, or the noised chains under ``chain_and_noise_separated``). Each
    ``(sample, prior_com)`` pair draws an independent offset.

    Uses a CPU ``torch.Generator`` reseeded with ``seed`` on every call, so the same ``seed``,
    ``num_samples``, and ``num_prior_coms`` always return the same offsets. This is the only draw of
    the prior-COM cloud; it is not seeded from ``prior_seed`` or a batch slice.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(num_samples, num_prior_coms, 3, generator=generator) * std_angstrom


def reference_features_for_sample(features: Features, sample_index: int) -> Features:
    """Return the featurization for this sample, or features when there is no per-sample list."""
    per_sample = features.featurization_metadata.get(PER_SAMPLE_FEATURES_METADATA_KEY)
    if per_sample is None:
        return features
    return per_sample[sample_index]


def narrow_per_sample_features(features: Features, sample_slice: slice) -> Features:
    """Return a shallow Features copy whose per-sample metadata is limited to sample_slice.

    Does not clone tensors or mutate the input. When no per-sample list is present, returns
    features unchanged.
    """
    per_sample = features.featurization_metadata.get(PER_SAMPLE_FEATURES_METADATA_KEY)
    if per_sample is None:
        return features
    out = features.model_copy(deep=False)
    metadata = dict(features.featurization_metadata)
    metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = per_sample[sample_slice]
    centers = metadata.get(SAMPLE_CENTERS_OF_MASS_METADATA_KEY)
    if centers is not None:
        metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = centers[sample_slice]
    out.featurization_metadata = metadata
    return out


def collate_sample_center_features(batch: list[Features]) -> Features:
    """Stack independently featurized samples along a leading batch axis.

    Each item must already be centered on its own binder COM.
    Tensor keys are stacked; non-tensor keys are taken from the first item.
    """
    out = batch[0].safe_copy()
    stacked: dict[str, torch.Tensor] = {}
    for key, value in batch[0].feature_dict.items():
        if isinstance(value, torch.Tensor):
            stacked[key] = torch.stack([item.feature_dict[key] for item in batch])
        else:
            stacked[key] = value
    out.feature_dict = stacked
    out.featurization_metadata.pop(PER_SAMPLE_FEATURES_METADATA_KEY, None)
    return out


def move_features_to_device(features: Features, device: torch.device) -> None:
    features.to_device(device)
    per_sample = features.featurization_metadata.get(PER_SAMPLE_FEATURES_METADATA_KEY)
    if per_sample is None:
        return
    for item in per_sample:
        item.to_device(device)


def expand_features_to_batch(features: Features, batch_size: int) -> Features:
    # Features already carry the batch-window list (narrowed by run_batch); collate the whole list.
    sample_centers = features.featurization_metadata.get(SAMPLE_CENTERS_OF_MASS_METADATA_KEY)
    per_sample = features.featurization_metadata.get(PER_SAMPLE_FEATURES_METADATA_KEY)
    if sample_centers is not None and per_sample is None:
        raise ValueError(
            f"sample_centers_of_mass requires independently featurized copies in {PER_SAMPLE_FEATURES_METADATA_KEY}"
        )
    if per_sample is not None:
        if len(per_sample) != batch_size:
            raise ValueError(f"per-sample feature copies have {len(per_sample)} rows but batch_size is {batch_size}")
        if sample_centers is not None and len(sample_centers) != batch_size:
            raise ValueError(f"sample_centers_of_mass has {len(sample_centers)} rows but batch_size is {batch_size}")
        device = next(value.device for value in features.feature_dict.values() if isinstance(value, torch.Tensor))
        features = collate_sample_center_features(per_sample)
        move_features_to_device(features, device)
        return features
    for key, value in features.feature_dict.items():
        if isinstance(value, torch.Tensor):
            features.feature_dict[key] = torch.stack([value] * batch_size)
    return features


def apply_random_global_translation(features: Features) -> None:
    """Draw one global translation per batch row and add it to coords and prior_mean."""
    translation_std = features.featurization_metadata.get(RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY)
    if translation_std is None or translation_std <= 0:
        return
    coords = features.feature_dict["coords"]
    batch_and_coord_axes = 2
    coord_dim = coords.shape[-1]
    middle = (1,) * (coords.ndim - batch_and_coord_axes)
    translation = torch.randn(coords.shape[0], *middle, coord_dim, device=coords.device, dtype=coords.dtype)
    translation = translation * translation_std
    features.feature_dict["coords"] = coords + translation
    features.feature_dict["prior_mean"] = features.feature_dict["prior_mean"] + translation
