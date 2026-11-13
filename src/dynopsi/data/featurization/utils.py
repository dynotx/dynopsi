import logging

import torch
from torch.nn.functional import one_hot, pad

from dynopsi.data.auth_chain_identity import (
    OPAQUE_AUTH_ASYM_ID_START,
    merge_auth_chain_id_tables,
    remap_auth_chain_id_table,
)
from dynopsi.data.conditioning import CATEGORICAL_CONDITIONS
from dynopsi.data.constants import AminoAcidSequenceFeature, AtomFilter, InpaintFeature, PocketFeature
from dynopsi.data.utils import add_oxygen_coords
from dynopsi.data.utils.sequence_utils import update_ids

logger = logging.getLogger(__name__)

## Feature keys organized by feature shape (atom or token)
## For example an ATOM_KEY is a feature with shape [num_atoms, ...]
## and a TOKEN_ATOM_KEY is a feature with shape [num_tokens, num_atoms, ...]
ATOM_KEYS_FLAT = [
    "atom_resolved_mask",
    "atom_pad_mask",
]  # [n_atoms]
ATOM_KEYS_3D = [
    "coords",
    "prior_mean",
]  # [1, n_atoms, 3]
TOKEN_KEYS_FLAT = [
    "token_index",
    "residue_index",
    "original_residue_index",
    "auth_residue_index",
    "asym_id",
    "auth_asym_id",
    "entity_id",
    "sym_id",
    "res_type",
    "res_encode",
    "token_pad_mask",
    "token_resolved_mask",
    "pocket_feature",
    "frame_resolved_mask",
    "noised_mask",
    "inpaint_feature",
    "binder_mask",
    "target_mask",
    "secondary_structure_feature",
    *(condition.key for condition in CATEGORICAL_CONDITIONS),
]
TOKEN_TO_ATOM_KEYS = ["frames_idx"]  # [n_tokens, frame_or_atom_idx_in_token]
ATOM_TO_TOKEN_KEYS = ["atom_to_token"]  # [n_atoms, n_tokens]


def strip_chain_padding(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove padding from features in the feature dictionary.

    If all tokens are valid (no padding), returns the features unchanged.
    Otherwise, crops the features to exclude padded positions.

    Parameters
    ----------
    features : dict[str, torch.Tensor]
        Feature dictionary containing a 'token_pad_mask' tensor indicating
        which tokens are padded vs. non-padded positions.

    Returns
    -------
    dict[str, torch.Tensor]
        Feature dictionary with padding removed. Residue indices are preserved.

    See Also
    --------
    crop_features : function used to remove padded positions.
    """
    if features["token_pad_mask"].bool().all():
        return features
    else:
        return crop_features(features, features["token_pad_mask"].bool(), reset_residue_index=False)


def pad_features(
    features: dict[str, torch.Tensor], max_tokens: int, atoms_to_include: AtomFilter
) -> dict[str, torch.Tensor]:
    """Pad features in the features dictionary to the maximum number of tokens.

    If needed, adds padding positions to the individual features in `features`
    to ensure each feature has size at least `max_tokens` or
    `max_tokens * num_atoms` for token and atom features, respectively.
    If `max_tokens` is not larger than the number of tokens in `token_index`,
    the features are returned unchanged.

    Parameters
    ----------
    features : dict[str, torch.Tensor]
        Feature dictionary containing a 'token_pad_mask' tensor indicating
        valid (non-padded) positions.

    Returns
    -------
    dict[str, torch.Tensor]
        Feature dictionary with padding removed. Residue indices are preserved
        (not reset to start from zero).

    Raises
    ------
    ValueError
        If `atoms_to_include` is not backbone-only or Ca-only.
    ValueError
        If `features` contains an invalid key not defined in one of
        `TOKEN_KEYS_FLAT`, `TOKEN_TO_ATOM_KEYS`, `ATOM_KEYS_FLAT`,
        `ATOM_KEYS_3D`, or `ATOM_TO_TOKEN_KEYS`.

    See Also
    --------
    pad : Underlying function used to add padded positions.
    """
    n_pad_tokens = max_tokens - features["token_index"].shape[0]

    # identify num_atoms
    if atoms_to_include == AtomFilter.FULL:
        raise ValueError("AtomFilter.FULL is not supported for pad_features")
    elif atoms_to_include == AtomFilter.BB:
        num_atoms = 3
    elif atoms_to_include == AtomFilter.CA:
        num_atoms = 1
    else:
        raise ValueError(f"Unsupported atoms_to_include: {atoms_to_include}")
    n_pad_atoms = n_pad_tokens * num_atoms

    if n_pad_tokens > 0:
        for key, value in features.items():
            pad_shape: tuple[int, ...]
            pad_value: bool | int | float
            if key in TOKEN_KEYS_FLAT:
                pad_shape = (0, n_pad_tokens)
                pad_value = False if value.dtype == torch.bool else 0
            elif key in TOKEN_TO_ATOM_KEYS:
                pad_shape = (0, 0, 0, n_pad_tokens)
                pad_value = 0
            elif key in ATOM_KEYS_FLAT:
                pad_shape = (0, n_pad_atoms)
                pad_value = False if value.dtype == torch.bool else 0
            elif key in ATOM_KEYS_3D:
                pad_shape = (0, 0, 0, n_pad_atoms, 0, 0)
                pad_value = 0
            elif key in ATOM_TO_TOKEN_KEYS:
                pad_shape = (0, n_pad_tokens, 0, n_pad_atoms)
                pad_value = 0
            else:
                raise ValueError(f"Unsupported key: {key}")
            features[key] = pad(value, pad_shape, mode="constant", value=pad_value)
    return features


def make_empty_chain(
    seq_length: int,
    atoms_to_include: AtomFilter,
    res_type: str = "UNK",
    asym_id: int = 0,
    auth_asym_id: int = 0,
    entity_id: int = 0,
    sym_id: int = 0,
    pocket_feature: int = PocketFeature.UNSPECIFIED,
    inpaint_feature: int = InpaintFeature.BINDER_INPAINT,
    coords_center_of_mass: tuple[float, float, float] | torch.Tensor | None = None,
    device: torch.device | None = None,
    keys_include: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Feature tensors for one chain.

    Parameters
    ----------
    seq_length : int
        The length of the sequence.
    atoms_to_include : AtomFilter
        The atoms to include in the example.
    res_type : str, optional
        The 3-letter code of the residue to use for the example. default is "UNK".
    asym_id : int, optional
        The asym_id of the example. default is 0.
    auth_asym_id : int, optional
        The auth_asym_id of the example. default is 0.
    entity_id : int, optional
        The entity_id of the example. default is 0.
    sym_id : int, optional
        The symmetry identifier for the chain. default is 0.
    pocket_feature : PocketFeature, optional
        The pocket feature for the chain. default is PocketFeature.UNSPECIFIED.
    inpaint_feature : InpaintFeature, optional
        The inpaint feature for the chain. default is InpaintFeature.BINDER_INPAINT.
    coords_center_of_mass : tuple[float, float, float] | torch.Tensor | None, optional
        The center of mass for atom coordinates in the example. If None, random coordinates are generated.
    device : torch.device | None, optional
        The device to place the tensors on.
    keys_include : list[str] | None, optional
        If provided, only include these keys in the returned dictionary.

    Returns
    -------
    dict[str, torch.Tensor]
        Feature dictionary for the mock chain.

    Raises
    ------
    ValueError
        If `atoms_to_include` is not backbone-only of Ca-only.
    """
    if atoms_to_include == AtomFilter.FULL:
        raise ValueError("AtomFilter.FULL is not supported for make_empty_chain")

    num_atoms = atoms_to_include.num_atoms()
    max_atoms = seq_length * num_atoms
    if coords_center_of_mass is None:
        coords = torch.randn(1, max_atoms, 3, dtype=torch.float32)
    else:
        if isinstance(coords_center_of_mass, torch.Tensor):
            coords_center_of_mass = coords_center_of_mass.clone().detach()
        else:
            coords_center_of_mass = torch.tensor(coords_center_of_mass, dtype=torch.float32)
        coords = torch.tile(coords_center_of_mass, (max_atoms, 1)).unsqueeze(0)

    is_binder = (inpaint_feature == InpaintFeature.BINDER_INPAINT) or (inpaint_feature == InpaintFeature.BINDER_CONTEXT)

    example = {
        # Structural token-level features
        "token_index": torch.arange(seq_length, dtype=torch.long),
        "residue_index": torch.arange(seq_length, dtype=torch.long),
        "original_residue_index": torch.arange(seq_length, dtype=torch.long),
        "auth_residue_index": torch.arange(1, seq_length + 1, dtype=torch.long),
        "asym_id": torch.full((seq_length,), asym_id, dtype=torch.long),
        "auth_asym_id": torch.full((seq_length,), auth_asym_id, dtype=torch.long),
        "entity_id": torch.full((seq_length,), entity_id, dtype=torch.long),
        "sym_id": torch.full((seq_length,), sym_id, dtype=torch.long),
        "token_pad_mask": torch.ones(seq_length, dtype=torch.float32),
        "token_resolved_mask": torch.ones(seq_length, dtype=torch.float32),
        "frame_resolved_mask": torch.ones(seq_length, dtype=torch.bool),
        # Sequence token-level features
        "res_type": torch.full((seq_length,), AminoAcidSequenceFeature.from_resname(res_type).value, dtype=torch.long),
        # Model-specific features
        "pocket_feature": torch.full((seq_length,), pocket_feature, dtype=torch.long),
        "noised_mask": torch.ones(seq_length, dtype=torch.bool),
        "inpaint_feature": torch.full((seq_length,), inpaint_feature, dtype=torch.int32),
        # Atom-level features
        "coords": coords,
        "atom_pad_mask": torch.ones(max_atoms, dtype=torch.float32),
        "atom_resolved_mask": torch.ones(max_atoms, dtype=torch.bool),
        # Atom-Residue mapping features
        "frames_idx": torch.arange(max_atoms, dtype=torch.long).reshape(seq_length, num_atoms),
        "atom_to_token": torch.eye(seq_length, dtype=torch.long).repeat_interleave(
            num_atoms, dim=0
        ),  # max_atoms, seq_length
        # tracking binder/target
        "binder_mask": torch.full((seq_length,), is_binder, dtype=torch.bool),
        "target_mask": torch.full((seq_length,), not is_binder, dtype=torch.bool),
    }

    example["res_encode"] = one_hot(example["res_type"], num_classes=len(AminoAcidSequenceFeature))

    if keys_include:
        example = {k: example[k] for k in keys_include}
    if device is not None:
        example = {k: v.to(device) for k, v in example.items()}
    return example


_LETTER_CHAIN_ID_LIMIT = 26


def _unique_chain_ids(values: torch.Tensor) -> list[int]:
    return [int(chain_id) for chain_id in values.unique().tolist()]


def _assign_noncolliding_chain_ids(
    colliding_ids: list[int],
    reserved_ids: set[int],
    *,
    start: int,
    limit: int | None = None,
) -> dict[int, int]:
    assigned_ids: dict[int, int] = {}
    reserved_ids = set(reserved_ids)
    candidate = start
    for old_id in colliding_ids:
        if old_id in assigned_ids:
            continue
        while candidate in reserved_ids:
            candidate += 1
        if limit is not None and candidate >= limit:
            raise ValueError(f"Not enough unique label chain ids in 0-{limit - 1}")
        assigned_ids[old_id] = candidate
        reserved_ids.add(candidate)
        candidate += 1
    return assigned_ids


def _remap_colliding_chain_ids(
    features: list[dict[str, torch.Tensor]],
    auth_chain_id_tables: list[dict[int, str]] | None = None,
) -> None:
    """Mutate cloned ``features`` in place; optional parallel table copies are remapped too."""
    if auth_chain_id_tables is not None and len(auth_chain_id_tables) != len(features):
        raise ValueError("auth_chain_id_tables must have one table per concatenated feature dict")
    used_label_ids = set(_unique_chain_ids(features[0]["asym_id"]))
    used_auth_ids = set(_unique_chain_ids(features[0]["auth_asym_id"]))
    for feat_i, feat in enumerate(features[1:], start=1):
        feat_asym_ids = _unique_chain_ids(feat["asym_id"])
        feat_auth_ids = _unique_chain_ids(feat["auth_asym_id"])
        colliding_asym_ids = [asym_id for asym_id in feat_asym_ids if asym_id in used_label_ids]
        colliding_auth_ids = [auth_asym_id for auth_asym_id in feat_auth_ids if auth_asym_id in used_auth_ids]
        assigned_label_ids = _assign_noncolliding_chain_ids(
            colliding_ids=colliding_asym_ids,
            reserved_ids=used_label_ids | (set(feat_asym_ids) - set(colliding_asym_ids)),
            start=0,
            limit=_LETTER_CHAIN_ID_LIMIT,
        )
        assigned_auth_ids = _assign_noncolliding_chain_ids(
            colliding_ids=colliding_auth_ids,
            reserved_ids=used_auth_ids | (set(feat_auth_ids) - set(colliding_auth_ids)),
            start=OPAQUE_AUTH_ASYM_ID_START,
        )
        if assigned_label_ids:
            remapped_asym_ids = feat["asym_id"].clone()
            for old_id, new_id in assigned_label_ids.items():
                remapped_asym_ids[feat["asym_id"] == old_id] = new_id
            feat["asym_id"] = remapped_asym_ids
        if assigned_auth_ids:
            remapped_auth_ids = feat["auth_asym_id"].clone()
            for old_id, new_id in assigned_auth_ids.items():
                remapped_auth_ids[feat["auth_asym_id"] == old_id] = new_id
            feat["auth_asym_id"] = remapped_auth_ids
        if auth_chain_id_tables is not None:
            auth_chain_id_tables[feat_i] = remap_auth_chain_id_table(
                auth_chain_id_tables[feat_i],
                assigned_auth_ids,
            )
        used_label_ids |= set(_unique_chain_ids(feat["asym_id"]))
        used_auth_ids |= set(_unique_chain_ids(feat["auth_asym_id"]))


def concat_features(
    features: list[dict[str, torch.Tensor]],
    make_asym_ids_unique: bool,
    make_entity_ids_unique: bool,
    reset_residue_index: bool = True,
) -> dict[str, torch.Tensor]:
    """Concatenate multiple feature dictionaries.

    Combines features from multiple sources, stripping padding from
    each individual feature dictionary before concatenating.

    Parameters
    ----------
    features : list[dict[str, torch.Tensor]]
        The list of feature dictionaries to concatenate.
    make_asym_ids_unique : bool, optional
        Whether to update the asym_ids and auth_asym_ids of all input features to make them unique.
        If True, update asym_ids and reset residue_index to start from 0 for each
        asym_id. Label ``asym_id`` remaps below ``_LETTER_CHAIN_ID_LIMIT``; auth
        ``auth_asym_id`` remaps from ``OPAQUE_AUTH_ASYM_ID_START`` upward. The two
        namespaces are independent.
        If False, preserve original asym_ids and auth_asym_ids and make residue_index
        contiguous based on input order.
    make_entity_ids_unique : bool, optional
        Whether to update the entity_ids of all input features to make them unique.
        If True, assign unique entity_ids to all chains and set all sym_ids to 0.
        If False, preserve entity_ids and assign sym_ids based on input order.
    reset_residue_index : bool, optional
        Whether to reset the residue_index to start at 0 and be consecutive for each asym_id.
        If False, preserve original residue indices. This should almost always be True,
        unless you explicitly want missing residues (gaps in the residue_index). We do NOT
        change the auth_residue_index.

    Returns
    -------
    dict[str, torch.Tensor]
        The concatenated features.

    Raises
    ------
    ValueError
        If the keys in the individual features dictionaries in `features` do not agree.
    ValueError
        If `make_asym_ids_unique=False` and `make_entity_ids_unique=True`, which
        could result in one asym_id mapping to multiple entity_ids which is not allowed.
    ValueError
        If `make_entity_ids_unique=False` and chains with matching entity_ids
        have different lengths.

    Notes
    -----
    Any features that are 2D and have off-diagonal elements (e.g. disto_target) cannot be concatenated directly.

    In order to track the original residue_index, this function will
    - check if `original_residue_index` is present in the input features
    - if it is, then it will be concatenated and appear as `original_residue_index` in the output features
    - if it is not, then a new key will be created in the output features called `original_residue_index`,
      and it will be exactly the concatenated `residue_index` of the input features.

    Examples
    --------
    features1 = {
        "asym_id": [0, 0, 0, 1, 1, 1],
        "residue_index": [0, 1, 2, 0, 1, 2],
        "token_index": [0, 1, 2, 3, 4, 5],
        "entity_id": [0, 0, 0, 1, 1, 1],
    }
    features2 = {
        "asym_id": [1, 1, 1, 2, 2, 2],
        "residue_index": [0, 1, 2, 0, 1, 2],
        "token_index": [0, 1, 2, 3, 4, 5],
        "entity_id": [0, 0, 0, 1, 1, 1],
    }

    Make asym_ids unique:
    concat_features([features1, features2], make_asym_ids_unique=True, make_entity_ids_unique=True) ->
       {
          "asym_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          "residue_index": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
          "token_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          "entity_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          "sym_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       }

    Take asym_ids as is:
    concat_features([features1, features2], make_asym_ids_unique=False, make_entity_ids_unique=True) ->
       {
          "asym_id": [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
          "residue_index": [0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2],
          "token_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          "entity_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          "sym_id": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       }

    Without making entity_ids unique:
    NOTE: if the chain lengths for intended matching entity_ids are different, then a ValueError will be raised.
    concat_features([features1, features2], make_asym_ids_unique=True, make_entity_ids_unique=False) ->
       {
          "asym_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
          "residue_index": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
          "token_index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          "entity_id": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
          "sym_id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
       }
    """
    # remove empty features
    features = [{k: v.clone() for k, v in feat.items()} for feat in features if len(feat["asym_id"]) > 0]
    if len(features) == 0:
        return {}

    # strip padding from features
    features = [strip_chain_padding(feat) for feat in features]

    feature_keys = set(features[0].keys())
    n_feature_keys = len(feature_keys)
    for feat in features[1:]:
        feature_keys &= set(feat.keys())
    if len(feature_keys) != n_feature_keys:
        raise ValueError("Keys do not match across input features")

    # adjust ids to be non-overlapping and correct entity and asym ids (sym_id should not change)

    # entity_id
    if make_entity_ids_unique:
        output_entity_ids = update_ids([feat["entity_id"] for feat in features])
    else:
        output_entity_ids = torch.cat([feat["entity_id"] for feat in features])

    if make_asym_ids_unique:
        _remap_colliding_chain_ids(features)

    # adjust indices to be consecutive
    # rule for resetting indices:
    # token_index: reset to start at 0 and be consecutive across all features
    # frames_idx: reset to start at 0 and be consecutive across all features
    # residue_index: reset to start at 0 and be consecutive for each asym_id
    for k in ["token_index", "frames_idx"]:
        features[0][k] -= features[0][k].min().long()
        for i in range(1, len(features)):
            features[i][k] += (features[i - 1][k].max() - features[i][k].min() + 1).long()

    # check if original_residue_index is present in the input features
    # if not, add it
    for feat in features:
        if "original_residue_index" not in feat:
            feat["original_residue_index"] = feat["residue_index"].clone()

    # reset residue index to start at 0 and be consecutive for each asym_id
    if reset_residue_index:
        for asym_id in torch.unique(torch.cat([feat["asym_id"] for feat in features])):
            # for each feature and each asym_id, identify the indices that correspond to this asym_id
            masks = [(feat["asym_id"] == asym_id) for feat in features]

            # identify the first mask that has any True values
            # this is the first residue index that should be reset to 0
            nonzero_mask_idxs = [i for i, mask in enumerate(masks) if mask.any()]
            first_nonzero_mask_idx = nonzero_mask_idxs[0]
            first_nonzero_mask = masks[first_nonzero_mask_idx]
            features[first_nonzero_mask_idx]["residue_index"][first_nonzero_mask] -= features[first_nonzero_mask_idx][
                "residue_index"
            ][first_nonzero_mask].min()
            # for all subsequent masks for this asym_id, set the residue index to be consecutive with the previous mask
            for i in nonzero_mask_idxs[1:]:
                features[i]["residue_index"][masks[i]] += (
                    features[i - 1]["residue_index"][masks[i - 1]].max()
                    - features[i]["residue_index"][masks[i]].min()
                    + 1
                )

    feats_out = {}
    try:
        for k in feature_keys:
            if k in ATOM_KEYS_3D:
                feats_out[k] = torch.cat([feats[k] for feats in features], dim=1)
            elif k in (ATOM_KEYS_FLAT + TOKEN_KEYS_FLAT + TOKEN_TO_ATOM_KEYS):
                feats_out[k] = torch.cat([feats[k] for feats in features], dim=0)
            elif k in (ATOM_TO_TOKEN_KEYS):
                try:
                    n = [feats[k].shape[0] for feats in features]
                    m = [feats[k].shape[1] for feats in features]
                    new_shape = [sum(n), sum(m)] + list(features[0][k].shape[2:])
                    feats_out[k] = torch.zeros(new_shape, dtype=features[0][k].dtype)
                    for i in range(len(features)):
                        n_start = sum(n[:i])
                        m_start = sum(m[:i])
                        feats_out[k][n_start : n_start + n[i], m_start : m_start + m[i]] = features[i][k]
                except Exception as e:
                    logger.error(f"Error concatenating {k}")
                    for feats in features:
                        logger.error(f"feats[k].shape: {feats[k].shape}")
                    raise e
            else:
                raise ValueError(f"Unsupported key: {k}")
    except Exception as e:
        logger.error(f"Error concatenating {k}")
        for feats in features:
            logger.error(f"feats[k].shape: {feats[k].shape}")
        raise e

    if not isinstance(output_entity_ids, torch.Tensor):
        raise ValueError(f"output_entity_ids expected to be tensor, but was {type(output_entity_ids)}")
    feats_out["entity_id"] = output_entity_ids

    # checks and updates to ids
    # check that one asym_id is associated with only one entity_id
    asym_id_to_entity_id_map = {}
    for asym_id in feats_out["asym_id"].unique():
        entity_ids = feats_out["entity_id"][feats_out["asym_id"] == asym_id].unique()
        if len(entity_ids) > 1:
            raise ValueError(f"Asym_id {asym_id} has multiple entity_ids: {entity_ids}")
        asym_id_to_entity_id_map[asym_id] = entity_ids[0].item()

    # now assign sym_ids, such that entity_ids with multiple asym_ids have unique sym_ids
    entity_id_to_asym_id_map = {}
    for asym_id, entity_id in asym_id_to_entity_id_map.items():
        if entity_id not in entity_id_to_asym_id_map:
            entity_id_to_asym_id_map[entity_id] = [asym_id]
        else:
            entity_id_to_asym_id_map[entity_id].append(asym_id)

    for entity_id, asym_ids in entity_id_to_asym_id_map.items():
        if len(asym_ids) > 1:
            for i in range(len(asym_ids)):
                feats_out["sym_id"][feats_out["asym_id"] == asym_ids[i]] = i

    return feats_out


def concat_features_with_auth_tables(
    features: list[dict[str, torch.Tensor]],
    auth_chain_id_tables: list[dict[int, str]],
    make_asym_ids_unique: bool,
    make_entity_ids_unique: bool,
    reset_residue_index: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[int, str]]:
    """Copy and remap parallel auth tables, then concat. Caller tables are not mutated."""
    if len(auth_chain_id_tables) != len(features):
        raise ValueError("auth_chain_id_tables must have one table per concatenated feature dict")
    cloned: list[dict[str, torch.Tensor]] = []
    tables: list[dict[int, str]] = []
    for feat, table in zip(features, auth_chain_id_tables, strict=True):
        if len(feat["asym_id"]) == 0:
            continue
        cloned.append({k: v.clone() for k, v in feat.items()})
        tables.append(dict(table))
    cloned = [strip_chain_padding(feat) for feat in cloned]
    tables = [
        {
            int(auth_id): chain_id
            for auth_id, chain_id in table.items()
            if int(auth_id) in set(_unique_chain_ids(feat["auth_asym_id"]))
        }
        for feat, table in zip(cloned, tables, strict=True)
    ]
    if make_asym_ids_unique and cloned:
        _remap_colliding_chain_ids(cloned, tables)
    concatenated = concat_features(
        cloned,
        make_asym_ids_unique=False,
        make_entity_ids_unique=make_entity_ids_unique,
        reset_residue_index=reset_residue_index,
    )
    return concatenated, merge_auth_chain_id_tables(tables)


def crop_features(
    features: dict[str, torch.Tensor], token_mask: torch.Tensor, reset_residue_index: bool = True
) -> dict[str, torch.Tensor]:
    """Crops input features according to token-level mask.

    Parameters
    ----------
    features : dict[str, torch.Tensor]
        The features to crop.
        Must include `token_index`, `atom_to_token`, `asym_id`, `residue_index`, and `token_pad_mask`.
    token_mask : torch.Tensor
        The mask to apply to the token features.
    reset_residue_index : bool, optional
        Whether to reset the residue_index to start at 0. This should almost always be True,
        unless you explicitly want missing residues (gaps in the residue_index)

    Returns
    -------
    dict[str, torch.Tensor]
        The cropped features.

    Raises
    ------
    ValueError
        If the shape of the `token_mask` does not match the shape of the input features `token_index`.
    ValueError
        If `features` contains any invalid keys not defined in one of
        `TOKEN_KEYS_FLAT`, `ATOM_KEYS_FLAT`,  `ATOM_KEYS_3D`, or `ATOM_TO_TOKEN_KEYS`.
    """
    if token_mask.shape[0] != features["token_index"].shape[0]:
        raise ValueError(
            f"Token mask shape {token_mask.shape} does not match features['token_index'].shape {features['token_index'].shape}"
        )

    # use the atom_to_token features to compute the atom_mask
    if token_mask.any():
        atom_mask = features["atom_to_token"][:, token_mask].sum(dim=1).clone().bool()
    else:
        atom_mask = torch.zeros(features["atom_to_token"].shape[0], dtype=torch.bool)

    new_features = {}
    for k in features:
        try:
            if k in ATOM_KEYS_3D:
                new_features[k] = features[k][:, atom_mask]
            elif k == "frames_idx":
                # check the current size of frames_idx to decide ca or bb/full
                if features[k].shape[1] == 1:
                    # ca
                    frames_idx = torch.arange(atom_mask.sum().int().item()).unsqueeze(1)
                else:
                    # bb/full
                    # we make the assumption here that the first 3 atoms of a residue are
                    # the (backbone) frame atoms
                    if not atom_mask.any():
                        frames_idx = torch.zeros(0, 3, dtype=torch.long)
                    else:
                        masked_atom_to_token = features["atom_to_token"][atom_mask][:, token_mask]
                        resi_first_frame = masked_atom_to_token.argmax(dim=0).unsqueeze(1)
                        frames_idx = torch.cat([resi_first_frame, resi_first_frame + 1, resi_first_frame + 2], dim=1)
                new_features[k] = frames_idx

            elif k in ATOM_KEYS_FLAT:
                new_features[k] = features[k][atom_mask]
            elif k in TOKEN_KEYS_FLAT:
                new_features[k] = features[k][token_mask]
            elif k in ATOM_TO_TOKEN_KEYS:
                new_features[k] = features[k][atom_mask][:, token_mask]
            else:
                raise ValueError(f"Unsupported key: {k}")
        except Exception as e:
            logger.error(f"Error cropping {k}")
            logger.error(f"features[{k}].shape: {features[k].shape}")
            logger.error(f"atom_mask.shape: {atom_mask.shape}")
            logger.error(f"token_mask.shape: {token_mask.shape}")
            raise e

    # perform check for mask, otherwise this will raise an error from min() of an empty tensor
    if token_mask.any():
        new_features["token_index"] = torch.arange(new_features["asym_id"].shape[0])
        if reset_residue_index:
            for asym_id in torch.unique(new_features["asym_id"][new_features["token_pad_mask"].bool()]):
                mask = (new_features["asym_id"] == asym_id) & new_features["token_pad_mask"].bool()
                new_features["residue_index"][mask] -= new_features["residue_index"][mask].min().long()

    return new_features


def map_residue_to_atom_mask(
    residue_mask: torch.Tensor, atom_to_token: torch.Tensor, atom_mask: torch.Tensor
) -> torch.Tensor:
    """Map a residue-level mask to atom-level, intersected with an existing atom mask.

    Expands a per-residue boolean mask to per-atom using the atom-to-token
    mapping, then returns the intersection with the provided atom mask.

    Parameters
    ----------
    residue_mask : torch.Tensor
        Residue-level mask to map to an atom mask, shape `(num_residues,)`.
    atom_to_token : torch.Tensor
        One-hot mapping from atoms to tokens/residues, shape `(num_atoms, num_tokens)`.
    atom_mask : torch.Tensor
        Existing atom mask to intersect with, shape `(num_atoms,)`.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape `(num_atoms,)` where True indicates atoms
        belonging to selected residues in `residue_mask` AND present in the original atom mask.
    """
    token_idx = atom_to_token.argmax(dim=1)  # [num_atoms]
    residue_to_atom_mask = residue_mask[token_idx].bool()
    return residue_to_atom_mask & atom_mask.bool()


def add_oxygen_to_bb_features(feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add oxygen to backbone atoms to a backbone-only feature dictionary.

    Computes oxygen coordinates from existing N, CA, C backbone atoms
    and updates atom-level features accordingly. Padding is stripped if present.

    Parameters
    ----------
    feature_dict : dict[str, torch.Tensor]
        Feature dictionary with backbone-only atoms. Required keys:
        - coords: torch.Tensor of shape `[n_residues, 3, 3]`
        - token_pad_mask: torch.Tensor of shape `[n_residues]`
        - atom_resolved_mask: torch.Tensor of shape `[n_atoms]`
        - atom_to_token: torch.Tensor of shape `[n_atoms, n_residues]`
        - token_index: torch.Tensor of shape `[n_residues]`

    Returns
    -------
    dict[str, torch.Tensor]
        Copy of feature dictionary with oxygen atoms added. Updated keys:
        - `coords`` shape `(1, n_residues * 4, 3)` with N, CA, C, O atoms
        - `atom_to_token`: shape `(n_residues * 4, n_residues)`
        - `atom_resolved_mask`: shape `(n_residues * 4,)`, oxygen atoms
          are marked as resolved by default

    Notes
    -----
    Only `coords`, `atom_to_token`, and `atom_resolved_mask` are updated.
    Other atom-level features are not modified.

    See Also
    --------
    add_oxygen_coords : Computes oxygen coordinates from backbone atoms.
    strip_chain_padding : Removes padding before processing.
    """
    feature_dict = feature_dict.copy()
    # check for padding
    if "token_pad_mask" in feature_dict.keys():
        if (~feature_dict["token_pad_mask"].bool()).any():
            feature_dict = strip_chain_padding(feature_dict)

    n_residues = feature_dict["token_index"].shape[0]

    coords = feature_dict["coords"].reshape(n_residues, 3, 3)
    coords_with_oxygen = add_oxygen_coords(coords).reshape(1, n_residues * 4, 3)
    feature_dict["coords"] = coords_with_oxygen

    feature_dict["atom_to_token"] = torch.eye(n_residues).repeat_interleave(4, dim=0)

    atom_resolved_mask = feature_dict["atom_resolved_mask"]  # shape [n_atoms]
    atom_resolved_mask_with_oxygen = torch.cat(
        [
            atom_resolved_mask.reshape(-1, 3).bool(),  # shape [n_residues, 3]
            torch.full(
                (int(n_residues), 1), True, dtype=torch.bool, device=atom_resolved_mask.device
            ),  # shape [n_residues, 1], defaults to all O's resolved
        ],
        dim=1,
    ).reshape(-1)
    feature_dict["atom_resolved_mask"] = atom_resolved_mask_with_oxygen

    return feature_dict
