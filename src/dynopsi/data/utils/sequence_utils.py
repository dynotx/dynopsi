import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from dynopsi.data.auth_chain_identity import (
    ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP,  # noqa: F401
    CHAIN_ALPHABET,
    SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP,
    resolve_unique_letter_auth_int,
)
from dynopsi.utils.strict_config import StrictConfig

logger = logging.getLogger(__name__)


def get_asym_id_from_chain_id(chain_id: str) -> int:
    """Infer asym ID from chain ID using alphabetical ordering.

    The asym ID (`asym_id`) is inferred from the single letter chain ID (`chain_id`)
    using alphabetical ordering (see SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP for the canonical single-letter mapping).

    Parameters
    ----------
    chain_id : str
        The chain id to convert to an asym_id.
        Expected format: potentially multiple letters, optionally followed by a number
        (e.g. "A", "B", "A1", "B10", "AA", "AB", "BA", "c").

    Returns
    -------
    int
        The asym_id corresponding to the chain_id, using spreadsheet-style base-52 numbering,
        with upper and lower case letters treated as distinct (e.g. A -> 0, a -> 26). -1 is
        returned for author chain ids that are not in the expected format or are not an integer.

    Notes
    -----
    The expected format is one or more letters, optionally followed by a number. The value of the trailing
    number will be added to the value of the letters.
    (e.g. "A" -> 0, "B" -> 1, "A1" -> 1, "B10" -> 11, "AA" -> 52, "AB" -> 53, "AB2" -> 55, "ab" -> 1431, "ab2" -> 1433).
    If the `chain_id` is an integer, it is returned as is. This is not recommended, and will log a warning.
    If the `chain_id` is not in the expected format or is not an integer, it is returned as -1. This is not recommended, and will log a warning.
    """
    # do the fastest op if possible / single letter chain id
    if chain_id in SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP.keys():
        return SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[chain_id]
    else:
        chain_alpha_match = re.match(r"^(?P<alpha>[A-Za-z]+)(?P<number>[0-9]*)$", chain_id)
        if chain_alpha_match:
            chain_alpha = chain_alpha_match.group("alpha")
            chain_number = chain_alpha_match.group("number")
            if chain_number:
                chain_number = int(chain_number)
            else:
                chain_number = 0
            value = 0
            for char in chain_alpha:
                value = value * len(CHAIN_ALPHABET) + (CHAIN_ALPHABET.index(char) + 1)
            return value - 1 + chain_number

        else:
            chain_int_match = re.match(r"^(?P<int>[0-9]+)$", chain_id)
            if chain_int_match:
                logger.warning(f"Unexpected author chain id: {chain_id}, returning as is.")
                chain_int = chain_int_match.group("int")
                return int(chain_int)
            else:
                logger.warning(f"Unexpected author chain id: {chain_id}, returning -1.")
                return -1


def get_chain_id_from_asym_id(asym_id: int) -> str:
    """Convert an asym ID to a spreadsheet-style alphabetical chain ID."""
    if asym_id < 0:
        raise ValueError(f"asym_id must be non-negative, got {asym_id}")

    value = asym_id + 1
    chain_id_chars = []
    while value > 0:
        value, remainder = divmod(value - 1, len(CHAIN_ALPHABET))
        chain_id_chars.append(CHAIN_ALPHABET[remainder])
    return "".join(reversed(chain_id_chars))


class ChainResidueRepr(StrictConfig):
    """Represents a list of chain ids and residue indices.

    Instances can be created from a string representation via the `from_string`
    method. Validation also uses initialization from string representation,
    so such a string can be passed wherever a `ChainResidueRepr` instance
    is expected.

    Attributes
    ----------
    chain_id_to_residue_index_map: dict[str, list[int] | None]
        Map from chain ID to residue indices. None indicates all residues in that chain.

    Notes
    -----
    Residue indices are 1-indexed with respect to the first residue in the chain (PDB/mmCIF convention).
    For 0-based indices suitable for dyno-psi inputs, 1 needs to be subtracted from the
    residue_indices; this is supported via the `asym_id_to_residue_index_map_0index` property.
    """

    chain_id_to_residue_index_map: dict[str, list[int] | None] = Field(default_factory=dict)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Generate the Pydantic core schema for ChainResidueRepr validation.

        Applies the _coerce method to the input, which automatically tries to coerce any input to a ChainResidueRepr.
        """
        return core_schema.no_info_before_validator_function(
            cls._coerce,
            handler(source_type),
        )

    @classmethod
    def _coerce(cls, v: Any) -> Any:
        """Coerce strings via from_string; bare chain/auth dicts wrap into chain_id_to_residue_index_map.

        Raises
        ------
        ValueError
            If the input is not a string, dictionary, or ChainResidueRepr.
        """
        if isinstance(v, str):
            return cls.from_string(v)
        if isinstance(v, ChainResidueRepr):
            return v
        if isinstance(v, dict):
            if "chain_id_to_residue_index_map" in v:
                return v
            return {"chain_id_to_residue_index_map": v}
        raise ValueError(f"Invalid type: {type(v)}")

    @classmethod
    def from_string(cls, s: str) -> "ChainResidueRepr":
        """Build a ChainResidueRepr from a chain-position string representation.

        The string must be a valid comma-separated list of chain positions, as
        defined by `expand_chain_position_str`.

        Parameters
        ----------
        s : str
            Chain position string to parse.

        Returns
        -------
        ChainResidueRepr
            Parsed chain–residue representation.

        Raises
        ------
        ValueError
            If the string is invalid or a chain has both full and partial residue lists.

        See Also
        --------
        expand_chain_position_str : Defines the accepted string format.
        """
        rep = expand_chain_position_str(s)
        chain_id_to_residue_index_map: dict[str, list[int] | None] = {}
        for chain_id, residue_index in rep:
            if chain_id not in chain_id_to_residue_index_map:
                if residue_index is None:
                    chain_id_to_residue_index_map[chain_id] = None
                else:
                    chain_id_to_residue_index_map[chain_id] = [residue_index]
            else:
                if residue_index is None:
                    raise ValueError(f"Chain cannot have both full and partial residue indices: {s}")
                else:
                    chain_id_to_residue_index_map[chain_id].append(residue_index)
        return ChainResidueRepr(chain_id_to_residue_index_map=chain_id_to_residue_index_map)

    @property
    def chain_ids(self) -> set[str]:
        """Chain IDs present in this representation."""
        return set(self.chain_id_to_residue_index_map.keys())

    @property
    def asym_id_to_residue_index_map_1index(self) -> dict[int, list[int] | None]:
        """Map from asym ID to 1-indexed residue indices.

         To construct the mapping, chain IDs are converted to asym IDs
         via `get_asym_id_from_chain_id`.

        Returns
        -------
        dict[int, list[int] | None]
            Keys are asym IDs; values are 1-indexed residue index lists, or None for full chain.

        See Also
        --------
        chain_id_to_residue_index_map : Defines the residue indices (values) in the mapping.
        """
        unique_chain_id_map = {chain_id: get_asym_id_from_chain_id(chain_id) for chain_id in self.chain_ids}
        return {
            asym_id: self.chain_id_to_residue_index_map[chain_id] for chain_id, asym_id in unique_chain_id_map.items()
        }

    @property
    def asym_id_to_residue_index_map_0index(self) -> dict[int, list[int] | None]:
        """Map from asym_id to 0-indexed residue indices.

        Returns
        -------
         dict[int, list[int] | None]
             Keys are asym IDs; values are 0-indexed residue index lists, or None for full chain.

        See Also
        --------
        asym_id_to_residue_index_map_1index : Defines underlying mapping between
        the same set of asym IDs and 1-indexed residue index lists; used to construct
        the 0-indexed residue indices.
        """
        return_dict: dict[int, list[int] | None] = {}
        for asym_id, residue_indices in self.asym_id_to_residue_index_map_1index.items():
            if residue_indices is None:
                return_dict[asym_id] = None
            else:
                return_dict[asym_id] = [residue_index - 1 for residue_index in residue_indices]
        return return_dict

    @property
    def is_contiguous(self) -> bool:
        """Boolean indicating whether all residue indices are contiguous.

        Residue indices are considered contiguous if they are all within the
        same chain and are consecutive. A full chain (no explicit residue list)
        is considered contiguous.

        Returns
        -------
        bool
            True if one chain and residues are contiguous (or full chain); False otherwise.
        Raises
        ------
        ValueError
            If the stored residue indices are an invalid type.
        """
        chain_ids = list(self.chain_id_to_residue_index_map.keys())
        if len(chain_ids) > 1:
            return False
        else:
            chain_residues = self.chain_id_to_residue_index_map[chain_ids[0]]
            # if full chain, return True
            if chain_residues is None:
                return True
            elif isinstance(chain_residues, list):
                # sort
                sorted_residue_indices = sorted(chain_residues)
                # check that the residue indices are consecutive
                if np.all(np.diff(sorted_residue_indices) == 1):
                    return True
                else:
                    return False
            else:
                raise ValueError(f"chain_residues is unexpected type. Expected list, got {type(chain_residues)}")

    def overlaps(self, other: "ChainResidueRepr") -> bool:
        """Check if this ChainResidueRepr overlaps with another ChainResidueRepr."""
        overlapping_chains = self.chain_ids & other.chain_ids
        if len(overlapping_chains) == 0:
            return False
        else:
            for chain_id in overlapping_chains:
                self_chain_residues = self.chain_id_to_residue_index_map[chain_id]
                other_chain_residues = other.chain_id_to_residue_index_map[chain_id]
                if self_chain_residues is None or other_chain_residues is None:
                    return True
                if np.any(np.isin(self_chain_residues, other_chain_residues)):
                    return True
            return False

    @property
    def to_string(self) -> str:
        """Convert ChainResidueRepr to a string representation.

        Output string representation is formatted as a comma-separated list of
        letter chain IDs and residue indices.

        Returns
        -------
        str
            String representation of this ChainResidueRepr.

        See Also
        --------
        expand_chain_position_str : Defines the expected string format.
        """
        output_strs: list[str] = []
        for chain_id, residue_indices in self.chain_id_to_residue_index_map.items():
            if residue_indices is None:
                output_strs.append(chain_id)
            else:
                output_strs.extend([f"{chain_id}{residue_index}" for residue_index in residue_indices])
        return ",".join(output_strs)

    @classmethod
    def from_asym_id_mask(
        cls,
        mask: torch.Tensor,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor | None = None,
        index_type: Literal["original_residue_index", "residue_index", "auth_residue_index"] | None = None,
        asym_id_map: dict[int, str] | None = None,
    ) -> "ChainResidueRepr":
        """Create a ChainResidueRepr from mask and asym ID tensors.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape [N]; True for residues to include in the representation.
        asym_id : torch.Tensor
            Asym ID per residue, shape [N], int. If asym_id_map is not provided, asym IDs are converted to
            spreadsheet-style alphabetical chain IDs.
        residue_index : torch.Tensor | None, default None
            Residue index per residue, shape [N], int. If present, the values of this index, masked
            will be used to construct the ChainResidueRepr. If None, then per asym_id,
            the residue indices will be 1-indexed with respect to the first residue in the chain.
        index_type: Literal["original_residue_index", "residue_index", "auth_residue_index"] or None, default None
            The type of index to use for the residue indices. If None, then the residue indices are assumed to be in the range [0, N-1].
            If present: "original_residue_index" and "residue_index" are assumed to be 0-indexed with respect to the first residue in the chain.
            and "auth_residue_index" is assumed to be 1-indexed with respect to the first residue in the chain.
        asym_id_map: dict[int, str] | None, default None
            A map of int asym IDs to string chain IDs.
            If None, `get_chain_id_from_asym_id` is used.

        Returns
        -------
        ChainResidueRepr
            Representation of the selected chain(s) and residues.
        """
        chain_id_to_residue_index_map: dict[str, list[int] | None] = {}
        for aid in asym_id.unique():
            if residue_index is not None:
                if index_type is None:
                    raise ValueError("index_type must be provided if residue_index is not None")
                elif index_type == "residue_index" or index_type == "original_residue_index":
                    offset = 1
                elif index_type == "auth_residue_index":
                    offset = 0
                indices = residue_index[mask & (asym_id == aid)] + offset
            else:
                min_asym_id_idx = torch.where(asym_id == aid)[0][0]
                indices_by_token = torch.where((asym_id == aid) & mask)[0]
                indices = indices_by_token - min_asym_id_idx
                indices = indices + 1
            if len(indices) == 0:
                continue
            else:
                aid_int = aid.item()
                chain_id = get_chain_id_from_asym_id(aid_int) if asym_id_map is None else asym_id_map[aid_int]
                chain_id_to_residue_index_map[chain_id] = indices.tolist()

        return ChainResidueRepr(chain_id_to_residue_index_map=chain_id_to_residue_index_map)

    def _asym_id_to_residue_index_map_for_mask(
        self,
        index_type: Literal["original_residue_index", "residue_index", "auth_residue_index"],
        auth_chain_id_table: Mapping[int, str] | None,
    ) -> dict[int, list[int] | None]:
        """Map chain keys to asym/auth ints for mask construction (0-index for original/residue; auth table for auth_residue_index)."""
        if index_type in ("original_residue_index", "residue_index"):
            return self.asym_id_to_residue_index_map_0index

        table = auth_chain_id_table or {}
        residue_map: dict[int, list[int] | None] = {}
        for chain_id, residue_indices in self.chain_id_to_residue_index_map.items():
            try:
                auth_int = resolve_unique_letter_auth_int(chain_id, table)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            residue_map[auth_int] = residue_indices
        return residue_map

    def to_mask(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        pad_mask: torch.Tensor,
        index_type: Literal["original_residue_index", "residue_index", "auth_residue_index"] = "original_residue_index",
        auth_chain_id_table: Mapping[int, str] | None = None,
    ) -> torch.Tensor:
        """Produce a boolean mask corresponding to the ChainResidueRepr.
        Convert the ChainResidueRepr to a mask, given input asym_id and residue_index tensors.
        Parameters
        ----------
        asym_id: torch.Tensor
            The asym ids of the residues in the mask. Shape [N], int.
        residue_index: torch.Tensor
            The residue indices of the residues in the mask. Shape [N], int. There can be duplicates of the same residue index;
            we are just looking for matches to the residue indices in the ChainResidueRepr.
            Valid examples:
            - for original_residue_index, [0, 1, 2, 3], [0, 1, 1, 2]
            - for auth_residue_index, [1, 2, 3, 4], [1, 2, 2, 3], [2, 3, 4]
        pad_mask: torch.Tensor
            The pad mask of the residues in the mask. Shape [N], bool
        index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"
            The type of index to use for the residue indices.
            - "original_residue_index": expected to be 0-indexed with respect to the first residue in the chain.
            - "auth_residue_index": expected to be 1-indexed with respect to the first residue in the chain.
        auth_chain_id_table: Mapping[int, str] | None, default None
            Used when ``index_type`` is ``auth_residue_index`` so non-letter keys
            resolve to persisted ints. Ignored for original/residue index.

        Returns
        -------
        torch.Tensor
            The mask of the residues in the ChainResidueRepr. Shape [N], bool

        Raises
        ------
        ValueError if the combination of asym_id, residue_index, and pad_mask does not match the ChainResidueRepr.
        Usually this occurs if the provided masks are missing some residues relative to the ChainResidueRepr.
        """
        if asym_id.shape != residue_index.shape:
            raise ValueError(
                f"Asym id and residue index shapes must match, got {asym_id.shape} and {residue_index.shape}"
            )
        if asym_id.min() < 0:
            raise ValueError(f"Asym ids must be non-negative, got minimum {asym_id.min()}")
        mask = torch.zeros_like(asym_id, dtype=torch.bool)
        asym_id_to_residue_index_map = self._asym_id_to_residue_index_map_for_mask(index_type, auth_chain_id_table)
        for aid, residue_indices_in_mask in asym_id_to_residue_index_map.items():
            asym_id_mask = (asym_id == aid) & pad_mask.bool()
            if residue_indices_in_mask is None:
                mask[asym_id_mask] = True
            else:
                residue_index_mask = torch.isin(
                    residue_index[asym_id_mask], torch.tensor(residue_indices_in_mask, dtype=torch.long)
                )
                present = {int(index) for index in residue_index[asym_id_mask].tolist()}
                if not set(residue_indices_in_mask) <= present:
                    raise ValueError("Not all residue indices in the mask are present in the ChainResidueRepr")
                mask[asym_id_mask] = residue_index_mask
        return mask

    def get_asym_id_to_index_map(
        self,
        index_type: Literal["original_residue_index", "auth_residue_index"],
        auth_chain_id_table: Mapping[int, str] | None = None,
    ) -> dict[int, list[int] | None]:
        """Map chain keys to residue indices for ``original_residue_index`` or ``auth_residue_index``."""
        if index_type not in ("original_residue_index", "auth_residue_index"):
            raise ValueError(f"Invalid index_type: {index_type}")
        return self._asym_id_to_residue_index_map_for_mask(index_type, auth_chain_id_table)


def expand_chain_position_str(chain_position_str: str) -> list[tuple[str, int | None]]:
    """Parse a comma- and hyphen-delimited string of chain positions into (chain_id, position) pairs.

    Supports single positions (e.g. ``A1``), ranges (e.g. ``A1-5``), full chain (e.g. ``A``),
    and comma-separated lists (e.g. ``A1,A3-5,B``). Range bounds are inclusive on both ends.

    Parameters
    ----------
    chain_position_str : str
        String specifying chain positions: chain ID (letter) optionally followed by
        a single position, a range of positions, or a comma separated list of positions.
        segments.

    Returns
    -------
    list[tuple[str, int | None]]
        List of (chain_id, residue_index) tuples.
        `residue_index` is None for full chain (e.g. `"A"`).

    Raises
    ------
    ValueError
        If the string is empty, malformed, mixes full-chain and partial specs for the
        same chain, uses a range across chains, or has start >= end in a range.

    Notes
    -----
    Range bounds are inclusive on both sides (e.g. ``A2-5`` includes 2 and 5).

    Examples
    --------
    >>> expand_chain_position_str("A1-3")
    [("A", 1), ("A", 2), ("A", 3)]
    >>> expand_chain_position_str("A1,A3-4")
    [("A", 1), ("A", 3), ("A", 4)]
    >>> expand_chain_position_str("A")
    [("A", None)]
    >>> expand_chain_position_str("A1,A2-5,A20,B15-20")
    [("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5), ("A", 20), ("B", 15), ("B", 16), ("B", 17), ("B", 18), ("B", 19), ("B", 20)]
    """
    resi_regex = re.compile(r"^(?P<chain>[A-Z]+)(?P<position>[0-9]*)$")
    if len(chain_position_str) == 0:
        return []
    elif "," not in chain_position_str and "-" not in chain_position_str:
        match = re.match(resi_regex, chain_position_str)

        if match:
            pos = match.group("position")
            if pos is None or len(pos) == 0:
                pos = None
            else:
                pos = int(pos)
            chain = match.group("chain")
            chain = chain.strip()
            if len(chain) == 0:
                raise ValueError(f"Inappropriate string provided, chain must be specified: {chain_position_str}")

            return [(chain, pos)]
        else:
            raise ValueError(f"Inappropriate string provided: {chain_position_str}")
    else:
        # deal with , first
        if "," in chain_position_str:
            pieces = [expand_chain_position_str(i.strip()) for i in chain_position_str.split(",")]

            # check that we don't have both full and partial chain pieces
            full_chain_pieces = [x[0] for sub in pieces for x in sub if x[1] is None]
            if len(set(full_chain_pieces)) != len(full_chain_pieces):
                raise ValueError(
                    f"Inappropriate string provided, cannot have duplicate full chains: {chain_position_str}"
                )
            partial_chain_pieces = [x[0] for sub in pieces for x in sub if x[1] is not None]
            if len(set(full_chain_pieces) & set(partial_chain_pieces)) > 0:
                raise ValueError(
                    f"Inappropriate string provided, cannot have both full and partial chain pieces: {chain_position_str}"
                )

            return sorted(list(set([x for sub in pieces for x in sub])), key=lambda x: (x[0], x[1]))
        elif "-" in chain_position_str:
            start, end = chain_position_str.split("-")
            start_chain, start_pos = expand_chain_position_str(start.strip())[0]
            # add the start_chain to the end chain
            end = start_chain + end
            end_chain, end_pos = expand_chain_position_str(end.strip())[0]
            if start_chain != end_chain:
                raise ValueError(f"Inappropriate string provided, chains are different: {chain_position_str}")

            if start_pos is None or end_pos is None:
                raise ValueError(
                    f"Inappropriate string provided, need coordinates for positions with -: {chain_position_str}"
                )
            else:
                # compare positions
                if start_pos < end_pos:
                    # return an inclusive range
                    return [(start_chain, i) for i in range(start_pos, end_pos + 1)]
                else:
                    raise ValueError(
                        f"Inappropriate string provided; start position {start_pos} >= end position {end_pos}"
                    )
        else:
            raise ValueError(f"Inappropriate string provided, must contain either , or -: {chain_position_str}")


def update_ids(ids_seq: Sequence[torch.Tensor] | Sequence[np.ndarray]) -> torch.Tensor | np.ndarray:
    """Update ids to be unique and consecutive.

    Parameters
    ----------
    ids_seq: a sequence of tensors or numpy arrays of ids. These tensors or arrays must be one-dimensional.

    Returns
    -------
    torch.Tensor or np.ndarray, depending on the input type, of unique ids. The length of the output tensor or
    array should be the sum of the lengths of the input tensors or arrays.
    """
    if len(ids_seq) == 0:
        raise ValueError("No input ids provided")
    if isinstance(ids_seq[0], torch.Tensor):
        is_tensor = True
    else:
        is_tensor = False
    final_ids = [ids_seq[0].clone()] if is_tensor else [ids_seq[0].copy()]
    if len(ids_seq) == 1:
        return final_ids[0]
    used_ids = ids_seq[0].unique() if is_tensor else np.unique(ids_seq[0])
    for ids_arr_or_tensor in ids_seq[1:]:
        if is_tensor:
            assert isinstance(ids_arr_or_tensor, torch.Tensor) and isinstance(used_ids, torch.Tensor)
            if torch.isin(ids_arr_or_tensor, used_ids).any():
                new_ids_tensor = ids_arr_or_tensor + used_ids.max() - ids_arr_or_tensor.min() + 1
            else:
                new_ids_tensor = ids_arr_or_tensor
            final_ids.append(new_ids_tensor)
            used_ids = torch.cat([used_ids, new_ids_tensor])
        else:
            assert isinstance(ids_arr_or_tensor, np.ndarray) and isinstance(used_ids, np.ndarray)
            if np.isin(ids_arr_or_tensor, used_ids).any():
                new_ids_arr = ids_arr_or_tensor + used_ids.max() - ids_arr_or_tensor.min() + 1
            else:
                new_ids_arr = ids_arr_or_tensor
            final_ids.append(new_ids_arr)
            used_ids = np.concatenate([used_ids, new_ids_arr])
    return torch.cat(cast("list[torch.Tensor]", final_ids)) if is_tensor else np.concatenate(final_ids)


def get_contiguous_intervals(asym_ids: torch.Tensor, residue_indices: torch.Tensor) -> torch.Tensor:
    """Get contiguous intervals of residue indices.

    Contiguous intervals are defined as follows:
    - Intervals are contiguous if the residue indices are consecutive.
    - Intervals are contiguous if the chain ids are the same.

    Arguments
    ---------
    asym_ids : torch.Tensor
        The chain ids of the residues, expected to be integers and consecutive.
    residue_indices : torch.Tensor
        The residue indices of the residues, expected to be integers and consecutive within each chain.

    Returns
    -------
    torch.Tensor
        The contiguous intervals of residue indices.
    """
    if len(residue_indices) == 0:
        return torch.empty(0, dtype=torch.long, device=residue_indices.device)

    new_interval = (torch.diff(asym_ids) != 0) | (torch.diff(residue_indices) != 1)
    interval_ids = torch.zeros(len(residue_indices), dtype=torch.long, device=residue_indices.device)
    interval_ids[1:] = torch.cumsum(new_interval.to(torch.long), dim=0)

    return interval_ids
