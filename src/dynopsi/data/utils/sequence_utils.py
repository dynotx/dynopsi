import re
import string
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP: dict[str, int] = dict(zip(string.ascii_uppercase, range(26)))
ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP: dict[int, str] = {v: k for k, v in SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP.items()}


def get_asym_id_from_chain_id(chain_id: str) -> int:
    """Infer asym ID from chain ID using alphabetical ordering.

    The asym ID (`asym_id`) is inferred from the single letter chain ID (`chain_id`)
    using alphabetical ordering (see SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP
    for the canonical mapping).

    Parameters
    ----------
    chain_id : str
        The chain id to convert to an asym_id.
        Expected format: a single letter, optionally followed by a number
        (e.g. "A", "B", "A1", "B10").

    Returns
    -------
    int
        The asym_id corresponding to the chain_id, in the range [0, 25],
        corresponding to the index of the chain ID in the alphabet.

    Raises
    ------
    ValueError
        If the `chain_id` is not in the expected format.
    """
    chain_id = chain_id.upper()

    # do the fastest op if possible / single letter chain id
    if chain_id in SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP.keys():
        return SINGLE_LETTER_CHAIN_TO_ASYM_ID_MAP[chain_id]
    # otherwise need to match a regex
    else:
        chain_alpha_match = re.match(r"^(?P<alpha>[A-Z])[0-9]*$", chain_id)
        if chain_alpha_match:
            chain_alpha = chain_alpha_match.group("alpha")
            return string.ascii_uppercase.index(chain_alpha)

        else:
            raise ValueError(f"Invalid chain id: {chain_id}")


class ChainResidueRepr(BaseModel):
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
    def _coerce(cls, v: Any) -> "ChainResidueRepr":
        """Coerce an input to a ChainResidueRepr when possible.
        - If the input is a string, it is converted to a `ChainResidueRepr` instance
        using the `from_string` method.
        - If the input is a dictionary or existing `ChainResidueRepr`, it is returned as is.

        Raises
        ------
        ValueError
            If the input is not a type (string, dictionary, or `ChainResidueRepr`)
            that can be automatically coerced.
        """
        if isinstance(v, str):
            return cls.from_string(v)
        elif isinstance(v, dict) | isinstance(v, ChainResidueRepr):
            return v
        else:
            raise ValueError(f"Invalid type: {type(v)}")
        return v

    @classmethod
    def from_string(cls, s: str) -> "ChainResidueRepr":
        """Build a ChainResidueRepr from a chain-position string representation.

        The string must be a valid comma-separated list of chain positions, as
        defined by  as defined by `expand_chain_position_str`.

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
                raise ValueError(
                    f"Invalid residue index type: {type(self.chain_id_to_residue_index_map[chain_ids[0]])}"
                )

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
        cls, mask: torch.Tensor, asym_id: torch.Tensor, asym_id_map: dict[int, str] | None = None
    ) -> "ChainResidueRepr":
        """Create a ChainResidueRepr from mask and asym ID tensors.

        Parameters
        ----------
        mask : torch.Tensor
            Boolean mask of shape [N]; True for residues to include in the representation.
        asym_id : torch.Tensor
            Asym ID per residue, shape [N], int. If asym_id_map is not provided, it is assumed
            that all asym IDs must fall into the range [0, 25].
        asym_id_map: dict[int, str] | None, default None
            A map of int asym IDs to string chain IDs.
            If None, `ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP` is used.

        Returns
        -------
        ChainResidueRepr
            Representation of the selected chain(s) and residues.
        """
        if asym_id_map is None:
            asym_id_map = ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP

        chain_id_to_residue_index_map: dict[str, list[int] | None] = {}
        for aid in asym_id.unique():
            min_asym_id_idx = torch.where(asym_id == aid)[0][0]
            indices_by_token = torch.where((asym_id == aid) & mask)[0]
            indices = indices_by_token - min_asym_id_idx
            if len(indices) == 0:
                continue
            else:
                # place in 1-indexed format
                chain_id_to_residue_index_map[asym_id_map[aid.item()]] = (indices + 1).tolist()

        return ChainResidueRepr(chain_id_to_residue_index_map=chain_id_to_residue_index_map)

    def to_mask(
        self,
        asym_id: torch.Tensor,
        residue_index: torch.Tensor,
        pad_mask: torch.Tensor,
        index_type: Literal["original_residue_index", "residue_index", "auth_residue_index"] = "original_residue_index",
    ) -> torch.Tensor:
        """Produce a boolean mask corresponding to the ChainResidueRepr.
        Convert the ChainResidueRepr to a mask, given input asym_id and residue_index tensors.
        Parameters
        ----------
        asym_id: torch.Tensor
            The asym ids of the residues in the mask. Shape [N], int. These will be converted to chain_ids using
            their position in the alphabet (see ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP). Expected to be in the range [0, 25].
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
        if asym_id.max() > 25 or asym_id.min() < 0:
            raise ValueError(f"Asym ids must be in the range [0, 25], got {asym_id.max()} and {asym_id.min()}")
        mask = torch.zeros_like(asym_id, dtype=torch.bool)
        asym_id_to_residue_index_map = (
            self.asym_id_to_residue_index_map_0index
            if index_type in ["original_residue_index", "residue_index"]
            else self.asym_id_to_residue_index_map_1index
        )
        for aid, residue_indices_in_mask in asym_id_to_residue_index_map.items():
            asym_id_mask = (asym_id == aid) & pad_mask.bool()
            if residue_indices_in_mask is None:
                mask[asym_id_mask] = True
            else:
                residue_index_mask = torch.isin(
                    residue_index[asym_id_mask], torch.tensor(residue_indices_in_mask, dtype=torch.long)
                )
                if residue_index_mask.sum() < len(residue_indices_in_mask):
                    raise ValueError("Not all residue indices in the mask are present in the ChainResidueRepr")
                else:
                    mask[asym_id_mask] = residue_index_mask
        return mask

    def get_asym_id_to_index_map(
        self, index_type: Literal["original_residue_index", "auth_residue_index"]
    ) -> dict[int, list[int] | None]:
        """Get a map from asym id to original_residue_index or auth_residue_index (in the features)."""
        if index_type == "original_residue_index":
            return self.asym_id_to_residue_index_map_0index
        elif index_type == "auth_residue_index":
            return self.asym_id_to_residue_index_map_1index
        else:
            raise ValueError(f"Invalid index_type: {index_type}")


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
