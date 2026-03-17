from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import from_numpy
from torch.nn.functional import one_hot

import dynopsi.data.constants as constants
from dynopsi.data.constants import AtomFilter
from dynopsi.data.structure import Structure

#######################################
### Classes + Definitions of Tokens ###
#######################################


@dataclass(frozen=True)
class TokenFeatures:
    """Represents token features."""

    def asdict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class OneHotSequenceTokenFeatures(TokenFeatures):
    """
    Represents one-hot encoded sequence token features.
    res_type: np.ndarray, shape [num_residues]. The integer index of the residue type, see constants.TOKENS for the mapping.
    res_encode: torch.Tensor, shape [num_residues, 33]. The one-hot encoded residue type.
    """

    res_type: torch.Tensor
    res_encode: torch.Tensor


@dataclass(frozen=True)
class StructureTokenFeatures(TokenFeatures):
    """
    Represents the tokenized version of a structure.
    Relevant for Ca, backbone, and all-atom tokenization.
    NOTE: Not padded.

    Attributes
    ----------
    # index and atom-to-token features
    token_index: torch.Tensor, ints, [num_residues]. The 0-indexed token index for the residue, which includes pad tokens.
    residue_index: torch.Tensor, ints, [num_residues]. The 0-indexed residue index for the residue, which does not include pad tokens.
    auth_residue_index: torch.Tensor, ints, [num_residues]. The residue index as defined by the mmCif input file. This may contain gaps.

    # chain id features
    asym_id: torch.Tensor, ints, [num_residues]
    auth_asym_id: torch.Tensor, ints, [num_residues]
    entity_id: torch.Tensor, ints, [num_residues]
    sym_id: torch.Tensor, ints, [num_residues]

    # mask features
    token_resolved_mask: torch.Tensor, torch.float32, size [num_residues]
       - 0.0 for all positions pad, UNK, and missing residues
       - 1.0 if the Ca atom (="center atom") is present in the residue

    # padding
    token_pad_mask: torch.Tensor, torch.float32, size [num_residues]
        - 0.0 for pad tokens, 1.0 otherwise

    # coordinates
    coords: torch.Tensor # floats, size [1, num_atoms, 3]. The x, y, z coordinates of the atoms.
    """

    token_index: torch.Tensor
    residue_index: torch.Tensor
    auth_residue_index: torch.Tensor

    asym_id: torch.Tensor
    auth_asym_id: torch.Tensor
    entity_id: torch.Tensor
    sym_id: torch.Tensor

    token_resolved_mask: torch.Tensor

    token_pad_mask: torch.Tensor

    coords: torch.Tensor


@dataclass(frozen=True)
class FrameStructureTokenFeatures(StructureTokenFeatures):
    """
    Represents the tokenized version of a structure that includes residue frames.
    Relevant for backbone only and all-atom tokenization.

    frames_idx: torch.Tensor, ints, size [num_residues, 3]
       - atom indices, grouped by frame (residue)
    atom_to_token: torch.Tensor, torch.int64, size [num_atoms, num_residues] --> one-hot encoded token index
       - 1 if the atom is in the token, 0 otherwise

    frame_resolved_mask: torch.Tensor, boolean tensor of shape [num_residues]
       - False for all positions pad, UNK, and missing residues
       - True only if all backbone atoms for the residue are present
    atom_resolved_mask: torch.Tensor, boolean tensor of shape [num_atoms] indicating which atoms are resolved
       - False for pad, UNK, and missing residues

    atom_pad_mask: torch.Tensor, torch.float32, size [num_atoms]
       - 0.0 for pad, UNK, and missing residues
       - 1.0 for resolved atoms
    """

    frames_idx: torch.Tensor
    atom_to_token: torch.Tensor

    frame_resolved_mask: torch.Tensor
    atom_resolved_mask: torch.Tensor

    atom_pad_mask: torch.Tensor


##################
### Tokenizers ###
##################


class Tokenizer(ABC):
    """Base class for tokenizers."""

    @abstractmethod
    def tokenize(self, structure: Structure) -> tuple[TokenFeatures, ...]:
        """Tokenize a structure and return TokenFeatures."""
        raise NotImplementedError


class DynoPsiTokenizer(Tokenizer):
    def __init__(self, atom_filter: AtomFilter) -> None:
        """
        Attributes
        ----------
            atom_filter: AtomFilter, an Enum that defines which atoms will be extracted during tokenization
            - FULL: all atoms --> structure component will return a FrameStructureTokenFeatures object
            - BB: backbone atoms (N, CA, C) --> structure component will return a FrameStructureTokenFeatures object
            - CA: CA atoms only --> structure component will return a StructureTokenFeatures object
        """
        super().__init__()
        self.atom_filter = atom_filter

    def tokenize(self, structure: Structure) -> tuple[TokenFeatures, ...]:
        """
        Tokenize a structure and return a list of TokenFeatures.
        Currently broken down into structure and sequence token features in that order, but order should not matter or be assumed.

        Arguments:
            structure: Structure, the structure to tokenize

        Returns
        -------
            list[TokenFeatures], a list of TokenFeatures objects
        """
        # per-residue chain id features
        asym_id = torch.cat([torch.full((chain["num_residues"],), chain["asym_id"]) for chain in structure.chains])
        auth_asym_id = torch.cat(
            [torch.full((chain["num_residues"],), chain["auth_asym_id"]) for chain in structure.chains]
        )
        entity_id = torch.cat([torch.full((chain["num_residues"],), chain["entity_id"]) for chain in structure.chains])
        sym_id = torch.cat([torch.full((chain["num_residues"],), chain["sym_id"]) for chain in structure.chains])

        # per-residue core features
        token_index = torch.arange(len(structure.residues), dtype=torch.long)
        residue_index = from_numpy(structure.residues["res_idx"].copy()).long()
        auth_residue_index = from_numpy(structure.residues["auth_res_idx"].copy()).long()

        # masks
        token_resolved_mask = from_numpy(structure.residues["ca_present"].copy()).to(torch.float32)
        # padding mask (initialized to all True)
        # should we do this here or in a separate cropping + padding step?
        token_pad_mask = torch.full((len(structure.residues),), 1.0, dtype=torch.float32)

        # per-residue sequence type features
        res_type = from_numpy(structure.residues["res_type"].copy()).long()
        res_encode = one_hot(res_type, num_classes=len(constants.TOKENS))

        # atom index features require filtering to the appropriate atoms first
        include_atom_mask = np.in1d(structure.atoms["atom_name"], np.array(self.atom_filter.to_atom_names()))
        structure_atoms = structure.atoms[include_atom_mask]

        # coordinates
        coords = from_numpy(structure_atoms["coords"].copy()).unsqueeze(0)

        # compute frames and atom features
        if self.atom_filter == AtomFilter.CA:
            structure_token_features = StructureTokenFeatures(
                token_index=token_index,
                residue_index=residue_index,
                auth_residue_index=auth_residue_index,
                asym_id=asym_id,
                auth_asym_id=auth_asym_id,
                entity_id=entity_id,
                sym_id=sym_id,
                token_resolved_mask=token_resolved_mask,
                token_pad_mask=token_pad_mask,
                coords=coords,
            )

        elif self.atom_filter == AtomFilter.BB:
            frames_idx = torch.arange(len(structure_atoms)).reshape(-1, 3)
            atom_to_token = torch.eye(len(structure.residues)).repeat_interleave(3, dim=0).long()
            # masks
            frame_resolved_mask = from_numpy(structure.residues["backbone_present"].copy()).bool()
            atom_resolved_mask = from_numpy(structure_atoms["is_present"].copy()).bool()

            # padding mask (initialized to all True)
            # should we do this here or in a separate cropping + padding step?
            atom_pad_mask = torch.full((len(structure_atoms),), 1.0, dtype=torch.float32)

            structure_token_features = FrameStructureTokenFeatures(
                token_index=token_index,
                residue_index=residue_index,
                auth_residue_index=auth_residue_index,
                asym_id=asym_id,
                auth_asym_id=auth_asym_id,
                entity_id=entity_id,
                sym_id=sym_id,
                token_resolved_mask=token_resolved_mask,
                token_pad_mask=token_pad_mask,
                coords=coords,
                frames_idx=frames_idx,
                atom_to_token=atom_to_token,
                frame_resolved_mask=frame_resolved_mask,
                atom_resolved_mask=atom_resolved_mask,
                atom_pad_mask=atom_pad_mask,
            )

        elif self.atom_filter == AtomFilter.FULL:
            raise NotImplementedError("Full atom filter not implemented yet")

        else:
            raise ValueError(f"Invalid atom filter: {self.atom_filter}")

        # get sequence features
        sequence_token_features = OneHotSequenceTokenFeatures(res_type=res_type, res_encode=res_encode)

        return (structure_token_features, sequence_token_features)
