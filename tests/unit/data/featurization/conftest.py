from typing import Any

import numpy as np
import pytest
import torch

from dynopsi.data.constants import TOKEN_IDS, InpaintFeature, PocketFeature
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Structure,
)
from dynopsi.data.tokenizer import FrameStructureTokenFeatures
from dynopsi.data.utils import ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP


@pytest.fixture
def features_A() -> dict[str, torch.Tensor]:
    atom_to_token = torch.cat(
        [torch.cat([torch.eye(7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long)], dim=0), torch.zeros(8, 1)],
        dim=1,
    ).repeat_interleave(3, dim=0)
    return {
        "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "auth_asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "entity_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 0]),
        "sym_id": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 0], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long),
        "auth_residue_index": torch.tensor([5, 6, 7, 8, 2, 3, 4, 0], dtype=torch.long),
        # atom features
        "coords": torch.cat([torch.zeros(4 * 3, 3), torch.ones(3 * 3, 3), torch.zeros(1 * 3, 3)]).unsqueeze(
            0
        ),  # 0's for asym_id 0, 1's for asym_id 1
        "atom_resolved_mask": torch.tensor([True] * 5 * 3 + [False] * 2 * 3 + [False] * 1 * 3),
        "atom_pad_mask": torch.tensor([True] * 7 * 3 + [False] * 1 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 7 + [False]),
        "token_resolved_mask": torch.tensor([True] * 5 + [False] * 2 + [False]),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 4 + [PocketFeature.POCKET] * 3 + [PocketFeature.UNSPECIFIED]
        ),
        "frame_resolved_mask": torch.tensor([True] * 5 + [False] * 2 + [False]),
        "noised_mask": torch.tensor([False] * 5 + [True] * 2 + [False]),
        "frames_idx": torch.cat([torch.arange(7 * 3).reshape(7, 3), torch.zeros(1, 3)]).to(torch.long),
        "inpaint_feature": torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 4 + [InpaintFeature.TARGET_CONTEXT] * 3 + [0]
        ),
        "binder_mask": torch.tensor([True] * 4 + [False] * 3 + [False]),
        "target_mask": torch.tensor([False] * 4 + [True] * 3 + [False]),
        # atom to token
        "atom_to_token": atom_to_token,
        "res_type": torch.tensor([1, 22, 3, 5, 6, 7, 8, 0]),
    }


@pytest.fixture
def features_B() -> dict[str, torch.Tensor]:
    return {
        "asym_id": torch.tensor([1, 1, 1]),
        "auth_asym_id": torch.tensor([21, 21, 21]),
        "entity_id": torch.tensor([1, 1, 1]),
        "sym_id": torch.tensor([0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "auth_residue_index": torch.tensor([13, 14, 15], dtype=torch.long),
        # atom features
        "coords": torch.ones(3 * 3, 3).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 3 * 3),
        "atom_pad_mask": torch.tensor([True] * 3 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 3),
        "token_resolved_mask": torch.tensor([True] * 3),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 3),
        "frame_resolved_mask": torch.tensor([True] * 3),
        "noised_mask": torch.tensor([False] * 3),
        "frames_idx": torch.arange(3 * 3).reshape(3, 3),
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT] * 3),
        "binder_mask": torch.tensor([False] * 3),
        "target_mask": torch.tensor([True] * 3),
        # atom to token
        "atom_to_token": torch.eye(3, dtype=torch.long).repeat_interleave(3, dim=0),
        "res_type": torch.tensor([6, 2, 8]),
    }


@pytest.fixture
def features_C() -> dict[str, torch.Tensor]:
    return {
        "asym_id": torch.tensor([1, 1, 1, 1]),
        "auth_asym_id": torch.tensor([21, 21, 21, 21]),
        "entity_id": torch.tensor([0, 0, 0, 0]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "auth_residue_index": torch.tensor([13, 14, 15, 16], dtype=torch.long),
        # atom features
        "coords": torch.ones(4 * 3, 3).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 4 * 3),
        "atom_pad_mask": torch.tensor([True] * 4 * 3),
        # other token features
        "token_pad_mask": torch.tensor([True] * 4),
        "token_resolved_mask": torch.tensor([True] * 4),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 4),
        "frame_resolved_mask": torch.tensor([True] * 4),
        "noised_mask": torch.tensor([False] * 4),
        "frames_idx": torch.arange(4 * 3).reshape(4, 3),
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT] * 4),
        "binder_mask": torch.tensor([False] * 4),
        "target_mask": torch.tensor([True] * 4),
        # atom to token
        "atom_to_token": torch.eye(4, dtype=torch.long).repeat_interleave(3, dim=0),
    }


@pytest.fixture
def features_all_design() -> dict[str, torch.Tensor]:
    """Features with all design residues."""
    atom_to_token = torch.cat(
        [torch.cat([torch.eye(7, dtype=torch.long), torch.zeros(1, 7, dtype=torch.long)], dim=0), torch.zeros(8, 1)],
        dim=1,
    ).repeat_interleave(3, dim=0)
    return {
        "asym_id": torch.zeros(8),
        "auth_asym_id": torch.zeros(8),
        "entity_id": torch.zeros(8),
        "sym_id": torch.zeros(8),
        "residue_index": torch.arange(8, dtype=torch.long),
        "token_index": torch.arange(8, dtype=torch.long),
        "auth_residue_index": torch.arange(8, dtype=torch.long),
        # atom features
        # residues 0-5 and part of 6 (N, CA) resolved and have coordinates 5.5 -> 6 * 3 - 1 atoms resolved
        # residue 6 unresolved atom (C) and residue 7 is fully unresolved have coordinates 6.6 -> 3 + 1 atoms resolved
        # pad tokens have coordinates 0.0 -> 3 atoms with 0.0
        "coords": torch.cat([torch.ones(6 * 3 - 1, 3) * 5.5, torch.ones(3 + 1, 3) * 6.6, torch.zeros(3, 3)]).unsqueeze(
            0
        ),
        "atom_resolved_mask": torch.cat([torch.ones(6 * 3 - 1), torch.zeros(2 * 3 + 1)]),
        "atom_pad_mask": torch.cat([torch.ones(7 * 3), torch.zeros(1 * 3)]),
        # other token features
        "token_pad_mask": torch.cat([torch.ones(7), torch.zeros(1)]),
        "token_resolved_mask": torch.cat([torch.ones(6), torch.zeros(1), torch.zeros(1)]),
        "pocket_feature": torch.tensor([PocketFeature.UNSPECIFIED] * 8),
        "frame_resolved_mask": torch.cat([torch.ones(5), torch.zeros(3)]),
        "noised_mask": torch.cat([torch.ones(7, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]),
        "frames_idx": torch.cat([torch.arange(7 * 3).reshape(7, 3), torch.zeros(1, 3)]),
        "inpaint_feature": torch.tensor([InpaintFeature.BINDER_INPAINT] * 7 + [0]),
        "binder_mask": torch.tensor([True] * 7 + [False]),
        "target_mask": torch.zeros(8, dtype=torch.bool),
        # atom to token
        "atom_to_token": atom_to_token,
    }


@pytest.fixture
def features_all_inpaint_options() -> dict[str, torch.Tensor]:
    """Features with a mix of context and design residues.
    - all resolved
    - all inpaint options
    - coordinates are different for each inpaint option
    - noise is split across inpaint options
    - single pad token
    """
    atom_to_token = torch.cat(
        [
            torch.cat([torch.eye(8, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long)], dim=0),
            torch.zeros(9, 1, dtype=torch.long),
        ],
        dim=1,
    ).repeat_interleave(3, dim=0)
    return {
        "asym_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.long),
        "auth_asym_id": torch.tensor([20, 20, 20, 20, 21, 21, 21, 21, 0], dtype=torch.long),
        "entity_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.long),
        "sym_id": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long),
        "residue_index": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long),
        "token_index": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long),
        "auth_residue_index": torch.tensor([5, 6, 7, 8, 2, 3, 4, 5, 0], dtype=torch.long),
        # atom features
        "coords": torch.cat(
            [
                torch.ones(2 * 3, 3) * 1.5,  # binder context
                torch.ones(2 * 3, 3) * 3.3,  # binder inpaint
                torch.ones(2 * 3, 3) * 4.4,  # target context
                torch.ones(2 * 3, 3) * 6.1,  # target inpaint
                torch.zeros(1 * 3, 3),  # pad
            ]
        ).unsqueeze(0),
        "atom_resolved_mask": torch.tensor([True] * 8 * 3 + [False] * 1 * 3, dtype=torch.bool),
        "atom_pad_mask": torch.tensor([True] * 8 * 3 + [False] * 1 * 3, dtype=torch.bool),
        # other token features
        "token_pad_mask": torch.tensor([True] * 8 + [False], dtype=torch.bool),
        "token_resolved_mask": torch.tensor([True] * 8 + [False], dtype=torch.bool),
        "pocket_feature": torch.tensor(
            [PocketFeature.BINDER] * 4
            + [PocketFeature.UNSELECTED, PocketFeature.POCKET, PocketFeature.UNSELECTED, PocketFeature.POCKET]
            + [PocketFeature.UNSPECIFIED],
            dtype=torch.long,
        ),
        "frame_resolved_mask": torch.tensor([True] * 8 + [False]),
        "noised_mask": torch.tensor([True, False] * 4 + [False]),  # alternating to cover all inpaint options
        "frames_idx": torch.cat(
            [torch.arange(8 * 3, dtype=torch.long).reshape(8, 3), torch.zeros(1, 3, dtype=torch.long)]
        ),
        "inpaint_feature": torch.tensor(
            [InpaintFeature.BINDER_CONTEXT] * 2
            + [InpaintFeature.BINDER_INPAINT] * 2
            + [InpaintFeature.TARGET_CONTEXT] * 2
            + [InpaintFeature.TARGET_INPAINT] * 2
            + [0],
            dtype=torch.long,
        ),
        "binder_mask": torch.tensor([True] * 4 + [False] * 4 + [False], dtype=torch.bool),
        "target_mask": torch.tensor([False] * 4 + [True] * 4 + [False], dtype=torch.bool),
        # atom to token
        "atom_to_token": atom_to_token,
    }


### Reference structure fixtures ###
@pytest.fixture
def base_struct_info_D() -> dict[str, Any]:
    return {
        "asym_id": torch.tensor([3, 3, 1, 4]),
        "auth_asym_id": torch.tensor([23, 23, 21, 24]),
        "entity_id": torch.tensor([0, 0, 1, 2]),
        "sym_id": torch.tensor([0, 0, 0, 0]),
        "residue_index": torch.tensor([0, 1, 0, 0]),
        "auth_residue_index": torch.tensor([12, 13, 5, 8]),
        "coords": torch.randn(4 * 3, 3),  # 3 atoms per residue
        "atom_resolved_mask": torch.ones(4 * 3, dtype=torch.bool),
        "token_resolved_mask": torch.ones(4, dtype=torch.bool),
        "resname": np.array(["ALA", "LEU", "PHE", "MET"]),
    }


@pytest.fixture
def frame_structure_token_features_D(base_struct_info_D: dict[str, Any]) -> FrameStructureTokenFeatures:
    return FrameStructureTokenFeatures(
        **{
            "token_index": torch.arange(len(base_struct_info_D["asym_id"])),
            "residue_index": base_struct_info_D["residue_index"],
            "auth_residue_index": base_struct_info_D["auth_residue_index"],
            "asym_id": base_struct_info_D["asym_id"],
            "auth_asym_id": base_struct_info_D["auth_asym_id"],
            "entity_id": base_struct_info_D["entity_id"],
            "sym_id": base_struct_info_D["sym_id"],
            "token_resolved_mask": base_struct_info_D["token_resolved_mask"],
            "token_pad_mask": torch.ones(4, dtype=torch.float),
            "coords": base_struct_info_D["coords"].unsqueeze(0),
            "frames_idx": torch.arange(4 * 3).reshape(-1, 3),
            "atom_to_token": torch.repeat_interleave(torch.eye(4), 3, dim=0),
            "frame_resolved_mask": base_struct_info_D["token_resolved_mask"],
            "atom_resolved_mask": base_struct_info_D["atom_resolved_mask"],
            "atom_pad_mask": torch.ones_like(base_struct_info_D["atom_resolved_mask"], dtype=torch.float),
        }
    )


@pytest.fixture
def structure_D(base_struct_info_D: dict[str, Any]) -> Structure:

    atoms = np.empty(
        len(base_struct_info_D["atom_resolved_mask"]),
        dtype=ATOMS_DTYPE,
    )
    atoms["asym_id"] = np.repeat(base_struct_info_D["asym_id"].numpy(), 3)
    atoms["res_idx"] = np.repeat(base_struct_info_D["residue_index"].numpy(), 3)
    atoms["atom_name"] = np.repeat(["N", "CA", "C"], len(base_struct_info_D["asym_id"]))
    atoms["coords"] = base_struct_info_D["coords"].numpy()
    atoms["is_present"] = np.full(len(base_struct_info_D["atom_resolved_mask"]), True)
    atoms["bfactor"] = np.full(len(base_struct_info_D["atom_resolved_mask"]), 100.0)
    atoms["plddt"] = np.full(len(base_struct_info_D["atom_resolved_mask"]), np.nan)

    residues = np.empty(
        len(base_struct_info_D["residue_index"]),
        dtype=RESIDUES_DTYPE,
    )

    residues["res_name"] = base_struct_info_D["resname"].copy()
    residues["res_type"] = [TOKEN_IDS[resname] for resname in base_struct_info_D["resname"].copy()]
    residues["asym_id"] = base_struct_info_D["asym_id"].numpy()
    residues["res_idx"] = base_struct_info_D["residue_index"].numpy()
    residues["auth_res_idx"] = base_struct_info_D["auth_residue_index"].numpy()
    residues["atom_start_idx"] = np.arange(len(base_struct_info_D["asym_id"])) * 3
    residues["num_atoms"] = np.full(len(base_struct_info_D["residue_index"]), 3)
    residues["atom_center"] = residues["atom_start_idx"] + 1
    residues["is_standard"] = np.full(len(base_struct_info_D["residue_index"]), True)
    residues["is_present"] = np.full(len(base_struct_info_D["residue_index"]), True)
    residues["ca_present"] = np.full(len(base_struct_info_D["residue_index"]), True)
    residues["backbone_present"] = np.full(len(base_struct_info_D["residue_index"]), True)

    num_asym_ids = len(base_struct_info_D["asym_id"].unique())
    chains = np.empty(
        num_asym_ids,
        dtype=CHAINS_DTYPE,
    )
    ordered_asym_ids = base_struct_info_D["asym_id"][base_struct_info_D["residue_index"] == 0].tolist()
    ordered_auth_asym_ids = base_struct_info_D["auth_asym_id"][base_struct_info_D["residue_index"] == 0].tolist()
    chains["chain_id"] = np.array([ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[asym_id] for asym_id in ordered_asym_ids])
    chains["auth_chain_id"] = np.array(
        [ASYM_ID_TO_SINGLE_LETTER_CHAIN_MAP[auth_asym_id] for auth_asym_id in ordered_auth_asym_ids]
    )
    chains["asym_id"] = np.array(ordered_asym_ids)
    chains["auth_asym_id"] = np.array(ordered_auth_asym_ids)
    chains["entity_id"] = base_struct_info_D["entity_id"][base_struct_info_D["residue_index"] == 0].numpy()
    chains["sym_id"] = base_struct_info_D["sym_id"][base_struct_info_D["residue_index"] == 0].numpy()
    _, idx, counts = np.unique(base_struct_info_D["asym_id"].numpy(), return_index=True, return_counts=True)
    chains["res_start_idx"] = np.sort(idx)
    chains["num_residues"] = counts[np.argsort(idx)]
    chains["atom_start_idx"] = chains["res_start_idx"] * 3
    chains["num_atoms"] = chains["num_residues"] * 3

    return Structure(
        atoms=atoms,
        residues=residues,
        chains=chains,
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )
