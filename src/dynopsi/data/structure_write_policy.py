"""Write-time structure contracts.

PDB writes require unique ``[A-Z]`` auth chain ids per model. With ``auth_res_idx``,
``(auth_chain_id, auth_res_idx)`` pairs must be unique among written residues.

mmCIF writes build one polymer entity table. Chains that share an entity must
share one residue-name sequence.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from dynopsi.data.auth_chain_identity import is_unique_single_letter_auth_id
from dynopsi.data.structure import Structure

PrimaryResidueIndexKey = Literal["res_idx", "auth_res_idx"]


@dataclass(frozen=True)
class PolymerEntity:
    """One polymer entity and the sequence shared by its chains."""

    entity_id: int
    subchains: tuple[str, ...]
    full_sequence: tuple[str, ...]


def polymer_entities(structure: Structure) -> list[PolymerEntity]:
    """Build the entity table once. Same-entity chains must agree on sequence."""
    records: list[PolymerEntity] = []
    seen: set[int] = set()
    for chain in structure.chains:
        entity_id = int(chain["entity_id"])
        if entity_id in seen:
            continue
        seen.add(entity_id)
        rows = structure.chains[structure.chains["entity_id"] == entity_id]
        sequences: list[tuple[str, ...]] = []
        subchains: list[str] = []
        for row in rows:
            residues = structure.chain_residues(row)
            sequences.append(tuple(str(residue["res_name"]) for residue in residues))
            subchains.append(str(row["chain_id"]))
        if any(sequence != sequences[0] for sequence in sequences):
            raise ValueError(f"Chains {subchains} share entity {entity_id} but do not have the same sequence.")
        records.append(PolymerEntity(entity_id, tuple(subchains), sequences[0]))
    return records


def persisted_auth_chain_ids(structures: Sequence[Structure]) -> tuple[str, ...]:
    return tuple(str(chain["auth_chain_id"]) for structure in structures for chain in structure.chains)


def _written_auth_seqid_pairs(structure: Structure) -> list[tuple[str, int]]:
    """``(auth_chain_id, auth_res_idx)`` pairs the PDB writer would emit."""
    pairs: list[tuple[str, int]] = []
    for chain in structure.chains:
        residues = structure.chain_residues(chain)
        auth_chain_id = str(chain["auth_chain_id"])
        for residue in residues:
            if not residue["is_present"]:
                continue
            pairs.append((auth_chain_id, int(residue["auth_res_idx"])))
    return pairs


def enforce_pdb_write_policy(
    structures: Sequence[Structure],
    primary_residue_index_key: PrimaryResidueIndexKey = "res_idx",
) -> None:
    """Allow the same [A-Z] letter across structures in one save; insertion codes are unused."""
    auth_ids = persisted_auth_chain_ids(structures)
    illegal = tuple(dict.fromkeys(auth_id for auth_id in auth_ids if not is_unique_single_letter_auth_id(auth_id)))
    if illegal:
        raise ValueError(
            "Cannot write PDB: every persisted auth_chain_id must be a unique [A-Z] letter; "
            f"got {list(illegal)}. PDB write never aliases, truncates, or remaps chain IDs."
        )

    for structure in structures:
        seen_auth_ids: set[str] = set()
        repeated: list[str] = []
        for auth_id in persisted_auth_chain_ids([structure]):
            if auth_id in seen_auth_ids:
                if auth_id not in repeated:
                    repeated.append(auth_id)
            else:
                seen_auth_ids.add(auth_id)
        if repeated:
            raise ValueError(
                "Cannot write PDB: persisted auth_chain_id strings must be unique among themselves; "
                f"repeated {repeated}. PDB write never aliases, truncates, or remaps chain IDs."
            )

    if primary_residue_index_key != "auth_res_idx":
        return

    for structure in structures:
        seen: set[tuple[str, int]] = set()
        duplicates: list[tuple[str, int]] = []
        for pair in _written_auth_seqid_pairs(structure):
            if pair in seen:
                if pair not in duplicates:
                    duplicates.append(pair)
            else:
                seen.add(pair)
        if duplicates:
            raise ValueError(
                "Cannot write PDB: primary_residue_index_key='auth_res_idx' requires unique "
                f"(auth_chain_id, auth_res_idx) among written residues; repeated {duplicates}."
            )
