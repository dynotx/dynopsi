import logging
import shutil
import string
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np
import requests
import smart_open
from pdbeccdutils.core import ccd_reader
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from scipy.spatial import cKDTree

import dynopsi.data.constants as constants
from dynopsi.data.paths import AA_CCD_CIF_FILE
from dynopsi.data.structure import (
    ATOMS_DTYPE,
    CHAIN_INTERFACE_DTYPE,
    CHAINS_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    RESIDUES_DTYPE,
    Chain,
    Structure,
)
from dynopsi.data.utils import get_asym_id_from_chain_id

logger = logging.getLogger(__name__)


## Loading Structures
def load_or_parse_ccd(ccd_path: Path | str = AA_CCD_CIF_FILE) -> dict[str, Mol]:
    """Load the CCD dictionary of residue name to rdkit.Chem.rdchem.Mol."""
    components = ccd_reader.read_pdb_components_file(str(AA_CCD_CIF_FILE))
    ccd: dict[str, Mol] = {}
    for name, component in components.items():
        mol = component.component.mol
        mol.SetProp("PDB_NAME", name)
        ccd[name] = mol

    return ccd


def load_structure_from_cif(
    cif_path: Path | str, atomic_interface_threshold: float = 10.0, compute_residue_interfaces: bool = True
) -> Structure:
    """
    Load a Structure from a CIF file.

    Arguments:
    ----------
    cif_path : Path
        The path to the CIF file.
    atomic_interface_threshold: float, default 10.0
        The threshold for the atom-atom distance for a pair of atoms to be considered at an interface (all sidechains are considered).
    compute_residue_interfaces: bool, default True
        Whether to compute the residues at the interface or just the chains at the interface. See find_chain_interfaces for more details.

    Returns
    -------
    Structure
        The loaded structure.
    """
    ccd = load_or_parse_ccd()
    return parse_mmcif(
        cif_path,
        ccd,
        atomic_interface_threshold=atomic_interface_threshold,
        compute_residue_interfaces=compute_residue_interfaces,
    )


def load_structure_from_pdb(
    pdb_path: Path | str,
    chain_id_to_sequence: dict[str, str] | None = None,
    atomic_interface_threshold: float = 10.0,
    compute_residue_interfaces: bool = True,
) -> Structure:
    """
    Load a Structure from a PDB file.

    Arguments:
    ----------
    pdb_path : Path
        The path to the PDB file.
    cif_path : Path | None
        The path to the mmCif file intermediate. If None, then the mmCif file will be saved to a temporary file
        and then deleted after the structure is loaded.
    chain_id_to_sequence: dict[str, str] | None, default None
        A dictionary mapping chain ids to sequences. These sequences should be ungapped.
        If provided, then chains with the same sequence will be assigned to the same entity. It is recommended that
        you check the cif output file to ensure that the sequences and positions are correct if you provide this argument.
        If None, then the sequences will be inferred from the residues and each chain will be assigned a separate entity.
    atomic_interface_threshold: float, default 10.0
        The threshold for the atom-atom distance for a pair of atoms to be considered at an interface (all sidechains are considered).
    compute_residue_interfaces: bool, default True
        Whether to compute the residues at the interface or just the chains at the interface. See find_chain_interfaces for more details.

    Returns
    -------
    Structure
        The loaded structure.
    """
    ccd = load_or_parse_ccd()
    return parse_pdb(
        pdb_path,
        ccd,
        chain_id_to_sequence=chain_id_to_sequence,
        atomic_interface_threshold=atomic_interface_threshold,
        compute_residue_interfaces=compute_residue_interfaces,
    )


def load_structure_from_rcsb(
    record_id: str,
    target_dir: str | Path | None = None,
    atomic_interface_threshold: float = 10.0,
    compute_residue_interfaces: bool = True,
) -> Structure:
    """
    Load the structure from RCSB PDB.

    Parameters
    ----------
    record_id : str
        The record id of the structure to load.
    target_dir : str | Path | None
        The directory to save the mmcif file to. If None, then the mmcif file
        will be downloaded to a temporary directory and then deleted after the structure is loaded.

    Returns
    -------
    Structure
        The loaded structure.
    """
    if target_dir is None:
        cache_dir = tempfile.mkdtemp()
    else:
        cache_dir = str(target_dir)

    try:
        file_path = download_mmcif_from_rcsb(record_id, cache_dir)
        structure = load_structure_from_cif(
            file_path,
            atomic_interface_threshold=atomic_interface_threshold,
            compute_residue_interfaces=compute_residue_interfaces,
        )
        if target_dir is None:
            shutil.rmtree(cache_dir)
        return structure
    except Exception as e:
        if target_dir is None:
            shutil.rmtree(cache_dir)
        raise e


def load_structure_from_afdb(
    uniprot_id: str,
    target_dir: str | Path | None = None,
    atomic_interface_threshold: float = 10.0,
    compute_residue_interfaces: bool = True,
) -> Structure:
    """
    Load the structure from AFDB.

    Parameters
    ----------
    uniprot_id : str
        The uniprot id of the structure to load.
    target_dir : str | Path | None
        The directory to save the mmcif file to. If None, then the mmcif file
        will be downloaded to a temporary directory and then deleted after the structure is loaded.

    Returns
    -------
    Structure
        The loaded structure.
    """
    if target_dir is None:
        cache_dir = tempfile.mkdtemp()
    else:
        cache_dir = str(target_dir)

    try:
        file_path = download_mmcif_from_afdb(uniprot_id, cache_dir)
        structure = load_structure_from_cif(
            file_path,
            atomic_interface_threshold=atomic_interface_threshold,
            compute_residue_interfaces=compute_residue_interfaces,
        )
        if target_dir is None:
            shutil.rmtree(cache_dir)
        return structure
    except Exception as e:
        if target_dir is None:
            shutil.rmtree(cache_dir)
        raise e


#####################################
##   Parsing PDB and mmCif Files   ##
#####################################


def parse_chain_to_residues(
    chain: gemmi.ResidueSpan,
    ungapped_sequence: list[str],
    ccd: dict[str, Mol],
    asym_id: int,
    chain_atom_start_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a protein chain.

    Arguments:
    ----------
    chain: gemmi.ResidueSpan, the chain to parse. This chain will only contain residues that exist in the ATOM loop_
    ungapped_sequence: list[str], a list of the three-letter residue codes of the expected sequence of the chain.
        This sequence should contain all residues that are present in the N-C crop
        of the chain and should be ungapped. This sequence can come from the entity.full_sequence field
        (which comes from the _entity_poly_seq field in the mmCif); however, there is no requirement that this field
        be ungapped relative to the sequence that can be read from the _atom_site fields in the mmCif.
    ccd : dict[str, Mol], a dictionary mapping residue names to the corresponding rdkit.Chem.rdchem.Mol molecule.
    asym_id: int, the asym_id of the chain.
    chain_atom_start_idx: int, the starting index of the atoms in the chain (includes all expected atoms, including
        unresolved atoms, based on residue identity).

    Returns
    -------
    list[Residue], one for each residue in the chain, including gaps. Residues that are
    expected but not present in the chain (gaps) will have a Residue object with all expected atoms,
    but all atoms will be set to is_present=False, bfactor=0, and coordinates=(0, 0, 0).

    If the chain contains nonstandard amino acids that are not allowed by dynopsi, then the chain will
    be skipped and a warning will be logged.

    Raises
    ------
    - ValueError if an alignment between the sequence defined in the entity section of the mmCif file and
    the sequence derived from the atoms that are present in the chain. We expect the alignment to be perfect,
    or for the atom-derived sequence to contain gaps relative to the expected sequence. If neither of these are true,
    then a ValueError will be raised.
    - ValueError if the alignment produces a mismatch between the expected residue name and the observed residue name.
    This secondary check is important because the alignment strategy at present tends to be a bit

    NOTE: author residue index parsing:
    - if the residue is present in the chain, it will receive exactly the author's residue index.
    - if the residue is NOT present in the chain, it will be assigned to the last present author's residue index + 1.
    - for the special case of gaps at the beginning of the chain, the author's residue index will be set to 1.
    The reason for doing this is to avoid collisions between inferred auth_res_idx and actual author auth_res_idx in
    the case of missing residues, while preserving the actual author auth_res_idx for the present residues.
    We expect that for any featurization down the line, any desired range will be a contiguous block of auth_res_idx,
    which will automatically include the missing residues if they are consecutive with respect to the author's residue index.
    Any specific positions (like hotspots) would be
    specified on existing auth_res_idx.
    - example:
        - expected sequence: A, B, C, D, E
        - observed residues: -, B, -, -, E
        - observed auth_res_idx: -, 5, -, -, 8
        - parsed auth_res_idx: 1, 5, 6, 6, 8


    NOTE: There is some ambiguity with the sequence alignment strategy, so we double check that the
    residue observed in the _atom_site fields in the mmCif matches the expected residue name.
    """
    # there are sometimes point mutations in the chain delineated with comma-- like MET,TYR
    # just select the first character
    sequence = [gemmi.Entity.first_mon(res) for res in ungapped_sequence]

    result = gemmi.align_sequence_to_polymer(
        sequence,
        chain,
        gemmi.PolymerType.PeptideL,
        gemmi.AlignmentScoring(),
    )

    # 1. Resolve chain indexes
    #   There are three indices tracked:
    #   - obs_chain_idx: the index of the chain sequence, corresponds to residues with existing atoms
    #   - expect_idx: the index of the residue in the expected sequence, corresponds to to the ungapped sequence position.
    #   - auth_res_idx: the author's residue index, corresponds to the residue number in the mmCif/PDB file.
    # chain_idxs is a tuple of (obs_chain_idx, expect_chain_idx, auth_res_idx)
    # allow auth_res_idx to be nan if the residue is not present in the chain.
    chain_idxs: list[tuple[int | None, int, int]] = []
    obs_chain_idx = 0
    prev_auth_res_idx = 0
    curr_atom_idx = chain_atom_start_idx
    for expect_idx, match in enumerate(result.match_string):
        if match == "|":
            # get the author's residue index
            auth_res_idx = chain[obs_chain_idx].seqid.num
            if auth_res_idx is None:
                logger.warning(
                    f"Author residue index is None for residue {expect_idx}. Assigning {prev_auth_res_idx + 1}."
                )
                auth_res_idx = prev_auth_res_idx + 1
            else:
                auth_res_idx = int(auth_res_idx)
            chain_idxs.append((obs_chain_idx, expect_idx, auth_res_idx))
            obs_chain_idx += 1
            prev_auth_res_idx = auth_res_idx
        elif match == " ":
            # expect " " to indicate a gap in the observed chain relative to the expected sequence
            # the other option is that there's a gap in the expected sequence, but this can only be
            # observed as a mismatch error in a downstream residue. This will be caught below.

            # for setting auth_res_idx, we use the rule that it is set to previous auth_res_idx + 1
            # this means that if there are gaps of length n, the auth_res_idx will be the same for all n residues.
            chain_idxs.append((None, expect_idx, prev_auth_res_idx + 1))
        elif match == ".":
            # mismatch
            raise ValueError(
                f"Mismatch between expected and observed sequence: {sequence[expect_idx]} != {chain[obs_chain_idx].name} at sequence position {expect_idx}"
            )
        else:
            raise ValueError(f"Unexpected match: {match} at position {expect_idx}. Possibly malformed CIF file.")

    # 2. Parse residues
    residues: list[np.ndarray] = []
    atoms_lst: list[np.ndarray] = []

    if len(chain_idxs) != len(sequence):
        raise ValueError(
            f"Number of chain indexes ({len(chain_idxs)}) does not match number of expected residues ({len(sequence)})"
        )

    for obs_idx, expect_idx, auth_res_idx in chain_idxs:
        expected_res_name = sequence[expect_idx]

        if isinstance(obs_idx, int):
            observed_res = chain[obs_idx]

            if observed_res.name != expected_res_name:
                raise ValueError(
                    f"Alignment mismatch: Expected residue {expected_res_name} but got {observed_res.name} at expected sequence position {expect_idx}"
                )

            is_present = True

        else:
            observed_res = None
            is_present = False

        # possibly map MSE to MET
        if expected_res_name == "MSE":
            expected_res_name = "MET"
            is_standard = False
            is_allowed = True
        else:
            is_standard = expected_res_name in constants.STANDARD_RESIDUES
            is_allowed = expected_res_name in constants.STANDARD_RESIDUES

        if not is_allowed:
            logger.warning(
                f"Chain contains nonstandard amino acid {expected_res_name} that is not allowed by dynopsi-v1. Skipping..."
            )
            return np.empty(0, dtype=RESIDUES_DTYPE), np.empty(0, dtype=ATOMS_DTYPE)

        # parse the atoms
        atoms = parse_residue_to_atoms(observed_res, expected_res_name, ccd, asym_id=asym_id, res_idx=expect_idx)
        # check if CA is present
        ca_present = any((atoms["atom_name"] == "CA") & atoms["is_present"])
        # check if all backbone atoms are present
        bb_atoms_present = np.sum(np.isin(atoms["atom_name"], ["N", "CA", "C", "O"]) & atoms["is_present"]) == 4
        # atom center idx
        atom_center_idx = np.where(atoms["atom_name"] == "CA")[0][0]

        # create the residue
        residue = np.array(
            [
                (
                    expected_res_name,  # res_name
                    constants.TOKEN_IDS[expected_res_name],  # res_type
                    asym_id,  # asym_id
                    expect_idx,  # res_idx
                    auth_res_idx,  # auth_res_idx
                    curr_atom_idx,  # atom_start_idx
                    len(atoms),  # num_atoms
                    curr_atom_idx + atom_center_idx,  # atom_center
                    is_standard,  # is_standard
                    is_present,  # is_present
                    ca_present,  # ca_present
                    bb_atoms_present,  # bb_atoms_present
                )
            ],
            dtype=RESIDUES_DTYPE,
        )

        curr_atom_idx += len(atoms)

        residues.append(residue)
        atoms_lst.append(atoms)

    return np.concatenate(residues), np.concatenate(atoms_lst)


def parse_residue_to_atoms(
    observed_residue: gemmi.Residue | None, residue_name: str, ccd: dict[str, Mol], asym_id: int, res_idx: int
) -> np.ndarray:
    """
    Parse a residue to atoms.

    Arguments:
    ----------
    observed_residue: gemmi.Residue | None, the residue to parse. None if the residue is not present and should be treated as a gap.
        residue_name: str, the name of the residue.
    ccd : dict[str, Mol], a dictionary mapping residue names to the corresponding rdkit.Chem.rdchem.Mol molecule.
    asym_id: int, the asym_id of the chain.

    Returns
    -------
    list[Atom], the atoms in the residue. Will include all expected atoms.
    Expected atoms will be ordered as they appear in the CCD entry.
    If an atom is not present in the residue:
    - coordinates will be set to (0, 0, 0)
    - is_present will be set to False
    - bfactor will be set to 0

    Raises
    ------
    ValueError: if the CCD entry is not found for the residue.
    """
    # use the ccd name to find the expected atoms (in order)
    mol = ccd.get(residue_name)
    if mol is None:
        raise ValueError(f"No CCD entry found for residue {residue_name}")

    mol = Chem.RemoveHs(mol, sanitize=False)

    # define the expected atoms and the order in which they should appear
    expected_atoms = constants.REF_ATOM_NAMES[residue_name]

    if observed_residue is None:
        observed_atoms = {}
    else:
        # get observed atom information
        observed_atoms = {a.name.upper(): a for a in observed_residue}
        # possibly map MSE to MET
        if observed_residue.name == "MSE":
            observed_atoms["SD"] = observed_atoms["SE"]
            observed_atoms.pop("SE")

    atoms: list[np.ndarray] = []
    for expect_atom in expected_atoms:
        if expect_atom in observed_atoms.keys():
            a = observed_atoms[expect_atom]
            atoms.append(
                np.array(
                    [
                        (
                            asym_id,  # asym_id
                            res_idx,  # res_idx
                            expect_atom,  # atom_name
                            (a.pos.x, a.pos.y, a.pos.z),  # coords
                            True,  # is_present
                            a.b_iso,  # bfactor
                            np.nan,  # plddt
                        )
                    ],
                    dtype=ATOMS_DTYPE,
                )
            )
        else:
            atoms.append(
                np.array(
                    [
                        (
                            asym_id,  # asym_id
                            res_idx,  # res_idx
                            expect_atom,  # atom_name
                            (0, 0, 0),  # coords
                            False,  # is_present
                            0,  # bfactor
                            np.nan,  # plddt
                        )
                    ],
                    dtype=ATOMS_DTYPE,
                )
            )

    return np.concatenate(atoms)


def find_chain_interfaces(
    atoms: np.ndarray,
    residues: np.ndarray,
    chains: np.ndarray,
    atomic_distance_threshold: float,
    compute_residue_interfaces: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the interfaces from a list of Chain objects.

    Arguments
    ---------
    atoms: np.ndarray of ATOMS_DTYPE, shape (n_atoms, )
    residues: np.ndarray of RESIDUES_DTYPE, shape (n_residues, )
    chains: np.ndarray of CHAINS_DTYPE, shape (n_chains, )
    atomic_distance_threshold: float, the distance threshold for atomic interfaces
    compute_residue_interfaces: bool, whether to compute residue interfaces and chain interfaces; if False, only compute chain interfaces

    Returns
    -------
    chain_interfaces: np.ndarray of CHAIN_INTERFACE_DTYPE, shape (n_chain_interfaces, )
        Only contains the chain pairs that have at least one interface residue.
    residue_interfaces: np.ndarray of RESIDUE_INTERFACE_DTYPE, shape (n_residue_interfaces, )
        Contains all the interface residues for all chain pairs. Will be None if compute_residue_interfaces is False.

    NOTE: This is 30% faster than the boltz implementation for structures with fewer chains.
    It's much slower for structures with many chains, so we might want to return None for interface residues if
    the number of chains is greater than some max_chains_threshold.
    """
    residue_interfaces_lst = []
    if compute_residue_interfaces:
        # build kdtree for each chain
        chain_trees: dict[int, cKDTree] = {}
        for chain in chains:
            chain_atoms = atoms[chain["atom_start_idx"] : (chain["atom_start_idx"] + chain["num_atoms"])]
            chain_presence = chain_atoms["is_present"]
            tree = cKDTree(chain_atoms["coords"][chain_presence])
            chain_trees[int(chain["asym_id"])] = tree

        chain_pairs = list(combinations(chains, 2))
        for chain_1, chain_2 in chain_pairs:
            # get atom arrays for each chain
            atom_1 = atoms[chain_1["atom_start_idx"] : (chain_1["atom_start_idx"] + chain_1["num_atoms"])]
            atom_2 = atoms[chain_2["atom_start_idx"] : (chain_2["atom_start_idx"] + chain_2["num_atoms"])]

            # get the presence arrays for each chain
            presence_1 = atom_1["is_present"]
            presence_2 = atom_2["is_present"]

            # get the atom to residue map for each chain, which is len(atoms) long per chain
            # with values being the residue index for that atom
            residue_1 = residues[chain_1["res_start_idx"] : (chain_1["res_start_idx"] + chain_1["num_residues"])]
            residue_2 = residues[chain_2["res_start_idx"] : (chain_2["res_start_idx"] + chain_2["num_residues"])]
            atom_to_residue_1 = np.repeat(np.arange(len(residue_1)), residue_1["num_atoms"])
            atom_to_residue_2 = np.repeat(np.arange(len(residue_2)), residue_2["num_atoms"])

            # build kdtree and query neighbors with only with present atoms
            tree1 = chain_trees[chain_1["asym_id"]]
            tree2 = chain_trees[chain_2["asym_id"]]
            pairs = tree1.query_ball_tree(tree2, r=atomic_distance_threshold)

            atom_interface_1 = np.repeat(np.arange(len(pairs)), [len(p) for p in pairs]).astype(np.int64)
            atom_interface_2 = np.concatenate(pairs).astype(np.int64)

            # map back to the original atom indices
            presence_idx_1 = np.where(presence_1)[0]
            presence_idx_2 = np.where(presence_2)[0]
            atom_interface_1 = presence_idx_1[atom_interface_1].astype(np.int64)
            atom_interface_2 = presence_idx_2[atom_interface_2].astype(np.int64)

            # map to the residue indices
            residue_interface_1 = atom_to_residue_1[atom_interface_1]
            residue_interface_2 = atom_to_residue_2[atom_interface_2]

            # format for output
            asym_id_1 = np.repeat(chain_1["asym_id"], len(residue_interface_1))
            asym_id_2 = np.repeat(chain_2["asym_id"], len(residue_interface_2))
            residue_interface = np.unique(
                np.stack([asym_id_1, asym_id_2, residue_interface_1, residue_interface_2], axis=1), axis=0
            )
            residue_interfaces_lst.append(residue_interface)

        if len(residue_interfaces_lst) > 0:
            residue_interfaces = np.unique(np.concatenate(residue_interfaces_lst), axis=0)
            chain_interfaces = np.array(
                [tuple(i) for i in np.unique(residue_interfaces[:, :2], axis=0)], dtype=CHAIN_INTERFACE_DTYPE
            )
            residue_interfaces = np.array([tuple(i) for i in residue_interfaces], dtype=RESIDUE_INTERFACE_DTYPE)
        else:
            chain_interfaces = np.empty(0, dtype=CHAIN_INTERFACE_DTYPE)
            residue_interfaces = np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE)

    else:
        # only compute which chains have interfaces
        # it's faster to build a ckdtree for all atoms
        chain_ids = np.concatenate([np.full(chain["num_atoms"], chain["asym_id"]) for chain in chains])
        coords = atoms["coords"]
        mask = atoms["is_present"]
        coords = coords[mask]
        chain_ids = chain_ids[mask]

        # Compute the distance matrix
        tree = cKDTree(coords)
        query = tree.query_ball_tree(tree, r=atomic_distance_threshold)

        # Get unique chain pairs
        interfaces: set[tuple[int, int]] = set()
        for c1, interface_pairs in zip(chain_ids, query):
            chains = np.unique(chain_ids[interface_pairs])
            chains = chains[chains != c1]
            interfaces.update((c1, c2) for c2 in chains)

        # Get unique chain pairs
        interfaces = {(min(i, j), max(i, j)) for i, j in interfaces}

        chain_interfaces = np.array(list(interfaces), dtype=CHAIN_INTERFACE_DTYPE)
        residue_interfaces = np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE)

    return chain_interfaces, residue_interfaces


def parse_structure_from_gemmi(
    structure: gemmi.Structure, ccd: dict[str, Mol]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a gemmi.Structure object, clean it up, infer missing residues, and return a list of Chain objects.

    Parameters
    ----------
    structure : gemmi.Structure
        The structure to parse.
    ccd : dict[str, Mol]
        A dictionary mapping residue names to the corresponding rdkit.Chem.rdchem.Mol molecule.
    """
    # Clean up the structure
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    # Parse by entities and their subchains to get asym_id, entity_id, and sym_id
    # Track the asym_ids to avoid collisions
    # chain_id_to_chain_info[subchain_id] = chain
    chain_id_to_chain_info: dict[str, Chain] = {}
    # define a Chain namedtuple
    for entity_id, entity in enumerate(structure.entities):
        if entity.polymer_type.name != "PeptideL":
            logger.warning(
                f"Entity {entity_id} is not a valid protein chain. It has type {entity.entity_type.name} and polymer type {entity.polymer_type.name}. Skipping..."
            )
            continue

        for sym_id, subchain_id in enumerate(entity.subchains):
            # Parse the asym_id from the subchain_id
            asym_id = get_asym_id_from_chain_id(subchain_id)
            # check for collisions
            # just raise a warning if there is a collision and assign the next available asym_id
            used_asym_ids = [c.asym_id for c in chain_id_to_chain_info.values()]
            if asym_id in used_asym_ids:
                unused_asym_id = max(used_asym_ids) + 1
                logger.warning(
                    f"Chain id {subchain_id} --> asym_id {asym_id} is already in use. Assigning {unused_asym_id} instead."
                )
                asym_id = unused_asym_id

            chain = Chain(
                name=subchain_id,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                ungapped_sequence=entity.full_sequence,
            )
            chain_id_to_chain_info[subchain_id] = chain

    # Now parse gemmi raw_chain (s) to get residues and atoms
    curr_atom_idx = 0
    curr_res_idx = 0
    chains_lst: list[np.ndarray] = []
    residues_lst: list[np.ndarray] = []
    atoms_lst: list[np.ndarray] = []
    for auth_chain in structure[0]:
        for raw_chain in auth_chain.subchains():
            if raw_chain.subchain_id() in chain_id_to_chain_info.keys():
                old_chain = chain_id_to_chain_info[raw_chain.subchain_id()]
                parsed_residues, parsed_atoms = parse_chain_to_residues(
                    raw_chain,
                    old_chain.ungapped_sequence,
                    ccd,
                    asym_id=old_chain.asym_id,
                    chain_atom_start_idx=curr_atom_idx,
                )
                if len(parsed_residues) == 0:
                    logger.warning(f"Chain {old_chain.name} failed to parse any residues. Skipping...")
                    continue

                chain_num_atoms = len(parsed_atoms)
                chain_num_residues = len(parsed_residues)
                new_chain = np.array(
                    [
                        (
                            old_chain.name,  # chain_id
                            auth_chain.name,  # auth_chain_id
                            old_chain.asym_id,  # asym_id
                            get_asym_id_from_chain_id(auth_chain.name),  # auth_asym_id
                            old_chain.entity_id,  # entity_id
                            old_chain.sym_id,  # sym_id
                            curr_atom_idx,  # atom_start_idx
                            chain_num_atoms,  # num_atoms
                            curr_res_idx,  # res_start_idx
                            chain_num_residues,  # num_residues
                        )
                    ],
                    dtype=CHAINS_DTYPE,
                )

                chains_lst.append(new_chain)
                residues_lst.append(parsed_residues)
                atoms_lst.append(parsed_atoms)
                curr_atom_idx += chain_num_atoms
                curr_res_idx += chain_num_residues

    return np.concatenate(chains_lst), np.concatenate(residues_lst), np.concatenate(atoms_lst)


def parse_mmcif(
    path: Path | str,
    ccd: dict[str, Mol],
    compute_residue_interfaces: bool = True,
    atomic_interface_threshold: float = 10.0,
) -> Structure:
    """
    Parse a mmcif file and return a list of Chain objects.

    Parameters
    ----------
    path : str
        The path to the mmcif file.
    ccd : dict[str, Mol], a dictionary mapping residue names to the corresponding rdkit.Chem.rdchem.Mol molecule.

    Returns
    -------
    list[Chain], the parsed chains that were successfully parsed.

    NOTE: asym_id(s) are assigned based on the chain_id. See get_asym_id_from_chain_id for more details. If there
    is a collision, the asym_id is assigned to the max(used_asym_ids) + 1.
    """
    with smart_open.open(path, "r") as io:
        block = gemmi.cif.read_string(io.read())[0]

    # Load structure object
    structure = gemmi.make_structure_from_block(block)

    chains, residues, atoms = parse_structure_from_gemmi(structure, ccd)
    chain_interfaces, residue_interfaces = find_chain_interfaces(
        atoms,
        residues,
        chains,
        atomic_distance_threshold=atomic_interface_threshold,
        compute_residue_interfaces=compute_residue_interfaces,
    )

    return Structure(
        chains=chains,
        residues=residues,
        atoms=atoms,
        residue_interfaces=residue_interfaces,
        interfaces=chain_interfaces,
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


def pdb_to_gemmi_structure(path: Path | str, chain_id_to_sequence: dict[str, str] | None = None) -> gemmi.Structure:
    """
    Parse a pdb file, potentially update residue indexes and entity information, and return a gemmi.Structure object.

    Parameters
    ----------
    path : str
        The path to the mmcif file.
    chain_id_to_sequence: dict[str, str] | None
        A dictionary mapping chain ids to sequences. These sequences should be ungapped.
        If provided, then chains with the same sequence will be assigned to the same entity. It is recommended that
        you check the cif output file to ensure that the sequences and positions are correct if you provide this argument.
        If None, then the sequences will be inferred from the residues and each chain will be assigned a separate entity.

    Returns
    -------
    gemmi.Structure, the parsed structure.

    Raises
    ------
    - ValueError: If the chain_id_to_sequence is provided and there is a mismatch between the expected chain ID and
    the actual chain ID mapped to the sequence.
    """
    with smart_open.open(path, "r") as io:
        structure = gemmi.read_pdb_string(io.read())

    structure.setup_entities()
    structure.assign_subchains()

    if chain_id_to_sequence is not None:
        # if chain_id_to_sequence is provided, use it to assign the sequences to the entities
        # first, map the provided sequences to the atoms present in the pdb file
        seqs_to_assign = list(set(chain_id_to_sequence.values()))
        structure.clear_sequences()
        structure.assign_best_sequences(seqs_to_assign)
        # second, reassign subchains and entities based on sequence matching
        structure.setup_entities()
        structure.assign_subchains()
        # third, create a map of sequence to chain ids
        sequence_to_chain_id: dict[str, list[str]] = defaultdict(list)
        for chain_id, sequence in chain_id_to_sequence.items():
            sequence_to_chain_id[sequence].append(chain_id)

    # assign _label_seq_id to each residue in each chain
    # these are integers that start at 1 and increment by 1 for each residue in the chain
    # if the chain sequence is provided, then _label_seq_id starts at 1 at the beginning of the provided sequence.
    # if the chain sequence is not provided, then _label_seq_id starts at 1 at the beginning of the chain sequence in _atom_site fields.
    structure.assign_label_seq_id()

    # obtain entities and rename subchains to be mmCif compatible
    # pdb_chain_id: {chain_alpha}xp --> mmcif_chain_id: {chain_alpha}
    subchain_id_to_entity: dict[str, gemmi.Entity] = {}
    entity_to_subchain_id: dict[gemmi.Entity, list[str]] = defaultdict(list)
    for entity in structure.entities:
        # change from alpha to numeric entity id
        if entity.name in string.ascii_uppercase:
            entity.name = str(string.ascii_uppercase.index(entity.name) + 1)

        for subchain_id in entity.subchains:
            subchain_id_to_entity[subchain_id] = entity
            entity_to_subchain_id[entity].append(subchain_id)

    # make a map for updating the subchain ids to maket he mmCif compatible
    # we will update these in the entities and residues
    # these are based on the chain name + a 1-indexed subchain number.
    # if a chain has multiple subchains, they are named {chain_alpha}1, {chain_alpha}2, etc.
    subchain_id_to_renamed_subchain_id: dict[str, str] = {}
    for chain in structure[0]:
        for i, subchain in enumerate(chain.subchains()):
            subchain_id_to_renamed_subchain_id[subchain.subchain_id()] = f"{chain.name}"
    for subchain in structure[0].subchains():
        # update the entity with full sequence information
        entity = subchain_id_to_entity[subchain.subchain_id()]
        new_subchain_id = subchain_id_to_renamed_subchain_id[subchain.subchain_id()]

        if chain_id_to_sequence is None:
            # get the full sequence (can be gapped)
            # turn gaps ('-') into UNK residues
            full_seq = subchain.make_one_letter_sequence()
            full_seq_triplet = [
                constants.AA_ONE_TO_THREE_MAP[resi_one_letter] if resi_one_letter != "-" else "UNK"
                for resi_one_letter in list(full_seq)
            ]
            entity.full_sequence = full_seq_triplet
        else:
            # check that the subchain_id matches the expected chain id
            full_seq_as_str = "".join([constants.AA_THREE_TO_ONE_MAP[resi] for resi in entity.full_sequence])
            expected_chain_ids = sequence_to_chain_id[full_seq_as_str]

            if new_subchain_id not in expected_chain_ids:
                raise ValueError(
                    f"Chain {new_subchain_id} does not match expected chain ids {expected_chain_ids} for sequence {full_seq_as_str}"
                )

        # redefine subchain names for each residue
        for resi in subchain:
            resi.subchain = new_subchain_id

    # redefine subchain names for each entity
    for entity in structure.entities:
        entity.subchains = [subchain_id_to_renamed_subchain_id[subchain] for subchain in entity_to_subchain_id[entity]]

    return structure


def parse_pdb(
    path: Path | str,
    ccd: dict[str, Mol],
    chain_id_to_sequence: dict[str, str] | None = None,
    compute_residue_interfaces: bool = True,
    atomic_interface_threshold: float = 10.0,
) -> Structure:
    """
    Parse a pdb file and return a list of Chain objects.

    Parameters
    ----------
    path : str
        The path to the mmcif file.
    ccd : dict[str, Mol], a dictionary mapping residue names to the corresponding rdkit.Chem.rdchem.Mol molecule.
    chain_id_to_sequence: dict[str, str] | None
        A dictionary mapping chain ids to sequences. These sequences should be ungapped.
        If provided, then chains with the same sequence will be assigned to the same entity. It is recommended that
        you check the cif output file to ensure that the sequences and positions are correct if you provide this argument.
        If None, then the sequences will be inferred from the residues and each chain will be assigned a separate entity.

    Returns
    -------
    list[Chain], the parsed chains that were successfully parsed.

    Raises
    ------
    - ValueError: If the chain_id_to_sequence is provided and there is a mismatch between the expected chain ID and
    the actual chain ID mapped to the sequence.

    NOTE: asym_id(s) are assigned based on the chain_id. See get_asym_id_from_chain_id for more details. If there
    is a collision, the asym_id is assigned to the max(used_asym_ids) + 1.
    """
    structure = pdb_to_gemmi_structure(path=path, chain_id_to_sequence=chain_id_to_sequence)

    chains, residues, atoms = parse_structure_from_gemmi(structure, ccd)
    chain_interfaces, residue_interfaces = find_chain_interfaces(
        atoms,
        residues,
        chains,
        atomic_distance_threshold=atomic_interface_threshold,
        compute_residue_interfaces=compute_residue_interfaces,
    )

    return Structure(
        chains=chains,
        residues=residues,
        atoms=atoms,
        residue_interfaces=residue_interfaces,
        interfaces=chain_interfaces,
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
    )


#############################
## PDB <> mmCif Conversion ##
#############################


def convert_pdb_to_mmcif(
    pdb_path: Path | str, cif_path: Path | str, chain_id_to_sequence: dict[str, str] | None = None
) -> None:
    """Convert a PDB file to a mmCIF file

    Parameters
    ----------
    cif_path : Path | str
        Path to the mmCIF file.
    pdb_path : Path | str
        Path to the PDB file.
    chain_id_to_sequence: dict[str, str] | None
        A dictionary mapping chain ids to sequences. These sequences should be ungapped.
        If provided, then chains with the same sequence will be assigned to the same entity. It is recommended that
        you check the cif output file to ensure that the sequences and positions are correct if you provide this argument.
        If None, then the sequences will be inferred from the residues and each chain will be assigned a separate entity.

    Raises
    ------
    - ValueError: If the chain_id_to_sequence is provided and there is a mismatch between the expected chain ID and
    the actual chain ID mapped to the sequence.
    """
    structure = pdb_to_gemmi_structure(path=pdb_path, chain_id_to_sequence=chain_id_to_sequence)

    doc = structure.make_mmcif_document()
    doc.write_file(str(cif_path))


#############################
## Downloading mmCif Files ##
#############################


def download_mmcif_from_rcsb(record_id: str, target_dir: str | Path) -> Path:
    """
    Download a mmcif structure from RCSB PDB and save it to the target path.

    Parameters
    ----------
    record_id : str
        The record id of the structure to download.
    target_dir : str | Path
        The directory to save the structure to. Structure will be saved as <record_id>.cif.

    Returns
    -------
    Path
        The path to the downloaded structure.
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{record_id.lower()}.cif"

    # the normal downloads endpoint is not working
    url = f"https://files.rcsb.org/view/{record_id.lower()}.cif"

    response = requests.get(url)
    if response.status_code != 200:
        print(response.text)
        raise ValueError(f"Failed to download structure {record_id} from PDB. Status code: {response.status_code}")
    with open(file_path, "w") as f:
        f.write(response.text)

    logger.info(f"Downloaded structure {record_id} from RCSB PDB to {file_path}")
    return file_path


def download_mmcif_from_afdb(uniprot_id: str, target_dir: str | Path) -> Path:
    """Download a mmcif structure from AFDB and save it to the target path.

    Parameters
    ----------
    uniprot_id : str
        The uniprot id of the structure to download. Can be the gene id or an isoform id.
        Example for TFR1: Q96P20 (gene), Q96P20-1, Q96P20-2, ... (isoforms).
    target_dir : str | Path
        The directory to save the structure to. Structure will be saved as AF-<uniprot_id>-model_v6.cif.

    Returns
    -------
    Path
        The path to the downloaded structure.
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"AF-{uniprot_id}-model_v6.cif"

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.cif"

    response = requests.get(url)
    if response.status_code != 200:
        if "-" in uniprot_id:
            old_uniprot_id = uniprot_id
            # if it fails, it could be because the query is for the canonical isoform, which AFDB names without the -1, -2, etc.
            # so try as if it was the canonical isoform
            uniprot_id = uniprot_id.split("-")[0]
            logger.warning(
                f"Failed to download structure {old_uniprot_id} from AFDB. Trying as canonical isoform {uniprot_id}."
            )
            file_path = download_mmcif_from_afdb(uniprot_id, target_dir)
            return file_path
        else:
            raise ValueError(
                f"Failed to download structure {uniprot_id} from AFDB. Status code: {response.status_code}"
            )
    with open(file_path, "w") as f:
        f.write(response.text)

    logger.info(f"Downloaded structure {uniprot_id} from AFDB to {file_path}")
    return file_path


################################
## Writing Structures to File ##
################################


def save_structures_to_file(structures: list[Structure], path: Path | str) -> None:
    """Save a structure to a file using gemmi

    Parameters
    ----------
    structures: list[Structure], each structure will be a different model in the output file.
    path: Path
        The path to save the structure to. It must end in .cif or .pdb and the output
        file type will be inferred from the extension.
    """
    # fail if output file is not a cif or pdb file
    if str(path).lower().endswith(".cif"):
        output_file_type = "cif"
    elif str(path).lower().endswith(".pdb"):
        output_file_type = "pdb"
    else:
        raise ValueError(f"Output file must end in .cif or .pdb, got {path}")

    gemmi_structure = gemmi.Structure()
    gemmi_structure.name = "model"
    for struct_idx, structure in enumerate(structures):
        model = gemmi.Model(struct_idx + 1)

        entity_id_to_entity: dict[int, gemmi.Entity] = {}
        for entity_id in structure.chains["entity_id"]:
            if entity_id not in entity_id_to_entity:
                new_entity = gemmi.Entity(str(entity_id))
                new_entity.entity_type = gemmi.EntityType.Polymer
                new_entity.polymer_type = gemmi.PolymerType.PeptideL

                entity_id_to_entity[entity_id] = new_entity

                gemmi_structure.entities.append(new_entity)

        chain_sequences: dict[str, list[str]] = defaultdict(list)
        for chain in structure.chains:
            gemmi_chain = gemmi.Chain(chain["auth_chain_id"])
            residues = structure.residues[chain["res_start_idx"] : (chain["res_start_idx"] + chain["num_residues"])]
            chain_sequence: list[str] = []

            for residue in residues:
                chain_sequence.append(residue["res_name"])
                # if the residue is not present, skip the record/atom records
                if not residue["is_present"]:
                    continue

                res_idx = residue["res_idx"]

                gemmi_res = gemmi.Residue()
                gemmi_res.entity_id = str(chain["entity_id"])
                gemmi_res.entity_type = gemmi.EntityType.Polymer

                gemmi_res.name = residue["res_name"]
                gemmi_res.seqid = gemmi.SeqId(residue["auth_res_idx"], " ")
                gemmi_res.label_seq = res_idx + 1
                gemmi_res.subchain = chain["chain_id"]

                # Add atoms
                atoms = structure.atoms[residue["atom_start_idx"] : (residue["atom_start_idx"] + residue["num_atoms"])]
                for atom in atoms:
                    if not atom["is_present"]:
                        continue
                    gemmi_atom = gemmi.Atom()
                    gemmi_atom.b_iso = atom["bfactor"]
                    gemmi_atom.element = gemmi.Element(constants.ATOM_NAME_TO_ELEMENT[atom["atom_name"]])
                    gemmi_atom.name = atom["atom_name"]
                    gemmi_atom.occ = 1.0
                    gemmi_atom.pos = gemmi.Position(atom["coords"][0], atom["coords"][1], atom["coords"][2])
                    gemmi_res.add_atom(gemmi_atom)

                gemmi_chain.add_residue(gemmi_res)

            model.add_chain(gemmi_chain)
            chain_sequences[chain["chain_id"]] = chain_sequence

        gemmi_structure.add_model(model)

        # update the entities with the full sequences
        for entity in gemmi_structure.entities:
            entity_id = int(entity.name)
            chain_ids = structure.chains[structure.chains["entity_id"] == entity_id]["chain_id"]
            if len(chain_ids) > 1:
                # check that the sequences are the same
                sequences = [chain_sequences[chain_id] for chain_id in chain_ids]
                if not all(sequence == sequences[0] for sequence in sequences):
                    logger.warning(f"Chains {chain_ids} have different sequences, but the same entity id {entity_id}")
            else:
                chain_id = chain_ids[0]
                sequence = chain_sequences[chain_id]
                entity.full_sequence = sequence

    if output_file_type == "cif":
        doc = gemmi_structure.make_mmcif_document()
        doc.write_file(str(path))
    else:
        gemmi_structure.write_pdb(str(path))


def save_structures_to_gif(structures: list[Structure], path: Path | str) -> None:
    """Save structures as a GIF."""
    pass
