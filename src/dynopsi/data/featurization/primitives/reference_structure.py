import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import PrivateAttr, field_validator, model_validator

from dynopsi.data.auth_chain_identity import (
    auth_chain_id_table_from_chains,
    merge_auth_chain_id_tables,
    resolve_unique_letter_auth_int,
)
from dynopsi.data.constants import AtomFilter, DesignDesignation
from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import register_featurizer
from dynopsi.data.featurization.utils import concat_features_with_auth_tables, crop_features
from dynopsi.data.structure import Structure
from dynopsi.data.structure_io import (
    load_structure_from_afdb,
    load_structure_from_cif,
    load_structure_from_pdb,
    load_structure_from_rcsb,
)
from dynopsi.data.tokenizer import DynoPsiTokenizer
from dynopsi.data.utils import get_asym_id_from_chain_id

logger = logging.getLogger(__name__)


@register_featurizer()
class ReferenceStructureFeaturizer(Featurizer):
    """Featurizer that adds a reference structure.

    Builds token- and atom-level features defining the reference structure
    and adds them into input `Features` object.

    Attributes
    ----------
    design_designation : DesignDesignation | dict[DesignDesignation, str] | None
        The design designation for the reference structure.
        If a dictionary, the keys are the design designations and the values lists of chain IDs.
        The dictionary must include all chain IDs in the structure, exclusively.
    atoms_to_include : AtomFilter | None
        The atoms to include in the reference structure.
        Must be set by the featurizer or by a parent featurization pipeline.
    rcsb_record_id : str | None
        RCSB record ID of the structure to download from the RCSB PDB server.
        Default is None.
    afdb_uniprot_id : str | None
        UniProt ID of the structure to download from the AFDB database (v6).
        Default is None.
    structure_path : str | Path | None
        The path to the PDB or mmCIF file structure file to use.
        Default is None.

    Notes
    -----
    Exactly one of `rcsb_record_id`, `afdb_uniprot_id`, or `structure_path` must be set.OTE: Exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path must be provided.

    Raises
    ------
    ValueError
        If none or more than one of `rcsb_record_id`, `afdb_uniprot_id`,
        or `structure_path` is provided.
    ValueError
        If a chain ID appears in more than one `DesignDesignation`.
    """

    type: Literal["reference_structure"] = "reference_structure"
    design_designation: DesignDesignation | dict[DesignDesignation, list[str]] | None = None
    index_type: Literal["original_residue_index", "auth_residue_index"] = "original_residue_index"
    atoms_to_include: AtomFilter | None = None
    rcsb_record_id: str | None = None
    afdb_uniprot_id: str | None = None
    structure_path: str | Path | None = None

    _reference_structure_feature_dict_cache: dict[str, Any] = PrivateAttr(default={})
    _structure: Structure | None = PrivateAttr(default=None)

    @property
    def design_designations(self) -> list[DesignDesignation]:
        """Design designations used by this featurizer.

        Returns
        -------
        list[DesignDesignation]
            A single designation as a list, or the keys of the
            design_designation dict.
        """
        if isinstance(self.design_designation, DesignDesignation):
            return [self.design_designation]
        else:
            return list(self.design_designation.keys())

    @field_validator("design_designation", mode="after")
    @classmethod
    def _check(
        cls, v: DesignDesignation | dict[DesignDesignation, list[str]]
    ) -> DesignDesignation | dict[DesignDesignation, list[str]]:
        """Ensure no chain ID is assigned to more than one design designation."""
        if isinstance(v, dict):
            used_chain_ids = set()
            for design_designation, chain_ids in v.items():
                for chain_id in chain_ids:
                    if chain_id in used_chain_ids:
                        raise ValueError(f"Chain ID {chain_id} is used in multiple design designations")
                    used_chain_ids.add(chain_id)
        return v

    @model_validator(mode="after")
    def validate_input_structure(self) -> Self:
        """Ensure exactly one structure source (rcsb_record_id, afdb_uniprot_id, or structure_path) is set."""
        non_null_inputs = sum(v is not None for v in [self.rcsb_record_id, self.afdb_uniprot_id, self.structure_path])
        if non_null_inputs != 1:
            raise ValueError("Exactly one of rcsb_record_id, afdb_uniprot_id, or structure_path must be provided")
        return self

    def _get_reference_structure_feature_dict(self, atoms_to_include: AtomFilter) -> dict[str, Any]:
        """Parse and tokenize the reference structure, then annotate binder/target masks using design designations.

        Structure is loaded from either RCSB, AFDB, or structure file,
        tokenized, and cached for reuse. Binder and target masks are
        set based one `design_designation`.

        Parameters
        ----------
        atoms_to_include : AtomFilter
            Atom set used for tokenization (e.g. backbone only).

        Returns
        -------
        dict[str, Any]
            Feature dictionary for the reference structure.

        Raises
        ------
        ValueError
            If the structure source is invalid, the file format is unsupported, or
            chain IDs in `design_designation` do not match the structure.
        """
        if len(self._reference_structure_feature_dict_cache) == 0:
            if self._structure is None:
                if self.rcsb_record_id is not None:
                    structure = load_structure_from_rcsb(self.rcsb_record_id)
                elif self.afdb_uniprot_id is not None:
                    structure = load_structure_from_afdb(self.afdb_uniprot_id)
                elif self.structure_path is not None:
                    suffix = Path(self.structure_path).suffix.lower()
                    if suffix == ".pdb":
                        structure = load_structure_from_pdb(self.structure_path)
                    elif suffix == ".cif" or suffix == ".mmcif":
                        structure = load_structure_from_cif(self.structure_path)
                    else:
                        raise ValueError(f"Invalid file extension: {self.structure_path}. Must be .pdb or .cif/.mmcif")
                self._structure = structure
            else:
                structure = self._structure
            token_features_lst = DynoPsiTokenizer(atom_filter=atoms_to_include).tokenize(structure=structure)
            for features in token_features_lst:
                self._reference_structure_feature_dict_cache.update(features.asdict())

            # get the binder and target masks
            n_tokens = self._reference_structure_feature_dict_cache["token_index"].shape[0]
            binder_mask = torch.full((n_tokens,), False, dtype=torch.bool)
            target_mask = torch.full((n_tokens,), False, dtype=torch.bool)

            if isinstance(self.design_designation, dict):
                if self.index_type == "original_residue_index":
                    asym_id_col = "asym_id"
                elif self.index_type == "auth_residue_index":
                    asym_id_col = "auth_asym_id"
                else:
                    raise ValueError(f"Invalid index type: {self.index_type}")
                # check that all chain_ids are present
                structure_asym_ids = (
                    self._reference_structure_feature_dict_cache[asym_id_col][
                        self._reference_structure_feature_dict_cache["token_pad_mask"].bool()
                    ]
                    .unique()
                    .tolist()
                )

                chains_specified = [
                    chain_id for chain_ids in self.design_designation.values() for chain_id in chain_ids
                ]
                auth_table = auth_chain_id_table_from_chains(structure.chains) if structure is not None else {}
                if self.index_type == "auth_residue_index":
                    asym_ids_specified = [
                        resolve_unique_letter_auth_int(chain_id, auth_table) for chain_id in chains_specified
                    ]
                else:
                    asym_ids_specified = [get_asym_id_from_chain_id(chain_id) for chain_id in chains_specified]
                if set(structure_asym_ids) != set(asym_ids_specified):
                    if set(asym_ids_specified) - set(structure_asym_ids) != set():
                        raise ValueError(
                            f"Chain IDs specified in design_designation are not present in the structure: {set(asym_ids_specified) - set(structure_asym_ids)}"
                        )
                    else:
                        logger.warning(
                            f"Chain IDs in the structure are not present in the design_designation: {set(structure_asym_ids) - set(asym_ids_specified)}\nSubsetting to only specified chain IDs"
                        )
                        crop_mask = torch.isin(
                            self._reference_structure_feature_dict_cache[asym_id_col], torch.tensor(asym_ids_specified)
                        )
                        self._reference_structure_feature_dict_cache = crop_features(
                            self._reference_structure_feature_dict_cache, crop_mask
                        )

                        # update the binder and target masks
                        n_tokens = self._reference_structure_feature_dict_cache["token_index"].shape[0]
                        binder_mask = torch.full((n_tokens,), False, dtype=torch.bool)
                        target_mask = torch.full((n_tokens,), False, dtype=torch.bool)

                for design_designation, chain_ids in self.design_designation.items():
                    if self.index_type == "auth_residue_index":
                        chain_ints = [resolve_unique_letter_auth_int(chain_id, auth_table) for chain_id in chain_ids]
                    else:
                        chain_ints = [get_asym_id_from_chain_id(chain_id) for chain_id in chain_ids]
                    mask = (
                        torch.isin(
                            self._reference_structure_feature_dict_cache[asym_id_col],
                            torch.tensor(chain_ints),
                        )
                        & self._reference_structure_feature_dict_cache["token_pad_mask"].bool()
                    )
                    if design_designation == DesignDesignation.FIXED:
                        target_mask[mask] = True
                    elif design_designation == DesignDesignation.DESIGN:
                        binder_mask[mask] = True
            elif self.design_designation == DesignDesignation.FIXED:
                target_mask[:] = True
            elif self.design_designation == DesignDesignation.DESIGN:
                binder_mask[:] = True
            else:
                raise ValueError(f"Invalid design designation: {self.design_designation}")
            self._reference_structure_feature_dict_cache["binder_mask"] = binder_mask
            self._reference_structure_feature_dict_cache["target_mask"] = target_mask

        return self._reference_structure_feature_dict_cache

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        """Add features from a reference structure.

        See tokenizer.py for more details on the features that are added.

        Parameters
        ----------
        inputs : Iterable[Features]
            Feature to which to add the reference structure.

        Yields
        ------
        Features
            Input Features with reference structure features added and updated metadata.

        Notes
        -----
        If the input `Feature`s already contain chains, asym IDs may be updated if
        needed to prevent asym ID collisions with a warning logged. Metadata will
        reflect the change in asym ID.
        """
        assert self.atoms_to_include is not None, (
            "atoms_to_include must be provided, either directly to ReferenceStructureFeaturizer or set by a parent featurization pipeline"
        )

        reference_structure_features = self._get_reference_structure_feature_dict(
            atoms_to_include=self.atoms_to_include
        )
        for input_features in inputs:
            new_feats = input_features.safe_copy()
            structure_auth_table = (
                auth_chain_id_table_from_chains(self._structure.chains) if self._structure is not None else {}
            )
            if len(new_feats.keys()) == 0:
                new_feats.feature_dict = {k: v.clone() for k, v in reference_structure_features.items()}
                ref_asym_id_map = None
                ref_auth_asym_id_map = None
                new_feats.auth_chain_id_table = merge_auth_chain_id_tables(
                    [new_feats.auth_chain_id_table, structure_auth_table]
                )

            else:
                new_feats.feature_dict, new_feats.auth_chain_id_table = concat_features_with_auth_tables(
                    [new_feats.feature_dict, reference_structure_features],
                    [new_feats.auth_chain_id_table, structure_auth_table],
                    make_asym_ids_unique=True,
                    make_entity_ids_unique=True,
                    reset_residue_index=False,
                )
                # check if there was a change in asym_id and add this to the metadata
                # in the format {original_asym_id: new_asym_id}
                ref_asym_ids = reference_structure_features["asym_id"]
                ref_auth_asym_ids = reference_structure_features["auth_asym_id"]
                n_nonpad_tokens_input = input_features["token_pad_mask"].bool().sum()
                new_asym_ids = new_feats["asym_id"][n_nonpad_tokens_input:]
                new_auth_asym_ids = new_feats["auth_asym_id"][n_nonpad_tokens_input:]
                ref_asym_id_map = {
                    int(ref_asym_id): int(expected_ref_asym_id)
                    for ref_asym_id, expected_ref_asym_id in zip(ref_asym_ids.numpy(), new_asym_ids.numpy())
                }
                ref_auth_asym_id_map = {
                    ref_auth_asym_id: expected_ref_auth_asym_id
                    for ref_auth_asym_id, expected_ref_auth_asym_id in zip(
                        ref_auth_asym_ids.numpy(), new_auth_asym_ids.numpy()
                    )
                }

            # update the medatata
            more_metadata: dict[str, Any] = {
                "ref_asym_id_map": ref_asym_id_map,
            }
            if self.verbose_metadata:
                more_metadata["structure"] = self._structure
            self._update_features_metadata(new_feats, more_metadata=more_metadata)

            # update reference structure that has sidechains
            if new_feats.input_reference_structure is None:
                new_feats.input_reference_structure = self._structure
            else:
                if ref_asym_id_map is not None and ref_auth_asym_id_map is not None:
                    structure = self._structure.update_asym_ids(
                        ref_asym_id_map=ref_asym_id_map, auth_asym_id_map=ref_auth_asym_id_map
                    )
                else:
                    structure = self._structure
                structure = new_feats.input_reference_structure.merge(
                    structure, mode="concatenate", reset_residue_index=False
                )
                new_feats.input_reference_structure = structure

            yield new_feats
