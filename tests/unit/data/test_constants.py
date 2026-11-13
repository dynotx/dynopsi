import pytest

from dynopsi.data.constants import (
    N_AMINO_ACID_SEQUENCE_FEATURES,
    NONCANONICAL_RESIDUE_TO_NATURAL,
    AminoAcidSequenceFeature,
    HomooligomerPartnerFeature,
)


def test_homooligomer_partner_feature_uses_embedding_safe_ids() -> None:
    assert int(HomooligomerPartnerFeature.MASK_OR_MISSING) == 0
    assert int(HomooligomerPartnerFeature.NO_PARTNER) == 1
    assert int(HomooligomerPartnerFeature.HAS_PARTNER) == 2
    assert [int(feature) for feature in HomooligomerPartnerFeature] == list(range(3))


def test_target_sequence_vocabulary_is_dense_and_embedding_safe() -> None:
    assert [int(feature) for feature in AminoAcidSequenceFeature] == list(range(N_AMINO_ACID_SEQUENCE_FEATURES))
    assert AminoAcidSequenceFeature.PAD == 0
    assert AminoAcidSequenceFeature.MASK == N_AMINO_ACID_SEQUENCE_FEATURES - 1


@pytest.mark.parametrize(
    "resname", [feature.name for feature in AminoAcidSequenceFeature if feature.name not in {"PAD", "MASK"}]
)
def test_target_sequence_feature_maps_supported_residue_names(resname: str) -> None:
    assert AminoAcidSequenceFeature.from_resname(resname) is AminoAcidSequenceFeature[resname]


@pytest.mark.parametrize(("noncanonical", "natural"), NONCANONICAL_RESIDUE_TO_NATURAL.items())
def test_target_sequence_feature_maps_noncanonical_residues(noncanonical: str, natural: str) -> None:
    assert AminoAcidSequenceFeature.from_resname(noncanonical) is AminoAcidSequenceFeature[natural]


@pytest.mark.parametrize("resname", ["", "HOH", "DA", "not-a-residue"])
def test_target_sequence_feature_maps_unsupported_names_to_unknown(resname: str) -> None:
    assert AminoAcidSequenceFeature.from_resname(resname) is AminoAcidSequenceFeature.UNK
