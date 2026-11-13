import numpy as np
import pytest
import torch

from dynopsi.data.auth_chain_identity import OPAQUE_AUTH_ASYM_ID_START
from dynopsi.data.featurization import Features, primitives
from dynopsi.data.structure import (
    CHAIN_INTERFACE_DTYPE,
    DYNO_PSI_FEATURES_DTYPE,
    RESIDUE_INTERFACE_DTYPE,
    Structure,
)
from tests.unit.data.conftest import (
    assert_feature_dict_equality,
    assert_metadata_equality,
    assert_structure_equality,
)


@pytest.fixture
def features(feature_dict_standard: dict[str, torch.Tensor], structure_standard: Structure) -> Features:
    return Features(
        feature_dict={k: v.clone() for k, v in feature_dict_standard.items()},
        input_reference_structure=structure_standard,
    )


@pytest.fixture
def cropped_structure(structure_standard: Structure) -> Structure:

    residues_cropped = structure_standard.residues[2:5]

    atom_start_idx = residues_cropped["atom_start_idx"][0]
    atom_end_idx = atom_start_idx + np.sum(residues_cropped["num_atoms"])
    atoms_cropped = structure_standard.atoms[atom_start_idx:atom_end_idx]

    # update atom start idx after assigning atoms_cropped
    residues_cropped["atom_start_idx"] = np.cumsum([0] + residues_cropped["num_atoms"].tolist()[:-1])
    residues_cropped["atom_center"] = residues_cropped["atom_start_idx"] + 1

    chains_cropped = structure_standard.chains[
        np.isin(structure_standard.chains["asym_id"], residues_cropped["asym_id"])
    ]
    chains_cropped["num_residues"] = np.array([2, 1])
    chains_cropped["res_start_idx"] = np.array([0, 2])
    chains_cropped["num_atoms"] = np.array(
        [
            np.sum(residues_cropped[residues_cropped["asym_id"] == 1]["num_atoms"]),
            np.sum(residues_cropped[residues_cropped["asym_id"] == 2]["num_atoms"]),
        ]
    )
    chains_cropped["atom_start_idx"] = np.array([0, chains_cropped["num_atoms"][0]])

    return Structure(
        residues=residues_cropped,
        atoms=atoms_cropped,
        chains=chains_cropped,
        features=np.empty(0, dtype=DYNO_PSI_FEATURES_DTYPE),
        interfaces=np.empty(0, dtype=CHAIN_INTERFACE_DTYPE),
        residue_interfaces=np.empty(0, dtype=RESIDUE_INTERFACE_DTYPE),
    )


@pytest.fixture
def cropped_feature_dict(feature_dict_standard: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    feature_dict_cropped = {
        "token_index": torch.arange(3),
        "residue_index": feature_dict_standard["residue_index"][2:5],
        "auth_residue_index": feature_dict_standard["auth_residue_index"][2:5],
        "original_residue_index": feature_dict_standard["original_residue_index"][2:5],
        "asym_id": feature_dict_standard["asym_id"][2:5],
        "auth_asym_id": feature_dict_standard["auth_asym_id"][2:5],
        "entity_id": feature_dict_standard["entity_id"][2:5],
        "sym_id": feature_dict_standard["sym_id"][2:5],
        "token_resolved_mask": feature_dict_standard["token_resolved_mask"][2:5],
        "token_pad_mask": feature_dict_standard["token_pad_mask"][2:5],
        "frames_idx": torch.arange(3 * 3).reshape(-1, 3),
        "atom_to_token": torch.repeat_interleave(torch.eye(3), repeats=3, dim=0),
        "frame_resolved_mask": feature_dict_standard["frame_resolved_mask"][2:5],
        "atom_resolved_mask": torch.ones(3 * 3, dtype=torch.bool),
        "pocket_feature": feature_dict_standard["pocket_feature"][2:5],
        "inpaint_feature": feature_dict_standard["inpaint_feature"][2:5],
        "noised_mask": feature_dict_standard["noised_mask"][2:5],
        "coords": feature_dict_standard["coords"][:, 2 * 3 : 5 * 3],
        "atom_pad_mask": torch.ones(3 * 3, dtype=torch.bool),
        "binder_mask": feature_dict_standard["binder_mask"][2:5],
        "target_mask": feature_dict_standard["target_mask"][2:5],
    }
    return feature_dict_cropped


class TestCropFeaturizer:
    @pytest.mark.parametrize("index_type", ["original_residue_index", "auth_residue_index"])
    @pytest.mark.parametrize("verbose_metadata", [True, False])
    def test_crop_featurizer(
        self,
        index_type: str,
        verbose_metadata: bool,
        features: Features,
        cropped_feature_dict: dict[str, torch.Tensor],
        cropped_structure: Structure,
    ) -> None:
        if index_type == "original_residue_index":
            residue_index_chain_spec = "B1-2,C1"
        else:
            residue_index_chain_spec = "V2-3,W4"

        featurizer = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec=residue_index_chain_spec,
            index_type=index_type,
            verbose_metadata=verbose_metadata,
        )

        out_feats = next(featurizer([features]))

        assert_feature_dict_equality(out_feats.feature_dict, cropped_feature_dict)
        assert_structure_equality(out_feats.input_reference_structure, cropped_structure)

        assert out_feats.featurization_steps == [("crop", "test_crop_featurizer")]
        if index_type == "original_residue_index":
            chain_id_to_residue_map = {
                "B": [1, 2],
                "C": [1],
            }
        else:
            chain_id_to_residue_map = {
                "V": [2, 3],
                "W": [4],
            }
        expected_metadata = {
            "test_crop_featurizer": {
                "name": "test_crop_featurizer",
                "type": "crop",
                "residue_index_chain_spec": {
                    "chain_id_to_residue_index_map": chain_id_to_residue_map,
                },
                "index_type": index_type,
                "verbose_metadata": verbose_metadata,
            }
        }
        if verbose_metadata:
            expected_metadata["test_crop_featurizer"]["structure"] = cropped_structure
        assert_metadata_equality(out_feats.featurization_metadata, expected_metadata)

    def test_crop_featurizer_invalid_crop(self, features: Features) -> None:
        featurizer = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec="A1-5,B1",
            index_type="original_residue_index",
            verbose_metadata=False,
        )
        with pytest.raises(ValueError):
            next(featurizer([features]))

    def test_crop_featurizer_invalid_index_type(self, features: Features) -> None:
        with pytest.raises(ValueError):
            primitives.CropFeaturizer(
                name="test_crop_featurizer",
                residue_index_chain_spec="A1-2,B1",
                index_type="invalid_index_type",
                verbose_metadata=False,
            )

    def test_crop_featurizer_missing_auth_residue_index(self, feature_dict_standard: dict[str, torch.Tensor]) -> None:
        feat_dict = {k: v.clone() for k, v in feature_dict_standard.items() if k != "auth_residue_index"}
        features = Features(feature_dict=feat_dict, input_reference_structure=None)
        featurizer = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec="V2-3",
            index_type="auth_residue_index",
            verbose_metadata=False,
        )
        with pytest.raises(ValueError):
            next(featurizer([features]))

    @pytest.mark.parametrize(
        ["index_type", "residue_index_chain_spec"], [("original_residue_index", "B1-2"), ("auth_residue_index", "V2-3")]
    )
    def test_crop_featurizer_adds_original_residue_index(
        self, feature_dict_standard: dict[str, torch.Tensor], index_type: str, residue_index_chain_spec: str
    ) -> None:
        feat_dict = {k: v.clone() for k, v in feature_dict_standard.items() if k != "original_residue_index"}
        features = Features(feature_dict=feat_dict, input_reference_structure=None)
        featurizer = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec=residue_index_chain_spec,
            index_type=index_type,
            verbose_metadata=False,
        )
        out_feats = next(featurizer([features]))
        assert "original_residue_index" in out_feats.feature_dict

    def test_auth_crop_unique_letter_chain_spec(
        self, features: Features, cropped_feature_dict: dict[str, torch.Tensor]
    ) -> None:
        featurizer = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec="V2-3,W4",
            index_type="auth_residue_index",
            verbose_metadata=False,
        )
        out_feats = next(featurizer([features]))
        assert out_feats["auth_residue_index"].tolist() == cropped_feature_dict["auth_residue_index"].tolist()

    def test_auth_crop_dict_a7_residue_and_whole_chain(self) -> None:
        n_res = 12
        n_atoms = n_res * 3
        feature_dict = {
            "token_index": torch.arange(n_res),
            "residue_index": torch.arange(n_res),
            "auth_residue_index": torch.arange(1, n_res + 1),
            "original_residue_index": torch.arange(n_res),
            "asym_id": torch.zeros(n_res, dtype=torch.long),
            "auth_asym_id": torch.full((n_res,), OPAQUE_AUTH_ASYM_ID_START, dtype=torch.long),
            "token_pad_mask": torch.ones(n_res, dtype=torch.bool),
            "atom_to_token": torch.repeat_interleave(torch.eye(n_res), repeats=3, dim=0),
            "coords": torch.zeros((1, n_atoms, 3)),
            "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
            "binder_mask": torch.zeros(n_res, dtype=torch.bool),
            "target_mask": torch.ones(n_res, dtype=torch.bool),
        }
        feats = Features(
            feature_dict=feature_dict,
            auth_chain_id_table={OPAQUE_AUTH_ASYM_ID_START: "A-7"},
        )
        residue_crop = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec={"A-7": [10]},
            index_type="auth_residue_index",
            verbose_metadata=False,
        )
        residue_out = next(residue_crop([feats]))
        assert residue_out["auth_residue_index"].tolist() == [10]
        assert residue_out["auth_asym_id"].tolist() == [OPAQUE_AUTH_ASYM_ID_START]

        whole_crop = primitives.CropFeaturizer(
            name="test_crop_featurizer",
            residue_index_chain_spec={"A-7": None},
            index_type="auth_residue_index",
            verbose_metadata=False,
        )
        whole_out = next(whole_crop([feats]))
        assert whole_out["auth_residue_index"].tolist() == list(range(1, n_res + 1))
