from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gemmi
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from dynopsi.data.constants import AtomFilter, InpaintFeature
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.utils import concat_features, crop_features, make_empty_chain
from dynopsi.data.structure import Structure
from dynopsi.design_pipeline import OutputSpecification
from dynopsi.design_pipeline_parser import DesignLanguageParser
from dynopsi.simulation.observable import Observable, ObservableOutputConfig

AUTH_SEQID = 81
RES_IDX_SEQID = 0
NEW_CHAIN_AUTH_SEQID = 1
REFERENCE_O_SENTINEL = np.array([999.0, 999.0, 999.0], dtype=np.float32)
REFERENCE_CB_SENTINEL = np.array([123.0, 456.0, 789.0], dtype=np.float32)


def _target_features_auth_differs_from_res_idx() -> Features:
    feature_dict = make_empty_chain(
        seq_length=1,
        atoms_to_include=AtomFilter.BB,
        res_type="ALA",
        inpaint_feature=InpaintFeature.TARGET_CONTEXT,
    )
    feature_dict["residue_index"] = torch.tensor([RES_IDX_SEQID], dtype=torch.long)
    feature_dict["auth_residue_index"] = torch.tensor([AUTH_SEQID], dtype=torch.long)
    return Features(feature_dict=feature_dict, atoms_to_include=AtomFilter.BB, auth_chain_id_table={0: "A"})


def _target_features_with_unnoised_reference_context() -> Features:
    """Unnoised TARGET_CONTEXT plus BINDER_INPAINT so save merges a cropped reference."""
    target = make_empty_chain(
        seq_length=1,
        atoms_to_include=AtomFilter.BB,
        res_type="ALA",
        asym_id=0,
        inpaint_feature=InpaintFeature.TARGET_CONTEXT,
    )
    target["residue_index"] = torch.tensor([RES_IDX_SEQID], dtype=torch.long)
    target["auth_residue_index"] = torch.tensor([AUTH_SEQID], dtype=torch.long)
    target["noised_mask"] = torch.zeros(1, dtype=torch.bool)
    binder = make_empty_chain(
        seq_length=1,
        atoms_to_include=AtomFilter.BB,
        res_type="ALA",
        asym_id=1,
        auth_asym_id=1,
        entity_id=1,
        inpaint_feature=InpaintFeature.BINDER_INPAINT,
    )
    feature_dict = concat_features(
        [target, binder],
        make_asym_ids_unique=False,
        make_entity_ids_unique=False,
        reset_residue_index=False,
    )
    features = Features(
        feature_dict=feature_dict,
        atoms_to_include=AtomFilter.BB,
        auth_chain_id_table={0: "A", 1: "B"},
    )
    context_mask = features.feature_dict["inpaint_feature"] == InpaintFeature.TARGET_CONTEXT
    context_struct = Features(
        feature_dict=crop_features(features.feature_dict, token_mask=context_mask, reset_residue_index=False),
        atoms_to_include=AtomFilter.BB,
        auth_chain_id_table={0: "A"},
    ).to_structure()
    oxygen_mask = context_struct.atoms["atom_name"] == "O"
    context_struct.atoms["coords"][oxygen_mask] = REFERENCE_O_SENTINEL
    cb_atom = np.zeros(1, dtype=context_struct.atoms.dtype)
    cb_atom["asym_id"] = 0
    cb_atom["res_idx"] = RES_IDX_SEQID
    cb_atom["atom_name"] = "CB"
    cb_atom["coords"] = REFERENCE_CB_SENTINEL
    cb_atom["is_present"] = True
    cb_atom["bfactor"] = 100.0
    residues = context_struct.residues.copy()
    residues["num_atoms"] += 1
    chains = context_struct.chains.copy()
    chains["num_atoms"] += 1
    context_struct = replace(
        context_struct,
        atoms=np.concatenate([context_struct.atoms, cb_atom]),
        residues=residues,
        chains=chains,
    )
    features.input_reference_structure = context_struct
    return features


def _target_features_with_noised_reference_context() -> Features:
    features = _target_features_with_unnoised_reference_context()
    features.feature_dict["noised_mask"] = torch.tensor([True, True], dtype=torch.bool)
    return features


def _pdb_seqids(path: Path, chain_name: str | None = None) -> list[int]:
    structure = gemmi.read_structure(str(path))
    seqids: list[int] = []
    for model in structure:
        for chain in model:
            if chain_name is not None and chain.name != chain_name:
                continue
            for residue in chain:
                num = residue.seqid.num
                assert num is not None
                seqids.append(num)
    return seqids


def _pdb_atom_coords(path: Path, chain_name: str, atom_name: str) -> np.ndarray:
    structure = gemmi.read_structure(str(path))
    coords: list[list[float]] = []
    for model in structure:
        chain = model.find_chain(chain_name)
        if chain is None:
            continue
        for residue in chain:
            atom = residue.find_atom(atom_name, "*")
            if atom is not None:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.asarray(coords, dtype=np.float32)


def _pdb_atom_names(path: Path, chain_name: str) -> list[str]:
    structure = gemmi.read_structure(str(path))
    return [
        atom.name
        for model in structure
        for chain in model
        if chain.name == chain_name
        for residue in chain
        for atom in residue
    ]


def _save_designed_pdb(tmp_path: Path, spec: OutputSpecification) -> tuple[Path, Path | None]:
    cfg = spec.observable_output_configs["X_T"]
    cfg.add_features_key("features")
    features = _target_features_auth_differs_from_res_idx()
    obs = Observable(observable_output_configs={"X_T": cfg})
    coords = features.feature_dict["coords"][0].reshape(1, 1, -1)
    obs.record("X_T", step_num=0, value={"coords": coords})
    files = obs.save(save_dir=tmp_path, start_index=0, reference_features={"features": features})
    pdb_path = next(Path(output["path"]) for output in files if output["output_format"] == "pdb")
    npz_paths = [
        Path(output["path"])
        for output in files
        if output["output_format"] == "npz" and Path(output["path"]).name.startswith("X_T_")
    ]
    return pdb_path, npz_paths[0] if npz_paths else None


class TestOutputSpecificationSaveKeyYaml:
    def test_omitted_key_defaults_to_res_idx(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {"output_dir": str(tmp_path), "formats": ["pdb"], "save_trajectory": False}
        )
        assert spec.observable_output_configs["X_T"].primary_residue_index_key == "res_idx"

    def test_yaml_auth_res_idx_is_copied_onto_generated_x_t_config(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["pdb"],
                "save_trajectory": False,
                "primary_residue_index_key": "auth_res_idx",
            }
        )
        assert spec.observable_output_configs["X_T"].primary_residue_index_key == "auth_res_idx"

    def test_explicit_observable_config_can_set_save_key(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "observable_output_configs": {
                    "X_T": {
                        "state_type": "clean_data_coords",
                        "outputs": {"final": ["pdb"]},
                        "primary_residue_index_key": "auth_res_idx",
                    }
                },
            }
        )
        assert spec.observable_output_configs["X_T"].primary_residue_index_key == "auth_res_idx"

    def test_invalid_key_value_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            OutputSpecification.model_validate(
                {
                    "output_dir": str(tmp_path),
                    "formats": ["pdb"],
                    "save_trajectory": False,
                    "primary_residue_index_key": "residue_index",
                }
            )

    def test_top_level_key_rejected_with_explicit_observable_configs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="primary_residue_index_key"):
            OutputSpecification.model_validate(
                {
                    "output_dir": str(tmp_path),
                    "primary_residue_index_key": "auth_res_idx",
                    "observable_output_configs": {
                        "X_T": {
                            "state_type": "clean_data_coords",
                            "outputs": {"final": ["pdb"]},
                        }
                    },
                }
            )

    def test_raw_context_output_is_copied_onto_generated_x_t_config(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["pdb"],
                "save_trajectory": False,
                "save_raw_context_output": True,
            }
        )
        assert spec.observable_output_configs["X_T"].save_raw_context_output is True

    def test_raw_context_output_defaults_off(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["mmcif"],
                "save_trajectory": True,
            }
        )

        assert spec.observable_output_configs["X_T"].save_raw_context_output is False


class TestDesignedPdbSeqids:
    def test_default_writes_res_idx_seqids(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {"output_dir": str(tmp_path), "formats": ["pdb"], "save_trajectory": False}
        )
        pdb_path, _ = _save_designed_pdb(tmp_path, spec)
        assert _pdb_seqids(pdb_path) == [RES_IDX_SEQID]

    def test_auth_res_idx_writes_author_seqids(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["pdb"],
                "save_trajectory": False,
                "primary_residue_index_key": "auth_res_idx",
            }
        )
        pdb_path, _ = _save_designed_pdb(tmp_path, spec)
        assert _pdb_seqids(pdb_path) == [AUTH_SEQID]

    def test_auth_res_idx_writes_author_seqids_on_reference_merge_path(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["pdb"],
                "save_trajectory": False,
                "primary_residue_index_key": "auth_res_idx",
            }
        )
        cfg = spec.observable_output_configs["X_T"]
        cfg.add_features_key("features")
        features = _target_features_with_unnoised_reference_context()
        obs = Observable(observable_output_configs={"X_T": cfg})
        n_res = int(features.feature_dict["residue_index"].shape[0])
        coords = features.feature_dict["coords"][0].reshape(1, n_res, -1)
        obs.record("X_T", step_num=0, value={"coords": coords})
        files = obs.save(save_dir=tmp_path, start_index=0, reference_features={"features": features})
        pdb_path = next(Path(output["path"]) for output in files if output["output_format"] == "pdb")
        assert _pdb_seqids(pdb_path, chain_name="A") == [AUTH_SEQID]
        assert _pdb_seqids(pdb_path, chain_name="B") == [NEW_CHAIN_AUTH_SEQID]
        np.testing.assert_allclose(_pdb_atom_coords(pdb_path, "A", "O"), REFERENCE_O_SENTINEL.reshape(1, 3), atol=1e-2)

    def test_auth_res_idx_does_not_change_designed_npz(self, tmp_path: Path) -> None:
        spec = OutputSpecification.model_validate(
            {
                "output_dir": str(tmp_path),
                "formats": ["pdb", "npz"],
                "save_trajectory": False,
                "primary_residue_index_key": "auth_res_idx",
            }
        )
        pdb_path, npz_path = _save_designed_pdb(tmp_path, spec)
        assert npz_path is not None
        loaded = Structure.load_from_npz(str(npz_path))
        assert loaded.residues["auth_res_idx"].tolist() == [RES_IDX_SEQID]
        assert _pdb_seqids(pdb_path) == [AUTH_SEQID]


class TestParserPathSaveKey:
    def test_parse_design_dict_threads_auth_res_idx(self, tmp_path: Path) -> None:
        design_dict = {
            "featurization_pipelines": {"feat": {"ref_structure": "path/to/structure.pdb"}},
            "output_specifications": {
                "out": {
                    "output_dir": str(tmp_path),
                    "formats": ["pdb"],
                    "save_trajectory": False,
                    "primary_residue_index_key": "auth_res_idx",
                    "save_raw_context_output": True,
                }
            },
            "simulation_pipelines": {"sim": {"type": "sde_simulation"}},
            "design_pipelines": {
                "design": {
                    "featurization_pipeline": "feat",
                    "simulation_pipeline": "sim",
                    "output_specification": "out",
                }
            },
        }
        with (
            patch(
                "dynopsi.design_pipeline_parser.get_featurization_pipeline_from_config",
                return_value=MagicMock(),
            ),
            patch(
                "dynopsi.design_pipeline_parser.get_simulation_pipeline_from_config",
                return_value=MagicMock(),
            ),
            patch("dynopsi.design_pipeline_parser.DesignPipeline") as mock_design,
        ):
            mock_design.return_value = MagicMock()
            DesignLanguageParser().parse_design_dict(design_dict)

        spec = mock_design.call_args.kwargs["output_specification"]
        assert isinstance(spec, OutputSpecification)
        assert spec.observable_output_configs["X_T"].primary_residue_index_key == "auth_res_idx"
        assert spec.observable_output_configs["X_T"].save_raw_context_output is True


def _features_with_binder_context_and_inpaint() -> Features:
    feature_dict = make_empty_chain(
        seq_length=4,
        atoms_to_include=AtomFilter.BB,
        res_type="ALA",
        inpaint_feature=InpaintFeature.BINDER_INPAINT,
    )
    feature_dict["inpaint_feature"] = torch.tensor(
        [
            InpaintFeature.BINDER_CONTEXT,
            InpaintFeature.BINDER_CONTEXT,
            InpaintFeature.BINDER_INPAINT,
            InpaintFeature.BINDER_INPAINT,
        ]
    )
    feature_dict["auth_residue_index"] = torch.tensor(
        [AUTH_SEQID, AUTH_SEQID + 1, AUTH_SEQID + 2, AUTH_SEQID + 3], dtype=torch.long
    )
    return Features(feature_dict=feature_dict, atoms_to_include=AtomFilter.BB, auth_chain_id_table={0: "A"})


def test_observable_output_config_defaults_to_res_idx() -> None:
    cfg = ObservableOutputConfig(state_type="clean_data_coords", outputs={"final": ("pdb",)})
    assert cfg.primary_residue_index_key == "res_idx"


def test_to_structure_fills_both_indices_when_binder_inpaint_and_context_coexist() -> None:
    features = _features_with_binder_context_and_inpaint()
    structure = features.to_structure()
    assert structure.residues["auth_res_idx"].tolist() == [
        AUTH_SEQID,
        AUTH_SEQID + 1,
        AUTH_SEQID + 2,
        AUTH_SEQID + 3,
    ]
    assert structure.residues["res_idx"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("noise_target_context", [False, True], ids=["binder-design", "noise-all"])
def test_canonical_pdb_keeps_full_reference_target_and_backbone_binder(
    tmp_path: Path, noise_target_context: bool
) -> None:
    features = _target_features_with_unnoised_reference_context()
    if noise_target_context:
        features.feature_dict["noised_mask"] = torch.tensor([True, True], dtype=torch.bool)
    observable = Observable()
    coordinates = features.feature_dict["coords"]
    observable.record("X_T", step_num=0, value={"coords": coordinates.reshape(1, 2, -1)})

    files = observable.save_clean_data_coords(
        save_dir=tmp_path,
        timesteps="final",
        output_formats=("pdb",),
        observable_key="X_T",
        reference_features_key="features",
        reference_features=features,
        start_index=0,
        save_raw_context_output=True,
    )

    canonical_path = next(Path(output["path"]) for output in files if not Path(output["path"]).name.startswith("raw"))
    reference_target = features.input_reference_structure
    assert reference_target is not None
    assert set(_pdb_atom_names(canonical_path, "A")) == set(reference_target.atoms["atom_name"])
    assert set(_pdb_atom_names(canonical_path, "B")) == {"N", "CA", "C", "O"}
    np.testing.assert_allclose(
        _pdb_atom_coords(canonical_path, "A", "CB"),
        REFERENCE_CB_SENTINEL.reshape(1, 3),
        atol=1e-2,
    )

    raw_paths = [Path(output["path"]) for output in files if Path(output["path"]).name.startswith("raw")]
    assert len(raw_paths) == int(noise_target_context)
    if raw_paths:
        assert set(_pdb_atom_names(raw_paths[0], "A")) == {"N", "CA", "C", "O"}


def test_observable_constructs_raw_context_once_for_multiple_formats(tmp_path: Path) -> None:
    canonical = SimpleNamespace(atoms={"coords": np.array([[0.0, 0.0, 0.0]])})
    raw = SimpleNamespace(atoms={"coords": np.array([[1.0, 0.0, 0.0]])})
    reference_features = MagicMock()
    reference_features.input_reference_structure = object()
    reference_features.feature_dict = {
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT]),
        "noised_mask": torch.tensor([True]),
        "token_pad_mask": torch.tensor([True]),
    }
    reference_features.design_coordinates_to_structure.side_effect = (
        lambda _coordinates, *, use_reference_for_noised_context=False: (
            [canonical] if use_reference_for_noised_context else [raw]
        )
    )
    reference_features.get_metadata.return_value = {}
    observable = Observable(
        observable_output_configs={
            "X_T": ObservableOutputConfig(
                state_type="clean_data_coords",
                outputs={"final": ("pdb", "mmcif")},
                features_keys=["features"],
                save_raw_context_output=True,
            )
        }
    )
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})

    with (
        patch("dynopsi.simulation.observable.reference_features_for_sample", return_value=reference_features),
        patch("dynopsi.simulation.observable.save_structures_to_file"),
    ):
        files = observable.save(save_dir=tmp_path, start_index=0, reference_features={"features": reference_features})

    assert {Path(output["path"]).name for output in files if output["output_format"] in {"pdb", "mmcif"}} == {
        "X_T_final_state_0.pdb",
        "rawcontext_X_T_final_state_0.pdb",
        "X_T_final_state_0.cif",
        "rawcontext_X_T_final_state_0.cif",
    }
    assert [
        call.kwargs.get("use_reference_for_noised_context", False)
        for call in reference_features.design_coordinates_to_structure.call_args_list
    ] == [True, False]


def test_observable_skips_raw_structure_without_noised_context(tmp_path: Path) -> None:
    canonical = SimpleNamespace(atoms={"coords": np.array([[0.0, 0.0, 0.0]])})
    reference_features = MagicMock()
    reference_features.input_reference_structure = object()
    reference_features.feature_dict = {
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT]),
        "noised_mask": torch.tensor([False]),
        "token_pad_mask": torch.tensor([True]),
    }
    reference_features.design_coordinates_to_structure.return_value = [canonical]
    reference_features.get_metadata.return_value = {}
    observable = Observable()
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})

    with (
        patch("dynopsi.simulation.observable.reference_features_for_sample", return_value=reference_features),
        patch("dynopsi.simulation.observable.save_structures_to_file"),
    ):
        files = observable.save_clean_data_coords(
            save_dir=tmp_path,
            timesteps="final",
            output_formats=("pdb",),
            observable_key="X_T",
            reference_features_key="features",
            reference_features=reference_features,
            start_index=0,
            save_raw_context_output=True,
        )

    assert [Path(output["path"]).name for output in files] == ["X_T_final_state_0.pdb"]
    reference_features.design_coordinates_to_structure.assert_called_once()
    assert reference_features.design_coordinates_to_structure.call_args.kwargs == {
        "use_reference_for_noised_context": True
    }


def _noised_context_features(
    structures_for_reference: list[MagicMock], structures_for_model: list[MagicMock]
) -> MagicMock:
    reference_features = MagicMock()
    reference_features.input_reference_structure = object()
    reference_features.feature_dict = {
        "inpaint_feature": torch.tensor([InpaintFeature.TARGET_CONTEXT]),
        "noised_mask": torch.tensor([True]),
        "token_pad_mask": torch.tensor([True]),
    }
    reference_features.design_coordinates_to_structure.side_effect = (
        lambda _coordinates, *, use_reference_for_noised_context=False: (
            structures_for_reference if use_reference_for_noised_context else structures_for_model
        )
    )
    reference_features.get_metadata.return_value = {}
    reference_features.featurization_metadata = {}
    return reference_features


@pytest.mark.parametrize("output_formats", [("npz",), ("pdb", "npz")])
def test_observable_writes_raw_context_npz_from_model_coordinates(
    tmp_path: Path, output_formats: tuple[str, ...]
) -> None:
    canonical = MagicMock()
    raw = MagicMock()
    canonical_sequential = MagicMock()
    raw_sequential = MagicMock()
    canonical.with_sequential_auth_res_idx.return_value = canonical_sequential
    raw.with_sequential_auth_res_idx.return_value = raw_sequential
    reference_features = _noised_context_features([canonical], [raw])
    observable = Observable()
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})

    with patch("dynopsi.simulation.observable.save_structures_to_file"):
        files = observable.save_clean_data_coords(
            save_dir=tmp_path,
            timesteps="final",
            output_formats=output_formats,
            observable_key="X_T",
            reference_features_key="features",
            reference_features=reference_features,
            start_index=0,
            save_raw_context_output=True,
        )

    names = {Path(output["path"]).name for output in files}
    expected = {"X_T_final_state_0.npz", "rawcontext_X_T_final_state_0.npz"}
    if "pdb" in output_formats:
        expected |= {"X_T_final_state_0.pdb", "rawcontext_X_T_final_state_0.pdb"}
    assert names == expected
    assert Path(canonical_sequential.dump_to_npz.call_args.args[0]).name == "X_T_final_state_0.npz"
    assert Path(raw_sequential.dump_to_npz.call_args.args[0]).name == "rawcontext_X_T_final_state_0.npz"
    assert [
        call.kwargs["use_reference_for_noised_context"]
        for call in reference_features.design_coordinates_to_structure.call_args_list
    ] == [True, False]


def test_observable_writes_raw_context_trajectory_npz_from_model_coordinates(tmp_path: Path) -> None:
    canonical = [MagicMock(), MagicMock()]
    raw = [MagicMock(), MagicMock()]
    reference_features = _noised_context_features(canonical, raw)
    observable = Observable()
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})
    observable.record("X_T", step_num=1, value={"coords": torch.ones(1, 1, 9)})

    with patch("dynopsi.simulation.observable.MultiModelStructure") as multi_cls:
        multi = MagicMock()
        multi_cls.from_structures.return_value = multi
        observable.save_clean_data_coords(
            save_dir=tmp_path,
            timesteps="any",
            output_formats=("npz",),
            observable_key="X_T",
            reference_features_key="features",
            reference_features=reference_features,
            start_index=0,
            save_raw_context_output=True,
        )

    canonical_sequential = [structure.with_sequential_auth_res_idx.return_value for structure in canonical]
    raw_sequential = [structure.with_sequential_auth_res_idx.return_value for structure in raw]
    assert multi_cls.from_structures.call_args_list[0].args[0] == canonical_sequential
    assert multi_cls.from_structures.call_args_list[1].args[0] == raw_sequential
    assert [Path(call.args[0]).name for call in multi.dump_to_npz.call_args_list] == [
        "X_T_trajectory_0.npz",
        "rawcontext_X_T_trajectory_0.npz",
    ]


def test_observable_omits_raw_context_npz_when_disabled(tmp_path: Path) -> None:
    structure = MagicMock()
    reference_features = MagicMock()
    reference_features.design_coordinates_to_structure.return_value = [structure]
    reference_features.get_metadata.return_value = {}
    reference_features.featurization_metadata = {}
    observable = Observable()
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})

    files = observable.save_clean_data_coords(
        save_dir=tmp_path,
        timesteps="final",
        output_formats=("npz",),
        observable_key="X_T",
        reference_features_key="features",
        reference_features=reference_features,
        start_index=0,
        save_raw_context_output=False,
    )

    assert [Path(output["path"]).name for output in files] == ["X_T_final_state_0.npz"]
    reference_features.design_coordinates_to_structure.assert_called_once()
    assert reference_features.design_coordinates_to_structure.call_args.kwargs == {
        "use_reference_for_noised_context": True
    }


def test_observable_writes_multimodel_mmcif_trajectory_at_sparse_timesteps(tmp_path: Path) -> None:
    features = _target_features_with_noised_reference_context()
    observable = Observable()
    n_residues = int(features.feature_dict["residue_index"].shape[0])
    coordinates = features.feature_dict["coords"].reshape(1, n_residues, -1)
    observable.record("X_T", step_num=3, value={"coords": coordinates})
    observable.record("X_T", step_num=11, value={"coords": coordinates + 1.0})

    files = observable.save_clean_data_coords(
        save_dir=tmp_path,
        timesteps="any",
        output_formats=("mmcif",),
        observable_key="X_T",
        reference_features_key="features",
        reference_features=features,
        start_index=0,
        save_raw_context_output=True,
    )

    paths = {Path(output["path"]).name: Path(output["path"]) for output in files}
    assert set(paths) == {
        "X_T_trajectory_0.cif",
        "rawcontext_X_T_trajectory_0.cif",
    }
    assert len(gemmi.read_structure(str(paths["X_T_trajectory_0.cif"]))) == 2
    assert len(gemmi.read_structure(str(paths["rawcontext_X_T_trajectory_0.cif"]))) == 2


def test_observable_omits_raw_context_output_when_disabled(tmp_path: Path) -> None:
    canonical = SimpleNamespace(atoms={"coords": np.array([[0.0, 0.0, 0.0]])})
    reference_features = MagicMock()
    reference_features.design_coordinates_to_structure.return_value = [canonical]
    reference_features.get_metadata.return_value = {}
    observable = Observable(
        observable_output_configs={
            "X_T": ObservableOutputConfig(
                state_type="clean_data_coords",
                outputs={"final": ("pdb",)},
                save_raw_context_output=False,
            )
        }
    )
    observable.record("X_T", step_num=0, value={"coords": torch.zeros(1, 1, 9)})

    with (
        patch("dynopsi.simulation.observable.reference_features_for_sample", return_value=reference_features),
        patch("dynopsi.simulation.observable.save_structures_to_file"),
    ):
        files = observable.save_clean_data_coords(
            save_dir=tmp_path,
            timesteps="final",
            output_formats=("pdb",),
            observable_key="X_T",
            reference_features_key="features",
            reference_features=reference_features,
            start_index=0,
            save_raw_context_output=False,
        )

    assert [Path(output["path"]).name for output in files] == ["X_T_final_state_0.pdb"]
    reference_features.design_coordinates_to_structure.assert_called_once()
