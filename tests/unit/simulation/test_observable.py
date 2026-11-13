import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from dynopsi.simulation.observable import Observable, ObservableOutputConfig

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_config(
    state_type: str = "any",
    outputs: dict | None = None,
    features_keys: list[str] | None = None,
) -> ObservableOutputConfig:
    cfg = ObservableOutputConfig(
        state_type=state_type,
        outputs=outputs or {"any": ("npz",)},
    )
    for k in features_keys or []:
        cfg.add_features_key(k)
    return cfg


def make_observable(
    keys_any: list[str] = (),
    keys_final: list[str] = (),
    total_batch_size: int | None = None,
) -> Observable:
    configs: dict[str, ObservableOutputConfig] = {}
    for k in keys_any:
        configs[k] = make_config(outputs={"any": ("npz",)})
    for k in keys_final:
        configs[k] = make_config(outputs={"final": ("npz",)})
    return Observable(observable_output_configs=configs, total_batch_size=total_batch_size)


def make_value(batch_size: int = 2, seq_len: int = 4, feat_dim: int = 3) -> dict[str, torch.Tensor]:
    return {"coords": torch.ones(batch_size, seq_len, feat_dim)}


# ---------------------------------------------------------------------------
# ObservableOutputConfig
# ---------------------------------------------------------------------------


class TestObservableOutputConfig:
    @pytest.mark.parametrize(
        "state_type",
        ["clean_data_coords", "conditional_vector_fields", "step_updates", "any"],
    )
    def test_valid_construction(self, state_type: str) -> None:
        cfg = ObservableOutputConfig(state_type=state_type, outputs={"any": ("npz",)})
        assert cfg.state_type == state_type

    def test_clean_data_coords_allows_pdb_mmcif(self) -> None:
        cfg = ObservableOutputConfig(
            state_type="clean_data_coords",
            outputs={"final": ("pdb", "mmcif", "npz")},
        )
        assert "pdb" in cfg.outputs["final"]

    @pytest.mark.parametrize("state_type", ["conditional_vector_fields", "step_updates", "any"])
    @pytest.mark.parametrize("output_format", ["pdb", "mmcif"])
    def test_non_clean_data_coords_rejects_structure_file_format(self, state_type: str, output_format: str) -> None:
        with pytest.raises(ValidationError, match="Only npz format"):
            ObservableOutputConfig(state_type=state_type, outputs={"any": (output_format,)})

    def test_features_keys_empty_by_default(self) -> None:
        cfg = make_config()
        assert cfg.features_keys == []

    def test_raw_context_output_defaults_off(self) -> None:
        cfg = make_config()
        assert cfg.save_raw_context_output is False

    def test_add_features_key_appends(self) -> None:
        cfg = make_config()
        cfg.add_features_key("cond")
        assert "cond" in cfg.features_keys

    def test_add_features_key_idempotent(self) -> None:
        cfg = make_config()
        cfg.add_features_key("cond")
        cfg.add_features_key("cond")
        assert cfg.features_keys.count("cond") == 1


# ---------------------------------------------------------------------------
# Observable properties
# ---------------------------------------------------------------------------


class TestObservableProperties:
    def test_keys_to_record_any_timestep(self) -> None:
        obs = make_observable(keys_any=["X", "Y"], keys_final=["Z"])
        assert set(obs.keys_to_record_any_timestep) == {"X", "Y"}

    def test_keys_to_record_final_timestep(self) -> None:
        obs = make_observable(keys_any=["X"], keys_final=["Y", "Z"])
        assert set(obs.keys_to_record_final_timestep) == {"Y", "Z"}

    def test_key_with_both_any_and_final(self) -> None:
        # A config can declare outputs for both timesteps simultaneously.
        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",), "final": ("npz",)})
        obs = Observable(observable_output_configs={"X": cfg})
        assert "X" in obs.keys_to_record_any_timestep
        assert "X" in obs.keys_to_record_final_timestep

    def test_empty_configs_returns_empty_lists(self) -> None:
        obs = Observable()
        assert obs.keys_to_record_any_timestep == []
        assert obs.keys_to_record_final_timestep == []


# ---------------------------------------------------------------------------
# Observable.record()
# ---------------------------------------------------------------------------


class TestObservableRecord:
    def test_first_record_no_batch_slice_sets_total_batch_size(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value=make_value(batch_size=3))
        assert obs.total_batch_size == 3

    def test_first_record_with_batch_slice_sets_total_batch_size_from_stop(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value=make_value(batch_size=2), batch_slice=slice(0, 5))
        # total_batch_size comes from batch_slice.stop, not value.shape[0]
        assert obs.total_batch_size == 5

    def test_buffer_allocated_with_full_batch_size(self) -> None:
        obs = Observable(total_batch_size=4)
        obs.record("X", step_num=0, value=make_value(batch_size=2), batch_slice=slice(0, 2))
        buf = obs.get_observable_step("X", 0)
        assert buf is not None
        assert buf["coords"].shape[0] == 4

    def test_buffer_shape_matches_inner_dims(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value={"coords": torch.zeros(2, 7, 3)})
        buf = obs.get_observable_step("X", 0)
        assert buf["coords"].shape == (2, 7, 3)

    def test_recorded_value_is_written_into_slice(self) -> None:
        obs = Observable(total_batch_size=4)
        value = {"coords": torch.full((2, 4, 3), 7.0)}
        obs.record("X", step_num=0, value=value, batch_slice=slice(0, 2))
        buf = obs.get_observable_step("X", 0)
        assert torch.all(buf["coords"][:2] == 7.0)
        # Indices outside the slice remain zero from buffer initialization.
        assert torch.all(buf["coords"][2:] == 0.0)

    def test_concurrent_records_keep_both_slices(self) -> None:
        obs = Observable(total_batch_size=4)
        active = 0
        peak = 0
        mu = threading.Lock()
        second_inside = threading.Event()
        real_clone = torch.Tensor.clone

        def slow_clone(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            nonlocal active, peak
            del args, kwargs
            with mu:
                active += 1
                peak = max(peak, active)
                if active > 1:
                    second_inside.set()
            try:
                second_inside.wait(timeout=0.2)
                return real_clone(tensor)
            finally:
                with mu:
                    active -= 1

        def record_slice(start: int, fill: float) -> None:
            obs.record(
                "X",
                step_num=0,
                value={"coords": torch.full((2, 4, 3), fill)},
                batch_slice=slice(start, start + 2),
            )

        with patch.object(torch.Tensor, "clone", slow_clone):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(record_slice, 0, 1.0), pool.submit(record_slice, 2, 7.0)]
                for future in futures:
                    future.result(timeout=5)

        assert peak == 1
        buf = obs.get_observable_step("X", 0)
        assert buf is not None
        assert torch.equal(buf["coords"][:2], torch.full((2, 4, 3), 1.0))
        assert torch.equal(buf["coords"][2:], torch.full((2, 4, 3), 7.0))

    def test_chunked_execution_fills_full_buffer(self) -> None:
        # Simulate two chunks of 2 filling a batch of 4.
        obs = Observable(total_batch_size=4)
        chunk_a = {"coords": torch.full((2, 4, 3), 1.0)}
        chunk_b = {"coords": torch.full((2, 4, 3), 2.0)}
        obs.record("X", step_num=0, value=chunk_a, batch_slice=slice(0, 2))
        obs.record("X", step_num=0, value=chunk_b, batch_slice=slice(2, 4))
        buf = obs.get_observable_step("X", 0)
        assert torch.all(buf["coords"][:2] == 1.0)
        assert torch.all(buf["coords"][2:] == 2.0)

    def test_record_is_detached_clone(self) -> None:
        # Mutating the original tensor after record() must not affect the stored buffer.
        obs = Observable()
        original = torch.ones(2, 4, 3)
        obs.record("X", step_num=0, value={"coords": original})
        original.fill_(999.0)
        buf = obs.get_observable_step("X", 0)
        assert buf["coords"].max().item() != 999.0

    def test_second_record_same_key_different_step_allocates_new_buffer(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value=make_value(batch_size=2))
        obs.record("X", step_num=1, value=make_value(batch_size=2))
        assert obs.get_observable_step("X", 0) is not None
        assert obs.get_observable_step("X", 1) is not None

    def test_multiple_modalities_all_buffered(self) -> None:
        obs = Observable()
        value = {
            "coords": torch.zeros(2, 4, 3),
            "sequence": torch.zeros(2, 4, 21),
        }
        obs.record("X", step_num=0, value=value)
        buf = obs.get_observable_step("X", 0)
        assert "coords" in buf
        assert "sequence" in buf

    def test_second_record_same_key_same_step_overwrites_slice(self) -> None:
        # Edge case: recording the same (key, step) twice with the same slice
        # overwrites rather than accumulates.
        obs = Observable(total_batch_size=2)
        obs.record("X", step_num=0, value={"coords": torch.full((2, 4, 3), 1.0)}, batch_slice=slice(0, 2))
        obs.record("X", step_num=0, value={"coords": torch.full((2, 4, 3), 9.0)}, batch_slice=slice(0, 2))
        buf = obs.get_observable_step("X", 0)
        assert torch.all(buf["coords"] == 9.0)


# ---------------------------------------------------------------------------
# get_observable / get_observable_step
# ---------------------------------------------------------------------------


class TestGetObservable:
    def test_get_observable_missing_key_returns_none(self) -> None:
        obs = Observable()
        assert obs.get_observable("nonexistent") is None

    def test_get_observable_returns_steps_dict(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value=make_value())
        obs.record("X", step_num=1, value=make_value())
        result = obs.get_observable("X")
        assert result is not None
        assert set(result.keys()) == {0, 1}

    def test_get_observable_step_missing_key_returns_none(self) -> None:
        obs = Observable()
        assert obs.get_observable_step("nonexistent", 0) is None

    def test_get_observable_step_missing_step_returns_none(self) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value=make_value())
        assert obs.get_observable_step("X", 99) is None

    def test_get_observable_step_returns_modality_dict(self) -> None:
        obs = Observable()
        obs.record("X", step_num=3, value={"coords": torch.zeros(2, 4, 3)})
        result = obs.get_observable_step("X", 3)
        assert result is not None
        assert "coords" in result


# ---------------------------------------------------------------------------
# save_observable_to_npz
# ---------------------------------------------------------------------------


class TestSaveObservableToNpz:
    def test_creates_one_file_per_batch_item(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X"])
        obs.record("X", step_num=0, value=make_value(batch_size=3))
        files = obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        assert len(files) == 3
        for i in range(3):
            assert (tmp_path / f"observable_outputs_{i}.npz").exists()

    def test_start_index_offsets_filenames(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X"])
        obs.record("X", step_num=0, value=make_value(batch_size=2))
        obs.save_observable_to_npz(save_dir=tmp_path, start_index=10)
        assert (tmp_path / "observable_outputs_10.npz").exists()
        assert (tmp_path / "observable_outputs_11.npz").exists()

    def test_npz_keys_follow_state_step_modality_scheme(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X"])
        obs.record("X", step_num=2, value={"coords": torch.zeros(1, 4, 3)})
        obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        data = np.load(tmp_path / "observable_outputs_0.npz")
        assert "X/2/coords" in data

    def test_npz_values_match_recorded_tensors(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X"])
        value = torch.full((1, 4, 3), 5.0)
        obs.record("X", step_num=0, value={"coords": value})
        obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        data = np.load(tmp_path / "observable_outputs_0.npz")
        assert np.allclose(data["X/0/coords"], value[0].numpy())

    def test_key_with_no_recorded_values_is_skipped_with_warning(self, tmp_path, caplog) -> None:
        # "X" is configured but never recorded — should log a warning and produce no file.
        obs = make_observable(keys_any=["X"])
        with caplog.at_level(logging.WARNING):
            files = obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        assert files == []
        assert "X" in caplog.text

    def test_empty_observable_produces_no_files(self, tmp_path) -> None:
        obs = Observable()
        files = obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        assert files == []
        assert list(tmp_path.iterdir()) == []

    def test_multiple_steps_all_appear_in_npz(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X"])
        obs.record("X", step_num=0, value={"coords": torch.zeros(1, 4, 3)})
        obs.record("X", step_num=1, value={"coords": torch.ones(1, 4, 3)})
        obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        data = np.load(tmp_path / "observable_outputs_0.npz")
        assert "X/0/coords" in data
        assert "X/1/coords" in data

    def test_multiple_state_keys_all_appear_in_npz(self, tmp_path) -> None:
        obs = make_observable(keys_any=["X", "Y"])
        obs.record("X", step_num=0, value={"coords": torch.zeros(1, 4, 3)})
        obs.record("Y", step_num=0, value={"coords": torch.ones(1, 4, 3)})
        obs.save_observable_to_npz(save_dir=tmp_path, start_index=0)
        data = np.load(tmp_path / "observable_outputs_0.npz")
        assert "X/0/coords" in data
        assert "Y/0/coords" in data


# ---------------------------------------------------------------------------
# save_clean_data_coords
# ---------------------------------------------------------------------------


class TestSaveCleanDataCoords:
    # Helpers

    @staticmethod
    def _make_ref_features(num_structures: int = 1) -> MagicMock:
        structures = [MagicMock() for _ in range(num_structures)]
        ref = MagicMock()
        ref.design_coordinates_to_structure.return_value = structures
        ref.get_metadata.return_value = {}
        ref.featurization_metadata = {}
        return ref

    @staticmethod
    def _record_coords(
        obs: Observable,
        key: str = "X",
        steps: tuple[int, ...] = (0,),
        batch_size: int = 1,
        seq_len: int = 4,
        flat: int = 9,
    ) -> None:
        for step in steps:
            obs.record(key, step_num=step, value={"coords": torch.zeros(batch_size, seq_len, flat)})

    def _call(
        self,
        obs: Observable,
        ref: MagicMock,
        *,
        tmp_path,
        timesteps: str = "final",
        output_formats: tuple = ("npz",),
        observable_key: str = "X",
        features_key: str = "feat",
        start_index: int = 0,
        features_label: str | None = None,
        primary_residue_index_key: str = "res_idx",
    ):
        return obs.save_clean_data_coords(
            save_dir=tmp_path,
            timesteps=timesteps,
            output_formats=output_formats,
            observable_key=observable_key,
            reference_features_key=features_key,
            reference_features=ref,
            start_index=start_index,
            features_label=features_label,
            primary_residue_index_key=primary_residue_index_key,
        )

    # Tests

    def test_returns_empty_when_observable_key_missing(self, tmp_path) -> None:
        obs = Observable()
        result = self._call(obs, MagicMock(), tmp_path=tmp_path)
        assert result == []

    def test_final_mode_passes_single_timestep_coords_to_design(self, tmp_path) -> None:
        obs = Observable()
        # Record three steps; only the max (step 2) should be forwarded as a single-frame stack.
        self._record_coords(obs, steps=(0, 1, 2))
        ref = self._make_ref_features(num_structures=1)
        self._call(obs, ref, tmp_path=tmp_path, timesteps="final")
        ref.design_coordinates_to_structure.assert_called_once()
        coords_arg = ref.design_coordinates_to_structure.call_args_list[0][0][0]
        assert coords_arg.shape[0] == 1  # 1 timestep

    def test_final_mode_uses_max_step(self, tmp_path) -> None:
        obs = Observable()
        obs.record("X", step_num=0, value={"coords": torch.full((1, 4, 9), 1.0)})
        obs.record("X", step_num=1, value={"coords": torch.full((1, 4, 9), 99.0)})
        ref = self._make_ref_features(num_structures=1)
        self._call(obs, ref, tmp_path=tmp_path, timesteps="final")
        coords_arg = ref.design_coordinates_to_structure.call_args[0][0]
        # max step (1) has value 99; step 0 has value 1
        assert coords_arg[0].mean().item() == pytest.approx(99.0)

    def test_any_mode_passes_all_steps_sorted(self, tmp_path) -> None:
        obs = Observable()
        # Insert out of order; sorted output should be (0, 1, 2)
        self._record_coords(obs, steps=(2, 0, 1))
        ref = self._make_ref_features(num_structures=3)
        self._call(obs, ref, tmp_path=tmp_path, timesteps="any")
        ref.design_coordinates_to_structure.assert_called_once()
        coords_arg = ref.design_coordinates_to_structure.call_args_list[0][0][0]
        assert coords_arg.shape[0] == 3  # 3 timesteps

    def test_any_mode_step_order_is_ascending(self, tmp_path) -> None:
        obs = Observable()
        obs.record("X", step_num=1, value={"coords": torch.full((1, 4, 9), 1.0)})
        obs.record("X", step_num=0, value={"coords": torch.full((1, 4, 9), 0.0)})
        ref = self._make_ref_features(num_structures=2)
        self._call(obs, ref, tmp_path=tmp_path, timesteps="any")
        coords_arg = ref.design_coordinates_to_structure.call_args[0][0]
        # timestep index 0 → step 0 (zeros), index 1 → step 1 (ones)
        assert coords_arg[0].mean().item() == pytest.approx(0.0)
        assert coords_arg[1].mean().item() == pytest.approx(1.0)

    def test_batch_calls_design_once_per_item(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs, batch_size=3)
        ref = self._make_ref_features(num_structures=1)
        self._call(obs, ref, tmp_path=tmp_path)
        assert ref.design_coordinates_to_structure.call_count == 3

    def test_file_naming_final_state_no_label(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        files = self._call(obs, ref, tmp_path=tmp_path, timesteps="final")
        assert Path(files[0]["path"]).name == "X_final_state_0.npz"

    def test_file_naming_trajectory_no_label(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs, steps=(0, 1))
        ref = self._make_ref_features(num_structures=2)
        files = self._call(obs, ref, tmp_path=tmp_path, timesteps="any")
        assert Path(files[0]["path"]).name == "X_trajectory_0.npz"

    def test_file_naming_includes_features_label(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        files = self._call(obs, ref, tmp_path=tmp_path, features_label="feat")
        assert Path(files[0]["path"]).name == "X_feat_final_state_0.npz"

    def test_start_index_offsets_filename(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs, batch_size=2)
        ref = self._make_ref_features()
        files = self._call(obs, ref, tmp_path=tmp_path, start_index=5)
        names = [Path(f["path"]).name for f in files]
        assert "X_final_state_5.npz" in names
        assert "X_final_state_6.npz" in names

    def test_single_structure_calls_dump_to_npz_directly(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        mock_structure = MagicMock()
        ref = MagicMock()
        ref.design_coordinates_to_structure.return_value = [mock_structure]
        ref.get_metadata.return_value = {}
        ref.featurization_metadata = {}
        self._call(obs, ref, tmp_path=tmp_path)
        sequential = mock_structure.with_sequential_auth_res_idx.return_value
        sequential.dump_to_npz.assert_called_once()

    def test_multiple_structures_uses_multimodel_npz(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs, steps=(0, 1))
        mock_structures = [MagicMock(), MagicMock()]
        ref = MagicMock()
        ref.design_coordinates_to_structure.return_value = mock_structures
        ref.get_metadata.return_value = {}
        ref.featurization_metadata = {}
        with patch("dynopsi.simulation.observable.MultiModelStructure") as mock_cls:
            mock_multi = MagicMock()
            mock_cls.from_structures.return_value = mock_multi
            self._call(obs, ref, tmp_path=tmp_path, timesteps="any")
        sequential_structures = [structure.with_sequential_auth_res_idx.return_value for structure in mock_structures]
        mock_cls.from_structures.assert_called_once_with(sequential_structures)
        mock_multi.dump_to_npz.assert_called_once()

    def test_pdb_output_calls_save_structures_to_file_with_pdb_extension(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        with patch("dynopsi.simulation.observable.save_structures_to_file") as mock_save:
            self._call(obs, ref, tmp_path=tmp_path, output_formats=("pdb",))
        mock_save.assert_called_once()
        assert str(mock_save.call_args[0][1]).endswith(".pdb")

    def test_pdb_output_passes_structures_and_author_index_key_directly(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        structures = ref.design_coordinates_to_structure.return_value
        with patch("dynopsi.simulation.observable.save_structures_to_file") as mock_save:
            self._call(
                obs,
                ref,
                tmp_path=tmp_path,
                output_formats=("pdb",),
                primary_residue_index_key="auth_res_idx",
            )
        mock_save.assert_called_once()
        assert mock_save.call_args.args[0] is structures
        assert mock_save.call_args.kwargs["primary_residue_index_key"] == "auth_res_idx"

    def test_mmcif_output_uses_cif_extension(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        with patch("dynopsi.simulation.observable.save_structures_to_file") as mock_save:
            self._call(obs, ref, tmp_path=tmp_path, output_formats=("mmcif",))
        mock_save.assert_called_once()
        assert str(mock_save.call_args[0][1]).endswith(".cif")

    def test_multiple_output_formats_produces_one_entry_each(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = self._make_ref_features()
        with patch("dynopsi.simulation.observable.save_structures_to_file"):
            files = self._call(obs, ref, tmp_path=tmp_path, output_formats=("npz", "pdb", "mmcif"))
        formats = {f["output_format"] for f in files}
        assert formats == {"npz", "pdb", "mmcif"}

    def test_returns_correct_file_output_config_fields(self, tmp_path) -> None:
        obs = Observable()
        self._record_coords(obs)
        ref = MagicMock()
        ref.design_coordinates_to_structure.return_value = [MagicMock()]
        ref.get_metadata.return_value = {"chain": "A"}
        ref.featurization_metadata = {}
        files = self._call(obs, ref, tmp_path=tmp_path, features_key="feat")
        assert len(files) == 1
        f = files[0]
        assert f["state_key"] == "X"
        assert f["features_key"] == "feat"
        assert f["timestep"] == "final"
        assert f["output_format"] == "npz"
        assert f["metadata"] == {"chain": "A"}


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_raises_if_features_keys_empty(self, tmp_path) -> None:
        # features_keys must be populated via add_features_key before saving.
        obs = make_observable(keys_any=["X"])
        obs.record("X", step_num=0, value=make_value(batch_size=1))
        with pytest.raises(ValueError, match="features_keys.*empty"):
            obs.save(save_dir=tmp_path, start_index=0, reference_features={})

    def test_save_non_clean_data_coords_only_writes_npz(self, tmp_path) -> None:
        # "any" state type never enters the clean_data_coords branch, so
        # save() produces the same npz files as save_observable_to_npz.
        cfg = ObservableOutputConfig(state_type="any", outputs={"any": ("npz",)})
        cfg.add_features_key("feat")
        obs = Observable(observable_output_configs={"X": cfg})
        obs.record("X", step_num=0, value=make_value(batch_size=1))
        files = obs.save(save_dir=tmp_path, start_index=0, reference_features={"feat": None})
        assert len(files) == 1
        assert files[0]["output_format"] == "npz"
