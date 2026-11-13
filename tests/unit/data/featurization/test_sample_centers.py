from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from dynopsi.data.constants import AtomFilter
from dynopsi.data.featurization.base import Features
from dynopsi.data.featurization.inference_featurization_pipeline.binder_design import (
    DynoPsiBinderDesignFeaturizationPipeline,
)
from dynopsi.data.featurization.inference_featurization_pipeline.registry import (
    get_featurization_pipeline_from_config,
)
from dynopsi.data.featurization.primitives import BinderRgCenterOfMassEstimator, PredefinedCenterOfMassEstimator
from dynopsi.data.featurization.sample_centers import (
    PER_SAMPLE_FEATURES_METADATA_KEY,
    RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY,
    SAMPLE_CENTERS_OF_MASS_METADATA_KEY,
    draw_prior_com_offsets,
    narrow_per_sample_features,
    reference_features_for_sample,
)
from dynopsi.simulation.build_initial_state import InitializeFeatures
from tests.unit.simulation.test_build_initial_state import make_store

STRUCTURES_DIR = Path(__file__).resolve().parent / "missing_fixtures"

_PDL1_CIF = STRUCTURES_DIR / "pdl1_5o45.cif"
_PDL1_COM_0 = (22.265675, 11.73316, -24.249834)
_PDL1_COM_X = (26.265675, 9.73316, -21.249834)
_FAR_OFFSET_ANGSTROM = 10.0
_FEATURE_ATOL = 1e-4
_BACKBONE_ATOMS_PER_RESIDUE = 3
_COORD_DIM = 3
_PDL1_COM_FAR = (_PDL1_COM_0[0] + _FAR_OFFSET_ANGSTROM, _PDL1_COM_0[1], _PDL1_COM_0[2])


def _cpu_row_features(*, target_x: float) -> Features:
    n_res = 2
    n_atoms = n_res * _BACKBONE_ATOMS_PER_RESIDUE
    atom_to_token = torch.zeros(n_atoms, n_res)
    for res in range(n_res):
        start = res * _BACKBONE_ATOMS_PER_RESIDUE
        atom_to_token[start : start + _BACKBONE_ATOMS_PER_RESIDUE, res] = 1.0
    coords = torch.zeros(1, n_atoms, _COORD_DIM)
    coords[0, _BACKBONE_ATOMS_PER_RESIDUE:, 0] = target_x
    return Features(
        feature_dict={
            "coords": coords,
            "frames_idx": torch.arange(n_atoms, dtype=torch.long).reshape(n_res, _BACKBONE_ATOMS_PER_RESIDUE),
            "noised_mask": torch.tensor([True, False]),
            "token_pad_mask": torch.ones(n_res, dtype=torch.bool),
            "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
            "atom_to_token": atom_to_token,
            "prior_mean": torch.zeros(1, n_atoms, _COORD_DIM),
            "frame_resolved_mask": torch.ones(n_res),
            "token_index": torch.arange(n_res),
        },
        atoms_to_include=AtomFilter.BB,
    )


def _binder_pipeline(
    *,
    predefined_com: tuple[float, float, float],
    sample_centers: list[tuple[float, float, float]] | None,
    random_global_translation_std_angstrom: float = 0.0,
    prior_com_translation_std_angstrom: float = 0.0,
    prior_com_translation_seed: int | None = None,
) -> DynoPsiBinderDesignFeaturizationPipeline:
    return DynoPsiBinderDesignFeaturizationPipeline(
        name="pdl1_sample_center_equiv",
        index_type="auth_residue_index",
        structure_path=str(_PDL1_CIF),
        crops={"crop_A19_132": "A19-132"},
        hotspots={"hotspot_IYQM": "A54,A56,A66,A115"},
        new_chains_lengths=[8],
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[predefined_com])},
        binder_atoms_to_include=AtomFilter.BB,
        sample_centers_of_mass=sample_centers,
        random_global_translation_std_angstrom=random_global_translation_std_angstrom,
        prior_com_translation_std_angstrom=prior_com_translation_std_angstrom,
        prior_com_translation_seed=prior_com_translation_seed,
    )


def _binder_and_target_com(features: Features, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (design-chain COM, target COM) for a batch row of initialized binder features."""
    coords = features.feature_dict["coords"][row, 0]
    atom_to_token = features.feature_dict["atom_to_token"][row]
    noised = features.feature_dict["noised_mask"][row]
    token_pad = features.feature_dict["token_pad_mask"][row].bool()
    atom_valid = atom_to_token.sum(dim=-1) > 0
    atom_token = atom_to_token.argmax(dim=-1)
    binder_atoms = noised[atom_token] & atom_valid
    target_atoms = (~noised[atom_token]) & token_pad[atom_token] & atom_valid
    return coords[binder_atoms].mean(dim=0), coords[target_atoms].mean(dim=0)


def _stub_draw(monkeypatch: pytest.MonkeyPatch, offsets: torch.Tensor) -> None:
    monkeypatch.setattr(
        "dynopsi.data.featurization.inference_featurization_pipeline.binder_design.draw_prior_com_offsets",
        lambda num_samples, num_prior_coms, std_angstrom, *, seed: offsets,
    )


def _featurize(pipeline: DynoPsiBinderDesignFeaturizationPipeline) -> Features:
    return next(iter(pipeline([Features(atoms_to_include=AtomFilter.BB)])))["features"]


def _initialize(features: Features, *, batch_size: int) -> Features:
    store = make_store(features={"input_features": features}, batch_size=batch_size)
    InitializeFeatures(name="init", in_feat_loc_store="input_features", out_feat_loc_store="out").run(store)
    return store.get_features("out")


def test_sample_centers_require_matching_predefined_estimator() -> None:
    kwargs = dict(
        name="test",
        index_type="auth_residue_index",
        structure_path="unused.cif",
        crops={"crop": "A1-10"},
        hotspots={"hotspot": "A5"},
        new_chains_lengths=[40],
        binder_atoms_to_include=AtomFilter.BB,
        sample_centers_of_mass=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
    )
    with pytest.raises(ValidationError, match="exactly one predefined_center_of_mass"):
        DynoPsiBinderDesignFeaturizationPipeline(
            **kwargs,
            new_chains_centers_of_mass={"center": BinderRgCenterOfMassEstimator()},
        )
    with pytest.raises(ValidationError, match="must match the predefined"):
        DynoPsiBinderDesignFeaturizationPipeline(
            **kwargs,
            new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(0.0, 0.0, 0.0)])},
        )
    pipeline = DynoPsiBinderDesignFeaturizationPipeline(
        **kwargs,
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(1.0, 2.0, 3.0)])},
    )
    assert pipeline.sample_centers_of_mass is not None
    assert len(pipeline.sample_centers_of_mass) == 2


def test_sample_centers_require_single_featurization_product() -> None:
    with pytest.raises(
        ValidationError,
        match="single featurization product",
    ):
        DynoPsiBinderDesignFeaturizationPipeline(
            name="test",
            index_type="auth_residue_index",
            structure_path="unused.cif",
            crops={"crop": "A1-10"},
            hotspots={"hotspot": "A5"},
            new_chains_lengths=[40, 50],
            binder_atoms_to_include=AtomFilter.BB,
            sample_centers_of_mass=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
            new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(1.0, 2.0, 3.0)])},
        )


def test_sample_centers_derive_predefined_estimator_from_row_0_when_empty() -> None:
    pipeline = DynoPsiBinderDesignFeaturizationPipeline(
        name="test",
        index_type="auth_residue_index",
        structure_path="unused.cif",
        crops={"crop": "A1-10"},
        hotspots={"hotspot": "A5"},
        new_chains_lengths=[40],
        binder_atoms_to_include=AtomFilter.BB,
        sample_centers_of_mass=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        new_chains_centers_of_mass={},
    )
    estimators = list(pipeline.new_chains_centers_of_mass.values())
    assert len(estimators) == 1
    assert isinstance(estimators[0], PredefinedCenterOfMassEstimator)
    assert tuple(float(x) for x in estimators[0].centers_of_mass[0]) == (1.0, 2.0, 3.0)


def test_sample_centers_mutually_exclusive_with_random_global_translation() -> None:
    kwargs = dict(
        name="test",
        index_type="auth_residue_index",
        structure_path="unused.cif",
        crops={"crop": "A1-10"},
        hotspots={"hotspot": "A5"},
        new_chains_lengths=[40],
        binder_atoms_to_include=AtomFilter.BB,
        sample_centers_of_mass=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        random_global_translation_std_angstrom=2.0,
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(1.0, 2.0, 3.0)])},
    )
    with pytest.raises(
        ValidationError,
        match="sample_centers_of_mass.*random_global_translation_std_angstrom",
    ):
        DynoPsiBinderDesignFeaturizationPipeline(**kwargs)


def test_random_global_translation_std_rejects_negative() -> None:
    with pytest.raises(ValidationError, match="random_global_translation_std_angstrom must be non-negative"):
        DynoPsiBinderDesignFeaturizationPipeline(
            name="test",
            index_type="auth_residue_index",
            structure_path="unused.cif",
            crops={"crop": "A1-10"},
            hotspots={"hotspot": "A5"},
            new_chains_lengths=[40],
            binder_atoms_to_include=AtomFilter.BB,
            new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(1.0, 2.0, 3.0)])},
            random_global_translation_std_angstrom=-1.0,
        )


def test_registry_keeps_random_global_translation_std() -> None:
    pipeline = get_featurization_pipeline_from_config(
        {
            "type": "dynopsi_binder_design_featurization_pipeline",
            "name": "test_binder_design",
            "index_type": "original_residue_index",
            "structure_path": "path/to/structure.pdb",
            "hotspots": {"hs": "A5,A8"},
            "new_chains_lengths": [10],
            "new_chains_centers_of_mass": {
                "center": {
                    "type": "predefined_center_of_mass",
                    "centers_of_mass": [[25.0, 40.0, 30.0]],
                }
            },
            "binder_atoms_to_include": "ca",
            "random_global_translation_std_angstrom": 2.0,
        }
    )
    assert pipeline.random_global_translation_std_angstrom == 2.0


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_scalar_pipeline_writes_random_global_translation_std_metadata() -> None:
    with_std = _featurize(
        _binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=None, random_global_translation_std_angstrom=2.0)
    )
    assert with_std.featurization_metadata[RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY] == 2.0

    default = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=None))
    assert RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY not in default.featurization_metadata


def test_initialize_features_applies_per_sample_random_global_translation(monkeypatch: pytest.MonkeyPatch) -> None:
    features = _cpu_row_features(target_x=10.0)
    features.featurization_metadata[RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY] = 2.0
    coords_before = features.feature_dict["coords"].clone()
    prior_before = features.feature_dict["prior_mean"].clone()

    drawn = torch.tensor(
        [
            [[[1.0, 2.0, 3.0]]],
            [[[4.0, 5.0, 6.0]]],
        ]
    )

    def _fake_randn(*size: int, device=None, dtype=None):
        assert size == (2, 1, 1, 3)
        return drawn.to(device=device, dtype=dtype)

    monkeypatch.setattr("dynopsi.data.featurization.sample_centers.torch.randn", _fake_randn)
    out = _initialize(features, batch_size=2)
    for row in range(2):
        shift = drawn[row] * 2.0
        assert torch.allclose(out.feature_dict["coords"][row], coords_before + shift)
        assert torch.allclose(out.feature_dict["prior_mean"][row], prior_before + shift)
        assert torch.allclose(
            out.feature_dict["coords"][row] - coords_before, out.feature_dict["prior_mean"][row] - prior_before
        )


def test_initialize_features_skips_random_global_translation_without_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _cpu_row_features(target_x=10.0)
    called = False

    def _fake_randn(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.randn should not be called")

    monkeypatch.setattr("dynopsi.data.featurization.sample_centers.torch.randn", _fake_randn)
    _initialize(features, batch_size=2)
    assert not called


def test_initialize_features_skips_random_global_translation_when_std_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = _cpu_row_features(target_x=10.0)
    features.featurization_metadata[RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY] = 0.0
    called = False

    def _fake_randn(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.randn should not be called")

    monkeypatch.setattr("dynopsi.data.featurization.sample_centers.torch.randn", _fake_randn)
    _initialize(features, batch_size=2)
    assert not called


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_sample_center_independent_featurize_matches_scalar_at_com_x() -> None:
    at_com_x = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_X, sample_centers=None))
    at_list = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=[_PDL1_COM_0, _PDL1_COM_X]))

    feat_x = _initialize(at_com_x, batch_size=1)
    feat_list = _initialize(at_list, batch_size=2)

    assert feat_x.feature_dict.keys() == feat_list.feature_dict.keys()
    for key, value in feat_x.feature_dict.items():
        other = feat_list.feature_dict[key]
        if not isinstance(value, torch.Tensor):
            continue
        assert other.shape[0] == 2, key
        assert torch.allclose(value[0], other[1], atol=_FEATURE_ATOL), key


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_per_sample_featurize_does_not_mutate_shared_new_chains_estimator() -> None:
    pipeline = _binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=[_PDL1_COM_0, _PDL1_COM_X])
    original_estimator = pipeline._new_chains_featurizer[0].estimate_centers_of_mass_function
    _featurize(pipeline)
    assert pipeline._new_chains_featurizer[0].estimate_centers_of_mass_function is original_estimator


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_reference_features_for_sample_uses_per_sample_frame() -> None:
    carrier = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=[_PDL1_COM_0, _PDL1_COM_X]))
    per_sample = carrier.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY]
    assert per_sample[0].input_reference_structure is not None
    assert per_sample[1].input_reference_structure is not None
    assert not (
        per_sample[0].input_reference_structure.atoms["coords"]
        == per_sample[1].input_reference_structure.atoms["coords"]
    ).all()

    feature_coords = per_sample[1].feature_dict["coords"]
    # feature coords are (1, n_atoms, 3); align_coordinates expects (batch, n_atoms, 3)
    coords_stack = feature_coords[0].unsqueeze(0)
    assert coords_stack.ndim == 3
    assert coords_stack.shape[0] == 1

    via_selector = reference_features_for_sample(carrier, 1).design_coordinates_to_structure(
        coords_stack, use_reference_for_noised_context=False
    )
    via_direct = per_sample[1].design_coordinates_to_structure(coords_stack, use_reference_for_noised_context=False)
    assert len(via_selector) == 1
    assert len(via_direct) == 1
    assert (via_selector[0].atoms["coords"] == via_direct[0].atoms["coords"]).all()


def test_narrow_per_sample_features_slices_without_mutating_input() -> None:
    copies = [_cpu_row_features(target_x=float(i)) for i in range(4)]
    origin = copies[0].safe_copy()
    origin_list = list(copies)
    origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = origin_list
    origin.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = [(float(i), 0.0, 0.0) for i in range(4)]
    narrowed = narrow_per_sample_features(origin, slice(1, 3))
    assert origin_list is origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY]
    assert len(origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY]) == 4
    assert len(narrowed.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY]) == 2
    assert narrowed.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] == [
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_swapping_com_row_0_does_not_change_other_sample_features() -> None:
    shared = _PDL1_COM_X
    feat_inside_first = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=[_PDL1_COM_0, shared]))
    feat_far_first = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_FAR, sample_centers=[_PDL1_COM_FAR, shared]))

    out_inside = _initialize(feat_inside_first, batch_size=2)
    out_far = _initialize(feat_far_first, batch_size=2)

    assert out_inside.feature_dict.keys() == out_far.feature_dict.keys()
    for key, value in out_inside.feature_dict.items():
        other = out_far.feature_dict[key]
        if not isinstance(value, torch.Tensor):
            continue
        assert torch.allclose(value[1], other[1], atol=_FEATURE_ATOL), key
        if key == "coords":
            assert not torch.allclose(value[0], other[0], atol=_FEATURE_ATOL), key


def test_initialize_features_indexes_copies_by_sample_offset() -> None:
    copies = [_cpu_row_features(target_x=float(i)) for i in range(3)]
    origin = copies[0].safe_copy()
    origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = copies
    origin.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    narrowed = narrow_per_sample_features(origin, slice(2, 3))
    out = _initialize(narrowed, batch_size=1)
    assert torch.allclose(
        out.feature_dict["coords"][0, 0, _BACKBONE_ATOMS_PER_RESIDUE:, 0],
        torch.tensor([2.0, 2.0, 2.0]),
    )


def test_initialize_features_errors_when_copies_shorter_than_sample_slice() -> None:
    copies = [_cpu_row_features(target_x=0.0)]
    origin = copies[0].safe_copy()
    origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = copies
    origin.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = [(0.0, 0.0, 0.0)]
    with pytest.raises(ValueError, match="per-sample feature copies have 1 rows but batch_size is 2"):
        _initialize(origin, batch_size=2)


def test_initialize_features_raises_without_per_sample_copies() -> None:
    features = _cpu_row_features(target_x=10.0)
    features.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY] = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    with pytest.raises(ValueError, match="independently featurized copies"):
        _initialize(features, batch_size=2)


def test_initialize_features_collate_follows_origin_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    copies = [_cpu_row_features(target_x=10.0), _cpu_row_features(target_x=8.0)]
    origin = _cpu_row_features(target_x=10.0)
    origin.to_device(device)
    origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = copies
    if device.type == "cuda":
        assert copies[0].feature_dict["coords"].device.type == "cpu"
    out = _initialize(origin, batch_size=2)
    expected = origin.feature_dict["coords"].device
    assert out.feature_dict["coords"].device == expected
    assert out.feature_dict["true_coords"].device == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")
def test_store_to_device_moves_independent_sample_copies() -> None:
    copies = [_cpu_row_features(target_x=10.0), _cpu_row_features(target_x=8.0)]
    origin = _cpu_row_features(target_x=10.0)
    origin.featurization_metadata[PER_SAMPLE_FEATURES_METADATA_KEY] = copies
    store = make_store(features={"input_features": origin}, batch_size=2)
    store.to_device(torch.device("cuda"))
    assert origin.feature_dict["coords"].device.type == "cuda"
    assert copies[0].feature_dict["coords"].device.type == "cuda"
    assert copies[1].feature_dict["coords"].device.type == "cuda"


_D0 = (1.0, -2.0, 0.5)
_D1 = (-3.0, 4.0, -1.5)


def _mu_plus(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(m + o for m, o in zip(_PDL1_COM_0, offset, strict=True))  # type: ignore[return-value]


def _draw_features(pipeline: DynoPsiBinderDesignFeaturizationPipeline, *, num_samples: int) -> Features:
    feature_iter = pipeline.featurize_for_sampling([Features(atoms_to_include=AtomFilter.BB)], num_samples=num_samples)
    return next(iter(feature_iter))["features"]


def test_draw_prior_com_offsets_is_seeded() -> None:
    first = draw_prior_com_offsets(4, 2, 2.0, seed=7)
    repeat = draw_prior_com_offsets(4, 2, 2.0, seed=7)
    other_seed = draw_prior_com_offsets(4, 2, 2.0, seed=8)
    assert first.shape == (4, 2, _COORD_DIM)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other_seed)
    # Each (sample, prior_com) pair is an independent draw.
    assert not torch.equal(first[:, 0], first[:, 1])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    assert torch.equal(first, torch.randn(4, 2, _COORD_DIM, generator=generator) * 2.0)


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_sigma_zero_is_scalar_pin_without_per_sample_metadata() -> None:
    carrier = _featurize(_binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=None))
    assert SAMPLE_CENTERS_OF_MASS_METADATA_KEY not in carrier.featurization_metadata
    assert PER_SAMPLE_FEATURES_METADATA_KEY not in carrier.featurization_metadata
    assert RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY not in carrier.featurization_metadata


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_call_with_sigma_stays_on_scalar_pin() -> None:
    pipeline = _binder_pipeline(
        predefined_com=_PDL1_COM_0,
        sample_centers=None,
        prior_com_translation_std_angstrom=2.0,
        prior_com_translation_seed=0,
    )
    carrier = _featurize(pipeline)
    assert SAMPLE_CENTERS_OF_MASS_METADATA_KEY not in carrier.featurization_metadata
    assert PER_SAMPLE_FEATURES_METADATA_KEY not in carrier.featurization_metadata
    assert RANDOM_GLOBAL_TRANSLATION_STD_METADATA_KEY not in carrier.featurization_metadata


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_featurize_for_sampling_matches_explicit_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_draw(monkeypatch, torch.tensor([[list(_D0)], [list(_D1)]]))
    pipeline = _binder_pipeline(
        predefined_com=_PDL1_COM_0,
        sample_centers=None,
        prior_com_translation_std_angstrom=2.0,
        prior_com_translation_seed=0,
    )
    carrier = _draw_features(pipeline, num_samples=2)

    assert pipeline.sample_centers_of_mass is None
    centers = carrier.featurization_metadata[SAMPLE_CENTERS_OF_MASS_METADATA_KEY]
    # One binder center (K = 1) per sample.
    assert centers[0][0] == pytest.approx(_mu_plus(_D0))
    assert centers[0][0] != pytest.approx(_PDL1_COM_0)

    explicit = _featurize(_binder_pipeline(predefined_com=_mu_plus(_D0), sample_centers=[_mu_plus(_D0), _mu_plus(_D1)]))
    out_drawn = _initialize(carrier, batch_size=2)
    out_explicit = _initialize(explicit, batch_size=2)
    assert out_drawn.feature_dict.keys() == out_explicit.feature_dict.keys()
    for key, value in out_drawn.feature_dict.items():
        other = out_explicit.feature_dict[key]
        if isinstance(value, torch.Tensor):
            assert torch.allclose(value, other, atol=_FEATURE_ATOL), key


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_drawn_binder_com_is_centered_and_sitting_varies(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_draw(monkeypatch, torch.tensor([[list(_D0)], [list(_D1)]]))
    pipeline = _binder_pipeline(
        predefined_com=_PDL1_COM_0,
        sample_centers=None,
        prior_com_translation_std_angstrom=2.0,
        prior_com_translation_seed=0,
    )
    out = _initialize(_draw_features(pipeline, num_samples=2), batch_size=2)

    origin = torch.zeros(_COORD_DIM)
    binder_0, target_0 = _binder_and_target_com(out, 0)
    binder_1, target_1 = _binder_and_target_com(out, 1)
    assert torch.allclose(binder_0, origin, atol=_FEATURE_ATOL)
    assert torch.allclose(binder_1, origin, atol=_FEATURE_ATOL)
    assert not torch.allclose(target_0 - binder_0, target_1 - binder_1, atol=_FEATURE_ATOL)


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_scalar_global_translation_moves_complex_not_sitting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Featurize both pipelines before patching torch.randn, which the center-of-mass featurizer also uses.
    _stub_draw(monkeypatch, torch.tensor([[list(_D0)], [list(_D1)]]))
    drawn_carrier = _draw_features(
        _binder_pipeline(
            predefined_com=_PDL1_COM_0,
            sample_centers=None,
            prior_com_translation_std_angstrom=2.0,
            prior_com_translation_seed=0,
        ),
        num_samples=2,
    )
    scalar_carrier = _featurize(
        _binder_pipeline(predefined_com=_PDL1_COM_0, sample_centers=None, random_global_translation_std_angstrom=2.0)
    )
    drawn = _initialize(drawn_carrier, batch_size=2)

    translations = torch.tensor([[[list(_D0)]], [[list(_D1)]]])

    def _fake_randn(*size: int, device=None, dtype=None):
        return translations.to(device=device, dtype=dtype)

    monkeypatch.setattr("dynopsi.data.featurization.sample_centers.torch.randn", _fake_randn)
    global_translated = _initialize(scalar_carrier, batch_size=2)

    binder_0, target_0 = _binder_and_target_com(global_translated, 0)
    binder_1, target_1 = _binder_and_target_com(global_translated, 1)
    # A whole-complex lab-frame translation leaves the binder off the origin ...
    assert not torch.allclose(binder_0, torch.zeros(_COORD_DIM), atol=_FEATURE_ATOL)
    # ... and keeps the sitting identical across rows, unlike the binder-COM draw.
    assert torch.allclose(target_0 - binder_0, target_1 - binder_1, atol=_FEATURE_ATOL)
    assert not torch.allclose(
        global_translated.feature_dict["coords"], drawn.feature_dict["coords"], atol=_FEATURE_ATOL
    )


@pytest.mark.skipif(not _PDL1_CIF.is_file(), reason="pdl1_5o45.cif fixture missing")
def test_swapping_drawn_com_row_0_does_not_change_other_sample_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _binder_pipeline(
        predefined_com=_PDL1_COM_0,
        sample_centers=None,
        prior_com_translation_std_angstrom=2.0,
        prior_com_translation_seed=0,
    )
    _stub_draw(monkeypatch, torch.tensor([[list(_D0)], [list(_D1)]]))
    out_near = _initialize(_draw_features(pipeline, num_samples=2), batch_size=2)
    _stub_draw(monkeypatch, torch.tensor([[[20.0, 0.0, 0.0]], [list(_D1)]]))
    out_far = _initialize(_draw_features(pipeline, num_samples=2), batch_size=2)

    assert out_near.feature_dict.keys() == out_far.feature_dict.keys()
    for key, value in out_near.feature_dict.items():
        other = out_far.feature_dict[key]
        if not isinstance(value, torch.Tensor):
            continue
        assert torch.allclose(value[1], other[1], atol=_FEATURE_ATOL), key
        if key == "coords":
            assert not torch.allclose(value[0], other[0], atol=_FEATURE_ATOL), key


def _sigma_kwargs(**overrides) -> dict:
    kwargs = dict(
        name="test",
        index_type="auth_residue_index",
        structure_path="unused.cif",
        crops={"crop": "A1-10"},
        hotspots={"hotspot": "A5"},
        new_chains_lengths=[40],
        binder_atoms_to_include=AtomFilter.BB,
        new_chains_centers_of_mass={"center": PredefinedCenterOfMassEstimator(centers_of_mass=[(1.0, 2.0, 3.0)])},
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            dict(
                prior_com_translation_std_angstrom=2.0,
                prior_com_translation_seed=0,
                sample_centers_of_mass=[(1.0, 2.0, 3.0)],
            ),
            "mutually exclusive",
        ),
        (
            dict(
                prior_com_translation_std_angstrom=2.0,
                prior_com_translation_seed=0,
                random_global_translation_std_angstrom=2.0,
            ),
            "mutually exclusive",
        ),
        (
            dict(
                prior_com_translation_std_angstrom=2.0,
                prior_com_translation_seed=0,
                new_chains_centers_of_mass={"center": BinderRgCenterOfMassEstimator()},
            ),
            "exactly one predefined_center_of_mass",
        ),
        (
            dict(
                prior_com_translation_std_angstrom=2.0,
                prior_com_translation_seed=0,
                new_chains_centers_of_mass={},
            ),
            "New chains are required",
        ),
        (
            dict(
                prior_com_translation_std_angstrom=2.0,
                prior_com_translation_seed=0,
                new_chains_lengths=[40, 50],
            ),
            "single featurization product",
        ),
        (dict(prior_com_translation_std_angstrom=-1.0), "must be non-negative"),
        (dict(prior_com_translation_std_angstrom=2.0), "seed is required"),
        (dict(prior_com_translation_seed=0), "requires prior_com_translation_std_angstrom"),
    ],
)
def test_prior_com_translation_validation(overrides: dict, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        DynoPsiBinderDesignFeaturizationPipeline(**_sigma_kwargs(**overrides))


def test_registry_keeps_prior_com_translation() -> None:
    pipeline = get_featurization_pipeline_from_config(
        {
            "type": "dynopsi_binder_design_featurization_pipeline",
            "name": "test_binder_design",
            "index_type": "original_residue_index",
            "structure_path": "path/to/structure.pdb",
            "hotspots": {"hs": "A5,A8"},
            "new_chains_lengths": [10],
            "new_chains_centers_of_mass": {
                "center": {
                    "type": "predefined_center_of_mass",
                    "centers_of_mass": [[25.0, 40.0, 30.0]],
                }
            },
            "binder_atoms_to_include": "ca",
            "prior_com_translation_std_angstrom": 2.0,
            "prior_com_translation_seed": 11,
        }
    )
    assert pipeline.prior_com_translation_std_angstrom == 2.0
    assert pipeline.prior_com_translation_seed == 11
