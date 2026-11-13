"""Strict load of a fixture checkpoint that uses the published hyperparameter names."""

import hashlib
import inspect
from typing import Any

import pytest
import torch

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.loaders import CheckpointIntegrityError, CheckpointSelectionError, HuggingFaceLoader
from dynopsi.model.model_config import (
    CHECKPOINT_SHORT_NAMES,
    DEFAULT_DYNOPSI_MODEL_CONFIG,
    DynoPsiModelConfig,
    HuggingFaceLoaderConfig,
    PriorFreeConstructionError,
    default_checkpoint_label,
    resolve_checkpoint,
)


def _fixture_hyperparameters() -> dict[str, Any]:
    """Small architecture whose trunk, shared config, and distogram head use the v1 names."""
    return {
        "config_priors": {"coords": {"type": "isotropic_gaussian", "params": {"std": 10.0, "sample_shape": None}}},
        "config_interpolants": {
            "coords": {
                "type": "euclidean_interpolant",
                "params": {"interpolant_schedule": {"type": "linear_interpolant_schedule", "params": {}}},
            }
        },
        "config_denoiser": {
            "type": "ProteinModule",
            "params": {
                "embedder": {
                    "type": "dyno_psi_single_cond_pair_embedder",
                    "params": {
                        "embed_dim": 12,
                        "pair_embed_dim": 4,
                        "cond_embed_dim": 4,
                        "time_dim": 4,
                        "index_embed_dim": 4,
                        "use_self_cond": False,
                        "num_distance_bins": 4,
                        "num_distance_bins_sc": 4,
                        "num_seq_sep_bins": 5,
                        "scalar_feats": None,
                        "categorical_feats": None,
                        "per_residue_scalar_feats": None,
                        "per_residue_categorical_feats": None,
                        "use_inpaint_dgram": False,
                    },
                },
                "trunk": {
                    "type": "dyno_psi_transformer_trunk",
                    "params": {
                        "config_shared": {
                            "dim_single": 12,
                            "attn_bias_mode": "pair",
                            "dim_cond": 4,
                            "n_layers": 1,
                            "dim_pair": 4,
                        },
                        "config_attention_transition_block": {
                            "n_heads": 1,
                            "dim_head": 4,
                            "expansion_factor": 2,
                            "use_qk_layernorm": False,
                        },
                        "config_pair_update": {
                            "n_updates": 1,
                            "use_triangle_mult": False,
                            "dim_triangle": 4,
                            "transition_n": 2,
                        },
                    },
                },
                "output_head": {
                    "type": "dyno_psi_output_head",
                    "params": {
                        "config_coords_decoder_head": {"dim_single": 12, "use_layer_norm": False},
                        "config_distogram_head": {
                            "dim_pair": 4,
                            "no_bins": 4,
                            "use_layer_norm": False,
                        },
                    },
                },
            },
        },
        "num_atoms": 3,
        "is_cvf_prediction": {"coords": True},
    }


def test_fixture_accepts_v1_hparam_names_and_strict_loads(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    hp = _fixture_hyperparameters()
    shared = hp["config_denoiser"]["params"]["trunk"]["params"]["config_shared"]
    trunk = hp["config_denoiser"]["params"]["trunk"]["params"]
    distogram = hp["config_denoiser"]["params"]["output_head"]["params"]["config_distogram_head"]
    assert "dim_pair" in shared
    assert "config_pair_update" in trunk
    assert "dim_pair" in distogram

    built = DynoPsiModel(**hp)
    ckpt_path = tmp_path / "fixture.ckpt"
    torch.save({"state_dict": built.state_dict(), "hyper_parameters": hp}, ckpt_path)

    load_calls: list[dict[str, Any]] = []
    real_load = torch.load

    def _spy_load(*args: Any, **kwargs: Any) -> Any:
        load_calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("dynopsi.model.loaders.torch.load", _spy_load)
    monkeypatch.setattr("dynopsi.model.loaders.hf_hub_download", lambda **_kwargs: str(ckpt_path))

    with open(ckpt_path, "rb") as checkpoint_file:
        expected_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    loaded = HuggingFaceLoader(
        "dynotx/dynopsi",
        "fixture.ckpt",
        revision="fixture",
        expected_sha256=expected_sha256,
    ).load("cpu")

    assert inspect.signature(HuggingFaceLoader.load).parameters.keys() == {"self", "map_location"}
    assert load_calls
    assert all(call.get("weights_only") is True for call in load_calls)
    assert set(loaded.state_dict()) == set(built.state_dict())
    assert any(key.startswith("flow_matcher.denoiser.main_trunk.pair_update_layers") for key in loaded.state_dict())


def test_hugging_face_loader_rejects_checksum_mismatch(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / "fixture.ckpt"
    ckpt_path.write_bytes(b"not the published checkpoint")
    monkeypatch.setattr("dynopsi.model.loaders.hf_hub_download", lambda **_kwargs: str(ckpt_path))

    loader = HuggingFaceLoader(
        "dynotx/dynopsi",
        "fixture.ckpt",
        revision="fixture",
        expected_sha256="0" * 64,
    )
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        loader.read("cpu")


def test_resolve_checkpoint_accepts_published_names_and_defaults_to_cicada() -> None:
    v1 = resolve_checkpoint("v1", origin="the test")
    assert v1.loader.filename == "dynopsi-1.ckpt"
    assert v1.loader.revision == "346e377be69ca2a8a78e6f921c08e2f3aa7b6e18"
    assert CHECKPOINT_SHORT_NAMES["v1"].sha256 == "c99f19a445992b6f6d4792f2ba27a27625b20379554d5389c1f6d2b7f43d404f"

    cicada = resolve_checkpoint("v1.5-cicada", origin="the test")
    assert cicada.loader.filename == "dynopsi-1.5-cicada.ckpt"
    assert cicada.loader.revision == "f55d9e684659aaf627b5bfa4ab05f735a3398951"
    assert (
        CHECKPOINT_SHORT_NAMES["v1.5-cicada"].sha256
        == "7a5bbc43efde45bbf04489ef280d2067751ba6f8a652c3aab8b510b91d1b3d39"
    )
    assert default_checkpoint_label() == "v1.5-cicada"
    assert DEFAULT_DYNOPSI_MODEL_CONFIG.loader == cicada.loader

    with pytest.raises(CheckpointSelectionError, match="Unknown checkpoint"):
        resolve_checkpoint("missing", origin="the test")


def test_model_config_rejects_unpublished_checkpoint_before_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_download(**_kwargs: Any) -> str:
        raise AssertionError("unsupported checkpoint reached the downloader")

    monkeypatch.setattr("dynopsi.model.loaders.hf_hub_download", _unexpected_download)
    config = DynoPsiModelConfig(
        loader=HuggingFaceLoaderConfig(
            repo_id="third-party/model",
            filename="untrusted.ckpt",
            revision="main",
        )
    )

    with pytest.raises(CheckpointSelectionError, match="Unsupported checkpoint source"):
        _ = config.dynopsi_model


def test_prior_free_skips_download() -> None:
    config = DynoPsiModelConfig(
        loader=HuggingFaceLoaderConfig(repo_id="dynotx/dynopsi", filename="dynopsi-1.ckpt", revision="abc")
    )
    with DynoPsiModelConfig.prior_free():
        assert config.prior_stds == {"coords": 1.0}
        with pytest.raises(PriorFreeConstructionError, match="prior_free"):
            _ = config.dynopsi_model
