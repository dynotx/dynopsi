"""Strict load of a fixture checkpoint that uses the published hyperparameter names."""

import inspect
from typing import Any

import pytest
import torch

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.loaders import CheckpointSelectionError, HuggingFaceLoader
from dynopsi.model.model_config import (
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

    loaded = HuggingFaceLoader("dynotx/dynopsi", "fixture.ckpt", revision="fixture").load("cpu")

    assert inspect.signature(HuggingFaceLoader.load).parameters.keys() == {"self", "map_location"}
    assert load_calls
    assert all(call.get("weights_only") is True for call in load_calls)
    assert set(loaded.state_dict()) == set(built.state_dict())
    assert any(key.startswith("flow_matcher.denoiser.main_trunk.pair_update_layers") for key in loaded.state_dict())


def test_resolve_checkpoint_accepts_published_names_and_defaults_to_cicada() -> None:
    v1 = resolve_checkpoint("v1", origin="the test")
    assert v1.loader.filename == "dynopsi-1.ckpt"
    assert v1.loader.revision == "346e377be69ca2a8a78e6f921c08e2f3aa7b6e18"

    cicada = resolve_checkpoint("v1.5-cicada", origin="the test")
    assert cicada.loader.filename == "dynopsi-1.5-cicada.ckpt"
    assert cicada.loader.revision == "f55d9e684659aaf627b5bfa4ab05f735a3398951"
    assert default_checkpoint_label() == "v1.5-cicada"
    assert DEFAULT_DYNOPSI_MODEL_CONFIG.loader == cicada.loader

    with pytest.raises(CheckpointSelectionError, match="Unknown checkpoint"):
        resolve_checkpoint("missing", origin="the test")


def test_prior_free_skips_download() -> None:
    config = DynoPsiModelConfig(
        loader=HuggingFaceLoaderConfig(repo_id="dynotx/dynopsi", filename="dynopsi-1.ckpt", revision="abc")
    )
    with DynoPsiModelConfig.prior_free():
        assert config.prior_stds == {"coords": 1.0}
        with pytest.raises(PriorFreeConstructionError, match="prior_free"):
            _ = config.dynopsi_model
