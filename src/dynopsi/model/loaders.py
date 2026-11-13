from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from dynopsi.data.conditioning import CATEGORICAL_CONDITIONS, ConditioningCheckpointError
from dynopsi.model import DynoPsiModel


class CheckpointSelectionError(ValueError):
    """A checkpoint request did not name a published short name."""


class CheckpointIntegrityError(RuntimeError):
    """A downloaded checkpoint did not match its published SHA-256."""


@cache
def _verify_checkpoint_sha256(path: str, expected_sha256: str) -> None:
    with open(path, "rb") as checkpoint_file:
        actual_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise CheckpointIntegrityError(f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}.")


def _assert_state_dict_keys_match(
    ckpt_state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    ckpt_keys = set(ckpt_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    if ckpt_keys != model_keys:
        only_ckpt = ckpt_keys - model_keys
        only_model = model_keys - ckpt_keys
        msg = "Mismatch between checkpoint and model state_dict keys. "
        if only_ckpt:
            msg += f"Only in checkpoint ({len(only_ckpt)}): {sorted(only_ckpt)}. "
        if only_model:
            msg += f"Only in model ({len(only_model)}): {sorted(only_model)}."
        raise ValueError(msg)


def _read_checkpoint(ckpt_path: Path | str, map_location: str | torch.device) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if not isinstance(ckpt, dict) or "hyper_parameters" not in ckpt or "state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'hyper_parameters' and 'state_dict'.")
    return ckpt


def hyperparameters(ckpt: dict[str, Any]) -> dict[str, Any]:
    hp = ckpt["hyper_parameters"]
    if isinstance(hp, dict) and "kwargs" in hp:
        hp = hp["kwargs"]
    if not isinstance(hp, dict):
        raise TypeError(f"Checkpoint hyper_parameters must be a dict, got {type(hp).__name__}.")
    return hp


def model_from_checkpoint(ckpt: dict[str, Any]) -> tuple[DynoPsiModel, dict[str, Any]]:
    """Build a model from a checkpoint dict and strict-load its state dict."""
    hp = hyperparameters(ckpt)
    model = DynoPsiModel(**hp)
    _assert_state_dict_keys_match(ckpt["state_dict"], model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, hp


def prior_stds_from_hyperparameters(hp: dict[str, Any]) -> dict[str, float]:
    """Read per-modality isotropic-Gaussian prior stds from checkpoint hyperparameters."""
    try:
        config_priors = hp["config_priors"]
    except KeyError as error:
        raise KeyError("Checkpoint hyper_parameters missing 'config_priors'") from error
    out: dict[str, float] = {}
    for modality, cfg in config_priors.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"config_priors[{modality!r}] must be a dict, got {type(cfg).__name__}")
        prior_type = cfg.get("type")
        if prior_type != "isotropic_gaussian":
            raise ValueError(
                f"Unsupported prior type for modality {modality!r}: {prior_type!r} (expected 'isotropic_gaussian')"
            )
        params = cfg.get("params") or {}
        if "std" not in params:
            raise KeyError(f"config_priors[{modality!r}].params missing 'std'")
        std = params["std"]
        if isinstance(std, torch.Tensor):
            out[modality] = float(std.detach().cpu().item())
        else:
            out[modality] = float(std)
    if not out:
        raise ValueError("config_priors did not yield any prior std values.")
    return out


def registered_categorical_keys(hyper_parameters: dict[str, Any]) -> set[str]:
    """Return conditioning keys the checkpoint declares that this package registers."""
    try:
        feats = hyper_parameters["config_denoiser"]["params"]["embedder"]["params"]["per_residue_categorical_feats"]
    except KeyError as error:
        raise ConditioningCheckpointError(
            "Checkpoint hyper_parameters have no per_residue_categorical_feats entry."
        ) from error
    if isinstance(feats, dict):
        names = list(feats)
    elif isinstance(feats, list):
        names = feats
    else:
        raise ConditioningCheckpointError(
            f"Checkpoint per_residue_categorical_feats must be a list or a mapping, got {type(feats).__name__}."
        )
    registered = {condition.key for condition in CATEGORICAL_CONDITIONS}
    return {key for key in names if isinstance(key, str) and key in registered}


class DynoPsiLoader(ABC):
    @abstractmethod
    def read(self, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Return the raw checkpoint dict."""
        ...

    def load(self, map_location: str | torch.device = "cpu") -> DynoPsiModel:
        """Load and return a DynoPsiModel. Caller owns the returned instance."""
        model, _hp = model_from_checkpoint(self.read(map_location))
        return model


class HuggingFaceLoader(DynoPsiLoader):
    """Download a checkpoint from Hugging Face and rehydrate it into a DynoPsiModel."""

    def __init__(self, repo_id: str, filename: str, revision: str, expected_sha256: str) -> None:
        """
        Parameters
        ----------
        repo_id : str
            Hugging Face repository containing the published checkpoint.
        filename : str
            Checkpoint filename in the repo.
        revision : str
            Pinned repository commit containing the checkpoint.
        expected_sha256 : str
            Published SHA-256 of the checkpoint file.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self.expected_sha256 = expected_sha256

    def path(self) -> str:
        path = hf_hub_download(repo_id=self.repo_id, filename=self.filename, revision=self.revision)
        _verify_checkpoint_sha256(path, self.expected_sha256)
        return path

    def read(self, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Download and verify the checkpoint, then return the raw dict."""
        return _read_checkpoint(self.path(), map_location)
