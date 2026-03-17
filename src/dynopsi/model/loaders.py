from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from dynopsi.model import DynoPsiModel


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


def _load_ckpt(
    ckpt_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> DynoPsiModel:

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    hp = ckpt["hyper_parameters"]
    if isinstance(hp, dict) and "kwargs" in hp:
        hp = hp["kwargs"]

    model = DynoPsiModel(**hp)

    _assert_state_dict_keys_match(ckpt["state_dict"], model)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


class DynoPsiLoader(ABC):
    @abstractmethod
    def load(self, map_location: str | torch.device = "cpu") -> DynoPsiModel:
        """Load and return a DynoPsiModel. Caller owns the returned instance."""
        ...


class HuggingFaceLoader(DynoPsiLoader):
    """Download a .ckpt from Hugging Face and rehydrate it into a DynoPsiModel."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
    ) -> None:
        """
        Parameters
        ----------
        repo_id : str
            Hugging Face repo (e.g. ``owner/model-name``).
        filename : str
            Checkpoint filename in the repo.
        """
        self.repo_id = repo_id
        self.filename = filename

    def load(self, map_location: str | torch.device = "cpu") -> DynoPsiModel:

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
        )
        return _load_ckpt(path, map_location=map_location)
