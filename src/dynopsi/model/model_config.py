from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import PrivateAttr

from dynopsi.model.dyno_psi_model import DynoPsiModel
from dynopsi.model.loaders import (
    CheckpointSelectionError,
    HuggingFaceLoader,
    hyperparameters,
    model_from_checkpoint,
    prior_stds_from_hyperparameters,
    registered_categorical_keys,
)
from dynopsi.utils.strict_config import StrictConfig

CHECKPOINT_REPO_ID = "dynotx/dynopsi"
_V1_REVISION = "346e377be69ca2a8a78e6f921c08e2f3aa7b6e18"
_V1_SHA256 = "c99f19a445992b6f6d4792f2ba27a27625b20379554d5389c1f6d2b7f43d404f"
_V1_5_CICADA_REVISION = "f55d9e684659aaf627b5bfa4ab05f735a3398951"
_V1_5_CICADA_SHA256 = "7a5bbc43efde45bbf04489ef280d2067751ba6f8a652c3aab8b510b91d1b3d39"
_PRIOR_FREE_PLACEHOLDER_STDS: dict[str, float] = {"coords": 1.0}
_prior_free: ContextVar[bool] = ContextVar("dynopsi_model_config_prior_free", default=False)


class PriorFreeConstructionError(RuntimeError):
    """A checkpoint load was attempted inside :meth:`DynoPsiModelConfig.prior_free`."""


@dataclass(frozen=True)
class PublishedCheckpoint:
    """Hugging Face file, commit, and checksum a checkpoint short name resolves to.

    Attributes
    ----------
    filename : str
        Checkpoint filename in :data:`CHECKPOINT_REPO_ID`.
    revision : str
        Repo git commit SHA to download.
    sha256 : str
        Expected SHA-256 of the downloaded checkpoint.
    """

    filename: str
    revision: str
    sha256: str


DEFAULT_CHECKPOINT_NAME = "v1.5-cicada"
CHECKPOINT_SHORT_NAMES: dict[str, PublishedCheckpoint] = {
    "v1": PublishedCheckpoint(filename="dynopsi-1.ckpt", revision=_V1_REVISION, sha256=_V1_SHA256),
    DEFAULT_CHECKPOINT_NAME: PublishedCheckpoint(
        filename="dynopsi-1.5-cicada.ckpt",
        revision=_V1_5_CICADA_REVISION,
        sha256=_V1_5_CICADA_SHA256,
    ),
}


class HuggingFaceLoaderConfig(StrictConfig):
    """Identify a supported published checkpoint on the Hugging Face Hub.

    Loading accepts only the exact repository, filename, and pinned revision
    associated with a published DynoPsi checkpoint short name.

    Attributes
    ----------
    repo_id : str
        Hugging Face repository id.
    filename : str
        Checkpoint filename in that repository.
    type : str
        Loader name. The public package only loads Hugging Face checkpoints.
    """

    type: Literal["huggingface"] = "huggingface"
    repo_id: str
    filename: str
    revision: str | None = None


class DynoPsiModelConfig(StrictConfig):
    """Lazy configuration for a published DynoPsi checkpoint.

    Attributes
    ----------
    loader : HuggingFaceLoaderConfig
        Hugging Face checkpoint to download.
    device_name : str
        Device to load the model on.
    use_self_conditioning : bool
        Whether sampling uses self-conditioning.
    cvf_eps : float
        Epsilon used when converting predictions.
    """

    loader: HuggingFaceLoaderConfig
    device_name: str = "cpu"
    use_self_conditioning: bool = True
    cvf_eps: float = 1e-6

    _dynopsi_model: DynoPsiModel | None = PrivateAttr(default=None)
    _prior_stds: dict[str, float] | None = PrivateAttr(default=None)
    _checkpoint_request: str | None = PrivateAttr(default=None)
    _checkpoint_origin: str | None = PrivateAttr(default=None)

    @classmethod
    @contextlib.contextmanager
    def prior_free(cls) -> Iterator[None]:
        """Skip checkpoint downloads while a config is being parsed.

        Inside the block, ``prior_stds`` returns a stand-in and any other load raises
        :class:`PriorFreeConstructionError`.
        """
        token = _prior_free.set(True)
        try:
            yield
        finally:
            _prior_free.reset(token)

    def declared_categorical_keys(self) -> set[str]:
        """Return registered conditioning keys declared by this checkpoint."""
        self._refuse_load_inside_prior_free()
        ckpt = self._loader().read("cpu")
        return registered_categorical_keys(hyperparameters(ckpt))

    @property
    def dynopsi_model(self) -> DynoPsiModel:
        self._ensure_loaded()
        if self._dynopsi_model is None:
            raise RuntimeError("Model failed to load.")
        return self._dynopsi_model

    @property
    def loaded_model(self) -> DynoPsiModel | None:
        """Return the model when a load has already succeeded."""
        return self._dynopsi_model

    @property
    def prior_stds(self) -> dict[str, float]:
        if _prior_free.get() and self._prior_stds is None:
            return dict(_PRIOR_FREE_PLACEHOLDER_STDS)
        self._ensure_loaded()
        if self._prior_stds is None:
            raise RuntimeError("Prior stds failed to load.")
        return self._prior_stds

    def to_device(self, device: torch.device) -> None:
        self.dynopsi_model.to_device(device)
        self.device_name = str(device)

    def _published_checkpoint(self) -> PublishedCheckpoint:
        for published in CHECKPOINT_SHORT_NAMES.values():
            if (
                self.loader.repo_id == CHECKPOINT_REPO_ID
                and self.loader.filename == published.filename
                and self.loader.revision == published.revision
            ):
                return published
        known = ", ".join(sorted(CHECKPOINT_SHORT_NAMES))
        raise CheckpointSelectionError(
            f"Unsupported checkpoint source. Use resolve_checkpoint() with a published short name: {known}."
        )

    def _loader(self) -> HuggingFaceLoader:
        published = self._published_checkpoint()
        return HuggingFaceLoader(
            self.loader.repo_id,
            self.loader.filename,
            published.revision,
            published.sha256,
        )

    def _refuse_load_inside_prior_free(self) -> None:
        if _prior_free.get():
            raise PriorFreeConstructionError("Checkpoint loading is disabled inside DynoPsiModelConfig.prior_free().")

    def _ensure_loaded(self) -> None:
        if self._dynopsi_model is not None:
            return
        self._refuse_load_inside_prior_free()
        try:
            model, hp = model_from_checkpoint(self._loader().read(self.device_name))
        except Exception as exc:
            if self._checkpoint_request is None:
                raise
            raise CheckpointSelectionError(
                f"Failed to load checkpoint {self._checkpoint_request!r} requested via {self._checkpoint_origin}."
            ) from exc
        self._dynopsi_model = model
        self._prior_stds = prior_stds_from_hyperparameters(hp)


def _published_hf_loader(published: PublishedCheckpoint) -> HuggingFaceLoaderConfig:
    return HuggingFaceLoaderConfig(
        repo_id=CHECKPOINT_REPO_ID,
        filename=published.filename,
        revision=published.revision,
    )


def _config_for(
    published: PublishedCheckpoint,
    *,
    request: str | None = None,
    origin: str | None = None,
) -> DynoPsiModelConfig:
    config = DynoPsiModelConfig(loader=_published_hf_loader(published))
    config._checkpoint_request = request
    config._checkpoint_origin = origin
    return config


DEFAULT_DYNOPSI_MODEL_CONFIG = _config_for(CHECKPOINT_SHORT_NAMES[DEFAULT_CHECKPOINT_NAME])


def default_checkpoint_label() -> str:
    """Short name of the default published checkpoint."""
    return DEFAULT_CHECKPOINT_NAME


def resolve_checkpoint(request: str, *, origin: str = "the checkpoint request") -> DynoPsiModelConfig:
    """Resolve a checkpoint short name to a published :class:`DynoPsiModelConfig`.

    Parameters
    ----------
    request : str
        Short name of a published checkpoint.
    origin : str
        Where the request came from, named in any error.

    Raises
    ------
    CheckpointSelectionError
        If ``request`` is not a known short name.
    """
    published = CHECKPOINT_SHORT_NAMES.get(request)
    if published is None:
        known = ", ".join(sorted(CHECKPOINT_SHORT_NAMES))
        raise CheckpointSelectionError(
            f"Unknown checkpoint {request!r} requested via {origin}. Known short names: {known}."
        )
    return _config_for(published, request=request, origin=origin)
