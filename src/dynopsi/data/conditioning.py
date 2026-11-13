"""Registry of constant-write categorical conditioning features.

A :class:`CategoricalCondition` describes a single integer written onto designed
tokens: which feature-dict key it fills, the ``IntEnum`` vocabulary it draws from,
the token scope it writes, and the inference-time ``type`` string of the featurizer
that writes it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, NoReturn

from pydantic import PlainValidator

from dynopsi.data.constants import DSSPLoopFeature, HomooligomerPartnerFeature, InpaintFeature


@dataclass(frozen=True)
class CategoricalCondition:
    """A constant integer written onto a designed-token mask.

    Attributes
    ----------
    key:
        Feature-dict key the writer fills, e.g. ``"chain_is_loopy"``.
    vocab:
        ``IntEnum`` the written value is drawn from. Pad and non-designed tokens are
        filled with ``vocab.MASK_OR_MISSING``.
    write_inpaint:
        Inpaint features whose non-pad tokens this condition writes.
    inference_type:
        YAML ``type`` string of the featurizer that writes this condition.
    reject_docking:
        Whether presence of this condition is rejected on docking tasks.
    mask_if_nonpad_chains_le:
        When set, ``MASK_OR_MISSING`` is written instead of the value for examples whose
        non-pad chain count is at or below this threshold; ``None`` applies no guard.
    """

    key: str
    vocab: type[IntEnum]
    write_inpaint: tuple[InpaintFeature, ...]
    inference_type: str
    reject_docking: bool = True
    mask_if_nonpad_chains_le: int | None = None


CATEGORICAL_CONDITIONS: tuple[CategoricalCondition, ...] = (
    CategoricalCondition(
        key="chain_is_loopy",
        vocab=DSSPLoopFeature,
        write_inpaint=(InpaintFeature.BINDER_INPAINT,),
        inference_type="chain_is_loopy_conditioning",
    ),
    CategoricalCondition(
        key="has_homooligomer_partner",
        vocab=HomooligomerPartnerFeature,
        write_inpaint=(InpaintFeature.BINDER_INPAINT, InpaintFeature.TARGET_INPAINT),
        inference_type="constant_write_conditioning",
        mask_if_nonpad_chains_le=1,
    ),
)


def get_categorical_condition(key: str) -> CategoricalCondition:
    for condition in CATEGORICAL_CONDITIONS:
        if condition.key == key:
            return condition
    known = sorted(condition.key for condition in CATEGORICAL_CONDITIONS)
    raise ValueError(f"Unknown design_conditioning key {key!r}. Registered keys: {known}")


class ConditioningCheckpointError(ValueError):
    """Registered conditioning keys and the checkpoint's declared keys disagree."""


def _vocab_member_names(vocab: type[IntEnum]) -> str:
    return ", ".join(member.name for member in vocab)


def _reject_condition_value(key: str, vocab: type[IntEnum], raw: object) -> NoReturn:
    members = _vocab_member_names(vocab)
    raise ValueError(
        f"design_conditioning.{key}: {raw!r} is not a valid {vocab.__name__} member. Allowed members: {members}."
    )


def _coerce_condition_value(key: str, vocab: type[IntEnum], raw: object) -> IntEnum:
    if isinstance(raw, bool):
        _reject_condition_value(key, vocab, raw)
    if isinstance(raw, vocab):
        return raw
    if isinstance(raw, int):
        try:
            return vocab(raw)
        except ValueError:
            _reject_condition_value(key, vocab, raw)
    if isinstance(raw, str):
        try:
            return vocab[raw]
        except KeyError:
            _reject_condition_value(key, vocab, raw)
    _reject_condition_value(key, vocab, raw)


def assert_registered_conditioning_matches(
    *,
    checkpoint_label: str,
    declared_keys: set[str],
    design_conditioning: Mapping[str, Mapping[str, object]] | None,
) -> None:
    """Raise ConditioningCheckpointError when the supplied and declared key sets differ.

    A key is supplied only when its variant map is non-empty.
    """
    supplied = {key for key, variants in (design_conditioning or {}).items() if variants}
    parts: list[str] = []
    mismatches = [
        *((key, "missing") for key in sorted(declared_keys - supplied)),
        *((key, "extra") for key in sorted(supplied - declared_keys)),
    ]
    for key, reason in mismatches:
        condition = get_categorical_condition(key)
        members = _vocab_member_names(condition.vocab)
        vocab = condition.vocab.__name__
        if reason == "missing":
            parts.append(
                f"design_conditioning.{key} is declared by checkpoint {checkpoint_label!r} "
                f"and is missing from this config. Allowed {vocab} members: {members}."
            )
        else:
            parts.append(
                f"design_conditioning.{key} is set in this config and is not declared by "
                f"checkpoint {checkpoint_label!r}. Allowed {vocab} members: {members}."
            )
    if parts:
        raise ConditioningCheckpointError(" ".join(parts))


def _normalize_design_conditioning(
    value: dict[str, dict[str, object]] | None,
) -> dict[str, dict[str, IntEnum]] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"design_conditioning must be a mapping or None, got {type(value).__name__}")
    normalized: dict[str, dict[str, IntEnum]] = {}
    for key, variants in value.items():
        condition = get_categorical_condition(key)
        if not isinstance(variants, dict):
            raise ValueError(f"design_conditioning[{key!r}] must be a mapping of variant name to value")
        normalized[key] = {name: _coerce_condition_value(key, condition.vocab, raw) for name, raw in variants.items()}
    return normalized


DesignConditioning = Annotated[
    dict[str, dict[str, int]] | None,
    PlainValidator(_normalize_design_conditioning),
]
"""``{feature: {name: value}}`` constant conditioning on designed tokens.

Unknown features fail. A value may be the enum member, its integer, or its name.
Bools and numeric strings are rejected. Several names under one feature produce one
design per name. An omitted or empty mapping leaves every conditioning feature unset.
"""
