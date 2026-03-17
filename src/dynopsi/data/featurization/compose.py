import itertools
import logging
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator

from dynopsi.data.featurization.base import Features, Featurizer
from dynopsi.data.featurization.registry import get_featurizer_from_config, register_featurizer

logger = logging.getLogger(__name__)


###################
### Composers
###################
class ComposeFeaturizer(Featurizer):
    """Base class for composers.

    Attributes
    ----------
    featurizers: Sequence[Featurizer], the list of featurizers to apply.
    """

    featurizers: Sequence[Featurizer] = Field(default_factory=list)

    @field_validator("featurizers", mode="before")
    @classmethod
    def validate_featurizers(cls, v: list[Featurizer | dict[str, Any]]) -> list[Featurizer]:
        result = []
        for f in v:
            if isinstance(f, dict):
                featurizer = get_featurizer_from_config(f)
            else:
                featurizer = f

            result.append(featurizer)

        return result

    @field_serializer("featurizers")
    def serialize_featurizers(self, featurizers: list[Featurizer]) -> list[dict[str, Any]]:
        return [f.model_dump(serialize_as_any=True) for f in featurizers]

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        raise NotImplementedError


@register_featurizer()
class ComposeSequentialFeaturizer(ComposeFeaturizer):
    """Composer that applies a list of featurizers in sequence to each input.

    In function notation:
    ComposeSequential(f1, f2, ...)(x1, x2, ...) == [f1(f2(f3(x1))), f1(f2(f3(x2))), ...]

    Parameters
    ----------
    featurizers : list[Featurizer]
        Featurizers to apply in order; each receives the output of the previous.

    Yields
    ------
    Features
        Output of the final featurizer in the pipeline for each input.

    Notes
    -----
    If no featurizers are provided, inputs are passed through unchanged.
    """

    type: Literal["compose_sequential"] = "compose_sequential"

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        for f in self.featurizers:
            inputs = f(inputs)
        yield from inputs


@register_featurizer()
class ComposeFanoutFeaturizer(ComposeFeaturizer):
    """Composer that applies multiple featurizers independently to each input.

    In function notation:
    ComposeFanout(f1, f2, ...)(x1, x2, ...) == [f1(x1), f1(x2), ..., f2(x1), f2(x2), ...]

    Parameters
    ----------
    featurizers : list[Featurizer]
        List of featurizers to apply independently to each input.

    Yields
    ------
    Features
        All outputs of each featurizer in sequence


    Notes
    -----
    If no featurizers are provided, inputs are passed through unchanged with
    a warning logged.
    """

    type: Literal["compose_fanout"] = "compose_fanout"

    def __call__(self, inputs: Iterable[Features]) -> Iterator[Features]:
        featurizers = self.featurizers
        if len(featurizers) == 0:
            logger.warning("No featurizers to apply")
            yield from inputs
            return
        else:
            streams = itertools.tee(inputs, len(featurizers))
            for f, stream in zip(featurizers, streams):
                yield from f(stream)
