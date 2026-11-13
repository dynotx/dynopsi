from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynopsi.design_pipeline import DesignPipeline, OutputSpecification

__all__ = [
    "DesignPipeline",
    "DynoPsiBinderDesignPipeline",
    "OutputSpecification",
]


def __getattr__(name: str) -> object:
    if name in ("DesignPipeline", "DynoPsiBinderDesignPipeline", "OutputSpecification"):
        import warnings

        from dynopsi.design_pipeline import DesignPipeline, OutputSpecification  # noqa: F401

        if name == "DynoPsiBinderDesignPipeline":
            warnings.warn(
                "DynoPsiBinderDesignPipeline is deprecated; use DesignPipeline.",
                DeprecationWarning,
                stacklevel=2,
            )
            return DesignPipeline
        return {"DesignPipeline": DesignPipeline, "OutputSpecification": OutputSpecification}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
