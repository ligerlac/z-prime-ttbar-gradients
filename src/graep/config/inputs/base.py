"""Common base for inputs (datasets) configuration.

Concrete subclasses live in sibling modules of :mod:`graep.config.inputs`,
one per resolver:

- :mod:`graep.config.inputs.open_data`
- :mod:`graep.config.inputs.rucio`

Adding a new resolver: add a sibling module here, subclass
:class:`FilesetSpec`, add resolver-specific fields with
``Field(description=...)``, and implement :meth:`make_resolver`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _default_cache_dir() -> Path:
    """Default on-disk fileset cache: ``/tmp/graep/.cache``."""
    return Path("/tmp/graep/.cache")


class FilesetSpec(BaseModel):
    """Base class for inputs configuration.

    Holds the cross-resolver fields. Concrete subclasses add
    resolver-specific fields and implement :meth:`make_resolver`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    queries: list[Callable[..., Any]] = Field(
        ...,
        description=(
            "One or more zero-argument callables. Each returns a mapping "
            "of ProcessInfo to a dataset pattern (or a list of patterns). "
            "Multiple callables are merged via UniqueProcessInfoDict."
        ),
        min_length=1,
    )
    cache_dir: Path = Field(
        default_factory=_default_cache_dir,
        description=(
            "Directory where build_fileset writes its on-disk cache. "
            "Defaults to /tmp/graep/.cache."
        ),
    )
    force_refresh: bool = Field(
        default=False,
        description="If True, bypass any cached entry and recompute the fileset.",
    )
    max_files_per_sample: int | None = Field(
        default=None,
        description=(
            "Per-process cap on the number of file URIs returned. "
            "None means no cap."
        ),
    )
    variation: str = Field(
        default="nominal",
        description=(
            "Variation tag used to build the fileset key "
            "'<process>__<variation>'. Most analyses use 'nominal'."
        ),
    )

    def make_resolver(self) -> Any:
        """Construct the resolver this spec describes.

        Concrete subclasses override with the resolver-specific
        constructor call.
        """
        msg = f"{type(self).__name__} must implement make_resolver()"
        raise NotImplementedError(msg)
