"""Schema for the output (artefacts) section of the user config.

Holds a root output directory and an optional category map. The user
constructs an :class:`OutputSpec` in their config; the notebook builds
the runtime :class:`graep.output.manager.OutputManager` via
:meth:`OutputManager.from_spec`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from graep.output.manager import DEFAULT_CATEGORIES


def _default_root() -> Path:
    """Default root output directory: ``./output``."""
    return Path("output").resolve()


class OutputSpec(BaseModel):
    """Output (artefacts) section of the user config."""

    root: Path = Field(
        default_factory=_default_root,
        description=(
            "Directory under which all category subdirectories live. "
            "Defaults to ./output (resolved at config-construction time)."
        ),
    )
    categories: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_CATEGORIES),
        description=(
            "Mapping from category name to its subdirectory name "
            "(relative to root). Defaults match graep.output.manager."
            "DEFAULT_CATEGORIES. Override to rename a subdirectory or "
            "add a new category for this run."
        ),
    )
