"""Inputs spec for the rucio catalog.

Importing this module triggers the import of
:mod:`graep.inputs.rucio`, which requires the ``examples`` dependency
group (``uv sync --group examples``). Users without rucio installed
should import :mod:`graep.config.inputs.open_data` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import Field

from graep.config.inputs.base import FilesetSpec
from graep.inputs.rucio import RucioFilesetResolver


class RucioFilesetSpec(FilesetSpec):
    """Inputs configured for the rucio catalog. Requires an active VOMS proxy."""

    scope: str = Field(
        default="cms",
        description=(
            "Rucio scope passed to client.list_files. Defaults to 'cms'; "
            "set to your experiment's scope otherwise."
        ),
    )
    veto_rules: list[Callable[[str], bool]] = Field(
        default_factory=list,
        description=(
            "Callables (str -> bool). Resolved dataset names matching "
            "any rule are dropped before file listing."
        ),
    )

    def make_resolver(self) -> Any:
        return RucioFilesetResolver(
            cache_dir=self.cache_dir,
            force_refresh=self.force_refresh,
            scope=self.scope,
            veto_rules=self.veto_rules,
        )
