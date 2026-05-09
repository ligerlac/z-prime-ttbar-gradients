"""Inputs spec for the CERN Open Data Portal."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from graep.config.inputs.base import FilesetSpec
from graep.inputs.open_data import OpenDataPortalFilesetResolver


class OpenDataPortalFilesetSpec(FilesetSpec):
    """Inputs configured for the CERN Open Data Portal."""

    timeout: float = Field(
        default=30.0,
        description="HTTP timeout for portal requests, in seconds.",
    )

    def make_resolver(self) -> Any:
        return OpenDataPortalFilesetResolver(
            cache_dir=self.cache_dir,
            force_refresh=self.force_refresh,
            timeout=self.timeout,
        )
