"""Top-level configuration for the CMS Open Data Z' -> tt-bar example.

Each section spec is built in its own module and imported here. The
notebooks consume :data:`config` directly.
"""

from __future__ import annotations

from graep.config.main import Config

from .queries import fileset_spec

config = Config(
    inputs=fileset_spec,
)
