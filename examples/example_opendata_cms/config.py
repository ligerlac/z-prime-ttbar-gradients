"""Top-level configuration for the CMS Open Data ztt example.

Each section spec is built in its own module and imported here. The
notebooks consume :data:`config` directly.
"""

from __future__ import annotations

from pathlib import Path

from graep.config.main import Config
from graep.config.output import OutputSpec

from .queries import fileset_spec

_HERE = Path(__file__).parent

config = Config(
    inputs=fileset_spec,
    output=OutputSpec(root=_HERE / "output"),
)
