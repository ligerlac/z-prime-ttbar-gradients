"""Fileset resolution: turn dataset queries into runnable filesets.

Public surface:

- :class:`ProcessInfo`, :class:`UniqueProcessInfoDict`,
  :class:`DuplicateProcessInfoError`, :class:`FilesetResolver` — types and
  ABC.

- :func:`build_fileset` — top-level helper; merges multiple query
  callables and dispatches to a resolver.

The submodule contains the following resolvers:

- :class:`OpenDataPortalFilesetResolver`:
    resolves dataset query patterns via ``opendata.cern.ch``. No authentication.
    This resolver does not require extra dependencies to be used.
    It lives in :mod:`graep.inputs.open_data`.

- :class:`RucioFilesetResolver`:
    resolves dataset query patterns via ``rucio``. This resolver requires
    ``rucio`` to be installed. It lives in :mod:`graep.inputs.rucio`.

"""

from __future__ import annotations

from graep.inputs.export import export_fileset
from graep.inputs.queries import (
    DuplicateProcessInfoError,
    FilesetResolver,
    ProcessInfo,
    UniqueProcessInfoDict,
    build_fileset,
)

__all__ = [
    "DuplicateProcessInfoError",
    "FilesetResolver",
    "ProcessInfo",
    "UniqueProcessInfoDict",
    "build_fileset",
    "export_fileset",
]
