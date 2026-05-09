"""Resolver that talks to the rucio catalog.

Requires an active VOMS proxy and the ``rucio`` / ``coffea`` Python
packages — these live in the ``examples`` dependency group, not the
framework core, so importing this module fails fast with a friendly
hint when those aren't installed.

The rucio scope (e.g. ``"cms"``, ``"atlas"``) is a constructor
parameter — defaults to ``"cms"`` because that's this framework's
primary user, but ATLAS or other-experiment users override at
construction time.

Resolve-time ``nevts`` / ``nevts_wt`` are placeholders (zero) — rucio
doesn't expose per-file event counts cheaply and we don't open files
here. :func:`graep.inputs.export_fileset` is the canonical step that
opens files and populates per-file counts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graep.inputs.queries import (
    FilesetDict,
    FilesetResolver,
    ProcessInfo,
    _normalise_value,
)

# rucio lives in the optional `examples` dependency group. We import it
# at module top (per the project's "imports always at file top" rule)
# but wrap with a try/except so the failure mode is a single, clear
# ImportError that points the user at `uv sync --group examples`.
try:
    from coffea.dataset_tools import rucio_utils
except ImportError as exc:  # pragma: no cover — environment dependent
    msg = (
        "graep.inputs.rucio requires the 'examples' dependency group: "
        "install with `uv sync --group examples` "
        "(or `pip install rucio`)."
    )
    raise ImportError(msg) from exc

logger = logging.getLogger(__name__)


class RucioFilesetResolver(FilesetResolver):
    """Resolve wildcard dataset patterns against rucio.

    Resolve-time ``nevts`` / ``nevts_wt`` are placeholders (zero) so
    this resolver does not open any ROOT file —
    :func:`graep.inputs.export_fileset` is the canonical step that
    opens files and populates per-file counts.

    Parameters
    ----------
    cache_dir, force_refresh
        Forwarded to :class:`FilesetResolver`.
    scope
        Rucio scope to use for ``client.list_files`` and
        ``get_pfn_for_files``. Defaults to ``"cms"``; ATLAS or
        other-experiment users override.
    veto_rules
        Optional callables ``str -> bool``. Each rule that returns
        ``True`` for a resolved dataset name causes the dataset to be
        dropped before file listing.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        force_refresh: bool = False,
        scope: str = "cms",
        veto_rules: Sequence[Callable[[str], bool]] = (),
    ) -> None:
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.scope = scope
        self.veto_rules = tuple(veto_rules)

    def resolve(
        self,
        queries: Mapping[ProcessInfo, Any],
        *,
        variation: str = "nominal",
        max_files_per_sample: int | None = None,
    ) -> FilesetDict:
        opts: dict[str, Any] = {
            "variation": variation,
            "max_files_per_sample": max_files_per_sample,
        }
        return self._cached_resolve(
            queries,
            opts,
            lambda: self._compute(
                queries,
                variation=variation,
                max_files_per_sample=max_files_per_sample,
            ),
        )

    # --- private --------------------------------------------------------

    def _get_client(self) -> Any:
        # Bare-Exception is intentional: rucio raises a wide spread of
        # auth/SSL/connection error classes when the proxy is missing or
        # expired (CannotAuthenticate, RSEAccessDenied, ssl errors, ...).
        # Catching `Exception` lets us surface a single user-friendly
        # message regardless of which specific failure mode triggered.
        try:
            return rucio_utils.get_rucio_client()
        except Exception as exc:
            msg = (
                "failed to instantiate rucio client. Run "
                "`voms-proxy-init -voms cms -rfc --valid 168:0` first. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc

    def _resolve_pattern(self, client: Any, pattern: str) -> list[str]:
        """Wildcard dataset pattern -> sorted list of concrete dataset names."""
        try:
            concrete = rucio_utils.query_dataset(pattern, client=client)
        except Exception as exc:
            msg = (
                f"rucio query_dataset failed for {pattern!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        kept = [
            name
            for name in concrete
            if not any(rule(name) for rule in self.veto_rules)
        ]
        return sorted(kept)

    def _list_files(self, client: Any, concrete: str) -> list[str]:
        """Concrete dataset name -> sorted XRootD URLs (PFNs).

        ``max_files_per_sample`` is *not* applied here — when one
        ProcessInfo spans multiple patterns, slicing has to happen
        after all patterns are resolved and merged.
        """
        try:
            entries = list(client.list_files(scope=self.scope, name=concrete))
        except Exception as exc:
            msg = (
                f"rucio list_files failed for {concrete!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(msg) from exc
        lfns = [e["name"] for e in entries]

        # PFN resolution. coffea's surface for this has shifted across
        # 2024.x releases; try the canonical entry point first, fail
        # loudly if it isn't there so the user knows to spike a manual
        # implementation rather than silently shipping LFNs.
        get_pfn = getattr(rucio_utils, "get_pfn_for_files", None)
        if get_pfn is None:
            msg = (
                "coffea.dataset_tools.rucio_utils.get_pfn_for_files is not "
                "available in the installed coffea version; PFN resolution "
                "needs a manual implementation. See plan §rucio risks."
            )
            raise RuntimeError(msg)
        pfn_map = get_pfn(client, lfns, scope=self.scope)
        return sorted(pfn_map[lfn] for lfn in lfns)

    def _compute(
        self,
        queries: Mapping[ProcessInfo, Any],
        *,
        variation: str,
        max_files_per_sample: int | None,
    ) -> FilesetDict:
        client = self._get_client()
        fileset: FilesetDict = {}
        for pinfo, value in queries.items():
            patterns = _normalise_value(value, key=pinfo)
            files: list[str] = []
            for pattern in patterns:
                concrete = self._resolve_pattern(client, pattern)
                if len(concrete) != 1:
                    msg = (
                        f"pattern {pattern!r} resolved to {len(concrete)} "
                        f"datasets: {concrete}"
                    )
                    raise ValueError(msg)
                files.extend(self._list_files(client, concrete[0]))

            files = sorted(set(files))
            if max_files_per_sample is not None:
                files = files[:max_files_per_sample]

            key = f"{pinfo.name}__{variation}"
            fileset[key] = {
                "files": dict.fromkeys(files, "Events"),
                "metadata": {
                    "process": pinfo.name,
                    "variation": variation,
                    # placeholders; export_fileset populates per-file values
                    "nevts": 0,
                    "nevts_wt": 0.0,
                    "xsec": pinfo.xsec,
                    "is_data": pinfo.is_data,
                },
            }
        return fileset
