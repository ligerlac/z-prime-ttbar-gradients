"""Resolver that talks to the public CERN Open Data Portal HTTP API.

No authentication required. The portal hosts datasets from any
experiment that publishes through it (CMS, ATLAS, LHCb, ALICE, …);
the only experiment-specific concern at the framework level is the
shape of the dataset path patterns the user supplies — those live in
the example's ``queries.py``, not here.

Each user query value is a path-like dataset pattern (with shell-style
wildcards permitted). The resolver groups queries by their primary
name (the first slash-separated segment), runs one search call per
group, and ``fnmatch``-matches the returned record titles against the
full pattern to pick the right record per query.

Per-record metadata gives total event counts directly, so this
resolver opens **zero** ROOT files at cold-cache time.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from graep.inputs.queries import (
    FilesetDict,
    FilesetResolver,
    ProcessInfo,
    _normalise_value,
)

logger = logging.getLogger(__name__)


_API = "https://opendata.cern.ch/api/records"
_DEFAULT_TIMEOUT = 30.0
_PAGE_SIZE = 500


class OpenDataPortalFilesetResolver(FilesetResolver):
    """Resolve path-like dataset patterns against the CERN Open Data
    Portal HTTP API.

    No authentication required. Resolve-time ``nevts`` / ``nevts_wt``
    are placeholders (zero); :func:`graep.inputs.export_fileset` is the
    canonical step that opens files and populates per-file counts.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        force_refresh: bool = False,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(cache_dir=cache_dir, force_refresh=force_refresh)
        self.timeout = timeout

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

    def _compute(
        self,
        queries: Mapping[ProcessInfo, Any],
        *,
        variation: str,
        max_files_per_sample: int | None,
    ) -> FilesetDict:
        records_per_process = self._find_record_ids(queries)
        return self._resolve_records(
            records_per_process,
            variation=variation,
            max_files_per_sample=max_files_per_sample,
        )

    def _fetch_json(self, url: str) -> dict[str, Any]:
        # Two-stage validation so the URL passed to urlopen is provably
        # within the open-data API and provably https. The prefix check
        # bounds where the request can go; the scheme check rejects file://
        # or data:// schemes that bandit S310 warns about.
        if not url.startswith(_API):
            msg = f"refusing to fetch URL outside the open data API: {url!r}"
            raise ValueError(msg)
        request = urllib.request.Request(url)
        if request.type != "https":
            msg = f"refusing non-https URL: {url!r}"
            raise ValueError(msg)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            msg = f"portal request failed for {url!r}: {type(exc).__name__}: {exc}"
            raise RuntimeError(msg) from exc

    def _find_record_ids(
        self,
        queries: Mapping[ProcessInfo, Any],
    ) -> dict[ProcessInfo, list[int]]:
        # Expand multi-pattern values: each (pinfo, pattern) pair is
        # resolved independently; the per-pattern record IDs collect
        # under the originating ProcessInfo.
        #
        # Group all (pinfo, pattern) pairs by primary name (the path
        # segment between the first two slashes — the bare CMS dataset
        # name). One portal search per unique primary name.
        #
        # Earlier we tried tokenising the primary name on the first
        # underscore (e.g. `ZPrimeToTT`, `TTToSemiLeptonic`). That works
        # for some camelcase prefixes (`ZprimeToTT` returns the full mass
        # grid in a single call) but the portal indexer doesn't expose
        # every prefix; `TTToSemiLeptonic` returned zero hits while the
        # full primary name returned the right two records. Searching by
        # the full primary name is robust at the cost of giving up the
        # implicit family-batching for Z'-style grids.
        flat: list[tuple[ProcessInfo, str]] = []
        for pinfo, value in queries.items():
            for pattern in _normalise_value(value, key=pinfo):
                flat.append((pinfo, pattern))

        by_primary: dict[str, list[tuple[ProcessInfo, str]]] = defaultdict(list)
        for pinfo, pattern in flat:
            primary = pattern.strip("/").split("/", 1)[0]
            by_primary[primary].append((pinfo, pattern))

        out: dict[ProcessInfo, list[int]] = defaultdict(list)
        for primary, group in by_primary.items():
            url = f"{_API}?q={primary}&size={_PAGE_SIZE}"
            hits = self._fetch_json(url).get("hits", {}).get("hits", [])
            if len(hits) >= _PAGE_SIZE:
                logger.warning(
                    "portal search for %r hit pagination ceiling (%d). "
                    "TODO(refactor): implement pagination if this saturates.",
                    primary,
                    _PAGE_SIZE,
                )
            title_to_id = {h["metadata"]["title"]: h["id"] for h in hits}
            logger.debug("portal search %r -> %d candidates", primary, len(title_to_id))

            for pinfo, pattern in group:
                matches = [tid for title, tid in title_to_id.items() if fnmatch.fnmatchcase(title, pattern)]
                if not matches:
                    msg = f"no portal record matched {pattern!r}"
                    raise ValueError(msg)
                if len(matches) > 1:
                    msg = f"multiple portal records matched {pattern!r}: {matches}"
                    raise ValueError(msg)
                out[pinfo].append(matches[0])

        return dict(out)

    def _resolve_records(
        self,
        records_per_process: Mapping[ProcessInfo, list[int]],
        *,
        variation: str,
        max_files_per_sample: int | None,
    ) -> FilesetDict:
        fileset: FilesetDict = {}
        for pinfo, record_ids in records_per_process.items():
            uris: list[str] = []
            for record_id in record_ids:
                record = self._fetch_json(f"{_API}/{record_id}")["metadata"]
                uris.extend(
                    entry["uri"]
                    for index in record.get("_file_indices", [])
                    for entry in index.get("files", [])
                )

            # Dedup + sort so two patterns whose records share a file
            # collapse cleanly, and the cache JSON is byte-stable.
            uris = sorted(set(uris))
            if max_files_per_sample is not None:
                uris = uris[:max_files_per_sample]

            key = f"{pinfo.name}__{variation}"
            fileset[key] = {
                "files": dict.fromkeys(uris, "Events"),
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
