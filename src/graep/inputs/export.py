"""Write a resolved fileset to disk as legacy-shaped JSON artefacts.

Two outputs (legacy-compatible):

- ``<output_dir>/nanoaods.json`` — combined dict keyed by process,
  then variation, with per-file path/nevts/nevts_wt and per-process totals.
- ``<output_dir>/nanoaods_jsons_per_process/nanoaods_<process>_<variation>.json``
  — one slim JSON per (process, variation), same schema scoped to a
  single entry. Set ``per_process_jsons=False`` to skip these.

Per-file ``nevts`` / ``nevts_wt`` come from opening every ROOT file with
uproot. This is the same cost as the legacy ``utils/build_fileset_json.py``
had — slow on first export, but the artefacts are versionable.

Weights handling:

- The default weights branch is ``"genWeight"``. Pass another string or
  ``None`` to change.
- A process is treated as data (no weight summation, ``nevts_wt =
  nevts``) when its fileset metadata carries ``is_data=True`` *or* its
  process name is in ``skip_weights_for``. The first comes from
  :class:`~graep.inputs.ProcessInfo`; the second is a per-call override
  for unusual cases.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uproot

from graep.inputs.queries import FilesetDict

logger = logging.getLogger(__name__)


def export_fileset(
    fileset: FilesetDict,
    output_dir: str | Path,
    *,
    weights_branch: str | None = "genWeight",
    skip_weights_for: Sequence[str] = (),
    per_process_jsons: bool = True,
) -> None:
    """Open every file in ``fileset`` and write the legacy-shaped JSONs.

    See module docstring for the output layout. Errors opening individual
    files are logged and the offending file is recorded with
    ``nevts=0``, ``nevts_wt=0.0`` rather than aborting the export.
    """
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    skip = set(skip_weights_for)
    combined: dict[str, dict[str, dict[str, Any]]] = {}

    for entry in fileset.values():
        meta = entry["metadata"]
        process = meta["process"]
        variation = meta["variation"]
        is_data = bool(meta.get("is_data", False)) or process in skip

        per_file: list[dict[str, Any]] = []
        nevts_total = 0
        nevts_wt_total = 0.0

        for url in entry["files"]:
            n, wt = _count_file(url, weights_branch=weights_branch, is_data=is_data)
            per_file.append({"path": url, "nevts": n, "nevts_wt": wt})
            nevts_total += n
            nevts_wt_total += wt

        combined.setdefault(process, {})[variation] = {
            "files": per_file,
            "nevts_total": nevts_total,
            "nevts_wt_total": nevts_wt_total,
        }

    combined_path = output_root / "nanoaods.json"
    with combined_path.open("w") as fh:
        json.dump(combined, fh, indent=2)
    logger.info("wrote combined fileset JSON: %s", combined_path)

    if per_process_jsons:
        per_proc_dir = output_root / "nanoaods_jsons_per_process"
        per_proc_dir.mkdir(parents=True, exist_ok=True)
        for process, variations in combined.items():
            for variation, payload in variations.items():
                slim = {process: {variation: payload}}
                p = per_proc_dir / f"nanoaods_{process}_{variation}.json"
                with p.open("w") as fh:
                    json.dump(slim, fh, indent=2)
                logger.debug("wrote per-process JSON: %s", p)
        logger.info("wrote per-(process, variation) JSONs under: %s", per_proc_dir)


def _count_file(
    url: str,
    *,
    weights_branch: str | None,
    is_data: bool,
) -> tuple[int, float]:
    """Open one ROOT file, return ``(nevts, nevts_wt)``.

    Behaviour:

    - ``is_data=True`` or ``weights_branch=None``: silently use
      ``nevts_wt = nevts`` (deliberate skip).
    - ``weights_branch`` set but branch missing from the file: log a
      WARNING and fall back to ``nevts_wt = nevts``. This catches the
      common mistake of forgetting ``ProcessInfo.is_data=True`` for an
      unweighted sample, or naming the branch wrongly.
    - ``weights_branch`` set and present: sum the branch.

    Any exception during the open is logged and reported as
    ``(0, 0.0)`` so a single broken file doesn't kill the whole export.
    """
    try:
        with uproot.open(url) as handle:
            events = handle["Events"]
            n = int(events.num_entries)
            if is_data or weights_branch is None:
                return n, float(n)
            if weights_branch not in events:
                logger.warning(
                    "weights branch %r missing in %s; falling back to nevts "
                    "(set ProcessInfo.is_data=True or weights_branch=None to "
                    "silence this)",
                    weights_branch,
                    url,
                )
                return n, float(n)
            wt = float(events[weights_branch].array(library="np").sum())
            return n, wt
    except Exception as exc:
        logger.warning("failed to count events in %s: %s", url, exc)
        return 0, 0.0
