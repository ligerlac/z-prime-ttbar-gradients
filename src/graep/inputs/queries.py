"""This is a core module containing shared types, ABC,
caching helpers, and the top-level ``build_fileset`` function needed
to resolve dataset queries.

Concrete resolver implementations live in sibling modules and are only imported
when the user picks the corresponding resolver. This is done to avoid
unnecessary dependencies on optional packages. It also allows new resolvers
to be added while keeping this core module clean.

This core also defines caching helpers for query results.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any, NamedTuple

from graep.config.inputs.base import FilesetSpec

logger = logging.getLogger(__name__)


# --- type aliases (module-private convenience) ----------------------------

FilesetEntry = dict[str, Any]
FilesetDict = dict[str, FilesetEntry]
# A query value is either a single dataset pattern (string) or a list of
# patterns. Multiple patterns under one ProcessInfo are aggregated into
# a single fileset entry: file URIs are unioned, nevts/nevts_wt summed.
QueryValue = "str | Sequence[str]"
QueriesCallable = Callable[[], "Mapping[ProcessInfo, QueryValue]"]


# --- value type ------------------------------------------------------------


class ProcessInfo(NamedTuple):
    """This define a physics process.

    Used as the dict key for queries; hashable by virtue of being a
    ``NamedTuple``. ``xsec`` is in picobarns or``None`` for data.
    ``is_data`` flags real data (not MC) so
    downstream mthods know to skip MC-weight handling.
    """

    name: str
    xsec: float | None
    is_data: bool = False


# --- duplicate detection ---------------------------------------------------


class DuplicateProcessInfoError(KeyError):
    """A ``ProcessInfo`` was registered twice in a ``UniqueProcessInfoDict``."""


class UniqueProcessInfoDict(MutableMapping[ProcessInfo, str]):
    """``MutableMapping`` that raises on duplicate ``ProcessInfo`` keys.

    Used to merge multiple query callables (signal, MC backgrounds, data)
    while catching accidental cross-registration of the same process.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._data: dict[ProcessInfo, str] = {}
        if args or kwargs:
            self.update(*args, **kwargs)

    def __getitem__(self, key: ProcessInfo) -> str:
        return self._data[key]

    def __setitem__(self, key: ProcessInfo, value: str) -> None:
        if key in self._data:
            existing = self._data[key]
            msg = (
                f"ProcessInfo {key!r} already registered with value {existing!r}; refusing to overwrite with {value!r}"
            )
            raise DuplicateProcessInfoError(msg)
        self._data[key] = value

    def __delitem__(self, key: ProcessInfo) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[ProcessInfo]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


# --- query-value normalisation ---------------------------------------------


def _normalise_value(value: Any, *, key: ProcessInfo) -> tuple[str, ...]:
    """Coerce a query value to a tuple of pattern strings.

    A bare string becomes a 1-tuple; a sequence of strings becomes its
    tuple form. Any other shape raises with a message naming the
    problematic ``ProcessInfo``.
    """
    if isinstance(value, str):
        return (value,)
    try:
        items = tuple(value)
    except TypeError as exc:
        msg = f"queries[{key!r}] must be a string or sequence of strings; got {type(value).__name__}"
        raise TypeError(msg) from exc
    if not items:
        msg = f"queries[{key!r}] must be a non-empty pattern or list of patterns; got an empty sequence"
        raise ValueError(msg)
    if not all(isinstance(p, str) for p in items):
        bad = [type(p).__name__ for p in items if not isinstance(p, str)]
        msg = f"queries[{key!r}] must contain only strings; found non-string entry types: {bad}"
        raise TypeError(msg)
    return items


# --- caching helpers (not exported) --------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for non-native types we want to hash stably."""
    if isinstance(obj, ProcessInfo):
        return [obj.name, obj.xsec]
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _stable_hash_key(parts: Mapping[str, Any]) -> str:
    """Deterministic short hash of a JSON-serialisable mapping."""
    payload = json.dumps(parts, sort_keys=True, default=_json_default).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _load_or_compute(
    cache_dir: Path,
    key: str,
    compute: Callable[[], FilesetDict],
    *,
    force_refresh: bool,
) -> FilesetDict:
    """Read ``cache_dir/{key}.json`` or fall through to ``compute()``.

    Writes are atomic. ``compute`` writes to a ``.json.tmp`` file, then
    ``Path.replace`` swaps it into place. Two concurrent callers may
    both compute, but neither sees a broken cache file.
    """
    cache_path = cache_dir / f"{key}.json"
    if not force_refresh and cache_path.exists():
        logger.info("fileset cache hit: %s", cache_path)
        with cache_path.open("r") as fh:
            return json.load(fh)

    fileset = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as fh:
        json.dump(fileset, fh, indent=2, default=_json_default)
    tmp_path.replace(cache_path)
    logger.info("fileset cache miss → wrote %s", cache_path)
    return fileset


# --- ABC -------------------------------------------------------------------


def _xsec_sort_key(p: ProcessInfo) -> tuple[str, float]:
    """Sort key that handles ``xsec=None`` without comparing None to float."""
    return (p.name, p.xsec if p.xsec is not None else float("-inf"))


class FilesetResolver(ABC):
    """Abstract base for fileset resolvers.

    Subclasses implement :meth:`resolve` which turns a mapping of
    ``ProcessInfo`` to backend-specific patterns into the framework's
    canonical fileset dict shape::

        {
            "<process>__<variation>": {
                "files":    {uri: "Events", ...},
                "metadata": {process, variation, nevts, nevts_wt, xsec},
            },
            ...
        }

    Caching is shared across resolvers. A subclass's :meth:`resolve`
    should call :meth:`_cached_resolve` and hand it a callback that
    does the actual work. The helper looks up the on-disk cache
    (keyed by resolver class, queries, and options) and only runs
    the callback on a cache miss.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        force_refresh: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.force_refresh = force_refresh

    @abstractmethod
    def resolve(
        self,
        queries: Mapping[ProcessInfo, Any],
        **opts: Any,
    ) -> FilesetDict:
        """Resolve queries to a fileset dict.

        ``queries`` values are :data:`QueryValue`. These are either a
        single dataset pattern string, or a sequence of patterns.
        The exact pattern syntax depends on the concrete resolver.
        Multiple patterns under one :class:`ProcessInfo` are aggregated
        by the resolver: file URIs unioned and deduplicated, ``nevts`` /
        ``nevts_wt`` summed.

        Subclass-specific keyword opts (``variation``,
        ``max_files_per_sample``, etc.) are documented in the
        :meth:`resolve` override.
        """

    def _cached_resolve(
        self,
        queries: Mapping[ProcessInfo, Any],
        opts: Mapping[str, Any],
        compute: Callable[[], FilesetDict],
    ) -> FilesetDict:
        """Cache-aware wrapper around ``compute``.

        Cache key is a stable hash of ``(resolver class name, queries,
        opts)``. Inputs are sorted before hashing, so two queries with
        the same contents hash to the same key regardless of the order
        they were added.
        Pattern lists are normalised (single-string upgraded to 1-tuple)
        and sorted internally so ``["A", "B"]`` and ``["B", "A"]``
        collapse to the same cache entry. ``cache_dir`` and
        ``force_refresh`` are *not* hashed; they govern cache behaviour
        rather than cache content.
        """
        sorted_items = sorted(queries.items(), key=lambda kv: _xsec_sort_key(kv[0]))
        hash_input: dict[str, Any] = {
            "resolver": type(self).__name__,
            "queries": {
                f"{p.name}|{p.xsec}|{p.is_data}": sorted(_normalise_value(pat, key=p)) for p, pat in sorted_items
            },
            "opts": dict(sorted(opts.items())),
        }
        key = _stable_hash_key(hash_input)
        return _load_or_compute(self.cache_dir, key, compute, force_refresh=self.force_refresh)


# --- top-level entry point -------------------------------------------------


def build_fileset(
    spec: FilesetSpec,
    **call_overrides: Any,
) -> FilesetDict:
    """Resolve a fileset from a typed configuration spec.

    ``spec`` is an instance of a concrete
    :class:`graep.config.inputs.FilesetSpec` subclass (for example
    :class:`graep.config.inputs.OpenDataPortalFilesetSpec` or
    :class:`graep.config.inputs.RucioFilesetSpec`). The query callables
    on the spec are called with no arguments, merged via
    :class:`UniqueProcessInfoDict`, and handed to the resolver the
    spec describes.

    ``call_overrides`` win over the spec's ``variation`` /
    ``max_files_per_sample`` so a notebook can change them at the call
    site without editing the config object.
    """
    if not isinstance(spec, FilesetSpec):
        msg = (
            f"spec must be a FilesetSpec instance "
            f"(e.g. OpenDataPortalFilesetSpec, RucioFilesetSpec); "
            f"got {type(spec).__name__}"
        )
        raise TypeError(msg)

    merged = UniqueProcessInfoDict()
    for q_callable in spec.queries:
        chunk = q_callable()
        for pinfo, pattern in chunk.items():
            try:
                merged[pinfo] = pattern
            except DuplicateProcessInfoError as exc:
                name = getattr(q_callable, "__name__", repr(q_callable))
                msg = f"queries callable {name!r} produced a duplicate ProcessInfo: {exc}"
                raise DuplicateProcessInfoError(msg) from exc

    resolver = spec.make_resolver()

    resolve_kwargs: dict[str, Any] = {
        "variation": spec.variation,
        "max_files_per_sample": spec.max_files_per_sample,
    }
    resolve_kwargs.update(call_overrides)

    return resolver.resolve(merged, **resolve_kwargs)
