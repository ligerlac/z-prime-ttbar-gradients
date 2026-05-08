from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import ClassVar, TextIO

# Verbosity scale (low = quiet, high = loud).
# 0 CRITICAL — only catastrophic failures
# 1 ERROR    — errors only
# 2 WARNING  — warnings + errors
# 3 INFO     — informational messages (default)
# 4 DEBUG    — everything
_VERBOSITY: dict[int, int] = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
}

# Third-party loggers muted by default. Add to this list as new one surface.
_NOISY_THIRD_PARTY: tuple[str, ...] = (
    "jax._src.xla_bridge",
    "distributed",
    "fsspec",
    "asyncio",
    "matplotlib",
)

# ANSI escape codes for the level-coloured prefix.
_BLUE = "\033[0;34m"
_YELLOW = "\033[1;33m"
_RED = "\033[0;31m"
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that prefixes each record with a level-coloured header.

    The message body is left untouched, so callers may embed their own ANSI
    sequences (for example a magenta banner around a phase-boundary log).
    """

    log_format_prefix = "[%(levelname)s:%(name)s:%(funcName)s:L.%(lineno)d] "

    PREFIX_COLORS: ClassVar[dict[int, str]] = {
        logging.INFO: _BLUE,
        logging.WARNING: _YELLOW,
        logging.ERROR: _RED,
        logging.CRITICAL: _RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.PREFIX_COLORS.get(record.levelno, "")
        prefix = logging.Formatter(self.log_format_prefix).format(record)
        message = record.getMessage().lstrip("\n")
        return f"{color}{prefix}{_RESET}{message}"


def setup_logging(
    verbosity: int = 3,
    per_module: Mapping[str, int] | None = None,
    quiet_third_party: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Configure logging for the current process.

    Idempotent — safe to call repeatedly to change verbosity at runtime
    (e.g. from successive notebook cells); previously installed handlers
    and per-module level overrides are cleared first.

    Verbosity scale (low = quiet, high = loud):

    =====  ========  =========================================
    Value  Level     Meaning
    =====  ========  =========================================
    0      CRITICAL  only catastrophic failures
    1      ERROR     errors only
    2      WARNING   warnings + errors
    3      INFO      informational messages (default)
    4      DEBUG     everything
    =====  ========  =========================================

    Parameters
    ----------
    verbosity
        Integer 0..4. See scale above.
    per_module
        Per-module verbosity overrides keyed by dotted-path logger name,
        e.g. ``{"graep.mva": 4, "graep.workflow": 2}``.
    quiet_third_party
        If ``True``, mute commonly chatty third-party loggers (JAX, dask,
        fsspec, asyncio, matplotlib) at ERROR.
    stream
        Output stream; defaults to ``sys.stdout``.
    """
    level = _to_level(verbosity, "verbosity")

    root = logging.getLogger()

    # Wipe root handlers so re-running a notebook cell doesn't stack them.
    for h in list(root.handlers):
        root.removeHandler(h)

    # Reset state on every existing graep.* logger so a previous call's
    # per_module overrides don't leak through this one.
    for name, lg in list(logging.root.manager.loggerDict.items()):
        is_graep = name == "graep" or name.startswith("graep.")
        if is_graep and isinstance(lg, logging.Logger):
            lg.setLevel(logging.NOTSET)
            for h in list(lg.handlers):
                lg.removeHandler(h)
            lg.propagate = True

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(ColoredFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    if per_module:
        for mod, vmod in per_module.items():
            logging.getLogger(mod).setLevel(_to_level(vmod, f"per_module[{mod!r}]"))

    if quiet_third_party:
        for name in _NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.ERROR)


def _to_level(verbosity: int, label: str) -> int:
    if verbosity not in _VERBOSITY:
        msg = f"{label} must be in 0..{max(_VERBOSITY)}, got {verbosity!r}"
        raise ValueError(msg)
    return _VERBOSITY[verbosity]
