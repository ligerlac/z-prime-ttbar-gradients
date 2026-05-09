"""Shared plotting utilities.

Currently exposes :func:`save_figure`, a thin wrapper around
``Figure.savefig`` that creates the parent directory, logs the write,
and surfaces an ``OSError`` with context on failure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    description: str = "figure",
) -> None:
    """Save ``fig`` to ``path``, creating parents and logging the write.

    Does not close the figure. Callers that produce many figures in a
    loop should call ``plt.close(fig)`` after the save themselves.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save.
    path : str | Path
        Destination path. Parent directories are created if missing.
    dpi : int, optional
        Resolution passed to ``Figure.savefig``.
    description : str, optional
        Short string used in the log line. Default ``"figure"``.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        logger.info("Saved %s to %s", description, path)
    except Exception as exc:
        logger.error("Failed to save %s to %s: %s", description, path, exc)
        msg = f"Cannot save {description} to {path}: {exc}"
        raise OSError(msg) from exc
