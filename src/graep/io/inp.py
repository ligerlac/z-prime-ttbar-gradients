from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

# Configure module-level logger
logger = logging.getLogger(__name__)


def load_histograms_from_pickle(
    pickle_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load a nested dictionary of histograms from a pickle file.

    Parameters
    ----------
    pickle_path : str or Path
        Path to the input pickle file.

    Returns
    -------
    dict
        Nested mapping from channel names to observables to histogram objects.

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist.
    IOError
        If reading from the pickle file fails.
    """
    path = Path(pickle_path)
    if not path.exists():
        msg = f"Pickle file not found: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open("rb") as file:
            histograms = pickle.load(file)
        logger.info("Histograms successfully loaded from %s", path)
        return histograms
    except Exception as exc:
        logger.error("Failed to load histograms from %s: %s", path, exc)
        raise
