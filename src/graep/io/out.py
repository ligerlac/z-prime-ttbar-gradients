from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import uproot

# Configure module-level logger
logger = logging.getLogger(__name__)


# ================
# Histogram writing
# ==================
def save_histograms_to_pickle(histograms: dict[str, dict[str, Any]], pickle_path: str | Path) -> None:
    """
    Save a nested dictionary of histograms to a pickle file.

    Parameters
    ----------
    histograms : dict
        Mapping from channel names to observables to histogram objects.
    pickle_path : str or Path
        Path to the output pickle file. The directory will be
        created if it does not exist.

    Raises
    ------
    IOError
        If writing to the pickle file fails.
    """
    path = Path(pickle_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(histograms, file)
        logger.info("Histograms successfully pickled to %s", path)
    except Exception as exc:
        logger.error("Failed to pickle histograms to %s: %s", path, exc)
        raise


def save_histograms_to_root(
    histograms: dict[str, dict[str, Any]],
    root_path: str | Path,
    add_offset: bool = False,
) -> None:
    """
    Save histograms to a ROOT file using uproot.
    Histograms with no entries (after optional offset) are skipped.
    Filenames in the ROOT file follow the pattern:
        "<channel>__<observable>__<sample>[__<variation>]".
        where [__variation] is optional.

    Parameters
    ----------
    histograms : dict
        Nested mapping of channel names to observables to histogram objects.
    root_path : str or Path
        Path to the output ROOT (.root) file. The directory will be
        created if it does not exist.
    add_offset : bool, optional
        If True, add a small offset to each bin to avoid empty bins
        (default is False).
    """
    path = Path(root_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(str(path)) as root_file:
            for channel, obs_dict in histograms.items():
                for observable, original_hist in obs_dict.items():
                    # Optionally add a minimal floating-point offset
                    if add_offset:
                        hist = original_hist + 1e-6
                        num_bins = hist.axes[0].size
                        empty_threshold = num_bins * 1e-6 * 1.01
                    else:
                        hist = original_hist
                        empty_threshold = 0.0

                    # Iterate samples and systematic variations
                    for sample in hist.axes[1]:
                        sample_hist = hist[:, sample, :]
                        for variation in sample_hist.axes[1]:
                            # Skip non-nominal variations for data
                            if sample == "data" and variation != "nominal":
                                continue

                            # Construct key and histogram slice
                            suffix = "" if variation == "nominal" else f"__{variation}"
                            hist_slice = hist[:, sample, variation]

                            # Check for non-empty histogram
                            total_entries = sum(hist_slice.values())
                            if total_entries > empty_threshold:
                                key = f"{channel}__{observable}__{sample}{suffix}"
                                root_file[key] = hist_slice
                                logger.debug("Saved ROOT histogram: %s", key)
                            else:
                                logger.warning(
                                    "Skipping empty histogram: %s__%s__%s%s", channel, observable, sample, suffix
                                )
        logger.info("Histograms successfully written to ROOT file %s", path)
    except Exception as exc:
        logger.error("Failed to write ROOT file %s: %s", path, exc)
        raise
