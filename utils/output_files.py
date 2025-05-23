import pickle
import logging

import uproot

logger = logging.getLogger(__name__)

def pkl_histograms(histograms, output_file="outputs/histograms/histogram.pkl"):
    """
    Save a histograms dictionary to a specified pickle file.

    Parameters
    ----------
    histogram : Dict[str, Dict[str, Hist]]
        Mapping of region and observable to histograms.
    output_dir : str
        pickle file to save the histogram in.
    """
    # Save the histogram to a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(histograms, f)


def unpkl_histograms(pickled_file="outputs/histograms/histogram.pkl"):
    """
    Save read a histograms dictionary from a specified pickle file.

    Parameters
    ----------
    pickled_file : str
        pickle file to read the histogram from.

    Returns
    -------
    histograms : Dict[str, Dict[str, , Hist]]
        Mapping of region and observable to histograms
    """
    with open(pickled_file, "rb") as f:
        histograms = pickle.load(f)
    return histograms


def save_histograms(
    hist_dict,
    output_file="outputs/histograms/histograms.root",
    add_offset=False,
):
    """
    Save histograms to a specified directory.

    Parameters
    ----------
    hist_dict : dict
        Dictionary of histograms to save.
    output_dir : str
        Directory to save the histograms.
    """

    with uproot.recreate(output_file) as f:
        # save all available histograms to disk
        for channel, observables in hist_dict.items():
            for observable, histogram in observables.items():
                # optionally add minimal offset to avoid completely empty bins
                # (useful for the ML validation variables that would need
                # binning adjustment to avoid those)
                if add_offset:
                    histogram += 1e-6
                    # reference count empty histogram with float-point math tolerance
                    empty_hist_yield = histogram.axes[0].size * (1e-6) * 1.01
                else:
                    empty_hist_yield = 0

                for sample in histogram.axes[1]:
                    sample_histograms = histogram[:, sample, :]
                    for variation in sample_histograms.axes[1]:
                        if sample == "data" and variation != "nominal":
                            # no systematics for data
                            continue

                        variation_string = (
                            "" if variation == "nominal" else f"__{variation}"
                        )

                        current_1d_hist = histogram[:, sample, variation]
                        if sum(current_1d_hist.values()) > empty_hist_yield:
                            # only save histograms containing events
                            f[
                                f"{channel}__{observable}__{sample}{variation_string}"
                            ] = current_1d_hist
                        else:
                            print(current_1d_hist)
                            logger.warning( f"The {channel}__{observable}__{sample}{variation_string} histogram is empty. It will not be saved.")
