import uproot


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
        for channel, histogram in hist_dict.items():
            # optionally add minimal offset to avoid completely empty bins
            # (useful for the ML validation variables that would need binning adjustment
            # to avoid those)
            if add_offset:
                histogram += 1e-6
                # reference count for empty histogram with floating point math tolerance
                empty_hist_yield = histogram.axes[0].size * (1e-6) * 1.01
            else:
                empty_hist_yield = 0

            for sample in histogram.axes[1]:
                for variation in histogram[:, sample, :].axes[1]:
                    variation_string = (
                        "" if variation == "nominal" else f"_{variation}"
                    )
                    current_1d_hist = histogram[:, sample, variation]

                    if sum(current_1d_hist.values()) > empty_hist_yield:
                        # only save histograms containing events
                        f[f"{channel}_{sample}{variation_string}"] = (
                            current_1d_hist
                        )
