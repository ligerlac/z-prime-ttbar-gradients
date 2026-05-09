"""MVA-specific plots: feature distributions and score histograms."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from graep.plotting.style import (
    ArrayLike,
    apply_style,
    convert_to_numpy,
)
from graep.plotting.utils import save_figure

if TYPE_CHECKING:
    from graep.config.plotting import PlottingSpec
    from graep.output.manager import OutputManager

logger = logging.getLogger(__name__)


def _setup_process_ordering(
    data_dict: dict[str, Any], plot_config: dict[str, Any]
) -> list[str]:
    """Order processes for plotting: configured order first, then the rest."""
    if not data_dict:
        msg = "Data dictionary cannot be empty"
        raise ValueError(msg)

    process_order = plot_config.get("process_order", list(data_dict.keys()))
    ordered = [p for p in process_order if p in data_dict]
    ordered += [p for p in data_dict if p not in process_order]
    logger.debug("Process ordering: %s", ordered)
    return ordered


def feature_distributions(
    spec: PlottingSpec,
    mgr: OutputManager,
    feature_data: dict[str, dict[str, ArrayLike]],
    mva_config: dict[str, Any],
    plot_config: dict[str, Any],
    *,
    figsize: tuple[float, float] | None = None,
    label_fontsize: int = 14,
    legend_fontsize: int = 12,
    tick_fontsize: int = 12,
    rcparams: dict[str, Any] | None = None,
) -> None:
    """Plot distributions of MVA input features for different processes.

    Writes one file per (feature, scaling) under
    ``mgr["plots"] / "features"`` using ``spec.output_format``.

    Parameters
    ----------
    spec : PlottingSpec
        Plotting configuration.
    mgr : OutputManager
        Output manager. Plots land in ``mgr["plots"] / "features"``.
    feature_data : dict[str, dict[str, ArrayLike]]
        Nested mapping process -> feature -> array.
    mva_config : dict[str, Any]
        MVA model configuration containing feature definitions.
    plot_config : dict[str, Any]
        Plot styling: ``process_colors``, ``process_labels``,
        ``process_order``.
    figsize : tuple[float, float], optional
        Per-call override of ``spec.figsize``.
    rcparams : dict[str, Any], optional
        Per-call rcParams overrides applied via ``plt.rc_context``.
    """
    output_path = mgr["plots"] / "features"
    logger.info("Creating MVA feature distribution plots in %s", output_path)

    apply_style(spec)
    figsize = figsize if figsize is not None else spec.figsize

    output_path.mkdir(parents=True, exist_ok=True)

    process_colors = plot_config.get("process_colors", {})
    process_labels = plot_config.get("process_labels", {})
    ordered_processes = _setup_process_ordering(feature_data, plot_config)

    with plt.rc_context(rcparams or {}):
        for feature in mva_config.get("features", []):
            feature_name = feature["name"]
            feature_label = feature.get("label", feature_name)
            feature_binning = feature.get("binning")

            if feature_binning is not None:
                if isinstance(feature_binning, str):
                    parts = feature_binning.strip().split(",")
                    prelim_bins = np.linspace(
                        float(parts[0]), float(parts[1]), int(parts[2]) + 1
                    )
                elif isinstance(feature_binning, (list, tuple)):
                    prelim_bins = np.asarray(feature_binning)
            else:
                prelim_bins = None

            for scaling_version in ["scaled", "unscaled"]:
                fig, ax = plt.subplots(figsize=figsize)

                bins = prelim_bins
                if prelim_bins is None:
                    all_values = np.concatenate(
                        [
                            data[feature_name][scaling_version]
                            for proc, data in feature_data.items()
                            if feature_name in data
                        ]
                    )
                    bins = np.linspace(
                        np.min(all_values), np.max(all_values), 50
                    )
                elif scaling_version == "scaled":
                    scaling = getattr(feature, "scale", None)
                    bins = scaling(bins) if scaling is not None else prelim_bins
                else:
                    bins = prelim_bins

                for process_name in ordered_processes:
                    logger.debug(
                        "Plotting %s feature data %r for process %r",
                        scaling_version,
                        feature_name,
                        process_name,
                    )
                    if (
                        process_name not in feature_data
                        or feature_name not in feature_data[process_name]
                    ):
                        continue

                    values = convert_to_numpy(
                        feature_data[process_name][feature_name][scaling_version]
                    )

                    ax.hist(
                        values,
                        bins=bins,
                        color=process_colors.get(process_name, "gray"),
                        label=process_labels.get(process_name, process_name),
                        alpha=0.7,
                        density=True,
                        histtype="stepfilled",
                        linewidth=1.5,
                    )

                ax.set_xlabel(feature_label, fontsize=label_fontsize)
                ax.set_ylabel("a.u.", fontsize=label_fontsize)
                ax.legend(frameon=False, fontsize=legend_fontsize)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                fig.tight_layout()

                plot_filename = output_path / (
                    f"{mva_config['name']}_feats_"
                    f"{feature_name}_{scaling_version}.{spec.output_format}"
                )
                save_figure(
                    fig,
                    plot_filename,
                    dpi=spec.dpi,
                    description=(
                        f"MVA feature plot ({feature_name}, "
                        f"{scaling_version})"
                    ),
                )
                plt.close(fig)


def scores(
    spec: PlottingSpec,
    mgr: OutputManager,
    scores_data: dict[str, ArrayLike],
    plot_config: dict[str, Any],
    *,
    prefix: str = "",
    bins: int = 50,
    score_range: tuple[float, float] = (0.0, 1.0),
    figsize: tuple[float, float] | None = None,
    label_fontsize: int = 14,
    legend_fontsize: int = 12,
    tick_fontsize: int = 12,
    rcparams: dict[str, Any] | None = None,
) -> None:
    """Plot MVA scores for different processes.

    Writes ``mgr["plots"] / "scores" / f"{prefix}mva_score.{ext}"``.

    Parameters
    ----------
    spec : PlottingSpec
        Plotting configuration.
    mgr : OutputManager
        Output manager. The plot lands in ``mgr["plots"] / "scores"``.
    scores_data : dict[str, ArrayLike]
        Mapping process -> array of MVA scores.
    plot_config : dict[str, Any]
        Plot styling: ``process_colors``, ``process_labels``,
        ``process_order``.
    prefix : str, optional
        Filename prefix.
    bins : int, optional
        Number of histogram bins.
    score_range : tuple[float, float], optional
        Score range to plot.
    figsize : tuple[float, float], optional
        Per-call override of ``spec.figsize``.
    rcparams : dict[str, Any], optional
        Per-call rcParams overrides applied via ``plt.rc_context``.

    Raises
    ------
    ValueError
        If ``scores_data`` is empty or ``bins`` is not positive.
    """
    if not scores_data:
        msg = "Scores dictionary cannot be empty"
        raise ValueError(msg)
    if bins <= 0:
        msg = "Number of bins must be positive"
        raise ValueError(msg)

    output_path = mgr["plots"] / "scores"
    logger.info(
        "Creating MVA scores plot with prefix %r in %s", prefix, output_path
    )

    apply_style(spec)
    figsize = figsize if figsize is not None else spec.figsize

    try:
        with plt.rc_context(rcparams or {}):
            output_path.mkdir(parents=True, exist_ok=True)

            process_colors = plot_config.get("process_colors", {})
            process_labels = plot_config.get("process_labels", {})
            ordered_processes = _setup_process_ordering(
                scores_data, plot_config
            )

            fig, ax = plt.subplots(figsize=figsize)

            bin_edges = np.linspace(score_range[0], score_range[1], bins + 1)
            max_height = 0.0

            for process_name in ordered_processes:
                logger.debug(
                    "Plotting MVA scores for process %r", process_name
                )
                if process_name not in scores_data:
                    continue

                process_scores = convert_to_numpy(scores_data[process_name])
                if len(process_scores) == 0:
                    logger.warning(
                        "No scores found for process %r", process_name
                    )
                    continue

                counts, _ = np.histogram(
                    process_scores, bins=bin_edges, density=True
                )
                if len(counts) > 0:
                    max_height = max(max_height, counts.max())

                ax.hist(
                    process_scores,
                    bins=bin_edges,
                    color=process_colors.get(process_name, "gray"),
                    label=process_labels.get(process_name, process_name),
                    alpha=0.7,
                    density=True,
                    histtype="stepfilled",
                    linewidth=1.5,
                )

            ax.set_xlabel("MVA Score", fontsize=label_fontsize)
            ax.set_ylabel("a.u.", fontsize=label_fontsize)
            ax.set_ylim(0, max_height * 1.15)
            ax.legend(frameon=False, fontsize=legend_fontsize)
            ax.tick_params(axis="both", labelsize=tick_fontsize)
            fig.tight_layout()

            plot_filename = (
                output_path / f"{prefix}mva_score.{spec.output_format}"
            )
            save_figure(
                fig,
                plot_filename,
                dpi=spec.dpi,
                description=f"MVA score plot [{prefix}]",
            )
            plt.close(fig)

    except Exception as exc:
        logger.error("Failed to create MVA scores plot: %s", exc)
        raise
