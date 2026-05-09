"""Stacked histogram with a data/MC ratio panel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.gridspec import GridSpec

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


def data_mc(
    spec: PlottingSpec,
    mgr: OutputManager,
    bin_edges: ArrayLike,
    data: ArrayLike,
    templates: dict[str, ArrayLike],
    *,
    fitted_params: dict[str, float] | None = None,
    plot_settings: dict[str, Any] | None = None,
    show_signal: bool = True,
    figsize: tuple[float, float] | None = None,
    ratio_ylim: tuple[float, float] = (0.5, 1.5),
    xlabel: str = "",
    ylabel: str = "Events",
    title: str = "",
    label_fontsize: int = 20,
    title_fontsize: int = 18,
    legend_fontsize: int = 16,
    annotation_fontsize: int = 18,
    ratio_label_fontsize: int = 14,
    name: str = "data_mc",
    rcparams: dict[str, Any] | None = None,
) -> None:
    """Stacked data/MC histogram with a ratio panel below.

    Saves to ``mgr["plots"] / f"{name}.{spec.output_format}"``.

    Parameters
    ----------
    spec : PlottingSpec
        Plotting configuration. Used for style, default ``figsize``,
        rcParams baseline, ``dpi`` and ``output_format``.
    mgr : OutputManager
        Output manager. The figure is written under ``mgr["plots"]``.
    bin_edges : ArrayLike
        Bin edges for the histogram.
    data : ArrayLike
        Observed data values.
    templates : dict[str, ArrayLike]
        Mapping of template histograms (process name -> bin counts).
    fitted_params : dict[str, float], optional
        Fitted parameters used to scale templates.
    plot_settings : dict[str, Any], optional
        Customisation settings dictionary containing:
        - ``process_order``: ordered process names
        - ``process_colors``: color mapping for processes
        - ``process_labels``: label mapping for processes
        - ``jax.fit_param_labels``: LaTeX labels for parameters
    show_signal : bool, optional
        Whether to display the signal process, by default True.
    figsize : tuple[float, float], optional
        Per-call override of ``spec.figsize``.
    ratio_ylim : tuple[float, float], optional
        Y-axis limits for the ratio panel.
    xlabel, ylabel, title : str, optional
        Axis labels and figure title.
    name : str, optional
        Filename stem (no extension). Default ``"data_mc"``.
    rcparams : dict[str, Any], optional
        Per-call rcParams overrides applied via ``plt.rc_context``
        (scoped to this plot).

    Raises
    ------
    ValueError
        If the templates dictionary is empty or ``data``/``bin_edges``
        have incompatible shapes.
    """
    if not templates:
        msg = "Templates dictionary cannot be empty"
        raise ValueError(msg)

    apply_style(spec)
    figsize = figsize if figsize is not None else spec.figsize

    try:
        with plt.rc_context(rcparams or {}):
            data_array = convert_to_numpy(data)
            edges_array = convert_to_numpy(bin_edges)

            if len(data_array) != len(edges_array) - 1:
                msg = (
                    f"Data length ({len(data_array)}) must match "
                    f"number of bins ({len(edges_array) - 1})"
                )
                raise ValueError(msg)

            bin_centers = 0.5 * (edges_array[:-1] + edges_array[1:])

            config = plot_settings or {}
            process_names = list(templates.keys())
            process_order = config.get("process_order")

            if process_order:
                process_order = [
                    p
                    for p in process_order
                    if p in templates and (show_signal or p.lower() != "signal")
                ]
            else:
                background_processes = sorted(
                    p for p in process_names if p.lower() != "signal"
                )
                signal_present = "signal" in templates and show_signal
                process_order = background_processes + (
                    ["signal"] if signal_present else []
                )

            logger.debug("Process order: %s", process_order)

            signal_scale = fitted_params.get("mu", 1.0) if fitted_params else 1.0
            ttbar_scale = (
                fitted_params.get("norm_ttbar_semilep", 1.0)
                if fitted_params
                else 1.0
            )

            scaled_templates = {}
            for process, values in templates.items():
                scale_factor = 1.0
                if process.lower() == "signal":
                    scale_factor = signal_scale
                elif process == "ttbar_semilep":
                    scale_factor = ttbar_scale
                scaled_templates[process] = (
                    convert_to_numpy(values) * scale_factor
                )

            figure = plt.figure(figsize=figsize)
            grid_spec = GridSpec(
                2, 1, height_ratios=(3, 1), hspace=0.1, figure=figure
            )
            main_axis = figure.add_subplot(grid_spec[0])
            ratio_axis = figure.add_subplot(grid_spec[1], sharex=main_axis)

            color_map = config.get("process_colors", {})
            label_map = config.get("process_labels", {})

            hep.histplot(
                [scaled_templates[p] for p in process_order],
                edges_array,
                stack=True,
                ax=main_axis,
                label=[label_map.get(p, p) for p in process_order],
                color=[color_map.get(p) for p in process_order],
                edgecolor="k",
                histtype="fill",
                linewidth=0.5,
            )

            hep.histplot(
                data_array,
                edges_array,
                yerr=np.sqrt(data_array),
                ax=main_axis,
                marker="o",
                color="k",
                label="Data",
                markersize=5,
                capsize=2,
                histtype="errorbar",
            )

            main_axis.set_ylabel(ylabel, fontsize=label_fontsize)
            main_axis.set_title(title, fontsize=title_fontsize)
            main_axis.legend(
                frameon=False,
                fontsize=legend_fontsize,
                ncol=2,
                loc="upper right",
            )
            plt.setp(main_axis.get_xticklabels(), visible=False)

            if fitted_params:
                param_labels = config.get("jax", {}).get("fit_param_labels", {})
                annotation_lines = []
                for param, value in fitted_params.items():
                    latex_label = param_labels.get(param, param)
                    if not (
                        latex_label.startswith("$") and latex_label.endswith("$")
                    ):
                        latex_label = f"${latex_label}$"
                    annotation_lines.append(f"{latex_label} = {value:.3f}")

                main_axis.text(
                    0.05,
                    0.94,
                    "\n".join(annotation_lines),
                    transform=main_axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=annotation_fontsize,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "black",
                        "boxstyle": "round,pad=0.5",
                        "alpha": 0.8,
                    },
                )

            total_prediction = np.sum(
                [scaled_templates[p] for p in process_order], axis=0
            )
            ratio_values = np.divide(
                data_array,
                total_prediction,
                out=np.ones_like(data_array),
                where=total_prediction > 0,
            )
            ratio_errors = np.divide(
                np.sqrt(data_array),
                total_prediction,
                out=np.zeros_like(data_array),
                where=total_prediction > 0,
            )

            ratio_axis.errorbar(
                bin_centers,
                ratio_values,
                yerr=ratio_errors,
                fmt="o",
                color="k",
                capsize=2,
            )
            ratio_axis.axhline(1, color="r", linestyle="--")
            ratio_axis.set_ylim(ratio_ylim)
            ratio_axis.set_xlabel(xlabel, fontsize=ratio_label_fontsize)
            ratio_axis.set_ylabel(
                "Data/Pred.",
                fontsize=ratio_label_fontsize,
                ha="center",
                labelpad=15,
            )

            save_figure(
                figure,
                mgr["plots"] / f"{name}.{spec.output_format}",
                dpi=spec.dpi,
                description="data/MC plot",
            )
            plt.close(figure)

    except Exception as exc:
        logger.error("Failed to create data/MC histogram: %s", exc)
        raise
