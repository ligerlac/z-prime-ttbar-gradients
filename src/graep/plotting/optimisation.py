"""Plots that look at the optimisation trace.

``pvalue_vs_parameters`` plots p-value against each varying parameter.
``parameters_over_iterations`` plots each varying parameter (and the
p-value) against iteration index.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from graep.plotting.style import (
    apply_style,
    convert_to_numpy,
    format_scientific_latex,
)
from graep.plotting.utils import save_figure

if TYPE_CHECKING:
    from graep.config.plotting import PlottingSpec
    from graep.output.manager import OutputManager

logger = logging.getLogger(__name__)


def _collect_varying_parameters(
    auxiliary_history: dict[str, Sequence[float]],
    mle_history: dict[str, Sequence[float]],
) -> dict[str, Sequence[float]]:
    """Combine and filter to only parameters that vary across iterations."""
    parameter_history: dict[str, Sequence[float]] = {}
    for name, history in auxiliary_history.items():
        if "__NN" not in name:
            parameter_history[f"aux__{name}"] = history
    for name, history in mle_history.items():
        parameter_history[f"mle__{name}"] = history

    non_constant: dict[str, Sequence[float]] = {}
    for name, history in parameter_history.items():
        arr = np.asarray(history)
        if not np.allclose(arr, arr[0]):
            non_constant[name] = history
    return non_constant


def _trace_title(
    base_name: str,
    gradients: dict[str, Any],
    learning_rates: dict[str, float],
) -> str:
    """Build the small per-axis title with gradient and learning-rate labels."""
    parts: list[str] = []
    if base_name in gradients.get("aux", {}):
        grad_val = convert_to_numpy(gradients["aux"][base_name])
        parts.append(
            r"$\Delta_{\theta}(p_s) = "
            f"{format_scientific_latex(grad_val)}$"
        )
    if base_name in learning_rates:
        lr_val = learning_rates[base_name]
        parts.append(r"$\eta = " f"{format_scientific_latex(lr_val)}$")
    return ", ".join(parts) if parts else ""


def pvalue_vs_parameters(
    spec: PlottingSpec,
    mgr: OutputManager,
    pvalue_history: Sequence[float],
    auxiliary_history: dict[str, Sequence[float]],
    mle_history: dict[str, Sequence[float]],
    gradients: dict[str, Any],
    learning_rates: dict[str, float],
    *,
    plot_settings: dict[str, Any] | None = None,
    label_fontsize: int = 10,
    title_fontsize: int = 10,
    tick_fontsize: int = 10,
    name: str = "pvalue_vs_parameters",
    rcparams: dict[str, Any] | None = None,
) -> None:
    """Plot p-value against each varying parameter.

    Saves to ``mgr["plots"] / f"{name}.{spec.output_format}"``.

    Parameters
    ----------
    spec : PlottingSpec
        Plotting configuration. Used for style, ``dpi`` and
        ``output_format``.
    mgr : OutputManager
        Output manager. The figure is written under ``mgr["plots"]``.
    pvalue_history : Sequence[float]
        History of p-values during optimisation.
    auxiliary_history : dict[str, Sequence[float]]
        History of auxiliary parameters.
    mle_history : dict[str, Sequence[float]]
        History of MLE parameters.
    gradients : dict[str, Any]
        Gradient values for parameters.
    learning_rates : dict[str, float]
        Learning rates used during optimisation.
    plot_settings : dict[str, Any], optional
        Carries LaTeX label dictionaries under ``plot_settings["jax"]``:
        ``aux_param_labels`` and ``fit_param_labels``.
    name : str, optional
        Filename stem (no extension). Default ``"pvalue_vs_parameters"``.
    rcparams : dict[str, Any], optional
        Per-call rcParams overrides applied via ``plt.rc_context``.

    Raises
    ------
    ValueError
        If no varying parameters are found or input data is invalid.
    OSError
        If the file cannot be saved.
    """
    logger.info("Creating p-value vs parameters plot")

    apply_style(spec)

    try:
        with plt.rc_context(rcparams or {}):
            non_constant_params = _collect_varying_parameters(
                auxiliary_history, mle_history
            )

            if not non_constant_params:
                logger.warning("No varying parameters found - skipping plot")
                return

            logger.debug(
                "Found %d varying parameters", len(non_constant_params)
            )

            num_params = len(non_constant_params)
            grid_size = math.ceil(math.sqrt(num_params))
            fig, axes = plt.subplots(
                grid_size,
                grid_size,
                figsize=(4 * grid_size, 3 * grid_size),
                sharey=True,
                squeeze=False,
            )
            axes_flat = axes.flatten()

            config = plot_settings or {}
            jax_config = config.get("jax", {})
            aux_labels = jax_config.get("aux_param_labels", {})
            fit_labels = jax_config.get("fit_param_labels", {})
            param_labels = {**aux_labels, **fit_labels}

            for idx, (ax, (param_name, history)) in enumerate(
                zip(axes_flat, non_constant_params.items(), strict=False)
            ):
                base_name = param_name.split("__", 1)[-1]
                ax.set_title(
                    _trace_title(base_name, gradients, learning_rates),
                    fontsize=title_fontsize,
                )

                ax.plot(history, pvalue_history, "o-", ms=5)

                display_name = param_labels.get(base_name, base_name)
                ax.set_xlabel(
                    f"{display_name} Value", fontsize=label_fontsize
                )
                if idx % grid_size == 0:
                    ax.set_ylabel(r"$p$-value", fontsize=label_fontsize)
                ax.grid(True, alpha=0.3)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            for ax in axes_flat[:num_params]:
                ax.yaxis.set_major_formatter(formatter)
                ax.tick_params(axis="both", labelsize=tick_fontsize)
                ax.yaxis.offsetText.set_fontsize(tick_fontsize)

            for ax in axes_flat[num_params:]:
                ax.set_visible(False)

            plt.tight_layout()
            save_figure(
                fig,
                mgr["plots"] / f"{name}.{spec.output_format}",
                dpi=spec.dpi,
                description="p-value vs parameters plot",
            )
            plt.close(fig)

    except Exception as exc:
        logger.error("Failed to create p-value vs parameters plot: %s", exc)
        raise


def parameters_over_iterations(
    spec: PlottingSpec,
    mgr: OutputManager,
    pvalue_history: Sequence[float],
    auxiliary_history: dict[str, Sequence[float]],
    mle_history: dict[str, Sequence[float]],
    gradients: dict[str, Any],
    learning_rates: dict[str, float],
    *,
    plot_settings: dict[str, Any] | None = None,
    label_fontsize: int = 10,
    title_fontsize: int = 10,
    tick_fontsize: int = 10,
    name: str = "parameters_over_iterations",
    rcparams: dict[str, Any] | None = None,
) -> None:
    """Plot parameter values and p-value over optimisation iterations.

    Saves to ``mgr["plots"] / f"{name}.{spec.output_format}"``.

    Parameters
    ----------
    spec : PlottingSpec
        Plotting configuration. Used for style, ``dpi`` and
        ``output_format``.
    mgr : OutputManager
        Output manager. The figure is written under ``mgr["plots"]``.
    pvalue_history : Sequence[float]
        History of p-values during optimisation.
    auxiliary_history : dict[str, Sequence[float]]
        History of auxiliary parameters.
    mle_history : dict[str, Sequence[float]]
        History of MLE parameters.
    gradients : dict[str, Any]
        Gradient values for parameters.
    learning_rates : dict[str, float]
        Learning rates used during optimisation.
    plot_settings : dict[str, Any], optional
        Carries LaTeX label dictionaries under ``plot_settings["jax"]``:
        ``aux_param_labels`` and ``fit_param_labels``.
    name : str, optional
        Filename stem (no extension). Default
        ``"parameters_over_iterations"``.
    rcparams : dict[str, Any], optional
        Per-call rcParams overrides applied via ``plt.rc_context``.

    Raises
    ------
    ValueError
        If no varying parameters are found or input data is invalid.
    OSError
        If the file cannot be saved.
    """
    logger.info("Creating parameters vs iterations plot")

    apply_style(spec)

    try:
        with plt.rc_context(rcparams or {}):
            non_constant_params = _collect_varying_parameters(
                auxiliary_history, mle_history
            )

            non_constant_params["pvalue"] = np.asarray(pvalue_history)
            if not non_constant_params:
                logger.warning("No varying parameters found - skipping plot")
                return

            logger.debug(
                "Found %d varying parameters (including p-value)",
                len(non_constant_params),
            )

            num_params = len(non_constant_params)
            num_iterations = len(pvalue_history)
            iteration_steps = np.arange(num_iterations)
            grid_size = math.ceil(math.sqrt(num_params))
            fig, axes = plt.subplots(
                grid_size,
                grid_size,
                figsize=(4 * grid_size, 3 * grid_size),
                sharex=True,
                squeeze=False,
            )
            axes_flat = axes.flatten()

            config = plot_settings or {}
            jax_config = config.get("jax", {})
            aux_labels = jax_config.get("aux_param_labels", {})
            fit_labels = jax_config.get("fit_param_labels", {})
            param_labels = {**aux_labels, **fit_labels}

            for ax, (param_name, history) in zip(
                axes_flat, non_constant_params.items(), strict=False
            ):
                base_name = (
                    param_name.split("__", 1)[-1]
                    if "__" in param_name
                    else param_name
                )
                ax.set_title(
                    _trace_title(base_name, gradients, learning_rates),
                    fontsize=title_fontsize,
                )

                ax.plot(iteration_steps, history, "o-", ms=3)

                display_name = (
                    r"$p$-value"
                    if param_name == "pvalue"
                    else param_labels.get(base_name, base_name)
                )
                ax.set_ylabel(display_name, fontsize=label_fontsize)
                ax.set_xlabel("Iteration", fontsize=label_fontsize)
                ax.grid(True, alpha=0.3)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            for ax in axes_flat:
                ax.yaxis.set_major_formatter(formatter)
                ax.tick_params(axis="both", labelsize=tick_fontsize)

            for ax in axes_flat[num_params:]:
                ax.set_visible(False)

            plt.tight_layout()
            save_figure(
                fig,
                mgr["plots"] / f"{name}.{spec.output_format}",
                dpi=spec.dpi,
                description="parameters vs iterations plot",
            )
            plt.close(fig)

    except Exception as exc:
        logger.error(
            "Failed to create parameters vs iterations plot: %s", exc
        )
        raise
