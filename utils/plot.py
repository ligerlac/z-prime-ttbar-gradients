import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter

# Configure matplotlib global settings
rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": True,
    "font.family": "serif",
})

# Type alias for array-like objects
ArrayLike = Union[np.ndarray, Any]  # Accepts jax.numpy.ndarray, etc.


def format_scientific_latex(value: float, significant_digits: int = 2) -> str:
    """
    Convert a floating-point number to LaTeX scientific notation.

    Parameters
    ----------
    value : float
        Numeric value to format
    significant_digits : int, optional
        Number of significant digits to retain, by default 2

    Returns
    -------
    str
        LaTeX-formatted string in scientific notation

    Examples
    --------
    >>> format_scientific_latex(2.487e-5)
    '2.487\\\\times10^{-5}'
    """
    formatted = f"{value:.{significant_digits}e}"
    mantissa, exponent_part = formatted.split("e")
    exponent = int(exponent_part)
    return rf"{mantissa}\times10^{{{exponent}}}"


def convert_to_numpy(array_like: ArrayLike) -> np.ndarray:
    """
    Convert JAX arrays or tracers to host-memory NumPy arrays.

    Safe to call outside any JAX-transformed function. Handles device
    transfer and conversion to standard NumPy arrays.

    Parameters
    ----------
    array_like : ArrayLike
        Input array (JAX array, tracer, or NumPy array)

    Returns
    -------
    np.ndarray
        Converted NumPy array on host memory
    """
    device_array = jax.device_get(array_like)
    return np.asarray(device_array)


def create_cms_histogram(
    bin_edges: ArrayLike,
    data: ArrayLike,
    templates: Dict[str, ArrayLike],
    fitted_params: Optional[Dict[str, float]] = None,
    plot_settings: Optional[Dict[str, Any]] = None,
    show_signal: bool = True,
    figsize: Tuple[float, float] = (12, 12),
    ratio_ylim: Tuple[float, float] = (0.5, 1.5),
    xlabel: str = "",
    ylabel: str = "Events",
    title: str = "",
) -> plt.Figure:
    """
    Create a CMS-style pre-fit/post-fit histogram plot with ratio panel.

    Parameters
    ----------
    bin_edges : ArrayLike
        Bin edges for histogram
    data : ArrayLike
        Observed data values
    templates : Dict[str, ArrayLike]
        Dictionary of template histograms (process name → bin counts)
    fitted_params : Dict[str, float], optional
        Fitted parameters for scaling templates, by default None
    plot_settings : Dict[str, Any], optional
        Customization settings dictionary containing:
        - process_order: Ordered process names
        - process_colors: Color mapping for processes
        - process_labels: Label mapping for processes
        - jax.fit_param_labels: LaTeX labels for parameters
        by default None
    show_signal : bool, optional
        Whether to display signal process, by default True
    figsize : Tuple[float, float], optional
        Figure dimensions, by default (12, 12)
    ratio_ylim : Tuple[float, float], optional
        Y-axis limits for ratio panel, by default (0.5, 1.5)
    xlabel : str, optional
        X-axis label, by default ""
    ylabel : str, optional
        Y-axis label for main panel, by default "Events"
    title : str, optional
        Figure title, by default ""

    Returns
    -------
    plt.Figure
        Generated matplotlib figure

    Notes
    -----
    - Uses CMS plotting style via mplhep
    - Creates two panels: main histogram and data/prediction ratio
    - Includes parameter value annotations when fitted_params are provided
    """
    # Apply CMS style and convert inputs
    hep.style.use("CMS")
    data_array = convert_to_numpy(data)
    edges_array = convert_to_numpy(bin_edges)
    bin_centers = 0.5 * (edges_array[:-1] + edges_array[1:])

    # Determine process drawing order
    config = plot_settings or {}
    process_names = list(templates.keys())
    process_order = config.get("process_order")

    if process_order:
        process_order = [
            p for p in process_order
            if p in templates and (show_signal or p.lower() != "signal")
        ]
    else:
        background_processes = sorted(
            p for p in process_names if p.lower() != "signal"
        )
        signal_present = "signal" in templates and show_signal
        process_order = background_processes + (["signal"] if signal_present else [])

    # Scale templates according to fitted parameters
    signal_scale = fitted_params.get("mu", 1.0) if fitted_params else 1.0
    ttbar_scale = (
        fitted_params.get("norm_ttbar_semilep", 1.0) if fitted_params else 1.0
    )

    scaled_templates = {}
    for process, values in templates.items():
        scale_factor = 1.0
        if process.lower() == "signal":
            scale_factor = signal_scale
        elif process == "ttbar_semilep":
            scale_factor = ttbar_scale
        scaled_templates[process] = convert_to_numpy(values) * scale_factor

    # Create figure with two panels (main + ratio)
    figure = plt.figure(figsize=figsize)
    grid_spec = GridSpec(2, 1, height_ratios=(3, 1), hspace=0.1, figure=figure)
    main_axis = figure.add_subplot(grid_spec[0])
    ratio_axis = figure.add_subplot(grid_spec[1], sharex=main_axis)

    # Plot stacked backgrounds and data points
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

    main_axis.set_ylabel(ylabel, fontsize=20)
    main_axis.set_title(title, fontsize=18)
    main_axis.legend(frameon=False, fontsize=16, ncol=2, loc="upper right")
    plt.setp(main_axis.get_xticklabels(), visible=False)

    # Add parameter annotation box
    if fitted_params:
        param_labels = config.get("jax", {}).get("fit_param_labels", {})
        annotation_lines = []
        for param, value in fitted_params.items():
            latex_label = param_labels.get(param, param)
            if not (latex_label.startswith("$") and latex_label.endswith("$")):
                latex_label = f"${latex_label}$"
            annotation_lines.append(f"{latex_label} = {value:.3f}")

        main_axis.text(
            0.05,
            0.94,
            "\n".join(annotation_lines),
            transform=main_axis.transAxes,
            va="top",
            ha="left",
            fontsize=18,
            bbox={
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round,pad=0.5",
                "alpha": 0.8,
            },
        )

    # Compute and plot ratio
    total_prediction = np.sum([scaled_templates[p] for p in process_order], axis=0)
    ratio_values = np.divide(
        data_array, total_prediction, out=np.ones_like(data_array), where=total_prediction > 0
    )
    ratio_errors = np.divide(
        np.sqrt(data_array),
        total_prediction,
        out=np.zeros_like(data_array),
        where=total_prediction > 0,
    )

    ratio_axis.errorbar(
        bin_centers, ratio_values, yerr=ratio_errors, fmt="o", color="k", capsize=2
    )
    ratio_axis.axhline(1, color="r", linestyle="--")
    ratio_axis.set_ylim(ratio_ylim)
    ratio_axis.set_xlabel(xlabel, fontsize=14)
    ratio_axis.set_ylabel("Data/Pred.", fontsize=14, ha="center", labelpad=15)

    return figure


def plot_pvalue_vs_parameters(
    pvalue_history: Sequence[float],
    auxiliary_history: Dict[str, Sequence[float]],
    mle_history: Dict[str, Sequence[float]],
    gradients: Dict[str, float],
    learning_rates: Dict[str, float],
    filename: str = "pvalue_vs_parameters.png",
    plot_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot p-values versus parameter values during optimization.

    Creates a grid of subplots showing the relationship between parameter values
    and p-values. Each subplot shows the trajectory of one parameter versus p-value.

    Parameters
    ----------
    pvalue_history : Sequence[float]
        History of p-values during optimization
    auxiliary_history : Dict[str, Sequence[float]]
        History of auxiliary parameters
    mle_history : Dict[str, Sequence[float]]
        History of maximum likelihood estimation parameters
    gradients : Dict[str, float]
        Gradient values for parameters
    learning_rates : Dict[str, float]
        Learning rates used during optimization
    filename : str, optional
        Output filename, by default "pvalue_vs_parameters.png"
    plot_settings : Dict[str, Any], optional
        Dictionary containing:
        - jax.aux_param_labels: LaTeX labels for auxiliary parameters
        - jax.fit_param_labels: LaTeX labels for MLE parameters
        by default None

    Returns
    -------
    None
        Saves plot to specified file
    """
    # Combine parameter histories
    parameter_history = {}
    for name, history in auxiliary_history.items():
        if "__NN" not in name:  # Exclude neural network parameters
            parameter_history[f"aux__{name}"] = history
    for name, history in mle_history.items():
        parameter_history[f"mle__{name}"] = history

    # Filter out constant parameters
    non_constant_params = {}
    for name, history in parameter_history.items():
        arr = np.asarray(history)
        if not np.allclose(arr, arr[0]):
            non_constant_params[name] = history

    if not non_constant_params:
        print("No varying parameters found - skipping plot")
        return

    # Create subplot grid
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

    # Get labeling information
    config = plot_settings or {}
    jax_config = config.get("jax", {})
    aux_labels = jax_config.get("aux_param_labels", {})
    fit_labels = jax_config.get("fit_param_labels", {})
    param_labels = {**aux_labels, **fit_labels}

    # Plot each parameter's history against p-values
    for idx, (ax, (param_name, history)) in enumerate(
        zip(axes_flat, non_constant_params.items())
    ):
        # Extract base parameter name
        base_name = param_name.split("__", 1)[-1]

        # Create title string with gradient and learning rate
        title_parts = []
        if base_name in gradients.get("aux", {}):
            grad_val = gradients["aux"][base_name]
            title_parts.append(
                r"$\Delta_{\theta}(p_s) = "
                f"{format_scientific_latex(grad_val)}$"
            )
        if base_name in learning_rates:
            lr_val = learning_rates[base_name]
            title_parts.append(
                r"$\eta = " f"{format_scientific_latex(lr_val)}$"
            )
        title_text = ", ".join(title_parts) if title_parts else ""

        # Plot trajectory
        ax.plot(history, pvalue_history, "o-", ms=5)
        ax.set_title(title_text, fontsize=10)

        # Configure axes
        display_name = param_labels.get(base_name, base_name)
        ax.set_xlabel(f"{display_name} Value")
        if idx % grid_size == 0:
            ax.set_ylabel(r"$p$-value")
        ax.grid(True, alpha=0.3)


    # Configure axes formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    for ax in axes_flat[:num_params]:
        ax.yaxis.set_major_formatter(formatter)

    # Hide unused subplots
    for ax in axes_flat[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parameters_over_iterations(
    pvalue_history: Sequence[float],
    auxiliary_history: Dict[str, Sequence[float]],
    mle_history: Dict[str, Sequence[float]],
    gradients: Dict[str, float],
    learning_rates: Dict[str, float],
    filename: str = "parameters_vs_iterations.png",
    plot_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plot parameter values and p-value over optimization iterations.

    Creates a grid of subplots showing the evolution of parameters and p-value
    as a function of iteration number.

    Parameters
    ----------
    pvalue_history : Sequence[float]
        History of p-values during optimization
    auxiliary_history : Dict[str, Sequence[float]]
        History of auxiliary parameters
    mle_history : Dict[str, Sequence[float]]
        History of maximum likelihood estimation parameters
    gradients : Dict[str, float]
        Gradient values for parameters
    learning_rates : Dict[str, float]
        Learning rates used during optimization
    filename : str, optional
        Output filename, by default "parameters_vs_iterations.png"
    plot_settings : Dict[str, Any], optional
        Dictionary containing:
        - jax.aux_param_labels: LaTeX labels for auxiliary parameters
        - jax.fit_param_labels: LaTeX labels for MLE parameters
        by default None

    Returns
    -------
    None
        Saves plot to specified file
    """
    # Combine parameter histories
    parameter_history = {}
    for name, history in auxiliary_history.items():
        if "__NN" not in name:  # Exclude neural network parameters
            parameter_history[f"aux__{name}"] = history
    for name, history in mle_history.items():
        parameter_history[f"mle__{name}"] = history

    # Filter out constant parameters
    non_constant_params = {}
    for name, history in parameter_history.items():
        arr = np.asarray(history)
        if not np.allclose(arr, arr[0]):
            non_constant_params[name] = history

    # Add p-value to parameters
    non_constant_params["pvalue"] = np.asarray(pvalue_history)
    if not non_constant_params:
        print("No varying parameters found - skipping plot")
        return

    # Create subplot grid
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

    # Get labeling information
    config = plot_settings or {}
    jax_config = config.get("jax", {})
    aux_labels = jax_config.get("aux_param_labels", {})
    fit_labels = jax_config.get("fit_param_labels", {})
    param_labels = {**aux_labels, **fit_labels}

    # Plot each parameter's evolution
    for idx, (ax, (param_name, history)) in enumerate(
        zip(axes_flat, non_constant_params.items())
    ):
        # Extract base parameter name
        base_name = param_name.split("__", 1)[-1] if "__" in param_name else param_name

        # Create title string with gradient and learning rate
        title_parts = []
        if base_name in gradients.get("aux", {}):
            grad_val = gradients["aux"][base_name]
            title_parts.append(
                r"$\Delta_{\theta}(p_s) = "
                f"{format_scientific_latex(grad_val)}$"
            )
        if base_name in learning_rates:
            lr_val = learning_rates[base_name]
            title_parts.append(
                r"$\eta = " f"{format_scientific_latex(lr_val)}$"
            )
        title_text = ", ".join(title_parts) if title_parts else ""

        # Plot trajectory
        ax.plot(iteration_steps, history, "o-", ms=3)
        ax.set_title(title_text, fontsize=10)

        # Configure axes
        if param_name == "pvalue":
            display_name = r"$p$-value"
        else:
            display_name = param_labels.get(base_name, base_name)
        ax.set_ylabel(display_name)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)


    # Configure axes formatting
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    for ax in axes_flat:
        ax.yaxis.set_major_formatter(formatter)

    # Hide unused subplots
    for ax in axes_flat[num_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)