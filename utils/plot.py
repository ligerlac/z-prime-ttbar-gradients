import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from typing import Any, Dict, List, Optional, Tuple, Union
import mplhep as hep
import jax

ArrayLike = Union[np.ndarray, Any]  # accept jnp.ndarray etc.
rcParams.update({
    "axes.formatter.use_mathtext": True,
    "text.usetex": True,
    "font.family":  "serif",
    #"font.serif":   ["Computer Modern Roman"],
})

def to_tex_scientific(x: float, sig: int = 2) -> str:
    """
    Turn a float into a 'mantissa × 10^{exp}' LaTeX string.
    E.g. 2.487e-05 → '2.487\\times10^{-5}'
    """
    s = f"{x:.{sig}e}"          # e.g. '2.487e-05'
    m, e = s.split("e")         # ['2.487', '-05']
    exp = int(e)                # -5
    return rf"{m}\times10^{{{exp}}}"

def to_numpy(x):
    """
    Bring a JAX array or tracer back to host-memory and convert to np.ndarray.
    Safe to call outside any JAX-transformed function.
    """
    # ensure any in-flight computation is done
    x = jax.device_get(x)
    return np.asarray(x)

def plot_cms_histogram(
    bin_edges: ArrayLike,
    data: ArrayLike,
    templates: Dict[str, ArrayLike],
    fitted_pars: Optional[Dict[str, float]] = None,
    plot_settings: Optional[Dict[str, str]] = None,
    show_signal: bool = True,
    figsize: Tuple[float, float] = (12, 12),
    ratio_ylim: Tuple[float, float] = (0.5, 1.5),
    xlabel: str = "",
    ylabel: str = "Events",
    title: str = "",
) -> plt.Figure:
    """
    Draw a CMS‐style pre‐/post‐fit plot using mplhep styling.

    Applies the CMS style via `hep.style.use("CMS")`, stacked backgrounds,
    data points with errors, and a ratio pad. Post‐fit scaling follows
    the same logic as your JAX model: "signal" scaled by "mu",
    "ttbar_semilep" by "norm_ttbar_semilep", others fixed.
    """
    # Apply CMS style
    hep.style.use("CMS")

    # Convert arrays
    data = to_numpy(data)
    bin_edges = to_numpy(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Determine process order
    keys = list(templates.keys())
    process_order = plot_settings.get("process_order", None) if plot_settings else None
    if process_order is None:
        bg = sorted([k for k in keys if k.lower() != "signal"] )
        if "signal" in keys and show_signal:
            process_order = bg + ["signal"]
        else:
            process_order = bg
    else:
        process_order = [k for k in process_order
                         if k in templates and (k.lower() != "signal" or show_signal)]

    # Scale templates if post‐fit
    scaled: Dict[str, np.ndarray] = {}
    # extract mu and norm_ttbar from fitted_pars if provided
    mu = fitted_pars.get("mu", 1.0) if fitted_pars else 1.0
    norm_ttbar = fitted_pars.get("norm_ttbar_semilep", 1.0) if fitted_pars else 1.0

    for proc, arr in templates.items():
        arr_np = to_numpy(arr)
        # apply the same logic as AllBkgRelaxedModelScalar.expected_data
        if fitted_pars is not None:
            if proc.lower() == "signal":
                scale = mu
            elif proc == "ttbar_semilep":
                scale = norm_ttbar
            else:
                scale = 1.0
        else:
            scale = 1.0

        scaled[proc] = arr_np * scale

    # Build stack arrays
    stack_vals = [scaled[p] for p in process_order]

    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=(3, 1), hspace=0.05, figure=fig)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    process_colors = plot_settings.get("process_colors", None) if plot_settings else None
    process_labels = plot_settings.get("process_labels", None) if plot_settings else None
    # Main plot: stacked histograms
    hep.histplot(
        [scaled[p] for p in process_order],
        bin_edges,
        stack=True,
        ax=ax_main,
        label=[process_labels.get(p) if process_labels else p for p in process_order],
        color=[process_colors.get(p) if process_colors else None for p in process_order],
        edgecolor="black",
        histtype="fill",
        linewidth=0.5,
    )
    # Data points
    hep.histplot(
        data,
        bin_edges,
        yerr=np.sqrt(data),
        ax=ax_main,
        marker='o',
        color='black',
        label='Data',
        markersize=5,
        capsize=2,
        histtype='errorbar',
    )

    ax_main.set_ylabel(ylabel, fontsize=20)
    ax_main.set_title(title)
    ax_main.legend(frameon=False, fontsize=16, ncol=2, loc='upper right')

    # Ratio
    total = np.sum(stack_vals, axis=0)
    ratio = np.divide(data, total, out=np.ones_like(data), where=total>0)
    ratio_err = np.divide(np.sqrt(data), total, out=np.zeros_like(data), where=total>0)

    ax_ratio.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        fmt='o',
        color='black',
        markersize=4,
        capsize=2,
    )
    ax_ratio.axhline(1.0, color='red', linestyle='--')
    ax_ratio.set_ylim(ratio_ylim)
    ax_ratio.set_xlabel(xlabel, fontsize=20)
    ax_ratio.set_ylabel('Data / Pred.', fontsize=20, ha='center', va='center', labelpad=20)

    # Clean up ticks
    plt.setp(ax_main.get_xticklabels(), visible=False)

    return fig


def plot_pval_history(pval_history,
                      aux_history,
                      mle_history,
                      gradients,
                      learning_rates,
                      fname="pval_history.png",
                      plot_settings=None,

                      ):
    """
    Plots p-value vs parameter histories in an ⌈√N⌉×⌈√N⌉ grid, where N is the total number
    of parameters (aux + mle). Unused subplots are turned off.

    Args:
        pval_history (Sequence[float]): Sequence of p-values.
        aux_history (Dict[str, Sequence[float]]): Histories for auxiliary parameters.
        mle_history (Dict[str, Sequence[float]]): Histories for MLE parameters.
        fname (str): File name to save the figure to.
    """
    # Combine into one mapping
    param_history = {
        **{f"aux__{k}": hist for k, hist in aux_history.items()},
        **{f"mle__{k}": hist for k, hist in mle_history.items()},
    }

    # filter out any whose history is constant
    filtered = {}
    for name, hist in param_history.items():
        arr = np.asarray(hist)
        if not np.allclose(arr, arr[0]):
            filtered[name] = hist

    if not filtered:
        print("No parameters changed; nothing to plot.")
        return

    n_params = len(filtered.keys())
    grid_size = math.ceil(math.sqrt(n_params))

    # Create grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(4 * grid_size, 3 * grid_size),
                             sharey=True, squeeze=False)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Plot each parameter history
    for idx, ((name, history), ax) in enumerate(zip(filtered.items(), axes_flat)):

        # Get label
        label = ""
        gradient = gradients["aux"].get(name.strip("aux__"), None)
        learning_rate = learning_rates.get(name.strip("aux__"), None)
        label_parts = []
        if gradient is not None:
            label_parts.append(
                rf"$\Delta_{{\theta}}(p_s) = {to_tex_scientific(gradient)}$"
            )
        if learning_rate is not None:
            label_parts.append(
                rf"$\eta = {to_tex_scientific(learning_rate)}$"
            )

        label = ", ".join(label_parts) if label_parts else None
        ax.plot(history, pval_history, marker='o', linestyle='-', markersize=5, label=label)

        ax.set_title(label)
        aux_param_labels = plot_settings.jax.get("aux_param_labels", {}) if plot_settings else {}
        fit_param_labels = plot_settings.jax.get("fit_param_labels", {}) if plot_settings else {}
        param_labels = {**aux_param_labels, **fit_param_labels}
        if "aux__" in name:
            param = name.removeprefix("aux__")
        if "mle__" in name:
            param = name.removeprefix("mle__")
        param_label = param_labels.get(param) if param_labels else name
        ax.set_xlabel(f"{param_label} value")

        # if this is in the first column (col index = 0), show ylabel
        if idx % grid_size == 0:
            ax.set_ylabel(r"$p$-value")
        else:
            # hide y-label text and tick labels
            ax.set_ylabel("")
            ax.tick_params(labelleft=True)

        plt.ticklabel_format(axis="y", style='sci', scilimits=(-3, 10))
        ax.grid(True, alpha=0.3)

    # Turn off any unused axes
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)