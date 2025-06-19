import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
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
    bin_edges, data, templates,
    fitted_pars=None, plot_settings=None,
    show_signal=True, figsize=(12,12), ratio_ylim=(0.5,1.5),
    xlabel="", ylabel="Events", title="",
) -> plt.Figure:
    """
    CMS‐style pre/post‐fit plot with an overlay of fit parameters.
    """
    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Apply CMS style and convert inputs to numpy
    # ──────────────────────────────────────────────────────────────────────────
    hep.style.use("CMS")
    data = np.asarray(data)
    edges = np.asarray(bin_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Determine process draw order
    # ──────────────────────────────────────────────────────────────────────────
    cfg = plot_settings or {}
    keys = list(templates)
    order = cfg.get("process_order")
    if order:
        order = [
            k for k in order
            if k in templates and (show_signal or k.lower() != "signal")
        ]
    else:
        bg = sorted(k for k in keys if k.lower() != "signal")
        order = bg + (["signal"] if "signal" in keys and show_signal else [])

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Scale templates according to fitted parameters
    # ──────────────────────────────────────────────────────────────────────────
    mu  = fitted_pars.get("mu", 1.0)                 if fitted_pars else 1.0
    ntt = fitted_pars.get("norm_ttbar_semilep", 1.0) if fitted_pars else 1.0
    scaled = {
        p: np.asarray(arr) * (
            mu if p.lower() == "signal"
            else ntt if p == "ttbar_semilep"
            else 1.0
        )
        for p, arr in templates.items()
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Create figure and two panels (main + ratio)
    # ──────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(2, 1, height_ratios=(3,1), hspace=0.1, figure=fig)
    ax_main  = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Draw stacked backgrounds and data points
    # ──────────────────────────────────────────────────────────────────────────
    colors = cfg.get("process_colors", {})
    labels = cfg.get("process_labels", {})
    hep.histplot(
        [scaled[p] for p in order], edges,
        stack=True, ax=ax_main,
        label=[labels.get(p, p) for p in order],
        color=[colors.get(p) for p in order],
        edgecolor="k", histtype="fill", linewidth=0.5,
    )
    hep.histplot(
        data, edges, yerr=np.sqrt(data),
        ax=ax_main, marker='o', color='k',
        label="Data", markersize=5, capsize=2, histtype="errorbar",
    )
    ax_main.set_ylabel(ylabel, fontsize=20)
    ax_main.set_title(title, fontsize=18)
    ax_main.legend(frameon=False, fontsize=16, ncol=2, loc="upper right")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Overlay fit‐parameter tile with LaTeX labels
    # ──────────────────────────────────────────────────────────────────────────
    if fitted_pars:
        fit_lbls = cfg.get("jax", {}).get("fit_param_labels", {})
        lines = []
        for key, val in fitted_pars.items():
            tex = fit_lbls.get(key, key)
            if not (tex.startswith("$") and tex.endswith("$")):
                tex = f"${tex}$"
            lines.append(f"{tex} = {val:.3f}")
        ax_main.text(
            0.05, 0.94, "\n".join(lines),
            transform=ax_main.transAxes,
            va="top", ha="left",
            fontsize=18,
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="round,pad=0.5",
                alpha=0.8
            )
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Compute and draw ratio panel
    # ──────────────────────────────────────────────────────────────────────────
    total = np.sum([scaled[p] for p in order], axis=0)
    ratio = np.divide(data, total, out=np.ones_like(data), where=total>0)
    err   = np.divide(np.sqrt(data), total, out=np.zeros_like(data), where=total>0)
    ax_ratio.errorbar(
        centers, ratio, yerr=err,
        fmt='o', color='k', capsize=2
    )
    ax_ratio.axhline(1, color='r', linestyle='--')
    ax_ratio.set_ylim(ratio_ylim)
    ax_ratio.set_xlabel(xlabel, fontsize=14)
    ax_ratio.set_ylabel("Data/Pred.", fontsize=14, ha="center", labelpad=15)

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

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        for ax in axes_flat[:n_params]:
            ax.yaxis.set_major_formatter(formatter)

        ax.grid(True, alpha=0.3)

    # Turn off any unused axes
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)

def plot_params_per_iter(pval_history,
                      aux_history,
                      mle_history,
                      gradients,
                      learning_rates,
                      fname="params_iters.png",
                      plot_settings=None,

                      ):
    """
    Plots p-value/ parameter histories VS # iterations ⌈√N⌉×⌈√N⌉ grid, where N is the total number
    of parameters (aux + mle) + 1 (pval). Unused subplots are turned off.

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

    filtered["pval"] = np.asarray(pval_history)  # add pval history
    if not filtered:
        print("No parameters changed; nothing to plot.")
        return

    n_params = len(filtered.keys())
    n_iterations = len(pval_history)
    steps = np.arange(n_iterations)
    grid_size = math.ceil(math.sqrt(n_params))

    # Create grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size,
                             figsize=(4 * grid_size, 3 * grid_size),
                             sharex=True, squeeze=False)

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
        ax.plot(steps, history, marker='o', linestyle='-', markersize=3, label=label)

        ax.set_title(label)

        aux_param_labels = plot_settings.jax.get("aux_param_labels", {}) if plot_settings else {}
        fit_param_labels = plot_settings.jax.get("fit_param_labels", {}) if plot_settings else {}
        param_labels = {**aux_param_labels, **fit_param_labels}
        if "aux__" in name:
            param = name.removeprefix("aux__")
        if "mle__" in name:
            param = name.removeprefix("mle__")
        param_label = param_labels.get(param) if param_labels else name
        if name == "pval":
            param_label = r"$p$-value"

        ax.set_ylabel(f"{param_label}", labelpad=10)
        ax.set_xlabel("Number of iterations")
        ax.tick_params(labelbottom=True)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        for ax in axes_flat[:n_params]:
            ax.yaxis.set_major_formatter(formatter)

        ax.grid(True, alpha=0.3)

    # Turn off any unused axes
    for ax in axes_flat[n_params:]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)