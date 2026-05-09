"""Shared plotting style and array helpers.

Holds the default rcParams baseline, the explicit style application
that consumes a :class:`PlottingSpec`, and the JAX/NumPy conversion
shim every plot uses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import jax
import mplhep as hep
import numpy as np
from matplotlib import rcParams

if TYPE_CHECKING:
    from graep.config.plotting import PlottingSpec

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | Any


DEFAULT_RCPARAMS: dict[str, Any] = {
    "axes.formatter.use_mathtext": True,
    "text.usetex": True,
    "font.family": "serif",
    "axes.linewidth": 1.2,
}


def apply_style(spec: PlottingSpec) -> None:
    """Apply the spec's rcParams overrides and (optionally) an mplhep style.

    Plot functions call this themselves so users do not have to
    remember to set up styling. Mutates global matplotlib state, which
    is why it's a function rather than an import-time side effect.
    """
    rcParams.update(spec.rcparams)
    if spec.mplhep_style is not None:
        hep.style.use(spec.mplhep_style)


def format_scientific_latex(value: float, significant_digits: int = 2) -> str:
    """Convert a floating-point number to LaTeX scientific notation.

    Parameters
    ----------
    value : float
        Numeric value to format.
    significant_digits : int, optional
        Number of significant digits to retain, by default 2.

    Returns
    -------
    str
        LaTeX-formatted string in scientific notation.

    Examples
    --------
    >>> format_scientific_latex(2.487e-5)
    '2.49\\times10^{-5}'

    Raises
    ------
    ValueError
        If ``significant_digits`` is not positive.
    TypeError
        If ``value`` is not a number.
    """
    if not isinstance(value, (int, float, np.number, np.ndarray)):
        msg = f"Value must be a number, got {type(value)}"
        raise TypeError(msg)
    if significant_digits <= 0:
        msg = "significant_digits must be positive"
        raise ValueError(msg)
    if isinstance(value, np.ndarray):
        if value.shape != ():
            msg = f"Value must be a scalar, got shape {value.shape}"
            raise ValueError(msg)
        value = value.item()

    try:
        formatted = f"{value:.{significant_digits}e}"
        mantissa, exponent_part = formatted.split("e")
        exponent = int(exponent_part)
        return rf"{mantissa}\times10^{{{exponent}}}"
    except (ValueError, OverflowError) as exc:
        logger.error("Failed to format %s in scientific notation: %s", value, exc)
        return str(value)


def convert_to_numpy(array_like: ArrayLike) -> np.ndarray:
    """Convert JAX arrays or tracers to host-memory NumPy arrays.

    Safe to call outside any JAX-transformed function. Handles device
    transfer and conversion to standard NumPy arrays.

    Parameters
    ----------
    array_like : ArrayLike
        Input array (JAX array, tracer, or NumPy array).

    Returns
    -------
    np.ndarray
        Converted NumPy array on host memory.

    Raises
    ------
    ValueError
        If ``array_like`` cannot be converted to a numpy array.
    """
    try:
        device_array = jax.device_get(array_like)
        return np.asarray(device_array)
    except Exception as exc:
        logger.error("Failed to convert array to numpy: %s", exc)
        msg = f"Cannot convert input to numpy array: {exc}"
        raise ValueError(msg) from exc
