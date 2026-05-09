"""Schema for the plotting section of the user config.

Carries the cross-plot defaults (rcParams, mplhep style, figsize,
dpi, output format). Plot-specific options (axis limits, bin counts,
plot-internal fontsizes) stay as function kwargs in the plotting
modules. For global fontsize control, set the relevant rcParams
entries (e.g. ``axes.labelsize``, ``legend.fontsize``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from graep.plotting.style import DEFAULT_RCPARAMS


class PlottingSpec(BaseModel):
    """Plotting section of the user config."""

    rcparams: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_RCPARAMS),
        description=(
            "matplotlib rcParams overrides applied via "
            "graep.plotting.style.apply_style. Defaults to "
            "DEFAULT_RCPARAMS (serif font, math text, axes line "
            "width). Override to tweak the global look, including "
            "fontsize entries like 'axes.labelsize' or "
            "'legend.fontsize'."
        ),
    )
    mplhep_style: str | None = Field(
        default="CMS",
        description=(
            "mplhep style name (passed to hep.style.use). Set to "
            "None to skip mplhep styling. Common choices: 'CMS', "
            "'ATLAS', 'LHCb', 'ALICE'."
        ),
    )
    figsize: tuple[float, float] = Field(
        default=(8.0, 6.0),
        description=(
            "Default figure size in inches. Plot functions accept a "
            "figsize kwarg that overrides this on a per-call basis."
        ),
    )
    dpi: int = Field(
        default=300,
        description="Default DPI when plot functions save figures to disk.",
    )
    output_format: str = Field(
        default="pdf",
        description=(
            "Default file extension for saved figures (no leading "
            "dot). Common choices: 'pdf', 'png', 'svg'."
        ),
    )
