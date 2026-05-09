"""Root user-config object.

The user assembles one :class:`Config` instance per analysis,
populating each section with the appropriate spec. The config
object is what notebooks (or other entry points) consume.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from graep.config.inputs.base import FilesetSpec
from graep.config.output import OutputSpec


class Config(BaseModel):
    """Top-level configuration for one analysis.

    Each field is one section of the framework. Sections are added as
    the corresponding submodules become available.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inputs: FilesetSpec = Field(
        ...,
        description=(
            "Inputs (datasets) section. A concrete FilesetSpec subclass "
            "such as OpenDataPortalFilesetSpec or RucioFilesetSpec."
        ),
    )
    output: OutputSpec = Field(
        default_factory=OutputSpec,
        description=(
            "Output section. Holds the root output directory and the "
            "category-to-subdirectory map used by OutputManager."
        ),
    )
