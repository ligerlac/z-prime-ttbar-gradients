from typing import List, Tuple, Optional, Callable, Literal, Union, Annotated
from pydantic import BaseModel, Field, model_validator
import re

# Type alias for (object, variable) pairs
ObjVar = Tuple[str, str]

class SubscriptableModel(BaseModel):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

# ------------------------
# General configuration
# ------------------------
class GeneralConfig(SubscriptableModel):
    lumi: Annotated[
        float,
        Field(description="Integrated luminosity in /pb")
    ]
    weights_branch: Annotated[
        str,
        Field(description="Branch name for event weight")
    ]
    max_files: Annotated[
        Optional[int],
        Field(default=1, description="Maximum number of files to process")
    ]
    run_preprocessing: Annotated[
        bool,
        Field(default=False, description="Whether to run preprocessing step")
    ]
    run_histogramming: Annotated[
        bool,
        Field(default=True, description="Whether to run histogramming step")
    ]
    output_dir: Annotated[
        Optional[str],
        Field(default="output/", description="Directory for output files")
    ]
    preprocessor: Annotated[
        Literal["uproot", "dak"],
        Field(default="uproot", description="Preprocessor to use")
    ]
    preprocessed_dir: Annotated[
        Optional[str],
        Field(default=None, description="Directory containing preprocessed files")
        ]


# ------------------------
# Preprocessing configuration
# ------------------------
class PreprocessConfig(SubscriptableModel):
    branches: Annotated[
        dict[str, List[str]],
        Field(description="Branches to keep per NanoAOD object. "
              "'event' refers to non-collection branches.")
    ]
    ignore_missing: Annotated[
        bool,
        Field(
            default=False,
            description="If True, missing branches are ignored."
        )
    ]
    mc_branches: Annotated[
        dict[str, List[str]],
        Field(description="Branches to keep for MC only. "
              "'event' refers to non-collection branches.")
    ]

    @model_validator(mode="after")
    def validate_branches(self) -> "PreprocessConfig":
        # check for duplicate objects in branches
        if len(list(self.branches.keys())) != len(set(self.branches.keys())):
            raise ValueError(f"Duplicate objects found in branch list.")
        # check for duplicate objects in mc_branches
        if len(list(self.mc_branches.keys())) != len(set(self.mc_branches.keys())):
            raise ValueError(f"Duplicate objects found in mc_branch list.")

        # Check for duplicate branches in the same object in branches
        for obj, obj_branches in self.branches.items():
            if len(obj_branches) != len(set(obj_branches)):
                raise ValueError(f"Duplicate branches found in '{obj}'.")

        # Check for duplicate branches in the same object in mc_branches
        for obj, obj_branches in self.mc_branches.items():
            if len(obj_branches) != len(set(obj_branches)):
                raise ValueError(f"Duplicate branches found in '{obj}'.")

        # check that MC branches are in branches
        for obj, obj_branches in self.mc_branches.items():
            if obj not in self.branches:
                raise ValueError(f"'{obj}' is not present in branches.")
            for br in obj_branches:
                if br not in self.branches[obj]:
                    raise ValueError(f"'{br}' is not present in branches for '{obj}'.")

        return self

# ------------------------
# Channel configuration
# ------------------------
class ChannelConfig(SubscriptableModel):
    name: Annotated[
        str,
        Field(description="Name of the analysis channel")
    ]
    observable_name: Annotated[
        str,
        Field(description="Name of the histogram observable")
    ]
    observable_binning: Annotated[
        Union[str, List[float]],
        Field(description="Either a 'low,high,nbins' string or a list of bin edges")
    ]
    observable_label: Annotated[
        Optional[str],
        Field(default="observable", description="LaTeX label for plots")
    ]
    observable_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable computing the observable")
    ]
    use: Annotated[
        Optional[List[ObjVar]],
        Field(default=None, description="(object, variable) pairs for the function")
    ]

    @model_validator(mode="after")
    def validate_observable_fields(self) -> "ChannelConfig":
        if self.observable_function:
            if not self.use:
                raise ValueError(
                    "If 'observable_function' is provided, 'use' must also be specified."
                )
            if self.observable_name or self.observable_binning:
                raise ValueError(
                    "Cannot use 'observable_function' together with 'observable_name' or 'observable_binning'."
                )

        if isinstance(self.observable_binning, str):
            if not re.match(r"^\s*[\d.]+\s*,\s*[\d.]+\s*,\s*\d+\s*$", self.observable_binning):
                raise ValueError(
                    f"Invalid binning string: {self.observable_binning}. Expected format: 'low,high,nbins'"
                )

        if isinstance(self.observable_binning, list):
            if len(self.observable_binning) < 2:
                raise ValueError("At least two bin edges required.")
            if any(b <= a for a, b in zip(self.observable_binning, self.observable_binning[1:])):
                raise ValueError("Binning edges must be strictly increasing.")

        return self

# ------------------------
# Corrections configuration
# ------------------------
# should this just be restricted to correctionlib
class CorrectionConfig(SubscriptableModel):
    name: Annotated[
        str,
        Field(description="Name of the correction")
    ]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether correction is event/object-level")
    ]
    use: Annotated[
        Optional[Union[ObjVar,List[ObjVar]]],
        Field(default=[],description="(object, variable) inputs")
    ]
    op: Annotated[
        Optional[Literal["mult", "add", "subtract"]],
        Field(default="mult", description="How (operationa) to apply correction to targets")
    ]
    key: Annotated[
        Optional[str],
        Field(default=None, description="Correctionlib key (optional)")
    ]
    use_correctionlib: Annotated[
        bool,
        Field(default=True, description="True if using correctionlib to apply correction")
    ]
    file: Annotated[
        str,
        Field(description="Path to correction file")
    ]
    transform: Annotated[
        Optional[Callable],
        Field(default=None, description="Optional function to apply transformation to inputs before applying correction")
    ]
    up_and_down_idx: Annotated[
        Optional[List[str]],
        Field(default=["up","down"], description="Systematic variation keys (optional)")
    ]
    target: Annotated[
        Optional[Union[ObjVar,List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify")
    ]

    @model_validator(mode="after")
    def validate_corrections_fields(self) -> "CorrectionConfig":
        if self.use_correctionlib:
            if not self.file:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'file' must also be specified."
                )
            if not self.key:
                raise ValueError(
                    "If 'use_correctionlib' is True, 'key' must also be specified."
                )
        if self.type == "object":
            if not self.target:
                raise ValueError(
                    "If correction 'type' is 'object', 'target' must be specified."
                )
        return self

# ------------------------
# Systematics configuration
# ------------------------
class SystematicConfig(SubscriptableModel):
    name: Annotated[
        str,
        Field(description="Name of the systematic variation")
    ]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether variation is event/object-level")
    ]
    up_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'up' variation")
    ]
    down_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'down' variation")
    ]
    target: Annotated[
        Optional[Union[ObjVar,List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify")
    ]
    use: Annotated[
        Optional[Union[ObjVar,List[ObjVar]]],
        Field(default=[], description="Inputs to the variation function")
    ]
    symmetrise: Annotated[
        bool,
        Field(default=False, description="Whether to symmetrise variation")
    ]
    op: Annotated[
        Literal["mult", "add", "subtract"],
        Field(description="How (operationa) to apply systematic variation function to targets")
    ]

    @model_validator(mode="after")
    def validate_functions_and_consistency(self) -> "SystematicConfig":
        if not self.up_function and not self.down_function:
            raise ValueError(
                f"Systematic '{self.name}' must define at least one of 'up_function' or 'down_function'."
            )

        if self.type == "object":
            if not self.target:
                raise ValueError(
                    "If correction 'type' is 'object', 'target' must be specified."
                )

        return self

# ------------------------
# Top-level configuration
# ------------------------
class Config(SubscriptableModel):
    general: Annotated[
        GeneralConfig,
        Field(description="Global settings for the analysis")
    ]
    channels: Annotated[
        List[ChannelConfig],
        Field(description="List of analysis channels")
    ]
    corrections: Annotated[
        List[CorrectionConfig],
        Field(description="Corrections to apply to data")
    ]
    systematics: Annotated[
        List[SystematicConfig],
        Field(description="Systematic variations to apply")
    ]
    preprocess: Annotated[
        Optional[PreprocessConfig],
        Field(default=None, description="Preprocessing settings")
    ]

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # Check for duplicate channel names
        channel_names = [channel.name for channel in self.channels]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("Duplicate channel names found in configuration.")

        # Check for duplicate correction names
        correction_names = [correction.name for correction in self.corrections]
        if len(correction_names) != len(set(correction_names)):
            raise ValueError("Duplicate correction names found in configuration.")

        # Check for duplicate systematic names
        systematic_names = [systematic.name for systematic in self.systematics]
        if len(systematic_names) != len(set(systematic_names)):
            raise ValueError("Duplicate systematic names found in configuration.")

        if self.general.run_preprocessing and not self.preprocess:
            raise ValueError("Preprocessing is enabled but no preprocess configuration provided.")
        return self