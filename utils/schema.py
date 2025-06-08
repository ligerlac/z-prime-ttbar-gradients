import copy
from typing import Annotated, Callable, List, Literal, Optional, Tuple, Union

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator


# Type alias for (object, variable) pairs
ObjVar = Tuple[str, Optional[str]]


class SubscriptableModel(BaseModel):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class FunctorConfig(SubscriptableModel):
    function: Annotated[
        Callable,
        Field(
            description="Function to be computed."
        ),
    ]
    use: Annotated[
        Optional[List[ObjVar]],
        Field(
            default=None,
            description="(object, variable) pairs for the function.",
        ),
    ]

class GoodObjectMasksConfig(SubscriptableModel):
    object: Annotated[
        str, Field(description="Name of the object to compute masks for")
    ]
    function: Annotated[
        Callable,
        Field(description="Function to compute good object masks. \
              Will be applied to the object in block key.")
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            description="(object, variable) pairs for the good object mask function.",
        ),
    ]

    @model_validator(mode="after")
    def validate_fields(self) -> "GoodObjectMasksConfig":
        # check that object is one of Muon, Jet or FatJet
        if self.object not in ["Muon", "Jet", "FatJet"]:
            raise ValueError(
                f"Invalid object '{self.object}'. Must be one of 'Muon', \
                    'Jet', or 'FatJet'."
            )

        return self


# ------------------------
# General configuration
# ------------------------
class GeneralConfig(SubscriptableModel):
    lumi: Annotated[float, Field(description="Integrated luminosity in /pb")]
    weights_branch: Annotated[
        str, Field(description="Branch name for event weight")
    ]
    lumifile: Annotated[
        str,
        Field(
            description="Path to JSON file with good luminosity sections",
        ),
    ]
    analysis: Annotated[
        Optional[str],
        Field(default="nondiff",
              description="Analysis MODE: 'diff', 'nondiff' or 'both'"),
    ]
    max_files: Annotated[
        Optional[int],
        Field(default=1, description="Maximum number of files to process"),
    ]
    run_preprocessing: Annotated[
        bool,
        Field(default=False, description="Whether to run preprocessing step"),
    ]
    run_histogramming: Annotated[
        bool,
        Field(default=True, description="Whether to run histogramming step"),
    ]
    run_statistics: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to run statistical analysis step",
        ),
    ]
    run_systematics: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to run systematic variations step",
        ),
    ]
    run_plots_only: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to skip all steps except plotting",
        ),
    ]
    read_from_cache: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to read preprocessed files from cache",
        ),
    ]

    output_dir: Annotated[
        Optional[str],
        Field(default="output/", description="Directory for output files"),
    ]
    preprocessor: Annotated[
        Literal["uproot", "dak"],
        Field(default="uproot", description="Preprocessor to use"),
    ]
    preprocessed_dir: Annotated[
        Optional[str],
        Field(
            default=None, description="Directory containing preprocessed files"
        ),
    ]
    processes: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="List of processes to include in the analysis",
        ),
    ]
    channels: Annotated[
        Optional[List[str]],
        Field(
            default=None,
            description="List of channels to include in the analysis",
        ),
    ]

    @model_validator(mode="after")
    def validate_general(self) -> "GeneralConfig":
        if self.analysis not in ["diff", "nondiff", "both"]:
            raise ValueError(
                f"Invalid analysis mode '{self.analysis}'. Must be 'diff' or 'nondiff'."
            )

        return self

# ------------------------
# JAX configuration
# ------------------------
class JaxConfig(SubscriptableModel):
    soft_selection: Annotated[
        FunctorConfig,
        Field(
            description="Soft selection function for JAX-mode observable shaping. "
            "Should return a dictionary of soft selections."
        ),
    ]
    params: Annotated[
        dict[str, float],
        Field(description="Thresholds, weights, and scaling factors used in JAX backend."),
    ]
    optimize: Annotated[
        bool,
        Field(
            default=True,
            description="Whether to run JAX optimisation for soft selection parameters",
        ),
    ]
    learning_rate: Annotated[
        float,
        Field(
            default=0.01,
            description="Learning rate for JAX optimisation",
        ),
    ]
    max_iterations: Annotated[
        Optional[int],
        Field(
            default=25,
            description="Number of optimisation steps for JAX",
        ),
    ]
    param_updates: Annotated[
            dict[str, Callable[[float, float], float]],
            Field(
                default_factory=dict,
                description=(
                    "Optional per-parameter update rules. Maps parameter name to a callable "
                    "that accepts (value, delta) and returns updated value. "
                    "Delta is defined as `learning_rate * gradient`. "
                    "Example: {'met_threshold': lambda x, d: jnp.clip(x + d, 20.0, 150.0)}"
                ),
            ),
        ]

    learning_rates: Annotated[
        Optional[dict[str, float]],
        Field(
            default=None,
            description="Optional per-parameter learning rates for JAX optimisation. "
            "If None, uses `learning_rate` for all parameters.",
        ),
    ]

    explicit_optimization: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to run explicit optimization loop for JAX parameters",
        ),
    ]
# ------------------------
# Preprocessing configuration
# ------------------------
class PreprocessConfig(SubscriptableModel):
    branches: Annotated[
        dict[str, List[str]],
        Field(
            description="Branches to keep per NanoAOD object. "
            "'event' refers to non-collection branches."
        ),
    ]
    ignore_missing: Annotated[
        bool,
        Field(
            default=False, description="If True, missing branches are ignored."
        ),
    ]
    mc_branches: Annotated[
        dict[str, List[str]],
        Field(
            description="Branches to keep for MC only. "
            "'event' refers to non-collection branches."
        ),
    ]

    @model_validator(mode="after")
    def validate_branches(self) -> "PreprocessConfig":
        # check for duplicate objects in branches
        if len(list(self.branches.keys())) != len(set(self.branches.keys())):
            raise ValueError("Duplicate objects found in branch list.")
        # check for duplicate objects in mc_branches
        if len(list(self.mc_branches.keys())) != len(
            set(self.mc_branches.keys())
        ):
            raise ValueError("Duplicate objects found in mc_branch list.")

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
                    raise ValueError(
                        f"'{br}' is not present in branches for '{obj}'."
                    )

        return self


# ------------------------
# Statistical analysis configuration
# ------------------------]
class StatisticalConfig(SubscriptableModel):
    cabinetry_config: Annotated[
        str,
        Field(
            description="Path to YAML file with cabinetry settings",
        ),
    ]


# ------------------------
# Observable configuration
# ------------------------
class ObservableConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the observable")]
    binning: Annotated[
        Union[str, List[float]],
        Field(
            description="Either a 'low,high,nbins' string or a list of bin edges"
        ),
    ]
    function: Annotated[
        Callable,
        Field(description="Callable computing the observable"),
    ]
    use: Annotated[
        List[ObjVar],
        Field(
            description="(object, variable) pairs for the function. "
            "If variable is None, object is passed.",
        ),
    ]
    label: Annotated[
        Optional[str],
        Field(default="observable", description="LaTeX label for plots"),
    ]
    works_with_jax: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the function works with JAX backend",
        ),
    ]

    @model_validator(mode="after")
    def validate_binning(self) -> "ObservableConfig":
        if isinstance(self.binning, str):
            self.binning = (
                self.binning.strip("[").strip("]").strip("(").strip(")")
            )
            binning = self.binning.split(",")
            if len(binning) != 3:
                raise ValueError(
                    f"Invalid binning string: {self.binning}. Need 3 values."
                    + "Expected format: 'low,high,nbins'"
                )
            else:
                try:
                    low, high, nbins = map(float, binning)
                    nbins = int(nbins)
                except ValueError:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Need 3 floats."
                        + "Expected format: 'low,high,nbins'"
                    )
                if low >= high:
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. Low must be < high."
                        + "Expected format: 'low,high,nbins'"
                    )
                if nbins <= 0 or not isinstance(nbins, int):
                    raise ValueError(
                        f"Invalid binning string: {self.binning}. nbins must be != 0."
                        + "Expected format: 'low,high,nbins'"
                    )

        elif isinstance(self.binning, list):
            if len(self.binning) < 2:
                raise ValueError("At least two bin edges required.")
            if any(b <= a for a, b in zip(self.binning, self.binning[1:])):
                raise ValueError("Binning edges must be strictly increasing.")
        return self


# ------------------------
# Ghost observable configuration
# ------------------------
class GhostObservable(SubscriptableModel):
    names: Union[str, List[str]]
    collections: Union[str, List[str]]
    function: Callable
    use: List[ObjVar]
    works_with_jax: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the function works with JAX backend",
        ),
    ]


# ------------------------
# Channel configuration
# ------------------------
class ChannelConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the analysis channel")]
    observables: Annotated[
        List[ObservableConfig],
        Field(
            description="List of observables for this channel (must be ≥ 1)"
        ),
    ]
    fit_observable: Annotated[
        str,
        Field(description="Name of the observable to use for fitting"),
    ]
    selection: Annotated[
        Optional[FunctorConfig],
        Field(
            default=None,
            description="Event selection function for this channel. "
            + "If None, all events are selected. Function must "
            + "return a PackedSelection object.",
        ),
    ]
    use_in_diff: Annotated[
        Optional[bool],
        Field(
            default=False,
            description="Whether to use this channel in differentiable analysis. "
            + "If None, defaults to True.",
        ),
    ]

    @model_validator(mode="after")
    def validate_fields(self) -> "ChannelConfig":
        if self.selection.function and not self.selection.use:
            raise ValueError(
                "If 'selection.function' is provided, 'selection.use' must also "
                + "be specified."
            )
        if not self.observables:
            raise ValueError("Each channel must have at least one observable.")

        obs_names = [obs.name for obs in self.observables]
        if self.fit_observable not in obs_names:
            raise ValueError(
                f"'fit_observable'='{self.fit_observable}' is not in the list of "
                + f"observables: {sorted(obs_names)}"
            )

        if len(set(obs_names)) != len(obs_names):
            raise ValueError(
                "Duplicate observable names found in the channel configuration."
            )

        return self


# ------------------------
# Corrections configuration
# ------------------------
# should this just be restricted to correctionlib
class CorrectionConfig(SubscriptableModel):
    name: Annotated[str, Field(description="Name of the correction")]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether correction is event/object-level"),
    ]
    use: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=[], description="(object, variable) inputs"),
    ]
    op: Annotated[
        Optional[Literal["mult", "add", "subtract"]],
        Field(
            default="mult",
            description="How (operationa) to apply correction to targets",
        ),
    ]
    key: Annotated[
        Optional[str],
        Field(default=None, description="Correctionlib key (optional)"),
    ]
    use_correctionlib: Annotated[
        bool,
        Field(
            default=True,
            description="True if using correctionlib to apply correction",
        ),
    ]
    file: Annotated[str, Field(description="Path to correction file")]
    transform: Annotated[
        Optional[Callable],
        Field(
            default=None,
            description="Optional function to apply transformation to inputs "
            + "before applying correction",
        ),
    ]
    up_and_down_idx: Annotated[
        Optional[List[str]],
        Field(
            default=["up", "down"],
            description="Systematic variation keys (optional)",
        ),
    ]
    target: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify"),
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
    name: Annotated[str, Field(description="Name of the systematic variation")]
    type: Annotated[
        Literal["event", "object"],
        Field(description="Whether variation is event/object-level"),
    ]
    up_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'up' variation"),
    ]
    down_function: Annotated[
        Optional[Callable],
        Field(default=None, description="Callable for 'down' variation"),
    ]
    target: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(default=None, description="Target (object, variable) to modify"),
    ]
    use: Annotated[
        Optional[Union[ObjVar, List[ObjVar]]],
        Field(
            default=[],
            description="(object, variable) inputs to variation functions. If variable "
            + "is None, object is passed.",
        ),
    ]
    symmetrise: Annotated[
        bool,
        Field(default=False, description="Whether to symmetrise variation"),
    ]
    op: Annotated[
        Literal["mult", "add", "subtract"],
        Field(
            description="How (operation) to apply systematic variation function "
            + "to targets"
        ),
    ]

    @model_validator(mode="after")
    def validate_functions_and_consistency(self) -> "SystematicConfig":
        if not self.up_function and not self.down_function:
            raise ValueError(
                f"Systematic '{self.name}' must define at least one of 'up_function' "
                + "or 'down_function'."
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
        GeneralConfig, Field(description="Global settings for the analysis")
    ]
    jax: Annotated[
        Optional[JaxConfig],
        Field(default=None, description="JAX configuration block for differentiable analysis"),
    ]
    ghost_observables: Annotated[
        Optional[List[GhostObservable]],
        Field(
            default=[],
            description="Variables to compute and store ahead of channel selection."
            "This variables will not be histogrammed unless specified as observable in "
            "a channel.",
        ),
    ]
    baseline_selection: Annotated[
        Optional[FunctorConfig],
        Field(
            default=None,
            description="Baseline event selection applied before "
            "channel-specific logic",
        ),
    ]
    good_object_masks: Annotated[
        Optional[List[GoodObjectMasksConfig]],
        Field(
            default=[],
            description="Good object masks to apply before channel selection. "
            "These are applied to the object in the 'object' field"
        ),
    ]
    channels: Annotated[
        List[ChannelConfig], Field(description="List of analysis channels")
    ]
    corrections: Annotated[
        List[CorrectionConfig],
        Field(description="Corrections to apply to data"),
    ]
    systematics: Annotated[
        List[SystematicConfig],
        Field(description="Systematic variations to apply"),
    ]
    preprocess: Annotated[
        Optional[PreprocessConfig],
        Field(default=None, description="Preprocessing settings"),
    ]
    statistics: Annotated[
        Optional[StatisticalConfig],
        Field(default=None, description="Statistical analysis settings"),
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
            raise ValueError(
                "Duplicate correction names found in configuration."
            )

        # Check for duplicate systematic names
        systematic_names = [systematic.name for systematic in self.systematics]
        if len(systematic_names) != len(set(systematic_names)):
            raise ValueError(
                "Duplicate systematic names found in configuration."
            )

        if self.general.run_preprocessing and not self.preprocess:
            raise ValueError(
                "Preprocessing is enabled but no preprocess configuration provided."
            )

        if self.statistics is not None:
            if (
                self.general.run_statistics
                and not self.statistics.cabinetry_config
            ):
                raise ValueError(
                    "Statistical analysis run enabled but no cabinetry configuration "
                    + "provided."
                )

        seen_ghost_obs = set()
        for obs in self.ghost_observables:
            names = obs.names if isinstance(obs.names, list) else [obs.names]
            colls = (
                obs.collections
                if isinstance(obs.collections, list)
                else [obs.collections] * len(names)
            )

            if len(names) != len(colls):
                raise ValueError(
                    f"In GhostObservable with function `{obs.function}`, "
                    f"number of names and collections must match if both are lists."
                )

            for name, coll in zip(names, colls):
                pair = (coll, name)
                if pair in seen_ghost_obs:
                    raise ValueError(
                        f"Duplicate (collection, name) pair: {pair}"
                    )
                seen_ghost_obs.add(pair)

        # check for duplicate object names in object masks
        seen_objects = set()
        for mask in self.good_object_masks:
            if mask.object in seen_objects:
                raise ValueError(
                    f"Duplicate object '{mask.object}' found in good object masks."
                )
            seen_objects.add(mask.object)

        return self


def load_config_with_restricted_cli(
    base_cfg: dict, cli_args: list[str]
) -> dict:
    """
    Load base config and override only `general`, `preprocess`, or `statistics`
    keys via CLI arguments in dotlist form.

    Parameters
    ----------
    base_cfg : dict
        The full Python config with logic, lambdas, etc.
    cli_args : list of str
        CLI args in OmegaConf dotlist format (e.g. general.lumi=25000)

    Returns
    -------
    dict
        Full merged config (with overrides applied to whitelisted sections only).
    """
    ALLOWED_CLI_TOPLEVEL_KEYS = {"general", "preprocess", "statistics"}

    # Deep copy so we don’t modify the original
    base_copy = copy.deepcopy(base_cfg)

    # Filter CLI args to allowed keys only
    filtered_cli = []
    for arg in cli_args:
        top_key = arg.split("=", 1)[0].split(".", 1)[0]
        if top_key in ALLOWED_CLI_TOPLEVEL_KEYS:
            filtered_cli.append(arg)
        else:
            raise ValueError(
                f"Override of top-level key `{top_key}` is not allowed. "
                f"Allowed keys: {', '.join(ALLOWED_CLI_TOPLEVEL_KEYS)}"
            )

    # Merge CLI with OmegaConf
    cli_cfg = OmegaConf.from_dotlist(filtered_cli)
    safe_base = OmegaConf.create(
        {k: v for k, v in base_copy.items() if k in ALLOWED_CLI_TOPLEVEL_KEYS}
    )
    merged_cfg = OmegaConf.merge(safe_base, cli_cfg)
    updated_subsections = OmegaConf.to_container(merged_cfg, resolve=True)

    # Patch back into full config
    for k in ALLOWED_CLI_TOPLEVEL_KEYS:
        if k in updated_subsections:
            base_copy[k] = updated_subsections[k]

    return base_copy
