from collections import defaultdict
from collections.abc import Mapping, Sequence
import glob
import logging
import os
from typing import Any, Literal, Optional
import warnings

import awkward as ak
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import jax
import jax.numpy as jnp
import numpy as np
import uproot
import vector

from analysis.base import Analysis
from utils.cuts import lumi_mask

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("DiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

def merge_histograms(existing, new):
    """
    Recursively merge `new` histogram dict into `existing`.
    Both must follow the structure:
    histograms[variation][region][observable] = jnp.ndarray
    """
    for variation, region_dict in new.items():
        for region, obs_dict in region_dict.items():
            for observable, array in obs_dict.items():
                existing[variation].setdefault(region, {})
                if observable in existing[variation][region]:
                    existing[variation][region][observable] += array
                else:
                    existing[variation][region][observable] = array
    return existing


def recursive_to_backend(data, backend: str = "jax"):
    """
    Recursively move data structures with Awkward
    arrays to the specified backend.

    Parameters
    ----------
    data : Any
        Object, list, or dict containing awkward Arrays
    backend : str
        Target backend ('jax', 'cpu', etc.)

    Returns
    -------
    Same structure as input, with awkward Arrays moved to backend.
    """
    if isinstance(data, ak.Array):
        if ak.backend(data) != backend:
            return ak.to_backend(data, backend)
        return data
    elif isinstance(data, Mapping):
        return {k: recursive_to_backend(v, backend) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return [recursive_to_backend(v, backend) for v in data]
    else:
        return data

# --------------------------------
# Differnetiable Analysis
# --------------------------------
class DifferentiableAnalysis(Analysis):
    def __init__(self, config):
        super().__init__(config)
        # 4 indices
        self.histograms = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict))
            )

    def set_histograms(
        self,
        histograms: dict[str, dict[str, dict[str,jnp.array]]]
    ) -> None:
        """
        Set the histograms for the analysis for all processes, variations, channels,
        and observables.

        Parameters
        ----------
        histograms : dict
            Dictionary of histograms with channel, observable, and variation.
        """
        self.histograms = histograms

    def histogramming(
        self,
        object_copies: dict[str, ak.Array],
        events: ak.Array,
        process: str,
        variation: str,
        xsec_weight: float,
        analysis: str,
        params,
        event_syst: Optional[dict[str, Any]] = None,
        direction: Literal["up", "down", "nominal"] = "nominal",
    ) -> Optional[ak.Array]:
        """
        Apply physics selections and fill histograms.

        Parameters
        ----------
        object_copies : dict
            Corrected event-level objects.
        events : ak.Array
            Original NanoAOD event collection.
        process : str
            Sample name.
        variation : str
            Systematic variation label.
        xsec_weight : float
            Normalization weight.
        analysis : str
            Analysis name string.
        event_syst : dict, optional
            Event-level systematic to apply.
        direction : str, optional
            Systematic direction: 'up', 'down', or 'nominal'.

        Returns
        -------
        dict
            Updated histogram dictionary.
        """
        jax_config = self.config.jax
        histograms = defaultdict(dict)
        if process == "data" and variation != "nominal":
            return histograms

        events = recursive_to_backend(events, "jax")
        object_copies = recursive_to_backend(object_copies, "jax")

        object_copies ={
            collection: ak.Array(var.layout)
            for collection, var in object_copies.items()
        }

        diff_selection_args = self._get_function_arguments(
                    jax_config.soft_selection.use, object_copies
                )

        diff_selection_weights = (
            jax_config.soft_selection.function(*diff_selection_args,
                                                params)
        )

        for channel in self.channels:
            # skip channels in non-differentiable analysis
            if not channel.use_in_diff:
                logger.warning(
                    f"Skipping channel {channel.name} in differentiable analysis"
                )
                continue

            chname = channel["name"]
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue
            logger.info(
                f"Applying selection for {chname} in {process}")
            mask = 1
            if (
                selection_funciton := channel["selection_function"]
            ) is not None:
                selection_args = self._get_function_arguments(
                    channel["selection_use"], object_copies
                )
                packed_selection = selection_funciton(*selection_args)
                if not isinstance(packed_selection, PackedSelection):
                    raise ValueError(
                        f"PackedSelection expected, got {type(packed_selection)}"
                    )
                mask = ak.Array(
                    packed_selection.all(packed_selection.names[-1])
                )

            mask = recursive_to_backend(mask, "jax")
            if process == "data":
                good_runs = lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                    jax=True
                )
                mask = mask & ak.to_backend(good_runs, "jax")

            if ak.sum(mask) == 0:
                logger.warning(
                    f"{analysis}:: No events left in {chname} for {process} with "
                    + "variation {variation}"
                )
                continue
            object_copies_channel = {
                                collection: variable[mask]
                                for collection, variable in object_copies.items()
                            }

            if process != "data":
                weights = (
                    events[mask].genWeight
                    * xsec_weight
                    / abs(events[mask].genWeight)
                )
            else:
                weights = ak.Array(np.ones(ak.sum(mask)))


            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            weights = jnp.array(weights.to_numpy())
            diff_selection_weights = diff_selection_weights[ak.to_jax(mask)]

            logger.info(f"Number of weighted events in {chname}: {ak.sum(weights):.2f}")
            logger.info(f"Number of raw events in {chname}: {ak.sum(mask)}")

            for observable in channel["observables"]:
                # check if observable works with JAX
                if not observable.works_with_jax:
                    logger.warning(
                        f"Observable {observable['name']} does not work with JAX, skipping."
                    )
                    continue

                logger.info(f"Computing observable {observable['name']}")
                observable_name = observable["name"]

                observable_args = self._get_function_arguments(
                    observable["use"], object_copies_channel
                )
                observable_vals = observable["function"](*observable_args)
                observable_binning = observable["binning"]

                # WIP:: need to enforce presence of this in config
                bandwidth = jax_config.params['kde_bandwidth']
                if isinstance(observable_binning, str):
                    low, high, nbins = map(
                        float, observable_binning.split(",")
                    )
                    nbins = int(nbins)
                    observable_binning = jnp.linspace(low, high, nbins)
                else:
                    observable_binning = jnp.array(observable_binning)

                # KDE-style soft binning
                cdf = jax.scipy.stats.norm.cdf(
                    observable_binning.reshape(-1, 1),
                    loc=ak.to_jax(observable_vals).reshape(1, -1),
                    scale=bandwidth
                )
                # Weight each event's contribution by selection weight
                weighted_cdf = cdf * diff_selection_weights.reshape(1, -1) * weights.reshape(1, -1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)
                histograms[chname][observable_name] = histogram

        return histograms


    def process(
        self, events: ak.Array, metadata: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """
        Run the full analysis logic on a batch of events.

        Parameters
        ----------
        events : ak.Array
            Input NanoAOD events.
        metadata : dict
            Metadata with keys 'process', 'xsec', 'nevts', and 'dataset'.

        Returns
        -------
        dict
            Histogram dictionary after processing.
        """
        all_histograms = self.histograms.copy()
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # Nominal processing
        obj_copies = self.get_object_copies(events)

        # Apply object masks
        # object masks break in JAX due to jagged masking
        obj_copies = self.apply_object_masks(obj_copies)

        # Move all data to JAX backend
        events = recursive_to_backend(events, "jax")
        obj_copies = recursive_to_backend(obj_copies, "jax")

        # Apply baseline selection
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )

        packed_selection = self.config.baseline_selection["function"](
            *baseline_args
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        mask = recursive_to_backend(mask, "jax")
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }

        # Compute ghost observables and store them
        obj_copies = self.compute_ghost_observables(
            obj_copies,
        )

        # Apply object corrections
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )

        # Move objects to JAX backend
        obj_copies_corrected = recursive_to_backend(
            obj_copies_corrected, "jax"
        )

        # Produce histograms
        histograms = self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
            params,
        )
        all_histograms["nominal"] = histograms


        if self.config.general.run_systematics:
            # Systematic variations
            for syst in self.systematics + self.corrections:
                # Skip nominal variations
                if syst["name"] == "nominal":   continue

                # Loop over up/down variations
                for direction in ["up", "down"]:
                    # Get systematic name
                    varname = f"{syst['name']}_{direction}"

                    # Move back to CPU backend before object masking
                    # object masks break in JAX due to jagged masking
                    events = recursive_to_backend(events, "cpu")
                    obj_copies = recursive_to_backend(obj_copies, "cpu")

                    # Filter objects
                    obj_copies = self.apply_object_masks(obj_copies)

                    # Move to JAX backend
                    events = recursive_to_backend(events, "jax")
                    obj_copies = recursive_to_backend(obj_copies, "jax")

                    # Apply object corrections
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )

                    # Produce histograms
                    histograms = self.histogramming(
                        obj_copies_corrected,
                        events,
                        process,
                        varname,
                        xsec_weight,
                        analysis,
                        params,
                        event_syst=syst,
                        direction=direction,
                    )
                    all_histograms[varname] = histograms

        return all_histograms

    def _calculate_significance(self, ) -> jnp.ndarray:
        """
        Calculate signal significance using KDE-smoothed nominal histograms.

        Parameters
        ----------
        variation : str, optional
            Systematic variation key to use. Default is "nominal".

        Returns
        -------
        jnp.ndarray
            Signal significance (S / sqrt(B + δS² + δB²)).
        """
        params = self.config.jax.params
        signal_yield_total = 0.0
        background_yield_total = 0.0
        variation = "nominal" # nominal only test
        for channel in self.channels:
            if not getattr(channel, "use_in_diff", False):
                continue

            region = channel.name
            observable = getattr(channel, "fit_observable", None)
            if observable is None:
                logger.warning(f"[Significance] No fit_observable in {region}, skipping.")
                continue

            bin_vals_signal = None
            bin_vals_bkg = None

            for process, proc_hists in self.histograms.items():
                # Skip non-nominal or missing data
                if variation not in proc_hists:
                    continue
                if region not in proc_hists[variation]:
                    continue
                if observable not in proc_hists[variation][region]:
                    continue

                hist = proc_hists[variation][region][observable]

                # Lazy init based on first matching histogram
                if bin_vals_signal is None:
                    bin_vals_signal = jnp.zeros_like(hist)
                    bin_vals_bkg = jnp.zeros_like(hist)

                if process in {"signal", "zprime"}:
                    bin_vals_signal += hist
                elif process != "data":
                    bin_vals_bkg += hist

            if bin_vals_signal is None:
                logger.warning(f"[Significance] No histograms found for {region}/{observable}")
                continue

            signal_yield = jnp.sum(bin_vals_signal)
            background_yield = jnp.sum(bin_vals_bkg)

            signal_yield_total += signal_yield
            background_yield_total += background_yield

        # Systematic uncertainties
        signal_syst = params.get("signal_systematic", 0.05) * signal_yield_total
        background_syst = params.get("background_systematic", 0.1) * background_yield_total

        denom = jnp.sqrt(background_yield_total + signal_syst**2 + background_syst**2 + 1e-6)
        significance = signal_yield_total / denom

        return significance


    def run_analysis_chain(self, params, fileset):

        config = self.config
        process_histograms = defaultdict(dict)
        for dataset, content in fileset.items():

            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]
            if process_name not in process_histograms:
                process_histograms[process_name] = defaultdict(lambda: defaultdict(dict))

            if (req_processes := config.general.processes) is not None:
                if process_name not in req_processes:
                    continue

            os.makedirs(f"{config.general.output_dir}/{dataset}", exist_ok=True)

            logger.info("========================================")
            logger.info(f"🚀 Processing dataset: {dataset}")

            for idx, (file_path, tree) in enumerate(content["files"].items()):
                output_dir = (
                    f"output/{dataset}/file__{idx}/"
                    if not config.general.preprocessed_dir
                    else f"{config.general.preprocessed_dir}/{dataset}/file__{idx}/"
                )
                if (
                    config.general.max_files != -1
                    and idx >= config.general.max_files
                ):
                    continue

                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
                logger.info(f"✅ Events retained after filtering: {remaining:,}")

                for skimmed in skimmed_files:
                    logger.info(f"📘 Processing skimmed file: {skimmed}")
                    logger.info("📈 Processing histograms for differentiable analysis")
                    events = NanoEventsFactory.from_root(
                        skimmed, schemaclass=NanoAODSchema, delayed=False
                    ).events()
                    histograms = self.process(events, metadata, params)
                    logger.info("📈 Completed.")
                    process_histograms[process_name] = (
                        merge_histograms(process_histograms[process_name], dict(histograms))
                    )


            logger.info(f"🏁 Finished dataset: {dataset}\n")

        self.set_histograms(process_histograms)
        # compute signifcance
        significance = self._calculate_significance()

        # Report end of processing
        logger.info("✅ All datasets processed.")

        return significance

    def run_analysis_chain_with_gradients(self, fileset):
        """
       Run the analysis chain and extract gradients

        Parameters:
        -----------
        process_data_dict : dict
            Dictionary with process names as keys and JAX data as values
            e.g., {'signal': jax_data, 'ttbar': jax_data, 'wjets': jax_data}
        """


        # Compute significance
        significance = self.run_analysis_chain(self.config.jax.params, fileset)

        print("Running differentiable event loop for multiple processes...")
        print(f"Processes: {list(self.histograms.keys())}")

        # Create gradient function
        grad_fn = jax.grad(self.run_analysis_chain, argnums=0)
        gradients = grad_fn(self.config.jax.params, fileset)

        print(f"Signal significance: {significance}")
        print(f"Parameter gradients: {gradients}")

        return significance, gradients


    def optimize_analysis_cuts(self, fileset):
        """
        Example of how to optimize analysis cuts using multiple processes to maximize significance.
        """
        significance, gradients = self.run_analysis_chain_with_gradients(fileset)

        print(f"Initial significance: {significance:.4f}")
        print(f"Initial gradients: {gradients}")

        # The objective is the differentiable_event_loop itself
        def objective(params):
            return self.run_analysis_chain(params, fileset)

        print("\nRunning optimization to maximize significance...")

        # Simple gradient ascent with parameter constraints
        learning_rate = 0.01
        params = self.config.jax.params.copy()

        print(f"{'Step':>4} {'Significance':>12} {'MET Cut':>8} {'B-tag Cut':>10} {'Lep HT Cut':>11}")
        print("-" * 55)

        for i in range(25):
            significance = objective(params)
            grads = jax.grad(objective)(params)

            # Update parameters with constraints
            for key in params:
                if key.endswith('_threshold'):
                    # For cut thresholds, use smaller learning rate and constrain ranges
                    if key == 'met_threshold':
                        print(params[key] + learning_rate * grads[key])
                        params[key] = jnp.clip(
                            params[key] + learning_rate * grads[key],
                            20.0, 150.0
                        )
                    elif key == 'btag_threshold':
                        params[key] = jnp.clip(
                            params[key] + learning_rate * grads[key],
                            0.1, 0.9
                        )
                    elif key == 'lep_ht_threshold':
                        params[key] = jnp.clip(
                            params[key] + learning_rate * grads[key],
                            50.0, 300.0
                        )
                elif key.endswith('_weight'):
                    # For weights, constrain to positive values
                    params[key] = jnp.maximum(
                        params[key] + learning_rate * grads[key],
                        0.01
                    )
                elif key.endswith('_scale'):
                    # Process scales should stay positive and reasonable
                    params[key] = jnp.clip(
                        params[key] + learning_rate * grads[key],
                        0.1, 10.0
                    )
                else:
                    # Other parameters
                    params[key] = params[key] + learning_rate * grads[key]

            #if (i + 1) % 5 == 0 or i == 0:
            print(f"{i+1:4d} {significance:12.4f} {params['met_threshold']:8.1f} "
                f"{params['btag_threshold']:10.3f} {params['lep_ht_threshold']:11.1f}")

        final_significance = objective(params)
        print(f"\nOptimization complete!")
        print(f"Initial significance: {significance:.4f}")
        print(f"Final significance: {final_significance:.4f}")
        print(f"Improvement: {((final_significance/significance - 1) * 100):.1f}%")

        print(f"\nOptimized parameters:")
        for key, value in params.items():
            if isinstance(value, (int, float)) or hasattr(value, 'item'):
                print(f"  {key}: {float(value):.4f}")

        # # Show process contributions at optimal cuts
        # print(f"\nProcess contributions at optimal cuts:")
        # histograms = {}
        # for process_name, jax_data in process_data_dict.items():
        #     if len(jax_data['met_pt']) == 0:
        #         continue
        #     selection_weight, _ = analysis.diff_selections.soft_selection_cuts(params, jax_data)
        #     total_weight = jnp.sum(selection_weight)
        #     print(f"  {process_name}: {float(total_weight):.1f} events")

        return params, final_significance