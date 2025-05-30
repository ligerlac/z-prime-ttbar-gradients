from collections import defaultdict
from functools import reduce
import logging
import operator
import os
from typing import Any, Literal, Optional

import awkward as ak
import cabinetry
from coffea.analysis_tools import PackedSelection
import hist
import numpy as np
import vector

from analysis.base import Analysis


# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ZprimeAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

# --------------------------------
# Differnetiable Analysis
# --------------------------------
class DifferentiableAnalysis(Analysis):

    def __init__(self, config):
        super.__init__(config)
        # 4 indices
        self.histograms = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(dict))
            )

    def set_histograms(
        self,
        histograms: dict[str, dict[str, dict[str,jnp.Array]]]
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
        histograms = defaultdict(dict)
        if process == "data" and variation != "nominal":
            return histograms

        jax_config = self.config.jax

        diff_selection_args = self._get_function_arguments(
                    jax_config.soft_selection.use, object_copies
                )
        diff_selection_weights = (
            jax_config.soft_selection.function(*diff_selection_args,
                                                self.config.params)
        )

        for channel in self.channels:
            # skip channels in non-differentiable analysis
            if not channel.use_in_diff:
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

            if process == "data":
                mask = mask & lumi_mask(
                    self.config.general.lumifile,
                    object_copies["run"],
                    object_copies["luminosityBlock"],
                )

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
                weights = np.ones(ak.sum(mask))

            weights = jnp.array(weights.to_numpy())

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            logger.info(f"Number of weighted events in {chname}: {ak.sum(weights):.2f}")
            logger.info(f"Number of raw events in {chname}: {ak.sum(mask)}")

            for observable in channel["observables"]:
                # check if observable works with JAX
                if not observable.works_with_jax:
                    continue

                logger.info(f"Computing observable {observable['name']}")
                observable_name = observable["name"]
                observable_label = observable["label"]

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
                    loc=observable_vals.reshape(1, -1),
                    scale=bandwidth
                )

                # Weight each event's contribution by selection weight
                weighted_cdf = cdf * diff_selection_weights.reshape(1, -1) * weights.reshape(-1, 1)
                bin_weights = weighted_cdf[1:, :] - weighted_cdf[:-1, :]
                histogram = jnp.sum(bin_weights, axis=1)
                histograms[chname][observable_name] = histogram


    def process(
        self, events: ak.Array, metadata: dict[str, Any],
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
        all_histograms = copy.deepcopy(self.histograms)
        analysis = self.__class__.__name__

        process = metadata["process"]
        variation = metadata.get("variation", "nominal")
        logger.debug(f"Processing {process} with variation {variation}")
        xsec = metadata["xsec"]
        n_gen = metadata["nevts"]

        lumi = self.config["general"]["lumi"]
        xsec_weight = (xsec * lumi / n_gen) if process != "data" else 1.0

        # move everything to JAX backend
        ak.to_backend(events, "jax")
        # Nominal processing
        obj_copies = self.get_object_copies(events)
        # Filter objects
        # Get object masks from configuration:
        if (obj_masks := self.config.good_object_masks) != []:
            filtered_objs = self.get_good_objects(obj_copies, obj_masks)
            for obj, filtered in filtered_objs.items():
                if obj not in obj_copies:
                    raise KeyError(f"Object {obj} not found in object_copies")
                obj_copies[obj] = filtered

        # Apply baseline selection
        baseline_args = self._get_function_arguments(
            self.config.baseline_selection["use"], obj_copies
        )

        packed_selection = self.config.baseline_selection["function"](
            *baseline_args
        )
        mask = ak.Array(packed_selection.all(packed_selection.names[-1]))
        obj_copies = {
            collection: variable[mask]
            for collection, variable in obj_copies.items()
        }
        # apply ghost observables
        obj_copies = self.compute_ghost_observables(
            obj_copies,
        )

        # apply event-level corrections
        # apply nominal corrections
        obj_copies_corrected = self.apply_object_corrections(
            obj_copies, self.corrections, direction="nominal"
        )
        # apply selection and fill histograms
        obj_copies_corrected = {
            obj: var
            for obj, var in obj_copies_corrected.items()
        }
        histograms = self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
        )
        all_histograms["nominal"] = histograms

        # Systematic variations
        for syst in self.systematics + self.corrections:
            if syst["name"] == "nominal":
                continue
            for direction in ["up", "down"]:
                # Filter objects
                # Get object masks from configuration:
                if (obj_masks := self.config.good_object_masks) != []:
                    filtered_objs = self.get_good_objects(obj_copies, obj_masks)
                    for obj, filtered in filtered_objs.items():
                        if obj not in obj_copies:
                            raise KeyError(f"Object {obj} not found in object_copies")
                        obj_copies[obj] = filtered

                # apply corrections
                obj_copies_corrected = self.apply_object_corrections(
                    obj_copies, [syst], direction=direction
                )
                varname = f"{syst['name']}_{direction}"
                histograms = self.histogramming(
                    obj_copies_corrected,
                    events,
                    process,
                    varname,
                    xsec_weight,
                    analysis,
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