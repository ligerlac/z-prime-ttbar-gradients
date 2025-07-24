from collections import defaultdict
from functools import reduce
import glob
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional
import warnings

import awkward as ak
import cabinetry
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import hist
import numpy as np
import uproot
import vector


from analysis.base import Analysis
from utils.cuts import lumi_mask
from utils.output_files import save_histograms, pkl_histograms, unpkl_histograms
from utils.preproc import pre_process_dak, pre_process_uproot
from utils.stats import get_cabinetry_rebinning_router

# -----------------------------
# Register backends
# -----------------------------
ak.jax.register_and_check()
vector.register_awkward()

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(name)s] %(message)s")
logger = logging.getLogger("NonDiffAnalysis")
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

# -----------------------------
# ZprimeAnalysis Class Definition
# -----------------------------
class NonDiffAnalysis(Analysis):

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize ZprimeAnalysis with configuration for systematics, corrections,
        and channels.

        Parameters
        ----------
        config : dict
            Configuration dictionary with 'systematics', 'corrections', 'channels',
            and 'general'.
        """
        super().__init__(config)
        self.nD_hists_per_region = self._init_histograms()

    def _prepare_dirs(self):
            # 1) create top-level output
            super()._prepare_dirs()

            out = self.dirs["output"]

            # 2) histograms lives under <output>/histograms
            (out / "histograms").mkdir(parents=True, exist_ok=True)

            # 3) cabinetry workspaces & stats
            (out / "statistics").mkdir(parents=True, exist_ok=True)

            self.dirs.update({
                "histograms":  out / "histograms",
                "statistics":  out / "statistics",
            })

    def _init_histograms(self) -> dict[str, dict[str, hist.Hist]]:
        """
        Initialize histograms for each analysis channel based on configuration.

        Returns
        -------
        dict
            Dictionary of channel name to hist.Hist object.
        """
        histograms = defaultdict(dict)
        for channel in self.channels:
            chname = channel.name
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue

            for observable in channel["observables"]:

                observable_label = observable["label"]
                observable_binning = observable["binning"]
                observable_name = observable["name"]

                if isinstance(observable_binning, str):
                    low, high, nbins = map(
                        float, observable_binning.split(",")
                    )
                    axis = hist.axis.Regular(
                        int(nbins),
                        low,
                        high,
                        name="observable",
                        label=observable_label,
                    )
                else:
                    axis = hist.axis.Variable(
                        observable_binning,
                        name="observable",
                        label=observable_label,
                    )

                histograms[chname][observable_name] = hist.Hist(
                    axis,
                    hist.axis.StrCategory([], name="process", growth=True),
                    hist.axis.StrCategory([], name="variation", growth=True),
                    storage=hist.storage.Weight(),
                )
        return histograms


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
        if process == "data" and variation != "nominal":
            return

        for channel in self.channels:
            chname = channel["name"]
            if (req_channels := self.config.general.channels) is not None:
                if chname not in req_channels:
                    continue
            logger.info(
                f"Applying selection for {chname} in {process}")
            mask = 1
            if (
                selection_funciton := channel.selection.function
            ) is not None:
                selection_args = self._get_function_arguments(
                    channel.selection.use, object_copies
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

            if event_syst and process != "data":
                weights = self.apply_event_weight_correction(
                    weights, event_syst, direction, object_copies_channel
                )

            logger.info(f"Number of weighted events in {chname}: {ak.sum(weights):.2f}")
            logger.info(f"Number of raw events in {chname}: {ak.sum(mask)}")
            for observable in channel["observables"]:
                logger.info(f"Computing observable {observable['name']}")
                observable_name = observable["name"]
                observable_args = self._get_function_arguments(
                    observable["use"], object_copies_channel
                )
                observable_vals = observable["function"](*observable_args)
                self.nD_hists_per_region[chname][observable_name].fill(
                    observable=observable_vals,
                    process=process,
                    variation=variation,
                    weight=weights,
                )


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
        # Filter objects
        obj_copies = self.apply_object_masks(obj_copies)

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
        self.histogramming(
            obj_copies_corrected,
            events,
            process,
            "nominal",
            xsec_weight,
            analysis,
        )

        if self.config.general.run_systematics:
            # Systematic variations
            for syst in self.systematics + self.corrections:
                if syst["name"] == "nominal":
                    continue
                for direction in ["up", "down"]:
                    # Filter objects
                    obj_copies = self.apply_object_masks(obj_copies)

                    # apply corrections
                    obj_copies_corrected = self.apply_object_corrections(
                        obj_copies, [syst], direction=direction
                    )
                    varname = f"{syst['name']}_{direction}"
                    self.histogramming(
                        obj_copies_corrected,
                        events,
                        process,
                        varname,
                        xsec_weight,
                        analysis,
                        event_syst=syst,
                        direction=direction,
                    )


    def run_fit(
        self, cabinetry_config: dict[str, Any]
    ) -> tuple[Any, Any, Any, Any]:
        """
        Run the fit using cabinetry.

        Parameters
        ----------
        cabinetry_config : dict
            Configuration for cabinetry.
        """

        # what do we do with this
        rebinning_router = get_cabinetry_rebinning_router(
            cabinetry_config, rebinning=slice(110j, None, hist.rebin(2))
        )
        # build the templates
        cabinetry.templates.build(cabinetry_config, router=rebinning_router)
        # optional post-processing (e.g. smoothing, symmetrise)
        cabinetry.templates.postprocess(cabinetry_config)
        # build the workspace
        ws = cabinetry.workspace.build(cabinetry_config)
        # save the workspace
        workspace_path = self.config.general.output_dir + "/statistics/"
        os.makedirs(workspace_path, exist_ok=True)
        workspace_path += "workspace.json"
        cabinetry.workspace.save(ws, workspace_path)
        # build the model and data
        model, data = cabinetry.model_utils.model_and_data(ws)
        # get pre-fit predictions
        prefit_prediction = cabinetry.model_utils.prediction(model)
        # perform the fit
        results = cabinetry.fit.fit(
            model,
            data,
        )  # perform the fit
        postfit_prediction = cabinetry.model_utils.prediction(
            model, fit_results=results
        )

        return data, results, prefit_prediction, postfit_prediction

    def run_analysis_chain(self, fileset):
        config = self.config
        for dataset, content in fileset.items():
            metadata = content["metadata"]
            metadata["dataset"] = dataset
            process_name = metadata["process"]
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
                if config.general.run_preprocessing:
                    logger.info(f"🔍 Preprocessing input file: {file_path}")
                    logger.info(f"➡️  Writing to: {output_dir}")
                    if config.general.preprocessor == "uproot":
                        pre_process_uproot(
                            file_path,
                            tree,
                            output_dir,
                            config,
                            is_mc=(dataset != "data"),
                        )
                    elif config.general.preprocessor == "dask":
                        pre_process_dak(
                            file_path,
                            tree,
                            output_dir + f"/part{idx}.root",
                            config,
                            is_mc=(dataset != "data"),
                        )

                skimmed_files = glob.glob(f"{output_dir}/part*.root")
                skimmed_files = [f"{f}:{tree}" for f in skimmed_files]
                remaining = sum(uproot.open(f).num_entries for f in skimmed_files)
                logger.info(f"✅ Events retained after filtering: {remaining:,}")
                if config.general.run_histogramming:
                    for skimmed in skimmed_files:
                        logger.info(f"📘 Processing skimmed file: {skimmed}")
                        logger.info("📈 Processing for non-differentiable analysis")
                        events = NanoEventsFactory.from_root(
                            skimmed, schemaclass=NanoAODSchema, delayed=False
                        ).events()
                        self.process(events, metadata)
                        logger.info("📈 Non-differentiable histogram-filling complete.")


            logger.info(f"🏁 Finished dataset: {dataset}\n")

        # Report end of processing
        logger.info("✅ All datasets processed.")

        # Save histograms for non-differentiable analysis
        if config.general.run_histogramming:
            save_histograms(
                self.nD_hists_per_region,
                output_file=f"{config.general.output_dir}/histograms/histograms.root",
            )
            pkl_histograms(
                self.nD_hists_per_region,
                output_file=f"{config.general.output_dir}/histograms/histograms.pkl",
            )
        # Run statistics for non-differentiable analysis
        if config.general.run_statistics:
            cabinetry_config = cabinetry.configuration.load(
                config.statistics.cabinetry_config
            )
            data, fit_results, pre_fit_predictions, postfit_predictions = (
                self.run_fit(cabinetry_config=cabinetry_config)
            )
            cabinetry.visualize.data_mc(
                pre_fit_predictions,
                data,
                close_figure=False,
                config=cabinetry_config,
            )
            cabinetry.visualize.data_mc(
                postfit_predictions,
                data,
                close_figure=False,
                config=cabinetry_config,
            )
            cabinetry.visualize.pulls(fit_results, close_figure=False)