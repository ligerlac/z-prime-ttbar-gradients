"""Dataset queries for the CMS Open Data Z' → tt̄ example.

Each callable returns a ``Mapping[ProcessInfo, str]`` where the values
are DAS-format dataset patterns. Multiple callables are merged via
``UniqueProcessInfoDict`` in ``graep.inputs.build_fileset``.

Notes
-----
- Process *names* are kept identical to the legacy ``datasets/nanoaods.json``
  (``signal``, ``ttbar_semilep``, ``ttbar_had``, ``ttbar_lep``, ``wjets``,
  ``data``) so the resulting fileset keys match bit-for-bit.
- Cross-sections come from the AN's reference table (also reproduced in
  iris-hep's ``cms/example_cms/datasets/queries.py``).
- The legacy signal sample (``ZprimeToTT_M400_W4``, Γ/m=1 %) is
  **not** published on the CMS Open Data Portal. We use a sibling from
  the published Γ/m=10 % grid (currently ``ZPrimeToTT_M2000_W200``).
  iris-hep made the same substitution for the same reason.
- Production-tag wildcards (``v17-v*``) let the resolver pick whichever
  version is currently published without us having to track v1/v2 churn.
"""

from __future__ import annotations

from collections.abc import Mapping

from graep.config.inputs.open_data import OpenDataPortalFilesetSpec
from graep.inputs import ProcessInfo

# ---------------------------------------------------------------------------
# DAS-pattern templates
# ---------------------------------------------------------------------------

# CMS NanoAODv9 simulation, Run 2 UL16, both v1 and v2 production tags.
_NANOAODSIM_UL16 = (
    "/{0}/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v*/NANOAODSIM"
).format

# CMS NanoAODv9 collision data templates per era.
_NANOAOD_DATA_ERA = (
    "/{stream}/Run2016{era}-UL2016_MiniAODv2_NanoAODv9-v*/NANOAOD"
).format


# ---------------------------------------------------------------------------
# Query callables
# ---------------------------------------------------------------------------


def signal_queries() -> Mapping[ProcessInfo, str]:
    """Z' → tt̄ signal sample.

    Selected from the published Γ/m=10 % grid: M=2000 GeV, W=200 GeV.
    The legacy pipeline targeted M400_W4 (Γ/m=1 %) which is not on the
    CMS Open Data Portal; the published grid uses 10 % widths only.
    """
    return {
        ProcessInfo("signal", xsec=0.01895): _NANOAODSIM_UL16(
            "ZPrimeToTT_M2000_W200_TuneCP2_13TeV-madgraph-pythia8"
        ),
    }


def mc_background_queries() -> Mapping[ProcessInfo, str]:
    """tt̄ (split by decay channel) and W+jets MC."""
    return {
        ProcessInfo("ttbar_semilep", xsec=364.31): _NANOAODSIM_UL16(
            "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8"
        ),
        ProcessInfo("ttbar_had", xsec=380.11): _NANOAODSIM_UL16(
            "TTToHadronic_TuneCP5_13TeV-powheg-pythia8"
        ),
        ProcessInfo("ttbar_lep", xsec=87.33): _NANOAODSIM_UL16(
            "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8"
        ),
        ProcessInfo("wjets", xsec=61526.7): _NANOAODSIM_UL16(
            "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8"
        ),
    }


def data_queries() -> Mapping[ProcessInfo, list[str]]:
    """SingleMuon collision data, 2016G + 2016H eras.

    Two DAS patterns under one ``ProcessInfo``: the resolver unions
    file URIs and sums event counts so the resulting ``data__nominal``
    entry covers both eras (matching what the legacy ``nanoaods.json``
    already did manually).
    """
    return {
        ProcessInfo("data", xsec=1.0, is_data=True): [
            _NANOAOD_DATA_ERA(stream="SingleMuon", era="G"),
            _NANOAOD_DATA_ERA(stream="SingleMuon", era="H"),
        ],
    }


# ---------------------------------------------------------------------------
# Spec assembly
# ---------------------------------------------------------------------------

fileset_spec = OpenDataPortalFilesetSpec(
    queries=[signal_queries, mc_background_queries, data_queries],
)
