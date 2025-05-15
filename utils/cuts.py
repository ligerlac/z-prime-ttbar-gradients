import awkward as ak
from coffea.analysis_tools import PackedSelection
import numpy as np

ak.jax.register_and_check()


# https://github.com/cms-opendata-workshop/workshop2024-lesson-event-selection/blob/main/instructors/dpoa_workshop_utilities.py
def lumi_mask(lumifile, run, lumiBlock, verbose=False):

    # lumifile should be the name/path of the file
    good_luminosity_sections = ak.from_json(open(lumifile, "rb"))
    # Pull out the good runs as integers
    good_runs = np.array(good_luminosity_sections.fields).astype(int)

    # Get the good blocks as an awkward array
    # First loop over to get them as a list
    all_good_blocks = []
    for field in good_luminosity_sections.fields:
        all_good_blocks.append(good_luminosity_sections[field])

    # Turn the list into an awkward array
    all_good_blocks = ak.to_backend(ak.Array(all_good_blocks), ak.backend(run))

    # ChatGPT helped me with this part!
    # Find index of values in arr2 if those values appear in arr1
    def find_indices(arr1, arr2):
        arr1_np = np.asarray(ak.to_numpy(arr1))
        arr2_np = np.asarray(ak.to_numpy(arr2))

        # Sort arr1 and track original indices
        sorter = np.argsort(arr1_np)
        sorted_arr1 = arr1_np[sorter]

        # Search positions
        pos = np.searchsorted(sorted_arr1, arr2_np)

        # Check if arr2 values actually exist in arr1
        valid = (pos < len(arr1_np)) & (sorted_arr1[pos] == arr2_np)

        # Prepare result
        out = np.full(len(arr2_np), -1, dtype=int)
        out[valid] = sorter[pos[valid]]
        return ak.to_backend(ak.Array(out), ak.backend(arr2))

    # Get the indices that say where the good runs are in the lumi file
    # for the runs that appear in the tree
    good_runs_indices = find_indices(good_runs, run)

    # For each event, calculate the difference between the luminosity block
    # and the good luminosity blocks for that run for that event
    diff = lumiBlock - all_good_blocks[good_runs_indices]

    # If the lumi block appears between any of those good block numbers,
    # then one difference will be positive and the other will be negative
    #
    # If it it outside of the range, both differences will be positive or
    # both negative.
    #
    # The product will be negagive if the lumi block is in the range
    # and positive if it is not in the range
    prod_diff = ak.prod(diff, axis=2)
    mask = ak.any(prod_diff <= 0, axis=1)
    return mask


def Zprime_workshop_selection(muons, jets, fatjets, met, reco):
    """
    Select events based on the Zprime workshop selection criteria.
    """
    lep_ht = muons.pt + met.pt
    selections = PackedSelection(dtype="uint64")
    selections.add("exactly_1mu", ak.num(muons) == 1)
    selections.add("atleast_1b", ak.sum(jets.btagDeepB > 0.5, axis=1) > 0)
    selections.add("met_cut", met.pt > 50)
    selections.add("lep_ht_cut", ak.firsts(lep_ht) > 150)
    selections.add("exactly_1fatjet", ak.num(fatjets) == 1)
    selections.add(
        "Zprime_channel",
        selections.all(
            "exactly_1mu",
            "atleast_1b",
            "met_cut",
            "lep_ht_cut",
            "exactly_1fatjet",
        ),
    )

    return selections


def Zprime_baseline(muons, jets, fatjets, met):
    """
    Select events based on the Zprime workshop selection criteria.
    """
    selections = PackedSelection(dtype="uint64")
    selections.add("exactly_1mu", ak.num(muons) == 1)
    selections.add("atleast_2jets", ak.num(jets, axis=1) > 1)
    selections.add(
        "baseline",
        selections.all(
            "exactly_1mu",
            "atleast_2jets",
        ),
    )

    return selections


def Zprime_hardcuts(muons, jets, fatjets, met, reco):
    """
    Select events based on the Zprime workshop selection criteria.
    """
    selections = PackedSelection(dtype="uint64")
    selections.add("exactly_1mu", ak.num(muons) == 1)
    selections.add("exactly_1fatjet", ak.num(fatjets) == 1)
    selections.add(
        "Zprime_channel",
        selections.all(
            "exactly_1fatjet",
        ),
    )

    return selections


def Zprime_softcuts_nonjax(muons, jets, fatjets, met):
    """
    Select events based on the Zprime workshop selection criteria.
    """
    lep_ht = muons.pt + met.pt
    soft_cuts = {
        "atleast_1b": ak.sum(jets.btagDeepB > 0.5, axis=1) > 0,
        "met_cut": met.pt > 50,
        "lep_ht_cut": ak.fill_none(ak.firsts(lep_ht) > 150, False),
    }

    return soft_cuts


# def  Zprime_softcuts_jax(muons, jets, fatjets, met, met_cut=50.0):
#     """
#     Select events based on the Zprime workshop selection criteria.
#     """
#     import jax
#     lep_ht = muons.pt + met.pt
#     soft_cuts = {
#         "atleast_1b": ak.sum(jets.btagDeepB > 0.5, axis=1)
#         > 0,
#         # "met_cut": met.pt > 50,
#         # "met_cut": 0.5*jnp.tanh((ak.to_jax(region_met.pt)-50)/100)+0.5,
#         "met_cut": jax.nn.sigmoid(
#             (ak.to_jax(met.pt) - met_cut) / met_cut
#         ),
#         "lep_ht_cut": ak.fill_none(
#             ak.firsts(lep_ht) > 150, False
#         ),
#     }

#     return soft_cuts
