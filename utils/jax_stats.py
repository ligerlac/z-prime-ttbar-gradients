from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx
import relaxed
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# 1) A JAX‐compatible Poisson log‐PDF
# ----------------------------------------------------------------------
@jax.jit
def poisson_logpdf(counts: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    """
    Per‐bin Poisson log‐likelihood:
       log P(n | λ) = n·log(λ) − λ − log(n!).
    """
    return counts * jnp.log(lam + 1e-12) - lam - jsp.special.gammaln(counts + 1.0)


# ----------------------------------------------------------------------
# 2) ChannelData holds what we need for each channel:
#       - data_counts: observed data histogram (1D)
#       - processes: dict of {process_name: nominal_histogram} for that channel
# ----------------------------------------------------------------------
class ChannelData(eqx.Module):
    data_counts: jnp.ndarray
    processes: Dict[str, jnp.ndarray]

    def __init__(self, data_counts: jnp.ndarray, processes: Dict[str, jnp.ndarray]):
        # Both `data_counts` and each `processes[...]` are assumed to be jnp.ndarray already.
        self.data_counts = data_counts
        self.processes = processes


# ----------------------------------------------------------------------
# 3) Pure‐JAX “AllBkg” model with two scalar parameters:
#       - "mu": global signal strength
#       - "norm_ttbar_semilep": global ttbar_semilep normalization factor
# ----------------------------------------------------------------------
class AllBkgRelaxedModelScalar(eqx.Module):
    """
    A relaxed model that has exactly two free parameters:
      * "mu"                 — a scalar (signal strength)
      * "norm_ttbar_semilep" — a scalar (uniformly scales ttbar_semilep across all channels)
    All other backgrounds remain fixed.
    """

    channels: List[ChannelData]

    def __init__(self, channels: List[ChannelData]):
        self.channels = channels

    def expected_data(
        self, pars: Dict[str, jnp.ndarray]
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Returns (main_expectations, aux_expectations). Because we have no auxiliary Poisson
        constraints in this setup, aux_expectations is a list of zero‐length arrays.
        """
        mu = pars["mu"]                           # scalar
        norm_ttbar = pars["norm_ttbar_semilep"]   # scalar

        main_list: List[jnp.ndarray] = []
        aux_list: List[jnp.ndarray] = []  # no shape constraints here

        for idx, ch in enumerate(self.channels):
            # Start from zero, then add each process:
            total = jnp.zeros_like(ch.data_counts)

            for pname, nominal_hist in ch.processes.items():
                if pname == "signal":
                    # scale entire signal histogram by mu
                    total = total + mu * nominal_hist
                elif pname == "ttbar_semilep":
                    # scale entire ttbar_semilep histogram by a single factor
                    total = total + norm_ttbar * nominal_hist
                else:
                    # all other backgrounds are fixed at nominal
                    total = total + nominal_hist

            main_list.append(total)
            aux_list.append(jnp.zeros((0,)))  # empty, since no aux constraints

        return main_list, aux_list

    def logpdf(
        self,
        data: Tuple[List[jnp.ndarray], List[jnp.ndarray]],
        pars: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """
        data = (obs_main_list, obs_aux_list).  We ignore obs_aux_list (no constraints).
        Return total log‐probability over all channels.
        """
        obs_main_list, _ = data
        main_exp_list, _ = self.expected_data(pars)

        total_logp = 0.0
        for i in range(len(self.channels)):
            total_logp += jnp.sum(poisson_logpdf(obs_main_list[i], main_exp_list[i]))
        return total_logp


# ----------------------------------------------------------------------
# 4) Convert nested‐dict histograms → a list of ChannelData + observed data arrays
# ----------------------------------------------------------------------
def build_allbkg_channel_data_scalar(
    histograms: Dict[str, Dict[str, Dict[str, jnp.ndarray]]],
    channels: List[Any],
) -> Tuple[List[ChannelData], List[jnp.ndarray]]:
    """
    Inputs:
      - histograms:
          {
            "process_name": {
                "nominal": {
                    "channel_name": {
                        "observable_name": <jnp.ndarray>
                    }
                }
            }
          }
        (We assume only the "nominal" variation is used for building the model.)

      - channels: list of your channel‐config objects, each has:
          .name           (string, e.g. "mu+jets")
          .fit_observable (string, e.g. "m_tt")
          .use_in_diff    (bool)

    Returns:
      - channel_data_list: List[ChannelData]   (one per valid channel)
      - obs_main_list:      List[jnp.ndarray]  (observed data histogram per channel)
    """
    channel_data_list: List[ChannelData] = []
    obs_main_list: List[jnp.ndarray] = []

    for ch in channels:
        if not ch.use_in_diff or not hasattr(ch, "fit_observable"):
            continue

        chname = ch.name
        obsname = ch.fit_observable

        # 1) Observed data histogram for this channel
        main_obs = histograms.get("data", {}) \
                              .get("nominal", {}) \
                              .get(chname, {}) \
                              .get(obsname, None)
        if main_obs is None:
            continue  # skip channels with no data

        # 2) Collect **all** nominal templates for this channel
        proc_dict: Dict[str, jnp.ndarray] = {}
        for pname, variations in histograms.items():
            nom = variations.get("nominal", {}) \
                            .get(chname, {}) \
                            .get(obsname, None)
            if nom is not None:
                proc_dict[pname] = nom

        # If “signal” is missing, treat it as zeros:
        if "signal" not in proc_dict:
            proc_dict["signal"] = jnp.zeros_like(main_obs)

        # If “ttbar_semilep” is missing, treat it as zeros:
        if "ttbar_semilep" not in proc_dict:
            proc_dict["ttbar_semilep"] = jnp.zeros_like(main_obs)

        # Now create ChannelData
        channel_data_list.append(ChannelData(
            data_counts = main_obs,
            processes  = proc_dict
        ))
        obs_main_list.append(main_obs)

    return channel_data_list, obs_main_list


# ----------------------------------------------------------------------
# 5) A pure‐JAX calculate_significance_relaxed that uses scalars for ttbar_norm
# ----------------------------------------------------------------------
def calculate_significance_relaxed(
    histograms: Dict[str, Dict[str, Dict[str, jnp.ndarray]]],
    channels: List[Any],
    params: Dict[str, float] = {},
    test_mu: float = 0.0,
) -> jnp.ndarray:
    """
    1) Build ChannelData for each channel + observed data list.
    2) Construct AllBkgRelaxedModelScalar.
    3) Build init_pars = {"mu": test_mu, "norm_ttbar_semilep": 1.0}.
    4) Call relaxed.infer.hypotest purely in JAX.
    5) Return the JAX array p0.
    """
    # (a) Build channel data and obs_main_list:
    channel_data_list, obs_main_list = build_allbkg_channel_data_scalar(histograms, channels)

    if len(channel_data_list) == 0:
        # no valid channels → zero significance
        return jnp.array(0.0)

    # (b) Prepare “data_for_hypotest”: (obs_main_list, obs_aux_list)
    #     We have no auxiliary constraints → pass empty list for second
    data_for_hypotest = (obs_main_list, [])

    # === DEBUG: Print a small summary of the observed data ===
    # We’ll print the shape and sum of each “obs_main” array, joined with commas.
    # This runs at trace‐time so the exact same JAXTracers appear whether we call
    # this function from “standalone” or from inside the optimizer.
    # for i, arr in enumerate(obs_main_list):
    #     jax.debug.print(
    #         "[DEBUG] Channel {i}: obs_main_list[{i}].shape = {shape}, sum = {s:.4f}",
    #         shape=arr.shape, s=jnp.sum(arr), i=i
    #     )
    # for i, arr in enumerate(channel_data_list):
    #     for process in arr.processes:
    #         jax.debug.print(
    #             "[DEBUG] Channel {i}: process '{p}': shape = {shape}, sum = {s:.4f}",
    #             p=process, shape=arr.processes[process].shape, s=jnp.sum(arr.processes[process]), i=i
    #         )

    # (c) Build our model
    model = AllBkgRelaxedModelScalar(channels=channel_data_list)

    # (d) Create initial parameters dictionary:
    #     mu = params["mu"], norm_ttbar_semilep = params["norm_ttbar_semilep"]
    # init_pars: Dict[str, jnp.ndarray] = {
    #     par: jnp.array(value) for par, value in params.items()
    # }

    # === DEBUG: Print initial parameters summary ===
    # (We assume “mu” and “norm_ttbar_semilep” are present in params.)
    # jax.debug.print(
    #     "[DEBUG] init_pars: mu = {m:.4f}, norm_ttbar_semilep = {n:.4f}",
    #     m=params["mu"], n=params["norm_ttbar_semilep"]
    # )

    # (e) Call relaxed.infer.hypotest entirely in JAX:
    vals = relaxed.infer.hypotest(
        test_mu,
        data_for_hypotest,
        model,
        init_pars = params,
        return_mle_pars = True,
        # bounds = {
        #     "mu": (0.0, jnp.inf),  # mu >= 0
        #     "norm_ttbar_semilep": (0.0, jnp.inf),  # ttbar normalization >= 0
        # },
        test_stat = "q0",   # discovery test
    )

    if isinstance(vals, tuple):
        p0, mle_pars = vals
        jax.debug.print("MLE parameters: {mle_pars}", mle_pars=mle_pars)
    else:
        p0 = vals

    return p0


# ----------------------------------------------------------------------
# 6) Integrate into your DifferentiableAnalysis
# ----------------------------------------------------------------------

# In your DifferentiableAnalysis._calculate_significance, replace the old call:
#
#   old:
#     p0 = calculate_significance_relaxed(histograms, self.channels)
#
#   new:
#     p0 = calculate_significance_relaxed_allbkg_scalar(histograms, self.channels, test_mu=0.0)
#
# Ensure that `histograms[...]` has jnp.ndarray leaves (e.g. by calling jnp.asarray(...)
# when you first build them). Then in optimize or wherever you do:
#
#    pvals, grads = jax.value_and_grad(self.run_histogram_and_significance, argnums=0)(...)
#
# JAX will happily trace through the entire “L-BFGS inside relaxed” without any tracer leaks.