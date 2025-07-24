#!/usr/bin/env python

import os
import argparse
import shutil
import pickle
from pathlib import Path
import warnings
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import vector
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from sklearn.model_selection import train_test_split
import mplhep as hep
hep.style.use("CMS")

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
jax.config.update("jax_enable_x64", True) # in notebooks this is on by default

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PHYSICS_PROCESSES = {
    "wjets": "preproc_uproot/z-prime-ttbar-data/wjets__nominal/file__0/part0.root",
    "ttbar_had": "preproc_uproot/z-prime-ttbar-data/ttbar_had__nominal/file__0/part0.root",
    "ttbar_lep": "preproc_uproot/z-prime-ttbar-data/ttbar_lep__nominal/file__0/part0.root",
    "ttbar_semilep": "preproc_uproot/z-prime-ttbar-data/ttbar_semilep__nominal/file__0/part0.root",
    "signal": "preproc_uproot/z-prime-ttbar-data/signal__nominal/file__0/part0.root",
}

# Selected input variables for training (with scale)
VARS_TO_USE = {
    "n_jet": 1.0 / 10.0,
    "leading_jet_mass": 1.0 / 20.0,
    "subleading_jet_mass": 1.0 / 10.0,
}

PLOT_COLORS = {
    "wjets": "#5790fc",
    "ttbar": "#e42536",
    "signal": "#964a8b",
}

PREFIX = "__NN_wjets_vs_ttbar_nn_"

# -----------------------------------------------------------------------------
# Full logic variable computations
# -----------------------------------------------------------------------------
def compute_n_jet(jets, muons, fatjet, met):
    return ak.num(jets, axis=1).to_numpy()

def compute_leading_jet_mass(jets, muons, fatjet, met):
    return jets.mass[:, 0].to_numpy()

def compute_subleading_jet_mass(jets, muons, fatjet, met):
    return jets.mass[:, 1].to_numpy()

def compute_st(jets, muons, fatjet, met):
    return (ak.sum(jets.pt, axis=1) + ak.sum(muons.pt, axis=1)).to_numpy()

def compute_leading_jet_btag_score(jets, muons, fatjet, met):
    return jets.btagDeepB[:, 0].to_numpy()

def compute_subleading_jet_btag_score(jets, muons, fatjet, met):
    return jets.btagDeepB[:, 1].to_numpy()

def compute_S_zz(jets, muons, fatjet, met):
    denominator = ak.sum(jets.px**2 + jets.py**2 + jets.pz**2, axis=1)
    S_zz = ak.sum(jets.pz * jets.pz, axis=1) / denominator
    return S_zz.to_numpy()

def compute_deltaR(jets, muons, fatjet, met):
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r = ak.min(delta_r, axis=1)
    return min_delta_r.to_numpy()

def compute_pt_rel(jets, muons, fatjet, met):
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r_indices = ak.argmin(delta_r, axis=1, keepdims=True)
    angle = muons.deltaangle(jet_in_pair[min_delta_r_indices])
    return (muons.p * np.sin(angle)).to_numpy().flatten()

def compute_deltaR_times_pt(jets, muons, fatjet, met):
    muon_in_pair, jet_in_pair = ak.unzip(ak.cartesian([muons, jets]))
    delta_r = muon_in_pair.deltaR(jet_in_pair)
    min_delta_r = ak.min(delta_r, axis=1)
    min_delta_r_indices = ak.argmin(delta_r, axis=1, keepdims=True)
    closest_jet_pt = jet_in_pair.pt[min_delta_r_indices]
    return (min_delta_r * closest_jet_pt).to_numpy().flatten()

PHYSICS_VARIABLES = {
    "n_jet": compute_n_jet,
    "leading_jet_mass": compute_leading_jet_mass,
    "subleading_jet_mass": compute_subleading_jet_mass,
    "st": compute_st,
    "leading_jet_btag_score": compute_leading_jet_btag_score,
    "subleading_jet_btag_score": compute_subleading_jet_btag_score,
    "S_zz": compute_S_zz,
    "deltaR": compute_deltaR,
    "pt_rel": compute_pt_rel,
    "deltaR_times_pt": compute_deltaR_times_pt,
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def save_plot(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def save_pickle(params, path: Path):
    np_params = jax.tree.map(np.array, params)
    with open(path, "wb") as f:
        pickle.dump(np_params, f)

def load_pickle(path: Path):
    with open(path, "rb") as f:
        np_params = pickle.load(f)
    return jax.tree.map(jnp.array, np_params)

# -----------------------------------------------------------------------------
# Preprocessing logic
# -----------------------------------------------------------------------------
def load_or_process_physics_data(output_dir: Path, reprocess: bool):
    cache_dir = output_dir / "cache"
    if reprocess and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for proc, path in PHYSICS_PROCESSES.items():
        cache_file = cache_dir / f"{proc}.npz"
        if cache_file.exists() and not reprocess:
            print(f"Loading cached {proc}")
            data[proc] = dict(np.load(cache_file))
            continue

        print(f"Processing {proc} from {path}")
        events = NanoEventsFactory.from_root(
            f"{path}:Events", schemaclass=NanoAODSchema, delayed=False
        ).events()

        muons = events.Muon[
            (events.Muon.pt > 55)
            & (abs(events.Muon.eta) < 2.4)
            & events.Muon.tightId
            & (events.Muon.miniIsoId > 1)
        ]
        jets = events.Jet
        fatjet = events.FatJet
        met = events.PuppiMET

        mask = (ak.num(jets, axis=1) >= 2) & (ak.num(muons, axis=1) == 1)
        muons, jets, fatjet, met = muons[mask], jets[mask], fatjet[mask], met[mask]

        values = {
            var: np.asarray(func(jets, muons, fatjet, met))
            for var, func in PHYSICS_VARIABLES.items()
        }

        np.savez(cache_file, **values)
        data[proc] = values

    data["ttbar"] = {}
    for var in PHYSICS_VARIABLES:
        data["ttbar"][var] = np.concatenate([
            data[p][var] for p in ["ttbar_had", "ttbar_lep", "ttbar_semilep"] if p in data
        ])

    return data

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_model_scores(
    model_paths: list[str],
    data: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    prefix: str = PREFIX,
):
    X_wjets = np.column_stack([data["wjets"][v] * s for v, s in VARS_TO_USE.items()])
    X_ttbar = np.column_stack([data["ttbar"][v] * s for v, s in VARS_TO_USE.items()])
    X_signal = np.column_stack([data["signal"][v] * s for v, s in VARS_TO_USE.items()])

    y_wjets = np.zeros(len(X_wjets))
    y_ttbar = np.ones(len(X_ttbar))

    n = min(len(X_wjets), len(X_ttbar))
    X = jnp.array(np.vstack([X_wjets[:n], X_ttbar[:n]]), dtype=jnp.float32)
    y = jnp.array(np.concatenate([y_wjets[:n], y_ttbar[:n]]), dtype=jnp.float32)

    _, X_test, _, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    for model_path in model_paths:
        model_name = Path(model_path).stem
        params = load_pickle(model_path)

        wjets_score = forward(params, X_test[y_test == 0], prefix=prefix)
        ttbar_score = forward(params, X_test[y_test == 1], prefix=prefix)
        signal_score = forward(params, X_signal, prefix=prefix)

        test_acc = float(accuracy(params, X_test, y_test, prefix=prefix))
        signal_acc = float(accuracy(
            params,
            np.concatenate([X_signal, X_test]),
            np.concatenate([np.ones(len(X_signal)), np.zeros(len(X_test))]),
            prefix=prefix
        ))

        # Styled plot
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.linspace(0, 1, 50)
        counts_wjets = ax.hist(wjets_score, bins=bins, label="wjets", color=PLOT_COLORS["wjets"],
                alpha=0.5, density=True)
        counts_ttbar = ax.hist(ttbar_score, bins=bins, label="ttbar", color=PLOT_COLORS["ttbar"],
                alpha=0.5, density=True)
        counts_signal = ax.hist(signal_score, bins=bins, label="signal", color=PLOT_COLORS["signal"],
                alpha=0.5, density=True)

        ax.set_title(f"{model_name}  |  Test acc: {test_acc:.2f}  |  Signal acc: {signal_acc:.2f}",
                     fontsize=12, loc="left")

        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("a.u.", fontsize=12)
        ax.set_ylim(0, max(max(counts_wjets[0]), max(counts_ttbar[0]), max(counts_signal[0])) * 1.1)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()

        save_path = output_dir / "plots" / "scores" / f"model_scores_{model_name}.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved styled model score plot to {save_path}")

def plot_variable_distributions(data, output_dir: Path):
    for var in PHYSICS_VARIABLES:
        fig, ax = plt.subplots(figsize=(8, 4))
        print(f"Plotting variable: {var}")
        for proc in ["wjets", "ttbar", "signal"]:
            values = data[proc][var]
            print(f"  {proc}: shape={values.shape}, min={values.min():.3f}, max={values.max():.3f}")
            ax.hist(
                values,
                bins=50,
                color=PLOT_COLORS.get(proc, "gray"),
                alpha=0.5,
                label=proc,
                density=True,
            )
        ax.set_xlabel(var, fontsize=12)
        ax.set_ylabel("a.u.", fontsize=12)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        fig.tight_layout()

        out_path = output_dir / "plots" / "features" / f"{var}_dist.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

# -----------------------------------------------------------------------------
# JAX NN functions
# -----------------------------------------------------------------------------
def init_network(key, input_dim, hidden_dim=16, output_dim=1, prefix=""):
    keys = random.split(key, 6)
    return {
        f"{prefix}W1": random.normal(keys[0], (input_dim, hidden_dim)) * 0.1,
        f"{prefix}b1": jnp.zeros(hidden_dim),
        f"{prefix}W2": random.normal(keys[2], (hidden_dim, hidden_dim)) * 0.1,
        f"{prefix}b2": jnp.zeros(hidden_dim),
        f"{prefix}W3": random.normal(keys[4], (hidden_dim, output_dim)) * 0.1,
        f"{prefix}b3": jnp.zeros(output_dim),
    }

def forward(params, x, prefix=""):
    h1 = jnp.tanh(jnp.dot(x, params[f"{prefix}W1"]) + params[f"{prefix}b1"])
    h2 = jnp.tanh(jnp.dot(h1, params[f"{prefix}W2"]) + params[f"{prefix}b2"])
    out = jnp.dot(h2, params[f"{prefix}W3"]) + params[f"{prefix}b3"]
    return out.squeeze()

def binary_cross_entropy(params, x, y, prefix=""):
    logits = forward(params, x, prefix)
    return jnp.mean(jnp.maximum(logits, 0) - logits * y + jnp.log(1 + jnp.exp(-jnp.abs(logits))))

def accuracy(params, x, y, prefix=""):
    preds = (forward(params, x, prefix) > 0).astype(jnp.float32)
    return jnp.mean(preds == y)

@partial(jit, static_argnames=['prefix'])
def update(params, x, y, lr=0.01, prefix=""):
    loss, grads = jax.value_and_grad(binary_cross_entropy)(params, x, y, prefix)
    updated = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return updated, loss

# -----------------------------------------------------------------------------
# Training logic
# -----------------------------------------------------------------------------
def train_and_plot(data, output_dir: Path):
    X_wjets = np.column_stack([data["wjets"][v] * s for v, s in VARS_TO_USE.items()])
    X_ttbar = np.column_stack([data["ttbar"][v] * s for v, s in VARS_TO_USE.items()])
    y_wjets = np.zeros(len(X_wjets))
    y_ttbar = np.ones(len(X_ttbar))

    n = min(len(X_wjets), len(X_ttbar))
    X = jnp.array(np.vstack([X_wjets[:n], X_ttbar[:n]]), dtype=jnp.float32)
    y = jnp.array(np.concatenate([y_wjets[:n], y_ttbar[:n]]), dtype=jnp.float32)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    key = random.PRNGKey(42)
    keys = random.split(key, 3)
    params = init_network(keys[2], input_dim=X_train.shape[1], prefix=PREFIX)

    history = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1000):
        params, loss = update(params, X_train, y_train, lr=0.02, prefix=PREFIX)
        history["loss"].append(loss)
        history["train_acc"].append(accuracy(params, X_train, y_train, PREFIX))
        history["test_acc"].append(accuracy(params, X_test, y_test, PREFIX))
        if epoch % 100 == 0:
            print(f"[{epoch}] loss={loss:.4f} train={history['train_acc'][-1]:.3f} test={history['test_acc'][-1]:.3f}")

    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="loss")
    ax.plot(history["train_acc"], label="train acc")
    ax.plot(history["test_acc"], label="test acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metrics")
    ax.legend()
    save_plot(fig, output_dir / "plots" / "metrics" / "training_curve.png")

    save_pickle(params, output_dir / "model.pkl")
    print(f"Saved model to {output_dir / 'model.pkl'}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="output_jax_nn_standalone")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--model-path", action="append", default=[], help="Path(s) to model.pkl to plot scores from.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_or_process_physics_data(output_dir, reprocess=args.reprocess)
    plot_variable_distributions(data, output_dir)
    train_and_plot(data, output_dir)
    plot_model_scores([f"{output_dir}/model.pkl"], data, output_dir, prefix=PREFIX)

    if args.model_path:
        plot_model_scores(args.model_path, data, output_dir)

if __name__ == "__main__":
    main()