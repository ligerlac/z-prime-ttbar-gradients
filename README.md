# Differentiable Z' → tt̄ Analysis Framework

This project is a framework for High-Energy Physics (HEP) analysis that leverages automatic differentiation to optimise analysis selections for maximal statistical significance. It is built on top of the scientific Python ecosystem, including `coffea`, `awkward-array`, and `uproot` for data handling, and `JAX` for gradient-based optimisation.

The example analysis implemented here searches for a Z' boson decaying to a top-antitop quark pair (tt̄).

## Quick Start

This section guides you through running the default analysis configuration provided in the repository.

### 1. Prerequisites

#### Environment Setup

Before running, you must set up the Python environment and install the required dependencies. The recommended method is to use `conda`.

**Using `conda` (Recommended)**

The `environment.yml` file contains all the necessary packages. Create and activate the conda environment with the following commands:

```bash
conda env create -f environment.yml
conda activate zprime_diff_analysis
```

**Using `pip`**

We also provide a `requirements.txt` file, you can still leverage `conda` for environment management:

```bash
# Create a new environment with Python 3.10 (or adjust version as needed)
conda create -n zprime_diff_analysis python=3.10

# Activate the environment
conda activate zprime_diff_analysis

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```
Alternatively, you can use Python’s built-in virtual environment module:

```bash
# Create a virtual environment in a folder named .venv
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Data Pre-processing

The analysis expects pre-processed data files. If you do not have them, you can generate them by running the pre-processing step. This will download the necessary data from the CERN Open Data Portal and skim it according to the configuration.

```bash
# This command overrides the default config to run only the pre-processing step.
# It may take a while to download and process the data.
python run.py general.run_preprocessing=True general.run_mva_training=False general.analysis=nondiff general.run_histogramming=False general.run_statistics=False
```

### 2. Run the Differentiable Analysis

Once the pre-processed data is available, you can run the main analysis with a single command:

```bash
python run.py
```

### 3. What is Happening?

The default configuration (`utils/configuration.py`) is set up to perform a differentiable analysis. The command above will:
1.  **MVA Pre-training**: First, it trains a small, JAX-based neural network to distinguish between `W+jets` and `ttbar` background events. The trained model parameters are saved to disk.
2.  **Differentiable Optimisation**: It then runs the main analysis optimisation loop. The goal is to find the selection cuts that maximise the statistical significance of the Z' signal. At each step, it calculates the gradient of the significance with respect to the cut thresholds (e.g., `met_threshold`, `btag_threshold`) and uses the `optax` optimiser to update them.
3.  **Outputs**: The analysis will produce plots in the `outputs/` directory showing the evolution of the parameters and significance during optimisation, along with the final histograms. The final optimised significance will be printed to the console.

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Key Technologies](#key-technologies)
  - [The Differentiable Workflow](#the-differentiable-workflow)
- [How to Implement an Analysis](#how-to-implement-an-analysis)
  - [1. The Configuration File (`utils/configuration.py`)](#1-the-configuration-file-utilsconfigurationpy)
  - [2. Defining Analysis Logic](#2-defining-analysis-logic)
  - [3. Running the Analysis](#3-running-the-analysis)
- [Configuration Reference](#configuration-reference)
  - [`general` Block](#general-block)
  - [`preprocess` Block](#preprocess-block)
  - [`jax` Block](#jax-block)
  - [`mva` Block](#mva-block)
  - [`channels` Block](#channels-block)
  - [`corrections` and `systematics` Blocks](#corrections-and-systematics-blocks)
  - [Other Top-Level Blocks](#other-top-level-blocks)
- [Under the Hood: The Differentiable Engine](#under-the-hood-the-differentiable-engine)
- [The Differentiable Statistical Model in JAX](#the-differentiable-statistical-model-in-jax)
- [Multi-Variate Analysis (MVA) Integration](#multi-variate-analysis-mva-integration)
- [Extending the Analysis](#extending-the-analysis)
- [Non-Differentiable Analysis](#non-differentiable-analysis)
- [Directory Structure](#directory-structure)

## Core Concepts

The central idea is to treat the entire analysis chain—from event selection to statistical significance—as a single, differentiable function. The inputs to this function are not just the data, but also the analysis parameters themselves (e.g., selection cut thresholds). By calculating the gradient of the significance with respect to these parameters, we can use optimizers like `optax` to iteratively update them and find the optimal set of cuts.

### Key Technologies
*   **`coffea` & `awkward-array`**: For handling complex, jagged data structures typical in HEP.
*   **`JAX`**: For just-in-time (JIT) compilation and automatic differentiation of Python/NumPy code.
*   **`relaxed`**: A JAX-based library for differentiable statistical models, providing a differentiable approximation of the profile likelihood ratio.
*   **`optax`**: A library of gradient-based optimizers for JAX.

### The Differentiable Workflow

The analysis is orchestrated by the `DifferentiableAnalysis` class in `analysis/diff.py`. The workflow proceeds as follows:

1.  **Preprocessing**: Raw NanoAOD files are skimmed to keep only necessary branches and apply a baseline selection. This is a one-time, non-differentiable step to reduce data volume.
2.  **MVA Pre-training (Optional)**: If configured, a Machine Learning model (e.g., a neural network) is trained on pre-selected data to serve as a powerful discriminator. The weights of this model can themselves become optimizable parameters.
3.  **Event Processing**: For each event, object corrections and systematic variations are applied.
4.  **Differentiable Histogramming**: Instead of making hard cuts, we apply "soft" selections using sigmoid functions. This results in a per-event weight. Histograms are filled using a Kernel Density Estimation (KDE) approach, which is smooth and differentiable.
5.  **Statistical Significance**: The `relaxed` library is used to construct a statistical model from the histograms and compute an asymptotic significance (a differentiable quantity).
6.  **Gradient Calculation**: `JAX` computes the gradient of the significance with respect to all optimizable parameters (cut thresholds, MVA weights, etc.).
7.  **Parameter optimisation**: The `optax` optimizer takes a step in the direction of the gradient to update the parameters, aiming to maximize significance. Steps 4-7 are repeated for a set number of iterations.

---

## How to Implement an Analysis

Implementing a new analysis or modifying the existing one primarily involves three steps:
1.  Modifying the central configuration file.
2.  Defining the analysis logic (observables and selections) in Python functions.
3.  Running the analysis workflow.

### 1. The Configuration File (`utils/configuration.py`)

This file is the central hub for defining your entire analysis. The `config` dictionary controls every aspect of the workflow.

**Key Sections:**

*   `general`: Global settings like integrated luminosity, which analysis steps to run (`run_preprocessing`, `run_mva_training`), and file paths.
*   `preprocess`: Defines the branches to keep from the input NanoAOD files.
*   `good_object_masks`: Defines baseline "good" object criteria (e.g., muon pT > 55 GeV) that are applied before any channel-specific logic. This is useful for creating a common object collection for all analysis channels.
*   `baseline_selection`: A hard, non-differentiable selection applied to all events early on.
*   `channels`: Defines the different analysis regions (e.g., signal region, control regions). For each channel, you specify:
    *   `name`: A unique name for the channel.
    *   `selection`: The selection function to apply for this channel.
    *   `observables`: A list of variables to be histogrammed.
    *   `fit_observable`: The specific observable used for the final statistical fit.
*   `ghost_observables`: A powerful feature for computing derived quantities (e.g., ST, ΔR) once and attaching them to the event record. These can then be used by any downstream function.
*   `corrections` & `systematics`: Define object and event-level corrections and systematic uncertainties. The framework supports both `correctionlib` and custom Python functions.
*   `mva`: Configure MVA models. You can define the architecture, features, and training parameters for a JAX or TensorFlow/Keras network.
*   `jax`: **This is the core of the differentiable analysis.**
    *   `params`: A dictionary of all optimizable parameters and their initial values (e.g., `'met_threshold': 50.0`).
    *   `soft_selection`: Points to the Python function that implements your differentiable selection logic.
    *   `param_updates`: Defines clamping functions to keep parameters within physical bounds during optimisation (e.g., `jnp.clip(x + d, 0.0, 3.0)`).
    *   `learning_rates`: Allows you to set custom learning rates for different parameters.

### 2. Defining Analysis Logic

The `config` file points to Python functions that contain the actual physics logic. These typically live in `utils/`.

#### Observables (`utils/observables.py`)

An observable function takes `awkward-array` collections as input and returns a flat array of the computed values.

**Example: `get_mtt`**
```python
def get_mtt(
    muons: ak.Array,
    jets: ak.Array,
    fatjets: ak.Array,
    met: ak.Array,
) -> ak.Array:
    # ... logic to calculate four-vectors and sum them ...
    p4tot = p4mu + p4fj + p4j + p4met
    return p4tot.mass
```

#### Selections (`utils/cuts.py`)

There are two types of selection functions:

1.  **Standard Selections**: Used for non-differentiable analysis or initial hard cuts. They take `awkward` arrays and should return a `coffea.analysis_tools.PackedSelection` object.

2.  **Differentiable "Soft" Selections**: This is where the magic happens. Instead of returning a boolean mask, this function returns a continuous, per-event **weight** between 0 and 1. This is achieved by replacing hard cuts like `met.pt > 50` with a sigmoid function `jax.nn.sigmoid((met.pt - 50) / scale)`.

**Example: `Zprime_softcuts_jax_workshop`**

This function takes JAX-backed awkward arrays and a `params` dictionary (containing the optimizable parameters defined in the config).

```python
def Zprime_softcuts_jax_workshop(
    muons: ak.Array,
    jets: ak.Array,
    met: ak.Array,
    jet_mass: ak.Array,
    nn,
    params: dict
) -> jnp.ndarray:
    # ...

    # A differentiable cut on MET
    met_cut_weight = jax.nn.sigmoid(
        (ak.to_jax(met) - params["met_threshold"]) / 25.0
    )

    # A differentiable cut on a b-tagging score
    btag_cut_weight = jax.nn.sigmoid(
        (soft_b_counts - params["btag_threshold"]) * 10.0
    )

    # ... other cuts

    # Combine all weights multiplicatively (like a logical AND)
    selection_weight = jnp.prod(jnp.stack([met_cut_weight, btag_cut_weight, ...]))
    return selection_weight
```

### 3. Running the Analysis

With the configuration and functions in place, you can run the analysis using a top-level script.

#### Example `run.py` script
A typical script would:
1.  Load the base configuration from `utils/configuration.py`.
2.  Optionally, override configuration settings from the command line.
3.  Construct the fileset of data samples.
4.  Instantiate the `DifferentiableAnalysis` class from `analysis/diff.py`.
5.  Call the main `run_analysis_optimisation` method.

```python
# In a hypothetical run.py
import sys
from analysis.diff import DifferentiableAnalysis
from utils.configuration import config, load_config_with_restricted_cli
from utils.input_files import construct_fileset

if __name__ == "__main__":
    # Load base config and override with CLI args
    cfg = load_config_with_restricted_cli(config, sys.argv[1:])

    fileset = construct_fileset(n_files_max_per_sample=cfg.general.max_files)
    analysis = DifferentiableAnalysis(cfg)
    final_histograms, final_significance = analysis.run_analysis_optimisation(fileset)

    print(f"optimisation complete! Final significance: {final_significance:.3f}")
```

#### Overriding Configuration from the Command Line

You can override certain configuration options directly from the command line using a dot-list format. This is useful for quick tests and batch submissions without modifying the main configuration file.

**Example:**
```python
python run.py general.max_files=10 general.run_systematics=False
```

**Important Limitations:**
For safety and to prevent breaking the analysis logic, only a restricted set of configuration keys can be overridden from the command line. This is because the main Python configuration file contains complex objects like functions and lambdas, which cannot be expressed as simple command-line arguments.

The allowed top-level keys for CLI overrides are:
*   `general`
*   `preprocess`
*   `statistics`

Attempting to override other keys (e.g., `jax.params`) will result in an error. To change these, you must edit the `utils/configuration.py` file directly.

## Configuration Reference

The analysis is controlled by a central configuration dictionary, typically defined in `utils/configuration.py`.
The structure of this configuration is validated against a Pydantic schema in `utils/schema.py`.

Below is a comprehensive reference for all available options, grouped by their top-level key.

---

### `general` Block

Global settings that control the overall workflow of the analysis.

| Parameter            | Type         | Default                     | Description                                                |
|----------------------|--------------|-----------------------------|------------------------------------------------------------|
| `lumi`              | `float`      | *Required*                  | Integrated luminosity in inverse picobarns (/pb).         |
| `weights_branch`    | `str`        | *Required*                  | Branch name containing event weights (e.g. `genWeight`).  |
| `lumifile`          | `str`        | *Required*                  | Path to the JSON file containing certified good luminosity sections (Golden JSON). |
| `analysis`          | `str`        | `"nondiff"`                 | Analysis mode: `"nondiff"`, `"diff"`, or `"both"`.        |
| `max_files`         | `int`        | `-1`                        | Max number of files per dataset. `-1` = unlimited.        |
| `run_preprocessing` | `bool`       | `False`                     | Run NanoAOD skimming and filtering.                       |
| `run_histogramming` | `bool`       | `True`                      | Run histogramming for non-differentiable analysis.        |
| `run_statistics`    | `bool`       | `True`                      | Run statistical analysis step (e.g. `cabinetry` fit).     |
| `run_systematics`   | `bool`       | `True`                      | Process systematic variations for non-differentiable analysis. |
| `run_plots_only`    | `bool`       | `False`                     | Generate plots from cached results only.                  |
| `run_mva_training`  | `bool`       | `False`                     | Run MVA model pre-training.                               |
| `read_from_cache`   | `bool`       | `True`                      | Read preprocessed data from cache if available.           |
| `output_dir`        | `str`        | `"output/"`                 | Root directory for all analysis outputs.                  |
| `preprocessor`      | `str`        | `"uproot"`                  | Preprocessing engine: `"uproot"` or `"dask"`.            |
| `preprocessed_dir`  | `str`        | `None`                      | Directory with pre-processed (skimmed) files.            |
| `cache_dir`         | `str`        | `"/tmp/gradients_analysis/"`| Cache directory for differentiable analysis.             |
| `processes`         | `list[str]`  | `None`                      | Limit analysis to specific processes.                     |
| `channels`          | `list[str]`  | `None`                      | Limit analysis to specific channels.                      |

---

### `preprocess` Block

Settings for the initial data skimming and filtering step.

| Parameter        | Type       | Default     | Description                                         |
|------------------|------------|-------------|-----------------------------------------------------|
| `branches`       | `dict`     | *Required*  | Mapping of collection names to branch lists.        |
| `ignore_missing` | `bool`     | `False`     | Ignore missing branches if `True`.                  |
| `mc_branches`    | `dict`     | *Required*  | Additional branches for MC samples.                 |

---

### `jax` Block

Configuration for the differentiable analysis workflow.

| Parameter                | Type         | Default     | Description                                                        |
|--------------------------|--------------|-------------|--------------------------------------------------------------------|
| `soft_selection`         | `dict`      | *Required*  | Differentiable selection function.                                 |
| &nbsp;&nbsp;↳ `function` | `Callable`  | *Required*  | Selection function to apply.                                       |
| &nbsp;&nbsp;↳ `use`      | `list[str]` | *Required*  | Input variables passed to `function`.                              |
| `params`                 | `dict`      | *Required*  | Optimizable parameters (e.g. `{'met_threshold': 50.0}`).          |
| `optimize`               | `bool`      | `True`      | Run gradient-based optimisation if `True`.                         |
| `learning_rate`          | `float`     | `0.01`      | Default optimizer learning rate.                                   |
| `max_iterations`         | `int`       | `50`        | Number of optimisation steps.                                      |
| `param_updates`          | `dict`      | `{}`        | Parameter-specific clamping functions.                             |
| &nbsp;&nbsp;↳ `param_name` | `Callable`| -           | `(old_value, delta) -> new_value` function.                        |
| `learning_rates`         | `dict`      | `None`      | Parameter-specific learning rates.                                 |
| `explicit_optimisation`  | `bool`      | `False`     | Use manual optimisation loop if `True`.                            |

---

### `mva` Block

List of MVA model configurations.

| Parameter               | Type            | Default     | Description                                                       |
|-------------------------|-----------------|-------------|-------------------------------------------------------------------|
| `name`                  | `str`          | *Required*  | Unique model name.                                                |
| `framework`             | `str`          | *Required*  | `"jax"` or `"keras"`.                                            |
| `learning_rate`         | `float`       | `0.01`      | Learning rate for pre-training.                                  |
| `grad_optimisation`     | `dict`        | `{}`        | MVA optimisation settings.                                       |
| &nbsp;&nbsp;↳ `optimise`    | `bool`    | `False`     | Include MVA in global optimisation.                              |
| &nbsp;&nbsp;↳ `learning_rate` | `float` | `0.001`    | Learning rate for MVA in optimisation.                           |
| `layers`                | `list[dict]` | *Required*  | Network architecture layers.                                     |
| &nbsp;&nbsp;↳ `ndim`    | `int`        | *Required*  | Number of nodes.                                                 |
| &nbsp;&nbsp;↳ `activation` | `str`     | *Required*  | Activation function.                                             |
| &nbsp;&nbsp;↳ `weights` | `str`        | *Required*  | Name for weights parameter.                                      |
| &nbsp;&nbsp;↳ `bias`    | `str`        | *Required*  | Name for bias parameter.                                         |
| `loss`                  | `Callable` or `str` | *Required* | Loss function (callable for JAX, string for Keras).             |
| `features`              | `list[dict]` | *Required*  | Input features for the model.                                   |
| &nbsp;&nbsp;↳ `name`    | `str`        | *Required*  | Feature name.                                                    |
| &nbsp;&nbsp;↳ `function`| `Callable`   | *Required*  | Function to compute the feature.                                 |
| &nbsp;&nbsp;↳ `use`     | `list[str]` | *Required*  | Input dependencies for the feature.                              |
| `classes`               | `list`       | *Required*  | Target classes (e.g. `["wjets", {"ttbar": [...]}]`).            |
| `balance_strategy`      | `str`       | `"undersample"` | `"none"`, `"undersample"`, `"oversample"`, `"class_weight"`.  |
| `random_state`          | `int`       | `42`        | Random seed for reproducibility.                                 |
| `epochs`                | `int`       | `1000`      | Pre-training epochs.                                             |
| `batch_size`            | `int`       | `32`        | Training batch size.                                             |
| `validation_split`      | `float`     | `0.2`       | Fraction for validation.                                         |
| `log_interval`          | `int`       | `100`      | Log frequency during training.                                   |

---

### `channels` Block

List of analysis channels or regions.

| Parameter            | Type          | Default     | Description                                                      |
|----------------------|---------------|-------------|------------------------------------------------------------------|
| `name`              | `str`         | *Required*  | Channel name (e.g. `"signal_region"`).                          |
| `observables`       | `list[dict]`  | *Required*  | Observables to histogram.                                       |
| &nbsp;&nbsp;↳ `name`      | `str`    | *Required*  | Observable name.                                                |
| &nbsp;&nbsp;↳ `binning`   | `tuple`  | *Required*  | Histogram binning.                                              |
| &nbsp;&nbsp;↳ `function`  | `Callable` | *Required* | Function to compute observable.                                 |
| &nbsp;&nbsp;↳ `use`       | `list[str]` | *Required* | Inputs to the observable function.                              |
| `fit_observable`    | `str`         | *Required*  | Observable used in statistical fit.                             |
| `selection`         | `dict`       | `None`      | Channel selection function.                                     |
| &nbsp;&nbsp;↳ `function` | `Callable` | *Required* | Selection function.                                             |
| &nbsp;&nbsp;↳ `use`      | `list[str]` | *Required* | Inputs to selection function.                                   |
| `use_in_diff`       | `bool`       | `False`     | Include in differentiable analysis.                             |

---

### `corrections` and `systematics` Blocks

| Parameter            | Type       | Default        | Description                                           |
|----------------------|------------|----------------|-------------------------------------------------------|
| `name`              | `str`     | *Required*     | Correction or systematic name.                       |
| `type`              | `str`     | *Required*     | `"object"` or `"event"`.                             |
| `op`                | `str`     | `"mult"`       | Operation: `"mult"` or `"add"`.                      |
| `target`           | `tuple` or `list` | `None` | Object/variable to modify (e.g. `("Jet", "pt")`).   |
| `use`              | `list`    | `[]`          | Inputs required for the variation function.          |
| **Corrections Only** |           |                |                                                       |
| &nbsp;&nbsp;↳ `file`        | `str`     | *Required*     | Path to correction file.                              |
| &nbsp;&nbsp;↳ `key`         | `str`     | `None`         | Key within the file.                                  |
| &nbsp;&nbsp;↳ `use_correctionlib` | `bool` | `True` | Use `correctionlib`.                                 |
| &nbsp;&nbsp;↳ `transform`   | `Callable` | `None`        | Transform arguments before evaluation.               |
| &nbsp;&nbsp;↳ `up_and_down_idx` | `list[str]` | `["up", "down"]` | Variation labels in file.                     |
| **Systematics Only** |           |                |                                                       |
| &nbsp;&nbsp;↳ `up_function`    | `Callable` | `None` | Function for "up" variation.                        |
| &nbsp;&nbsp;↳ `down_function`  | `Callable` | `None` | Function for "down" variation.                      |
| &nbsp;&nbsp;↳ `symmetrise`     | `bool`    | `False`| Auto-generate "down" from "up" (not implemented).   |

---

## Under the Hood: The Differentiable Engine

The core of the differentiable workflow is the `_run_traced_analysis_chain` method in `analysis/diff.py`. This function is what `JAX` traces and differentiates.

```python
def _run_traced_analysis_chain(
    self,
    params: dict[str, Any],
    processed_data_events: dict,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    # ...
    # 1. Collect histograms for all processes using the current `params`
    histograms_by_process = self._collect_histograms(...)

    # 2. Calculate significance from these histograms
    significance, mle_params = self._calculate_significance(histograms_by_process, params["fit"])
    # ...
    return significance, mle_params
```

The optimisation loop in `run_analysis_optimisation` then does the following:

```python
# Define the objective function to be *maximized* (significance)
# Note: Optimizers typically *minimize*, so we would differentiate the *negative* significance.
# The `relaxed` library handles this internally.
def objective(params):
    return self._run_traced_analysis_chain(params, processed_data)

# Get the gradient of the objective function w.r.t. the parameters
gradients = jax.grad(objective)(all_parameters)

# Use an optimizer to update the parameters
# solver = OptaxSolver(fun=objective, opt=tx, ...)
# new_parameters, state = solver.update(parameters, state)
```

This loop continues until the significance converges or a maximum number of iterations is reached.

## The Differentiable Statistical Model in JAX

A key innovation of this framework is its end-to-end differentiable statistical model. This model is constructed in JAX, allowing the entire analysis—from event selection to statistical inference—to be differentiated.

### 1. Differentiable Histograms

Instead of standard histograms with hard bin counts, the analysis produces "soft" histograms using a Kernel Density Estimation (KDE) approach. For each event, instead of adding `1` to a single bin, a Gaussian kernel is placed at the event's observable value. The histogram's bin contents are then the integral of all event kernels over that bin's range. This process is smooth and differentiable with respect to both the event weights (from soft selections) and the observable values themselves.

### 2. Building the Statistical Model

The statistical model is built upon these differentiable histograms. The goal is to construct a likelihood function `L(data | params)` that can be maximized to find the best-fit parameters.

The core of the model is the prediction for the expected number of events (`expected_yields`) in each histogram bin for each analysis channel. This is a function of the model parameters. In the current implementation (`utils/jax_stats.py`), the model is simplified and includes two global scalar parameters:
*   `mu` ($\mu$): The signal strength parameter. $\mu=0$ corresponds to the background-only hypothesis, and $\mu=1$ corresponds to the nominal signal hypothesis.
*   `norm_ttbar_semilep` ($\kappa_{t\bar{t}}$): A normalisation factor for the `ttbar_semilep` background process, applied uniformly across all channels.

For a single channel, the expected yield $\lambda_i$ in bin $i$ is the sum of all signal and background processes:

$\lambda_i(\mu, \kappa_{t\bar{t}}) = \mu \cdot \lambda_{\text{sig}, i} + \kappa_{t\bar{t}} \cdot \lambda_{t\bar{t}, i} + \sum_{b \in \text{other bkg}} \lambda_{b, i}$

Note that in this simplified model, systematic uncertainties are not incorporated via nuisance parameters in the differentiable fit. The framework can produce systematically-varied histograms, but they are used in the non-differentiable analysis path.

### 3. Likelihood and Hypothesis Testing with `relaxed`

With the model for `expected_yields` defined, the final step is to perform statistical inference. This is where the `relaxed` library is used.

1.  **Likelihood Construction**: `relaxed` takes the JAX model of expected yields and the observed data histograms to construct the full likelihood function. For this simplified model, the likelihood is a product of Poisson probability mass functions over all bins:

    $L(\text{data} | \mu, \kappa_{t\bar{t}}) = \prod_{c \in \text{channels}} \prod_{i \in \text{bins}} \text{Pois}(N_{\text{obs}, c, i} | \lambda_{c, i}(\mu, \kappa_{t\bar{t}}))$

    Since the model in `utils/jax_stats.py` does not include systematic uncertainties as nuisance parameters, there are no constraint terms in the likelihood.

2.  **Differentiable Hypothesis Test**: The main goal is to test for the presence of a signal. This is done using a profile likelihood ratio test statistic, $q_0$. `relaxed.infer.hypotest` performs this test. It finds the values of the model parameters (in this case, $\kappa_{t\bar{t}}$) that maximize the likelihood for a given $\mu$ (profiling) and then computes $q_0$. The significance is then $Z = \sqrt{q_0}$.

This means we can compute the gradient of the final significance `Z` with respect to any parameter in the chain, including the selection cut thresholds (`met_threshold`, etc.) and even the weights of a JAX-based MVA. This gradient is what drives the optimisation.

## Multi-Variate Analysis (MVA) Integration

The framework includes support for integrating Machine Learning models (MVAs) into the analysis, with implementations for both JAX and TensorFlow/Keras. This allows for the use of non-linear discriminants that can be pre-trained and, in the case of JAX models, optimized *in-situ* with the rest of the analysis.

The core logic is handled by the `JAXNetwork` and `TFNetwork` classes in `utils/mva.py`.

### MVA Implementations

#### `JAXNetwork`
This is a neural network implementation written purely in JAX, providing deep integration with the differentiable analysis workflow.
*   **Explicit Parameter Management**: Unlike frameworks that encapsulate model weights, the `JAXNetwork` manages its weights and biases in a simple Python dictionary. This transparency is key to its integration. Parameter names follow a convention (e.g., `__NN_my_model_W1`) that allows the framework to automatically identify them.
*   **End-to-End optimisation**: When MVA optimisation is enabled in the configuration (`grad_optimisation.optimise: True`), the network's parameters are added to the global set of variables that the main optimizer tunes. This means the optimizer can simultaneously adjust the MVA's weights to improve signal/background separation *and* tune the analysis selection cuts, all to directly maximize the final statistical significance.
*   **Full Control**: The from-scratch implementation gives full control over the network's forward pass, loss function, and training loop, all within the JAX ecosystem.
*   **Configuration**: The network architecture (layers, activations) is defined in `utils/configuration.py`. Activations are provided as Python `lambda` functions, allowing for custom, non-standard activation functions if needed.

#### `TFNetwork`
This class provides a wrapper around a standard `tf.keras.Sequential` model.
*   **Leverage Keras**: It allows you to use the rich and mature Keras API for building and training models.
*   **Pre-training Only**: The primary use case is to pre-train a powerful discriminator. The trained model is then used to compute a score for each event, which is used as a static input feature in the main analysis. The weights of a TF/Keras model are **not** part of the global significance optimisation.
*   **Configuration**: The architecture is defined in the configuration file, with activations specified as strings (e.g., `"relu"`, `"tanh"`).

### The MVA Workflow: Pre-training and In-situ Optimisation

The framework handles MVAs in a two-stage process: an initial, one-off pre-training phase, followed by an optional, continuous optimisation phase that happens alongside the main analysis optimisation.

#### Stage 1: Pre-training

*   **When**: This happens once at the start of the analysis if `general.run_mva_training` is `True`.
*   **Data**: The framework allows for a completely separate object definition for MVA training. In `config.good_object_masks`, you can define an `mva` key with different object selection criteria than the `analysis` key. This is useful for training on a broader, less-biased dataset.
*   **Process**:
    1.  During event processing (`_prepare_data`), two parallel sets of object collections are created: one for the main analysis and one for MVA training, each with its own "good object" masks applied.
    2.  After all files are processed, the MVA-specific data is passed to `_run_mva_training`.
    3.  The models (both JAX and Keras) are trained on this dedicated dataset.
    4.  The resulting trained model (for Keras) or parameters (for JAX) are saved to disk.

#### Stage 2: Inference and In-situ Optimisation (Fine-tuning)

*   **When**: This happens at every step of the main gradient-based optimisation loop.
*   **Inference (On-the-fly)**:
    *   The MVA *instance* (containing the forward pass logic) and its input *features* are attached to a special collection in the event record.
    *   The differentiable selection function (e.g., `Zprime_softcuts_jax_workshop`) must be designed to accept this collection and the global `params` dictionary as inputs.
    *   Inside this traced function, the MVA's `forward_pass` is called on-the-fly, using the current state of the MVA's weights from the `params` dictionary. This ensures that the entire calculation, from MVA inputs to score, is part of the JAX computation graph.
*   **In-situ Optimisation (JAX only)**:
    *   If an MVA is configured with `framework: "jax"` and `grad_optimisation.optimise: True`, its pre-trained parameters are included in the set of globally optimisable parameters.
    *   Because the MVA's forward pass is executed on-the-fly within the main traced function, its output (the MVA score) is fully differentiable with respect to its weights and biases.
    *   The gradient of the final statistical significance is therefore also calculated with respect to these MVA parameters.
    *   The `optax` optimiser updates the MVA weights at each step, effectively "fine-tuning" the MVA to directly maximise the analysis significance, alongside all other selection cuts.
    *   Keras models are used for inference only; their weights are not optimised during this stage.

### Extending and Adding a New MVA

Adding a new MVA to the analysis is a configuration-driven process:

1.  **Add to Config**: Create a new dictionary entry in the `mva` list in `utils/configuration.py`. Give it a unique `name`.

2.  **Define Architecture & Framework**:
    *   Set `framework` to `"jax"` or `"keras"`.
    *   Define the `layers` list, specifying the dimensions, activation functions, and names for weights/biases for each layer.
    *   Define the `loss` function.

3.  **Define Input Features**:
    *   In the `features` list for your MVA, define each input variable.
    *   Each feature needs a `name`, a `function` to compute it (e.g., `lambda mva: mva.n_jet`), and a `use` key specifying the inputs to that function. The framework automatically computes "ghost observables" first, so you can define features that depend on them.

4.  **Use the MVA in Selection**:
    *   The framework will automatically train the MVA (if `run_mva_training` is `True`) and compute its output score for every event.
    *   This score is attached to a special object collection named after your MVA (e.g., `wjets_vs_ttbar_nn`).
    *   You can then use this score in your differentiable selection function in `utils/cuts.py` just like any other variable.

5.  **Enable Gradient optimisation (JAX only)**:
    *   To make the JAX MVA's weights optimizable, set `grad_optimisation.optimise: True` in its configuration.
    *   The framework will automatically find all parameters with the `__NN` prefix and add them to the set of variables that the optimizer will tune. You can even set a custom learning rate for the MVA weights.

## Extending the Analysis

#### Adding a New Optimizable Parameter

1.  **Add to Config**: Add a new key-value pair to `config["jax"]["params"]`.
    ```python
    "params": {
        'met_threshold': 50.0,
        'btag_threshold': 0.5,
        'my_new_cut': 100.0, # <-- Add new parameter
    },
    ```
2.  **Use in Soft Selection**: Use `params["my_new_cut"]` in your soft selection function in `utils/cuts.py`.
3.  **(Optional) Add a Clamp**: Add a rule for your new parameter in `config["jax"]["param_updates"]` to keep it within a sensible range.

#### Adding a New Systematic Uncertainty

1.  **Add to Config**: Add a new dictionary to the `config["systematics"]` list.
2.  **Define Logic**:
    *   If it's a simple scale factor, you can define the `up_function` and `down_function` directly in the config (e.g., `lambda: 1.05`).
    *   For more complex variations, define a function in `utils/systematics.py` that takes an object collection and returns a per-object weight.
    *   For `correctionlib`-based uncertainties, ensure the `file` and `key` are specified correctly.
3.  **Specify Target**: Define the `target` (which object and variable are affected) and the `op` (how the variation is applied, e.g., `mult` or `add`).

The framework will automatically propagate these systematics through the non-differentiable analysis path to produce varied histograms for tools like `cabinetry`.

## Non-Differentiable Analysis

Alongside the differentiable path, the framework fully supports a traditional, non-differentiable analysis via the `NonDiffAnalysis` class in `analysis/nondiff.py`. This path uses standard hard cuts and fills `hist` objects, which can then be used with tools like `cabinetry` for statistical inference. You can control which analysis runs via the `analysis` key in `config.general`.

---

## Directory Structure

```
├── analysis/
│   ├── base.py         # Base class with common analysis logic (corrections, etc.)
│   ├── diff.py         # Implements the full differentiable analysis workflow
│   └── nondiff.py      # Implements a traditional, non-differentiable analysis
├── utils/
│   ├── configuration.py # The main configuration file for the analysis
│   ├── cuts.py         # Defines selection logic (both hard and soft/differentiable)
│   ├── observables.py  # Defines functions to compute physics observables
│   ├── mva.py          # MVA (neural network) model definitions and training logic
│   ├── systematics.py  # Functions for systematic variations
│   ├── schema.py       # Pydantic schemas for validating the configuration
│   └── ...             # Other helper utilities
├── cabinetry/
│   └── ...             # Configuration for the `cabinetry` statistical tool
├── corrections/
│   └── ...             # Correction files (e.g., from `correctionlib`)
└── README.md
```