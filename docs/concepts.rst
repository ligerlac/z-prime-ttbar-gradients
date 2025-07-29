Core Concepts
=============

The framework treats the analysis chain as a differentiable function:

1. **Preprocessing**: Baseline filtering of NanoAOD data.
2. **MVA Pre-training**: Train a classifier to distinguish signal and background.
3. **Differentiable Histogramming**: Use KDE-based histogramming for smooth gradients.
4. **Statistical Model**: Construct a differentiable profile likelihood.
5. **Gradient Optimisation**: Update parameters using JAX and Optax.

Key Technologies
----------------
* **coffea** & **awkward-array**
* **JAX**
* **relaxed**
* **optax**
