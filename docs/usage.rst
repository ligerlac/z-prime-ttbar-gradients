Usage
=====

Quick Start
-----------

1. **Preprocess Data**:
   .. code-block:: bash

       python run.py general.run_preprocessing=True general.run_mva_training=False general.analysis=nondiff general.run_histogramming=False general.run_statistics=False

2. **Run Analysis**:
   .. code-block:: bash

       python run.py

3. **Outputs**:
   Plots and results will be generated in the ``outputs/`` directory.

What Happens Internally
-----------------------
* **MVA Pre-training**: Trains a neural network to separate signal and background.
* **Differentiable Optimisation**: Gradients are used to update cut thresholds.
* **Result**: Plots and significance results are saved.
