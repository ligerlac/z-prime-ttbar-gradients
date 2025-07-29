Configuration Reference
=======================

The analysis is configured using ``utils/configuration.py``. Below is a full reference.

General Block
-------------
.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - lumi
     - float
     - Required
     - Integrated luminosity in /pb.
   * - weights_branch
     - str
     - Required
     - Event weights branch.
   * - analysis
     - str
     - "nondiff"
     - Analysis mode: "nondiff", "diff", or "both".
   * - max_files
     - int
     - -1
     - Maximum files per dataset (-1 unlimited).
   * - run_preprocessing
     - bool
     - False
     - Run data preprocessing.
   * - run_histogramming
     - bool
     - True
     - Run histogramming.
   * - run_statistics
     - bool
     - True
     - Run statistical analysis.
   * - run_mva_training
     - bool
     - False
     - Train MVA models.
   * - output_dir
     - str
     - "output/"
     - Output directory.

Preprocess Block
----------------
Defines branches for data skimming.

JAX Block
---------
Configures differentiable analysis (params, learning rates, optimisation).

MVA Block
---------
Defines MVA models (architecture, features, training).

Channels Block
--------------
Defines channels, selections, observables.

Corrections & Systematics
-------------------------
Object/event-level corrections and uncertainties.
