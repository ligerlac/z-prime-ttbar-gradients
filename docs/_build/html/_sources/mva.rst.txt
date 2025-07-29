Multi-Variate Analysis (MVA)
============================

The framework supports both JAX and TensorFlow MVAs.

JAXNetwork
----------
* Fully differentiable and optimised alongside cut thresholds.
* Weights and biases are included in global parameter updates.

TFNetwork
---------
* Keras-based model for pre-training only.
* Used for inference during analysis without optimisation.

Workflow
--------
1. Pre-training (optional).
2. Inference during each analysis iteration.
3. Optional in-situ optimisation for JAX-based models.
