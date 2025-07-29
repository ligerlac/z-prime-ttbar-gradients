Extending the Analysis
======================

Adding a New Cut Parameter
--------------------------
1. Add to ``config["jax"]["params"]``.
2. Reference in ``utils/cuts.py``.
3. (Optional) Define clamping in ``config["jax"]["param_updates"]``.

Adding Systematic Uncertainties
-------------------------------
1. Add an entry to ``config["systematics"]``.
2. Implement logic in ``utils/systematics.py`` if needed.
