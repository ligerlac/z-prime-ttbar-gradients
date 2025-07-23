from collections import Counter
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, jit
from equinox import filter_jit
import tensorflow as tf
from sklearn.model_selection import train_test_split


from analysis.base import Analysis

# =============================================================================
# Supporting functions
# =============================================================================

# ===============================================================================
# Train-Test split
# ===============================================================================

# ----------------------------------------------------------------------------
# Abstract BaseNetwork class
# ----------------------------------------------------------------------------
class BaseNetwork(ABC, Analysis):
    """
    Abstract interface for MVA networks. Subclasses must implement framework-specific
    initialization, training, and prediction.
    """

    def __init__(self, mva_cfg: Dict[str, Any]) -> None:
        """
        Store the Pydantic configuration for this MVA.

        Args:
            mva_cfg: MVAConfig instance describing layers, training params, etc.
        """
        self.mva_cfg = mva_cfg
        self.name = mva_cfg.name
        # For JAX: dictionary of weight/bias arrays. For TF: not used.
        self.parameters: Dict[str, Any] = {}
        # For TF/Keras: holds the Sequential model. For JAX: remains None.
        self.model: Any = None

    def split_train_test(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        test_size=self.mva_cfg.validation_split
        random_state=self.mva_cfg.random_state
        shuffle=True
        stratify=y if self.mva_cfg.balance_strategy!="none" else None

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )

    def _extract_features(
        self,
        obj_copies: Dict[str, Any],
        feat_cfgs: List[Any]
    ) -> np.ndarray:
        """
        Compute and stack all features for one event batch.
        """
        arrays: List[np.ndarray] = []
        for feat in feat_cfgs:
            # grab raw inputs for this feature
            args = self._get_function_arguments(feat.use, obj_copies)
            vals = feat.function(*args)
            if feat.scale is not None:
                vals = feat.scale(vals)
            arrays.append(np.asarray(vals))
        # stack as columns
        return np.stack(arrays, axis=1).astype(float)

    def _make_labels(
        self,
        n_events: int,
        process: str,
        classes: list[Union[str, dict[str, tuple[str]]]],
    ) -> np.ndarray:
        """
        Produce integer labels for a given process, based on class definitions.

        Parameters
        ----------
        n_events : int
            Number of events for this process.
        process : str
            The process name (e.g. 'ttbar_semilep', 'wjets').
        classes : list
            List of class definitions, where each entry is either:
            - a string (single-process class), or
            - a dict mapping a class name to a list/tuple of process names to merge.

        Returns
        -------
        np.ndarray
            Array of shape (n_events,) filled with the integer label corresponding to the process.

        Raises
        ------
        RuntimeError
            If the process name is not found in any class definition.
        """
        for idx, cls in enumerate(classes):
            if isinstance(cls, str):
                if process == cls:
                    return np.full(n_events, idx, dtype=int)
            elif isinstance(cls, dict):
                class_name, proc_list = next(iter(cls.items()))
                if process == class_name:
                    return np.full(n_events, idx, dtype=int)

        raise RuntimeError(
            f"Process '{process}' not found in any class definition in: {classes}"
        )

    def _balance_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
        """
        Rebalance X,y according to `strategy` (none, undersample, oversample, class_weight).
        """
        strategy = self.mva_cfg.balance_strategy
        random_state = self.mva_cfg.random_state

        if strategy == "none":
            return X, y, None

        rng = np.random.RandomState(random_state)
        counts = Counter(y)
        labels = sorted(counts.keys())
        if strategy in ("undersample", "oversample"):
            # target count per class
            target = min(counts.values()) if strategy == "undersample" else max(counts.values())
            Xs, ys = [], []
            for lbl in labels:
                idxs = np.where(y == lbl)[0]
                sel = rng.choice(idxs, size=target, replace=(strategy=="oversample"))
                Xs.append(X[sel]); ys.append(y[sel])
            X_new = np.concatenate(Xs, axis=0)
            y_new = np.concatenate(ys, axis=0)
            perm = rng.permutation(len(y_new))
            return X_new[perm], y_new[perm], None

        if strategy == "class_weight":
            total = float(len(y))
            class_weights = {lbl: total/(len(labels)*counts[lbl]) for lbl in labels}
            return X, y, class_weights

        raise ValueError(f"Unknown balance_strategy={strategy}")

    def prepare_inputs(
        self,
        events_per_process: dict[str, list[tuple[dict, int]]],
    ) -> Any:
        """
        Prepare input features and labels for MVA training.

        Parameters
        ----------
        events_per_process : dict
            Mapping from process name to a list of (object_copies, n_events) tuples.
            These represent the corrected object collections and number of events for each process.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple containing:
            - Training features Xtr
            - Training labels ytr
            - Validation features Xvl
            - Validation labels yvl
            - Class weights cw (to rebalance losses)
        """
        X_list, y_list = [], []

        # Loop over class definitions in config
        for cls_idx, class_def in enumerate(self.mva_cfg.classes):
            # Handle plain string: single-process class like "wjets"
            if isinstance(class_def, str):
                class_name = class_def
                process_list = [class_def]

            # Handle dictionary: merged-process class like {"ttbar": ["ttbar_semilep", ...]}
            elif isinstance(class_def, dict):
                # Assumes only one key-value pair per dict
                class_name, process_list = next(iter(class_def.items()))

            else:
                raise TypeError(f"Invalid class definition: {class_def}")

            # Directly get entries for this class name from the training data
            all_entries = events_per_process.get(class_name, [])

            if not all_entries:
                logger.warning(f"No events found for MVA class '{class_name}'. Skipping.")
                continue

            # For each (object_copies, n_events) pair in this class
            for obj_copies, n_events in all_entries:
                # Extract feature matrix from object dictionaries
                X = self._extract_features(obj_copies, self.mva_cfg.features)

                # Generate label array for this batch of size n_events
                y = self._make_labels(n_events, class_name, self.mva_cfg.classes)

                print(X.shape, y.shape, class_name)
                # Append to full dataset lists
                X_list.append(X)
                y_list.append(y)

        # Sanity check: ensure we have data
        if not X_list or not y_list:
            raise ValueError("No valid training data found for any class.")

        # Stack all batches into unified arrays
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Step 2: balance the dataset (e.g. by undersampling/oversampling)
        Xb, yb, cw = self._balance_dataset(X, y)

        print(Xb.shape, yb.shape)

        # Step 3: train/validation split
        Xtr, Xvl, ytr, yvl = self.split_train_test(Xb, yb)

        return Xtr, ytr, Xvl, yvl, cw

    @abstractmethod
    def init_network(self) -> None:
        """
        Set up the network structure:
        - For JAX: initialize weight and bias parameters
        - For TF/Keras: build and compile tf.keras.Sequential model
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        inputs: Union[jnp.ndarray, Any],
        targets: Union[jnp.ndarray, Any],
        val_inputs: Optional[Union[jnp.ndarray, Any]] = None,
        val_targets: Optional[Union[jnp.ndarray, Any]] = None
    ) -> Any:
        """
        Perform training over the given data.

        Args:
            inputs: training features
            targets: training labels
            val_inputs: optional validation features
            val_targets: optional validation labels

        Returns:
            For JAX: dict of trained parameters
            For TF/Keras: trained tf.keras.Model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        inputs: Union[jnp.ndarray, Any],
        **kwargs
    ) -> Any:
        """
        Compute model outputs for new data.

        Args:
            inputs: feature array for inference
            **kwargs: additional framework-specific args (e.g., batch_size)

        Returns:
            Predicted outputs (logits/probabilities)
        """
        raise NotImplementedError

# ----------------------------------------------------------------------------
# JAXNetwork implementation
# ----------------------------------------------------------------------------
class JAXNetwork(BaseNetwork):
    """
    JAX-based MVA using manual parameter management and gradient updates.
    """

    def init_network(self) -> None:
        """
        Initialize weight and bias parameters for each layer.

        Uses a seed derived from the learning rate for reproducibility.
        """
        # Build list of layer dimensions: input + each hidden/output size
        dims = [len(self.mva_cfg.features)] + [layer.ndim for layer in self.mva_cfg.layers]
        # mo:: key implementation different from lino code generates slightly different results
        # check jupyer nb for linos implementation to debug!
        # Derive RNG seed from random_state configuration variable
        seed_int = int(self.mva_cfg.random_state)
        rng_key = random.PRNGKey(seed_int)
        # One key per layer
        keys = random.split(rng_key, 1 * len(self.mva_cfg.layers))
        #keys = random.split(keys[2], 6)  # split each key into two for weights/biases
        # Allocate and store parameters
        for idx, layer_cfg in enumerate(self.mva_cfg.layers):
            w_key, b_key = random.split(keys[idx], 2)
            #w_key, b_key = keys[2*idx], keys[2*idx + 1]
            in_dim, out_dim = dims[idx], dims[idx + 1]
            # Weights: small normal initialization
            self.parameters[f"__NN_{self.name}_{layer_cfg.weights}"] = random.normal(w_key, (in_dim, out_dim)) * 0.1
            # Biases: zeros
            self.parameters[f"__NN_{self.name}_{layer_cfg.bias}"]    = jnp.zeros(out_dim)


    def _forward_pass(self, params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute network output by sequentially applying each layer's activation.

        Args:
            x: input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size,) or (batch_size, output_dim) squeezed
        """
        h = x
        for layer_cfg in self.mva_cfg.layers:

            w = params[f"__NN_{self.name}_{layer_cfg.weights}"]
            b = params[f"__NN_{self.name}_{layer_cfg.bias}"]
            h = layer_cfg.activation(h, w, b)
        return h.squeeze()

    def _compute_loss(self, params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate configured loss function on predictions vs.
        true labels.
        """
        preds = self._forward_pass(params, x)
        return self.mva_cfg.loss(preds, y)

    def _compute_accuracy(self, params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute binary classification accuracy using threshold 0.
        """
        logits = self._forward_pass(params, x)
        predictions = (logits > 0).astype(jnp.float32)
        return jnp.mean(predictions == y)

    def _update_step(self, params, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Perform a single gradient descent update over a batch.

        Returns:
            Scalar loss value for the batch.
        """
        lr = self.mva_cfg.learning_rate
        loss_val, grads = value_and_grad(self._compute_loss)(params, x, y)
        # Update each parameter with gradient
        new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        return new_params, loss_val

    def train(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        val_inputs: Optional[jnp.ndarray] = None,
        val_targets: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Training loop with optional mini-batching and validation split.

        Respects config: epochs, batch_size, validation_split, log_interval.
        """
        # Load training settings
        epochs = getattr(self.mva_cfg, 'epochs', 1000)
        batch_size = getattr(self.mva_cfg, 'batch_size', None)
        val_split = getattr(self.mva_cfg, 'validation_split', 0.0)
        log_interval = getattr(self.mva_cfg, 'log_interval', 100)

        # JIT-compile the update step
        self._update_step = jit(self._update_step)


        # Split data into training/validation if requested
        num_samples = inputs.shape[0]
        if val_split > 0:
            split_idx = int(num_samples * (1 - val_split))
            train_x, train_y = inputs[:split_idx], targets[:split_idx]
            valid_x, valid_y = inputs[split_idx:], targets[split_idx:]
        else:
            train_x, train_y = inputs, targets
            valid_x = valid_y = None

        # Iterate epochs
        params = self.parameters
        for epoch in range(1, epochs + 1):
            # Shuffle dataset each epoch for SGD
            rng = random.PRNGKey(epoch)
            perm = random.permutation(rng, train_x.shape[0])
            x_shuf = train_x[perm]
            y_shuf = train_y[perm]

            # Full-batch or mini-batch updates
            if batch_size is None:
                params, loss_val = self._update_step(params, x_shuf, y_shuf)
            else:
                loss_val = 0.0
                for start in range(0, x_shuf.shape[0], batch_size):
                    end = start + batch_size
                    params, loss_val = self._update_step(params, x_shuf[start:end], y_shuf[start:end])

            # Logging progress
            if epoch % log_interval == 0 or epoch == epochs:
                train_acc = self._compute_accuracy(params, train_x, train_y)
                msg = f"{self.mva_cfg.name} | Epoch {epoch}: loss={loss_val:.4f}, acc={train_acc:.4f}"
                if valid_x is not None:
                    val_acc = self._compute_accuracy(params, valid_x, valid_y)
                    msg += f", val_acc={val_acc:.4f}"
                print(msg)

        self.parameters = params

        return self.parameters

    def predict(
        self,
        inputs: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """
        Run a forward pass to compute outputs for new inputs.

        Args:
            inputs: array of shape (num_examples, input_dim)
            **kwargs: ignored for JAX

        Returns:
            Array of predictions or logits.
        """
        return self._forward_pass(self.parameters, inputs)

# ----------------------------------------------------------------------------
# TFNetwork implementation
# ----------------------------------------------------------------------------
class TFNetwork(BaseNetwork):
    """
    TensorFlow/Keras-based MVA leveraging built-in .fit() and .predict().
    """

    def init_network(self) -> None:
        """
        Construct and compile a Keras Sequential model based on layer configs.
        """
        input_dim = len(self.mva_cfg.features)
        keras_layers = []

        # Build dense layers in sequence
        for idx, layer_cfg in enumerate(self.mva_cfg.layers):
            layer_args: Dict[str, Any] = {
                'units': layer_cfg.ndim,
                'activation': (
                    layer_cfg.activation.value
                    if hasattr(layer_cfg.activation, 'value') else layer_cfg.activation
                )
            }
            if idx == 0:
                # Only first layer needs input_shape
                layer_args['input_shape'] = (input_dim,)
            keras_layers.append(tf.keras.layers.Dense(**layer_args))

        # Assemble model
        self.model = tf.keras.Sequential(keras_layers)
        # Compile with specified optimizer, loss, and metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.mva_cfg.learning_rate
            ),
            loss=self.mva_cfg.loss,
            metrics=['accuracy']
        )

    def train(
        self,
        inputs: Any,
        targets: Any,
        val_inputs: Optional[Any] = None,
        val_targets: Optional[Any] = None
    ) -> tf.keras.Model:
        """
        Train the Keras model using .fit(), with config-driven parameters.
        """
        # Extract training params
        epochs = getattr(self.mva_cfg, 'epochs', 50)
        batch_size = getattr(self.mva_cfg, 'batch_size', 32)
        val_split = getattr(self.mva_cfg, 'validation_split', 0.0)

        fit_kwargs = {
            'x': inputs,
            'y': targets,
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': val_split,
            'verbose': 1
        }
        # If explicit validation data provided, override split
        if val_inputs is not None and val_targets is not None:
            fit_kwargs.pop('validation_split')
            fit_kwargs['validation_data'] = (val_inputs, val_targets)

        self.model.fit(**fit_kwargs)
        return self.model

    def predict(
        self,
        inputs: Any,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Generate predictions using the trained Keras model.

        Args:
            inputs: array-like features
            batch_size: optional batch size for prediction

        Returns:
            Numpy array of model outputs.
        """
        return self.model.predict(inputs, batch_size=batch_size)
