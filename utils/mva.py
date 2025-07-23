# =============================================================================
# Imports
# =============================================================================

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from equinox import filter_jit
from jax import random, value_and_grad, jit
from sklearn.model_selection import train_test_split

from analysis.base import Analysis


# =============================================================================
# Abstract Base Class
# =============================================================================

class BaseNetwork(ABC, Analysis):
    """
    Abstract base class for MVA networks (e.g. JAX, TensorFlow).
    Handles data preparation, feature extraction, class balancing,
    and defines abstract methods for training and prediction.
    """

    def __init__(self, mva_cfg: Dict[str, Any]) -> None:
        """
        Initialize the network with a configuration dictionary.

        Args:
            mva_cfg (Dict[str, Any]): Pydantic config specifying network architecture
                and training options.
        """
        self.mva_cfg = mva_cfg
        self.name = mva_cfg.name
        self.parameters: Dict[str, Any] = {}  # Used by JAX models
        self.model: Any = None  # Used by TensorFlow models

    def _split_train_test(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split features and labels into training and validation sets.

        Args:
            features (np.ndarray): Full feature matrix of shape (n_samples, n_features).
            labels (np.ndarray): Corresponding class labels of shape (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Training and validation splits: (X_train, X_val, y_train, y_val)
        """
        test_size = self.mva_cfg.validation_split
        random_state = self.mva_cfg.random_state

        # Stratify if balancing is active
        stratify = labels if self.mva_cfg.balance_strategy != "none" else None

        return train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )

    def _extract_features(
        self,
        object_collections: Dict[str, Any],
        feature_configs: List[Any]
    ) -> np.ndarray:
        """
        Compute and stack all features for one event batch.

        Args:
            object_collections (Dict[str, Any]): Dictionary of NanoAOD-style
                objects per event.
            feature_configs (List[Any]): List of feature configuration objects,
            each defining a callable and its required input objects.

        Returns:
            np.ndarray: 2D array of shape (n_events, n_features) with stacked features.
        """
        feature_arrays = []

        for feature_cfg in feature_configs:
            # Extract input arguments for the feature from the object collections
            args = self._get_function_arguments(feature_cfg.use, object_collections)

            # Compute raw feature values
            feature_values = feature_cfg.function(*args)

            # Optionally apply scaling (e.g., normalization)
            if feature_cfg.scale is not None:
                feature_values = feature_cfg.scale(feature_values)

            # Convert to NumPy array
            feature_arrays.append(np.asarray(feature_values))

        # Stack all features as columns into a matrix
        return np.stack(feature_arrays, axis=1).astype(float)

    def _make_labels(
        self,
        num_events: int,
        process_name: str,
        class_definitions: list[Union[str, dict[str, tuple[str]]]],
    ) -> np.ndarray:
        """
        Generate integer class labels for events belonging to a process.

        Args:
            num_events (int): Number of events to label.
            process_name (str): Name of the process (e.g., 'ttbar').
            class_definitions (list): List of class definitions, each a string or dict
                mapping class name to one or more process names.

        Returns:
            np.ndarray: Integer label array of shape (num_events,) filled with
                class index.

        Raises:
            RuntimeError: If the process is not matched in any class definition.
        """
        for class_index, class_entry in enumerate(class_definitions):
            # Simple class: match process directly
            if isinstance(class_entry, str) and process_name == class_entry:
                return np.full(num_events, class_index, dtype=int)

            # Merged class: match dictionary key
            elif isinstance(class_entry, dict):
                class_name, _ = next(iter(class_entry.items()))
                if process_name == class_name:
                    return np.full(num_events, class_index, dtype=int)

        # If not matched, raise an error
        raise RuntimeError(
            f"Process '{process_name}' not found in any class definition: \
                {class_definitions}"
        )

    def _balance_dataset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
        """
        Balance the dataset using undersampling, oversampling, or class reweighting.

        Args:
            features (np.ndarray): Input feature matrix of shape (n_samples,n_features).
            labels (np.ndarray): Integer class labels of shape (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
                Balanced feature matrix, balanced label array,
                and optionally a dictionary of class weights.
        """
        strategy = self.mva_cfg.balance_strategy
        random_state = self.mva_cfg.random_state

        # No balancing: return input as-is
        if strategy == "none":
            return features, labels, None

        # Create reproducible RNG and compute class counts
        rng = np.random.RandomState(random_state)
        class_counts = Counter(labels)
        unique_labels = sorted(class_counts)

        if strategy in ("undersample", "oversample"):
            # Determine how many samples per class to retain/generate
            target = (
                min(class_counts.values()) if strategy == "undersample"
                else max(class_counts.values())
            )

            balanced_features = []
            balanced_labels = []

            # Resample each class to the target count
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]

                # Oversampling allows replacement; undersampling does not
                resampled_indices = rng.choice(
                    label_indices,
                    size=target,
                    replace=(strategy == "oversample")
                )

                balanced_features.append(features[resampled_indices])
                balanced_labels.append(labels[resampled_indices])

            # Concatenate all resampled classes and shuffle
            features_stacked = np.concatenate(balanced_features, axis=0)
            labels_stacked = np.concatenate(balanced_labels, axis=0)
            permuted_indices = rng.permutation(len(labels_stacked))

            return (
                features_stacked[permuted_indices],
                labels_stacked[permuted_indices],
                None
            )

        if strategy == "class_weight":
            # Compute inverse frequency weights for use in the loss function
            total_samples = float(len(labels))
            class_weights = {
                label: total_samples / (len(unique_labels) * class_counts[label])
                for label in unique_labels
            }
            return features, labels, class_weights

        # If we reach here, strategy was invalid
        raise ValueError(f"Unknown balance_strategy={strategy}")

    def prepare_inputs(
        self,
        events_per_process: dict[str, list[tuple[dict, int]]],
    ) -> tuple[np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               Optional[Dict[int, float]]]:
        """
        Prepare training/validation data from event-level inputs.

        Args:
            events_per_process (dict):
                Mapping from process name to a list of tuples:
                (object_collections: dict, n_events: int)

        Returns:
            tuple:
                - features_train (np.ndarray)
                - labels_train (np.ndarray)
                - features_val (np.ndarray)
                - labels_val (np.ndarray)
                - class_weights (Optional[dict[int, float]])
        """
        all_features = []
        all_labels = []

        for class_def in self.mva_cfg.classes:
            if isinstance(class_def, str):
                class_name = class_def
                process_list = [class_def]
            elif isinstance(class_def, dict):
                class_name, process_list = next(iter(class_def.items()))
            else:
                raise TypeError(f"Invalid class definition: {class_def}")

            entries = events_per_process.get(class_name, [])
            if not entries:
                logger.warning(f"No events found for MVA class '{class_name}'. \
                               Skipping.")
                continue

            for object_dict, num_events in entries:
                feature_matrix = self._extract_features(object_dict,
                                                        self.mva_cfg.features)
                label_array = self._make_labels(num_events,
                                                class_name,
                                                self.mva_cfg.classes)

                all_features.append(feature_matrix)
                all_labels.append(label_array)

        if not all_features or not all_labels:
            raise ValueError("No valid training data found for any class.")

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        features_balanced, labels_balanced, class_weights = (
            self._balance_dataset(features, labels))

        X_train, X_val, y_train, y_val = (
            self._split_train_test(features_balanced, labels_balanced))

        return X_train, y_train, X_val, y_val, class_weights

    @abstractmethod
    def init_network(self) -> None:
        """
        Initialize network architecture and parameters.

        Must be implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        training_inputs: Union[jnp.ndarray, Any],
        training_labels: Union[jnp.ndarray, Any],
        validation_inputs: Optional[Union[jnp.ndarray, Any]] = None,
        validation_labels: Optional[Union[jnp.ndarray, Any]] = None
    ) -> Any:
        """
        Train the model.

        Args:
            inputs (array): Training feature array.
            targets (array): Training labels.
            val_inputs (Optional[array]): Validation features.
            val_targets (Optional[array]): Validation labels.

        Returns:
            Trained model or parameter state.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        inputs: Union[jnp.ndarray, Any],
        **kwargs
    ) -> Any:
        """
        Perform inference with trained model.

        Args:
            inputs (array): Feature input for prediction.
            **kwargs: Optional keyword args for inference.

        Returns:
            Prediction results (array or backend-specific output).
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------
# JAXNetwork implementation
# ----------------------------------------------------------------------------

class JAXNetwork(BaseNetwork):
    """
    JAX-based MVA implementation using explicit parameter management and manual gradient
    descent updates.
    """

    def init_network(self) -> None:
        """
        Initialize weight and bias parameters for each layer using
        random number generator.

        Uses the configured random seed for reproducibility.
        """
        layer_dimensions = [len(self.mva_cfg.features)]
                            + [layer.ndim for layer in self.mva_cfg.layers]
        seed_value = int(self.mva_cfg.random_state)
        rng_key = random.PRNGKey(seed_value)
        layer_keys = random.split(rng_key, len(self.mva_cfg.layers))

        for layer_index, layer_config in enumerate(self.mva_cfg.layers):
            weight_key, bias_key = random.split(layer_keys[layer_index], 2)
            input_dim, output_dim = (layer_dimensions[layer_index],
                                     layer_dimensions[layer_index + 1])

            weight_name = f"__NN_{self.name}_{layer_config.weights}"
            bias_name = f"__NN_{self.name}_{layer_config.bias}"

            self.parameters[weight_name] = random.normal(weight_key,
                                                         (input_dim, output_dim)) * 0.1
            self.parameters[bias_name] = jnp.zeros(output_dim)

    def forward_pass(self, parameters: dict, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Compute network output by sequentially applying each layer's activation.

        Args:
            parameters (dict): Dictionary of trained weights and biases.
            inputs (jnp.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            jnp.ndarray: Output predictions of shape (n_samples,).
        """
        activations = inputs
        for layer_config in self.mva_cfg.layers:
            weights = parameters[f"__NN_{self.name}_{layer_config.weights}"]
            biases = parameters[f"__NN_{self.name}_{layer_config.bias}"]
            activations = layer_config.activation(activations, weights, biases)
        return activations.squeeze()

    def compute_loss(self,
                     parameters: dict,
                     inputs: jnp.ndarray,
                     targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute scalar loss value from predictions and true labels.

        Args:
            parameters (dict): Network parameters.
            inputs (jnp.ndarray): Input features.
            targets (jnp.ndarray): True labels.

        Returns:
            jnp.ndarray: Scalar loss value.
        """
        predictions = self.forward_pass(parameters, inputs)
        return self.mva_cfg.loss(predictions, targets)

    def compute_accuracy(self,
                         parameters: dict,
                         inputs: jnp.ndarray,
                         targets: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute binary classification accuracy using a threshold at zero.

        Args:
            parameters (dict): Model parameters.
            inputs (jnp.ndarray): Input feature matrix.
            targets (jnp.ndarray): Ground truth labels.

        Returns:
            jnp.ndarray: Mean classification accuracy.
        """
        logits = self.forward_pass(parameters, inputs)
        predictions = (logits > 0).astype(jnp.float32)
        return jnp.mean(predictions == targets)

    def _update_step(self,
                     parameters: dict,
                     inputs: jnp.ndarray,
                     targets: jnp.ndarray
        ) -> tuple[dict, jnp.ndarray]:
        """
        Perform one step of gradient descent to update parameters.

        Args:
            parameters (dict): Current parameter dictionary.
            inputs (jnp.ndarray): Feature batch.
            targets (jnp.ndarray): Label batch.

        Returns:
            tuple: (updated parameters, loss value)
        """
        learning_rate = self.mva_cfg.learning_rate
        loss_value, gradients = value_and_grad(self.compute_loss)(parameters,
                                                                  inputs,
                                                                  targets)
        updated_parameters = jax.tree.map(lambda p, g: p - learning_rate * g,
                                          parameters, gradients)
        return updated_parameters, loss_value

    def train(
        self,
        training_inputs: jnp.ndarray,
        training_labels: jnp.ndarray,
        validation_inputs: Optional[jnp.ndarray] = None,
        validation_labels: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Train the model using optional mini-batching and validation.

        Args:
            training_inputs (jnp.ndarray): Training feature matrix.
            training_labels (jnp.ndarray): Training label vector.
            validation_inputs (Optional[jnp.ndarray]): Validation features.
            validation_labels (Optional[jnp.ndarray]): Validation labels.

        Returns:
            dict: Final trained parameter dictionary.
        """
        total_epochs = getattr(self.mva_cfg, 'epochs', 1000)
        batch_size = getattr(self.mva_cfg, 'batch_size', None)
        validation_fraction = getattr(self.mva_cfg, 'validation_split', 0.0)
        log_frequency = getattr(self.mva_cfg, 'log_interval', 100)

        self._update_step = jit(self._update_step)
        total_samples = training_inputs.shape[0]

        if validation_fraction > 0:
            split_index = int(total_samples * (1 - validation_fraction))
            train_x, train_y = (training_inputs[:split_index],
                                training_labels[:split_index])
            valid_x, valid_y = (training_inputs[split_index:],
                                training_labels[split_index:])
        else:
            train_x, train_y = training_inputs, training_labels
            valid_x = valid_y = None

        parameters = self.parameters
        for epoch in range(1, total_epochs + 1):
            rng_key = random.PRNGKey(epoch)
            shuffled_indices = random.permutation(rng_key, train_x.shape[0])
            shuffled_inputs = train_x[shuffled_indices]
            shuffled_labels = train_y[shuffled_indices]

            if batch_size is None:
                parameters, current_loss = self._update_step(parameters,
                                                             shuffled_inputs,
                                                             shuffled_labels)
            else:
                current_loss = 0.0
                for batch_start in range(0, shuffled_inputs.shape[0], batch_size):
                    batch_end = batch_start + batch_size
                    batch_inputs = shuffled_inputs[batch_start:batch_end]
                    batch_labels = shuffled_labels[batch_start:batch_end]
                    parameters, current_loss = self._update_step(parameters,
                                                                 batch_inputs,
                                                                 batch_labels)

            if epoch % log_frequency == 0 or epoch == total_epochs:
                training_accuracy = self.compute_accuracy(parameters, train_x, train_y)
                log_message = f"{self.mva_cfg.name} | \
                                Epoch {epoch}: loss={current_loss:.4f}, \
                                acc={training_accuracy:.4f}"
                if valid_x is not None:
                    validation_accuracy = self.compute_accuracy(parameters,
                                                                valid_x,
                                                                valid_y)
                    log_message += f", val_acc={validation_accuracy:.4f}"
                print(log_message)

        self.parameters = parameters
        return self.parameters

    def predict(
        self,
        inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Generate predictions from the trained network.

        Args:
            inputs (jnp.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            jnp.ndarray: Model output logits or probabilities.
        """
        return self.forward_pass(self.parameters, inputs)


# ----------------------------------------------------------------------------
# TFNetwork implementation
# ----------------------------------------------------------------------------
class TFNetwork(BaseNetwork):
    """
    TensorFlow/Keras-based MVA leveraging built-in .fit() and .predict().
    Implements network initialization, training, and prediction for Keras models.
    """

    def init_network(self) -> None:
        """
        Construct and compile a Keras Sequential model based on layer configuration.

        Uses the layer configuration from `mva_cfg` to define fully-connected layers,
        with activations and input shape for the first layer.
        """
        input_dimension = len(self.mva_cfg.features)
        keras_layers = []

        for layer_index, layer_config in enumerate(self.mva_cfg.layers):
            dense_layer_args: Dict[str, Any] = {
                'units': layer_config.ndim,
                'activation': (
                    layer_config.activation.value
                    if hasattr(layer_config.activation, 'value')
                    else layer_config.activation
                )
            }
            if layer_index == 0:
                dense_layer_args['input_shape'] = (input_dimension,)

            keras_layers.append(tf.keras.layers.Dense(**dense_layer_args))

        self.model = tf.keras.Sequential(keras_layers)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.mva_cfg.learning_rate),
            loss=self.mva_cfg.loss,
            metrics=['accuracy']
        )

    def train(
        self,
        training_inputs: Any,
        training_labels: Any,
        validation_inputs: Optional[Any] = None,
        validation_labels: Optional[Any] = None
    ) -> tf.keras.Model:
        """
        Train the Keras model using the .fit() method with optional validation data.

        Args:
            training_inputs (Any): Input training data.
            training_labels (Any): Training labels.
            validation_inputs (Optional[Any]): Validation feature array.
            validation_labels (Optional[Any]): Validation labels.

        Returns:
            tf.keras.Model: The trained Keras model instance.
        """
        total_epochs = getattr(self.mva_cfg, 'epochs', 50)
        batch_size = getattr(self.mva_cfg, 'batch_size', 32)
        validation_split_fraction = getattr(self.mva_cfg, 'validation_split', 0.0)

        training_arguments = {
            'x': training_inputs,
            'y': training_labels,
            'epochs': total_epochs,
            'batch_size': batch_size,
            'validation_split': validation_split_fraction,
            'verbose': 1
        }

        if validation_inputs is not None and validation_labels is not None:
            # Use explicit validation data instead of split
            training_arguments.pop('validation_split')
            training_arguments['validation_data'] = (validation_inputs,
                                                     validation_labels)

        self.model.fit(**training_arguments)
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
            inputs (Any): Input features to run inference on.
            batch_size (Optional[int]): Optional batch size for inference.
            **kwargs: Additional keyword arguments passed to Keras predict.

        Returns:
            Any: Prediction results from the model.
        """
        return self.model.predict(inputs, batch_size=batch_size, **kwargs)