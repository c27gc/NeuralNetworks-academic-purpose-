"""Dense networks with explicit forward, backward, and optimization passes."""

from __future__ import annotations

from math import exp, isfinite, log1p, sqrt, tanh
from pathlib import Path
from random import Random


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _matrix(values, width, name, allow_empty=False):
    rows = [list(map(float, row)) for row in values]
    if (not rows and not allow_empty) or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be a nonempty matrix with {width} columns")
    if any(not isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain finite numbers")
    return rows


def _activation(name, value):
    if name == "linear":
        return value
    if name == "sigmoid":
        z = exp(-abs(value))
        return 1.0 / (1.0 + z) if value >= 0 else z / (1.0 + z)
    if name == "relu":
        return max(0.0, value)
    if name == "tanh":
        return tanh(value)
    raise ValueError(f"unsupported activation: {name!r}")


def _derivative(name, output):
    if name == "sigmoid":
        return output * (1.0 - output)
    if name == "tanh":
        return 1.0 - output * output
    if name == "relu":
        return float(output > 0)
    return 1.0


class Input:
    """Input feature count, accepted by Sequential(input_shape=...)."""

    def __init__(self, number_of_inputs):
        _positive_integer(number_of_inputs, "number_of_inputs")
        self.number_of_inputs = number_of_inputs


class Dense:
    """Fully connected layer; weights have shape (input_features, units)."""

    def __init__(self, units, activation="linear", use_bias=True):
        _positive_integer(units, "units")
        _activation(activation, 0.0)
        self.units = units
        self.activation = activation
        self.use_bias = bool(use_bias)
        self.input_units = None
        self.weights = []
        self.bias = []

    def build(self, input_units, rng=None):
        _positive_integer(input_units, "input_units")
        rng = rng if rng is not None else Random(0)
        limit = sqrt(6.0 / (input_units + self.units))
        self.set_weights([[rng.uniform(-limit, limit) for _ in range(self.units)]
                          for _ in range(input_units)])

    def set_weights(self, weights, bias=None):
        rows = _matrix(weights, self.units, "weights")
        values = [0.0] * self.units if bias is None else list(map(float, bias))
        if len(values) != self.units or any(not isfinite(v) for v in values):
            raise ValueError("bias must have one finite value per unit")
        self.input_units = len(rows)
        self.weights = rows
        self.bias = values if self.use_bias else [0.0] * self.units

    def _affine(self, batch):
        return [[sum(x * row[j] for x, row in zip(sample, self.weights))
                 + (self.bias[j] if self.use_bias else 0.0)
                 for j in range(self.units)] for sample in batch]

    def forward(self, batch):
        if self.input_units is None:
            raise RuntimeError("build the layer before evaluation")
        rows = _matrix(batch, self.input_units, "inputs", allow_empty=True)
        return [[_activation(self.activation, z) for z in row] for row in self._affine(rows)]


class Sequential:
    """A trainable multilayer perceptron using only the Python standard library."""

    def __init__(self, input_shape=None, seed=42):
        if isinstance(input_shape, Input):
            input_shape = input_shape.number_of_inputs
        if input_shape is not None:
            _positive_integer(input_shape, "input_shape")
        self.layers = []
        self.input_shape = input_shape
        self.seed = seed
        self.compiled = False
        self._rng = Random(seed)
        self._moments = []
        self._step = 0

    def add(self, layer):
        if not isinstance(layer, Dense):
            raise TypeError("Sequential.add expects a Dense layer")
        if any(layer is existing for existing in self.layers):
            raise ValueError("a layer instance cannot be reused in the same model")
        self.layers.append(layer)
        self.compiled = False

    def compile(self, loss="mse", optimizer="sgd", learning_rate=0.01):
        self.compiled = False
        if loss not in {"mse", "binary_crossentropy"}:
            raise ValueError("loss must be 'mse' or 'binary_crossentropy'")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if not isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not self.layers:
            raise ValueError("a model needs at least one layer")
        width = self.input_shape or self.layers[0].input_units
        if width is None:
            raise ValueError("input_shape is required for unconfigured layers")
        self.input_shape = width
        for layer in self.layers:
            if layer.input_units is None:
                layer.build(width, self._rng)
            elif layer.input_units != width:
                raise ValueError("layer input width does not match the previous layer")
            width = layer.units
        if loss == "binary_crossentropy" and self.layers[-1].activation != "sigmoid":
            raise ValueError("binary_crossentropy requires a sigmoid output")
        self.loss, self.optimizer, self.learning_rate = loss, optimizer, float(learning_rate)
        self._moments = [[[0.0] * (len(layer.weights) * layer.units + layer.units)
                          for _ in range(2)] for layer in self.layers]
        self._step = 0
        self.compiled = True

    def _require_compiled(self):
        if not self.compiled:
            raise RuntimeError("compile the model first")

    def _data(self, inputs, targets):
        self._require_compiled()
        x = _matrix(inputs, self.input_shape, "inputs")
        y = _matrix(targets, self.layers[-1].units, "targets")
        if len(x) != len(y):
            raise ValueError("inputs and targets must have the same sample count")
        if self.loss == "binary_crossentropy" and any(v < 0 or v > 1 for row in y for v in row):
            raise ValueError("binary targets must be in [0, 1]")
        return x, y

    def _forward(self, inputs):
        activations, logits = [inputs], []
        for layer in self.layers:
            z = layer._affine(activations[-1])
            logits.append(z)
            activations.append([[_activation(layer.activation, v) for v in row] for row in z])
        return activations, logits

    def predict(self, batch):
        self._require_compiled()
        x = _matrix(batch, self.input_shape, "inputs", allow_empty=True)
        return self._forward(x)[0][-1]

    def evaluate(self, inputs, targets):
        x, y = self._data(inputs, targets)
        activations, logits = self._forward(x)
        prediction = activations[-1]
        count = len(y) * self.layers[-1].units
        if self.loss == "mse":
            return {"loss": sum((p - t) ** 2 for row, truth in zip(prediction, y)
                                for p, t in zip(row, truth)) / count}
        # Softplus from logits retains the correct loss even at saturation.
        loss = sum(max(z, 0) - z * t + log1p(exp(-abs(z)))
                   for row, truth in zip(logits[-1], y) for z, t in zip(row, truth)) / count
        correct = sum((p >= 0.5) == (t >= 0.5) for row, truth in zip(prediction, y)
                      for p, t in zip(row, truth))
        return {"loss": loss, "accuracy": correct / count}

    def gradients(self, inputs, targets):
        """Return averaged analytical gradients without updating parameters."""
        x, y = self._data(inputs, targets)
        activations, _ = self._forward(x)
        count = len(y) * self.layers[-1].units
        delta = [[(p - t) / count * (1 if self.loss == "binary_crossentropy" else
                  2 * _derivative(self.layers[-1].activation, p))
                  for p, t in zip(row, truth)] for row, truth in zip(activations[-1], y)]
        result = [None] * len(self.layers)
        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            previous = activations[index]
            result[index] = {
                "weights": [[sum(sample[i] * error[j] for sample, error in zip(previous, delta))
                             for j in range(layer.units)] for i in range(layer.input_units)],
                "bias": [sum(row[j] for row in delta) if layer.use_bias else 0.0
                         for j in range(layer.units)],
            }
            if index:
                delta = [[sum(error[j] * layer.weights[i][j] for j in range(layer.units))
                          * _derivative(self.layers[index - 1].activation, sample[i])
                          for i in range(layer.input_units)]
                         for sample, error in zip(previous, delta)]
        return result

    def _update(self, gradients):
        self._step += 1
        for index, (layer, gradient) in enumerate(zip(self.layers, gradients)):
            values = [v for row in layer.weights for v in row] + layer.bias
            grads = [v for row in gradient["weights"] for v in row] + gradient["bias"]
            m, v = self._moments[index]
            for j, g in enumerate(grads):
                if self.optimizer == "adam":
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    g = (m[j] / (1 - 0.9 ** self._step)) / (sqrt(v[j] / (1 - 0.999 ** self._step)) + 1e-8)
                values[j] -= self.learning_rate * g
                if not isfinite(values[j]):
                    raise FloatingPointError("nonfinite parameters; reduce the learning rate")
            split = layer.input_units * layer.units
            layer.weights = [values[i:i + layer.units] for i in range(0, split, layer.units)]
            layer.bias = values[split:]

    def fit(self, inputs, targets, epochs=100, batch_size=32, shuffle=True,
            validation_data=None, patience=None):
        """Mini-batch optimization; early stopping restores best validation weights."""
        _positive_integer(epochs, "epochs")
        _positive_integer(batch_size, "batch_size")
        x, y = self._data(inputs, targets)
        validation = self._data(*validation_data) if validation_data is not None else None
        if patience is not None:
            _positive_integer(patience, "patience")
            if validation is None:
                raise ValueError("patience requires validation_data")
        history = {"loss": [], "val_loss": []}
        best_loss, stale, best = float("inf"), 0, None
        indices = list(range(len(x)))
        for _ in range(epochs):
            if shuffle:
                self._rng.shuffle(indices)
            for start in range(0, len(x), batch_size):
                selected = indices[start:start + batch_size]
                self._update(self.gradients([x[i] for i in selected], [y[i] for i in selected]))
            history["loss"].append(self.evaluate(x, y)["loss"])
            if validation is not None:
                value = self.evaluate(*validation)["loss"]
                history["val_loss"].append(value)
                if patience is not None:
                    if value < best_loss:
                        best_loss, stale = value, 0
                        best = [([row[:] for row in layer.weights], layer.bias[:]) for layer in self.layers]
                    else:
                        stale += 1
                        if stale >= patience:
                            break
        if best is not None:
            for layer, (weights, bias) in zip(self.layers, best):
                layer.set_weights(weights, bias)
            self.compile(self.loss, self.optimizer, self.learning_rate)
        return history

    def summary(self):
        self._require_compiled()
        lines = ["Layer              Shape       Activation    Parameters"]
        total = 0
        for i, layer in enumerate(self.layers):
            parameters = layer.input_units * layer.units + (layer.units if layer.use_bias else 0)
            total += parameters
            lines.append(f"Dense {i:<12} {layer.input_units:>3} -> {layer.units:<3} {layer.activation:<13} {parameters}")
        return "\n".join(lines + [f"Total parameters: {total}"])

    def save(self, path):
        """Write portable inference parameters. Optimizer moments are not saved."""
        self._require_compiled()
        lines = ["LECTURE_NN 1", f"{self.input_shape} {len(self.layers)}",
                 f"{self.loss} {self.optimizer} {self.learning_rate:.17g}"]
        for layer in self.layers:
            lines += [f"{layer.units} {layer.activation} {int(layer.use_bias)}",
                      " ".join(f"{v:.17g}" for row in layer.weights for v in row),
                      " ".join(f"{v:.17g}" for v in layer.bias)]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        tokens = iter(Path(path).read_text(encoding="utf-8").split())
        try:
            if next(tokens) != "LECTURE_NN" or next(tokens) != "1":
                raise ValueError("unsupported model format")
            width, count = int(next(tokens)), int(next(tokens))
            if not 0 < width <= 4096 or not 0 < count <= 128:
                raise ValueError("invalid model dimensions")
            loss, optimizer, rate = next(tokens), next(tokens), float(next(tokens))
            model = cls(width)
            for _ in range(count):
                units, activation, flag = int(next(tokens)), next(tokens), next(tokens)
                if not 0 < units <= 4096 or width * units > 1_000_000 or flag not in {"0", "1"}:
                    raise ValueError("invalid layer dimensions or bias flag")
                layer = Dense(units, activation, flag == "1")
                layer.set_weights([[float(next(tokens)) for _ in range(units)] for _ in range(width)],
                                  [float(next(tokens)) for _ in range(units)])
                model.add(layer)
                width = units
            if next(tokens, None) is not None:
                raise ValueError("unexpected data after model")
            model.compile(loss, optimizer, rate)
            return model
        except (StopIteration, RuntimeError) as error:
            raise ValueError("truncated model file") from error
