"""A deliberately small dense-network API for experiments and lecture notes.

The implementation keeps the data model explicit: batches are represented as
``list[list[float]]`` where the outer list contains samples and the inner list
contains features.  This makes the examples readable without requiring a
numerical runtime.
"""

from __future__ import annotations

from math import exp
from typing import Iterable, List, Optional, Sequence


Vector = List[float]
Matrix = List[Vector]


def _activation(name: str, value: float) -> float:
    if name == "linear":
        return value
    if name == "sigmoid":
        # Avoid overflow for large negative values in small teaching examples.
        if value >= 0:
            z = exp(-value)
            return 1.0 / (1.0 + z)
        z = exp(value)
        return z / (1.0 + z)
    if name == "relu":
        return max(0.0, value)
    raise ValueError(f"Unsupported activation: {name!r}")


class Input:
    """Describe the number of features expected by a network."""

    def __init__(self, number_of_inputs: int):
        if number_of_inputs <= 0:
            raise ValueError("number_of_inputs must be positive")
        self.number_of_inputs = number_of_inputs


class Dense:
    """A fully-connected layer with explicit parameters."""

    def __init__(
        self,
        units: int,
        activation: str = "linear",
        use_bias: bool = True,
    ):
        if units <= 0:
            raise ValueError("units must be positive")
        _activation(activation, 0.0)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.input_units: Optional[int] = None
        self.weights: Matrix = []
        self.bias: Vector = []

    def build(self, input_units: int) -> None:
        if input_units <= 0:
            raise ValueError("input_units must be positive")
        self.input_units = input_units
        self.weights = [[0.0 for _ in range(self.units)] for _ in range(input_units)]
        self.bias = [0.0 for _ in range(self.units)]

    def set_weights(self, weights: Sequence[Sequence[float]], bias: Optional[Sequence[float]] = None) -> None:
        rows = [list(map(float, row)) for row in weights]
        if not rows or any(len(row) != self.units for row in rows):
            raise ValueError("weights must have shape (input_units, units)")
        if bias is None:
            values = [0.0] * self.units
        else:
            values = list(map(float, bias))
            if len(values) != self.units:
                raise ValueError("bias must have one value per unit")
        self.input_units = len(rows)
        self.weights = rows
        self.bias = values

    def forward(self, batch: Sequence[Sequence[float]]) -> Matrix:
        if self.input_units is None:
            raise RuntimeError("layer must be built before it is evaluated")
        result: Matrix = []
        for sample in batch:
            if len(sample) != self.input_units:
                raise ValueError("sample width does not match layer input")
            values = []
            for unit in range(self.units):
                total = self.bias[unit] if self.use_bias else 0.0
                total += sum(sample[index] * self.weights[index][unit] for index in range(self.input_units))
                values.append(_activation(self.activation, total))
            result.append(values)
        return result


class Sequential:
    """A compact sequential model with a Keras-like surface."""

    def __init__(self, input_shape: Optional[int] = None):
        self.layers: List[Dense] = []
        self.input_shape = input_shape
        self.compiled = False

    def add(self, layer: Dense) -> None:
        if not isinstance(layer, Dense):
            raise TypeError("Sequential.add expects a Dense layer")
        self.layers.append(layer)

    def compile(self, loss: str = "mse", optimizer: str = "sgd") -> None:
        if loss != "mse":
            raise ValueError("Only mse is available in the reference implementation")
        if optimizer not in {"sgd", "adam"}:
            raise ValueError("optimizer must be 'sgd' or 'adam'")
        if not self.layers:
            raise ValueError("a model needs at least one layer")
        input_units = self.input_shape
        for layer in self.layers:
            if input_units is None:
                if layer.input_units is None:
                    raise ValueError("input_shape is required for unconfigured layers")
                input_units = layer.input_units
            elif layer.input_units is None:
                layer.build(input_units)
            elif layer.input_units != input_units:
                raise ValueError("layer input width does not match the previous layer")
            input_units = layer.units
        self.compiled = True

    def predict(self, batch: Iterable[Iterable[float]]) -> Matrix:
        if not self.compiled:
            raise RuntimeError("compile the model before calling predict")
        output = [list(map(float, sample)) for sample in batch]
        for layer in self.layers:
            output = layer.forward(output)
        return output
