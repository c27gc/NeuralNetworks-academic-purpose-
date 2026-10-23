# Neural Networks — Lecture Notes

[![CI](https://github.com/c27gc/NeuralNetworks-academic-purpose-/actions/workflows/ci.yml/badge.svg)](https://github.com/c27gc/NeuralNetworks-academic-purpose-/actions/workflows/ci.yml)

Compact lecture notes and reference implementations for building dense neural
networks from first principles.

The project brings together two small, readable implementations with the same
mental model:

- **Python** — a dependency-free `Sequential` / `Dense` API for experiments,
  equations, and notebooks.
- **C++17** — a typed implementation with CMake, CTest, and a compatibility
  bridge to the original neuron primitive.

The repository also contains the earlier Orion and C++ exercises that document
the progression of the ideas. They remain useful as historical reference
material; the tested entry points live under `python/` and `cpp/`.

## Quick start

### Python

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -t . -v
```

```python
from lecture_nn import Dense, Sequential

model = Sequential(input_shape=2)
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="mse", optimizer="sgd")
predictions = model.predict([[0.2, 0.8]])
```

### C++17

```bash
cmake -S cpp -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The C++ reference layer uses the historical `NeuronClass` as its sigmoid
primitive while keeping ownership, validation, and the public API modern.

## Layout

| Path | Purpose |
| --- | --- |
| `python/lecture_nn` | Small Python reference API |
| `python/tests` | Python unit tests |
| `cpp/include` | Public C++17 headers |
| `cpp/src` | C++ implementation and compatibility layer |
| `cpp/tests` | CTest executable |
| `orion/` | Earlier Python experiments |
| `C version/`, `Separate Classes/` | Earlier C++ exercises |

## Scope

These implementations are intentionally educational. They focus on making
forward propagation, activations, parameters, and model composition easy to
inspect. Training algorithms and tensor acceleration are left as natural
extensions for future lecture notes.
