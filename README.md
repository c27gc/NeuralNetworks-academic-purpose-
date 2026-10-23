# Neural Networks — Lecture Notes

[![Tests](https://github.com/c27gc/NeuralNetworks-academic-purpose-/actions/workflows/ci.yml/badge.svg)](https://github.com/c27gc/NeuralNetworks-academic-purpose-/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-80b7e3)
![C++](https://img.shields.io/badge/C%2B%2B-17-b4edbc)

**Build it. Train it. Understand every derivative.**

A hands-on study of dense neural networks: mathematical notes, trainable Python
and C++ engines, and a visual laboratory for exploring how networks learn.
Forward propagation, backpropagation, and optimization are written out explicitly.
Python uses only the standard library; C++ uses the standard library and CMake.

## Open the laboratory

```bash
PYTHONPATH=python python3 -m lecture_nn lab
```

Open **http://127.0.0.1:8765**. Choose XOR quadrants, concentric circles, or two
spirals; adjust the hidden layers, activation, optimizer, learning rate, and seed.
Train or pause the network while watching its decision boundary, loss curve,
architecture, and training accuracy update. Everything runs locally.

To use the same laboratory with either engine, build C++ and start it with:

```bash
PYTHONPATH=python python3 -m lecture_nn lab --cpp ./build/lecture_nn_model
```

The **Engine** selector switches between Python and C++17. C++ training runs in a
persistent worker, retaining optimizer state while the experiment is paused.

## What you can explore

| Capability | Python | C++17 |
| :--- | :---: | :---: |
| Composable `Sequential` / `Dense` networks | ✓ | ✓ |
| Linear, ReLU, sigmoid, and tanh activations | ✓ | ✓ |
| Xavier initialization and seeded experiments | ✓ | ✓ |
| Backpropagation with inspectable gradients | ✓ | ✓ |
| Mini-batch SGD and Adam | ✓ | ✓ |
| Mean squared error and binary cross-entropy | ✓ | ✓ |
| Portable model files | ✓ | ✓ |
| Numerical gradient checks | ✓ | ✓ |
| Validation loss and early stopping | ✓ | — |
| Interactive decision-boundary laboratory | ✓ | ✓ |

## Train your first network

```python
from lecture_nn import Dense, Sequential

x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

model = Sequential(input_shape=2, seed=42)
model.add(Dense(8, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", learning_rate=0.03)

history = model.fit(x, y, epochs=600, batch_size=4, shuffle=False)
print(model.summary())
print(model.evaluate(x, y))
print(model.predict(x))
model.save("xor.nn")
```

Run Python scripts with `PYTHONPATH=python`, or install the package in a virtual
environment to use it from any directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install ./python
.venv/bin/lecture-nn lab
```

For a ready-to-run classification experiment:

```bash
PYTHONPATH=python python3 -m lecture_nn train --dataset circles --epochs 120 --save circles.nn
```

The Python and C++ XOR tests require perfect classification and low loss, rather
than merely checking that training completes. Dataset metrics describe the
training samples; use separate validation data when evaluating generalization.

## C++ workflow

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/lecture_nn_xor xor.nn
```

The same model-building vocabulary is available in C++:

```cpp
#include <lecture_nn/lecture_nn.hpp>

lecture_nn::Sequential model(2, 42);
model.add(lecture_nn::Dense(8, "tanh"));
model.add(lecture_nn::Dense(1, "sigmoid"));
model.compile("binary_crossentropy", "adam", 0.03F);
model.fit({{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {{0}, {1}, {1}, {0}}, 800, 4, false);
model.save("xor.nn");
```

## Move models between languages

Both engines read and write the versioned `.nn` text format, with the same weight
layout, activation names, and loss definitions. Train in Python and predict in
C++, or load a C++ model with `Sequential.load("xor.nn")` in Python.

```bash
./build/lecture_nn_model predict model.nn inputs.csv
./build/lecture_nn_model train model.nn inputs.csv targets.csv trained.nn 100 16
```

CSV files contain numeric rows without headers. Prediction writes one output row
per sample. The C++ command trains in row order; the library API also supports
shuffling. Model files store architecture and parameters, **not optimizer moments
or random-generator state**. Loading starts a new optimization run.

Seeded runs are reproducible within each engine. Their random generators differ;
cross-language equivalence is checked using identical stored parameters and
unshuffled batches. C++ uses 32-bit floats and Python uses double precision, so
results are compared with numerical tolerances.

## Read the notes

1. [Forward propagation and shapes](notes/01-forward-propagation.md) — dense layers,
   nonlinearities, initialization, and objectives.
2. [Deriving backpropagation](notes/02-backpropagation.md) — gradients, reduction,
   numerical checks, SGD, and Adam.
3. [Designing an experiment](notes/03-experiments.md) — XOR, decision boundaries,
   validation, reproducibility, and model exchange.

## Verify the mathematics and behavior

```bash
PYTHONPATH=python python3 -m unittest discover -s python/tests -t . -v

# Include cross-language inference and optimizer comparisons:
LECTURE_NN_CPP="$PWD/build/lecture_nn_model" PYTHONPATH=python \
  python3 -m unittest python.tests.test_interop -v
```

Tests cover numerical gradients, XOR and regression convergence, saturated
cross-entropy, bias-free layers, malformed data, architecture validation,
serialization, HTTP training, and Python/C++ agreement after SGD and Adam updates.
C++ checks run in both Debug and Release builds.

## Project map

| Path | Contents |
| :--- | :--- |
| `python/lecture_nn/` | Training engine, experiment CLI, datasets, and laboratory |
| `python/tests/` | Unit, numerical, integration, and interoperability tests |
| `cpp/include/lecture_nn/` | Public C++ API |
| `cpp/src/` | Dense layers, optimizers, training, and model I/O |
| `cpp/examples/` | Executable XOR experiment |
| `cpp/tools/` | CSV prediction and training commands |
| `notes/` | Equations, derivations, and experimental methodology |

Designed for small, inspectable CPU experiments. The engines use dense scalar
loops; GPU kernels, automatic differentiation, and large-scale workloads are
outside the scope of these notes.
