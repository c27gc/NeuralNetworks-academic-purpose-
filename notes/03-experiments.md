# 03 · Designing an experiment

## Start with a falsifiable expectation

Train a `2 → 8 tanh → 1 sigmoid` network on the four XOR truth-table rows.
With Adam, seed 42, and a learning rate of 0.03, the supplied tests require all
four decisions to be correct and the loss to be small. A linear hidden layer
cannot represent XOR, regardless of how many linear units it contains.

The laboratory's XOR quadrants dataset uses continuous inputs in $[-1,1]^2$.
Its label is one when the two coordinates have the same sign. The truth table
and the quadrant dataset are distinct experiments.

## Change one assumption at a time

| Experiment | Change | Observe |
| :--- | :--- | :--- |
| Capacity | Hidden width/depth | Shape and complexity of the boundary |
| Optimization | SGD versus Adam | Speed and smoothness of loss reduction |
| Step size | Learning rate | Slow progress, convergence, or instability |
| Representation | Tanh, sigmoid, ReLU, linear | Regions the network can separate |
| Initialization | Seed | Sensitivity to a different starting point |

Concentric circles ask the network to learn a closed boundary. Two spirals need
a more complex boundary and often more training. More capacity does not guarantee
better generalization.

## Separate fitting from evaluation

The laboratory reports training accuracy and training loss. For a generalization
experiment, generate a second dataset using a different seed:

```python
from lecture_nn.datasets import classification

train_x, train_y = classification("circles", samples=96, seed=42)
valid_x, valid_y = classification("circles", samples=96, seed=73)
history = model.fit(
    train_x, train_y, epochs=500, batch_size=16,
    validation_data=(valid_x, valid_y), patience=20,
)
print(model.evaluate(valid_x, valid_y))
```

Early stopping monitors validation loss and restores the best validation
parameters, then resets the optimizer moments. Keep a third, untouched test set
when using validation to choose the architecture or learning rate.

## Make comparisons reproducible

Record architecture, seed, dataset, optimizer, learning rate, batch size, and
number of epochs. Use identical saved parameters for Python/C++ comparisons;
their random generators and floating-point precision differ. Disable shuffling
when comparing individual optimizer updates across engines.

The `.nn` format starts with `LECTURE_NN 1`, followed by input width and layer
count, then loss/optimizer/rate. Each layer stores units, activation, bias flag,
row-major weights, and bias. Both loaders validate dimensions, finite parameters,
and trailing data. New format revisions must use a new version identifier.

The model file does not contain training data, optimizer moments, random state,
or dataset scaling. If inputs require normalization, apply the same transform at
inference and store its metadata alongside the model.
