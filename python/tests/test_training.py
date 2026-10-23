import math
import tempfile
import unittest
from pathlib import Path

from lecture_nn import Dense, Input, Sequential
from lecture_nn.datasets import classification


def network(activation="tanh", loss="mse", optimizer="sgd", bias=True):
    model = Sequential(Input(2), seed=7)
    first = Dense(3, activation, use_bias=bias)
    first.set_weights([[0.4, -0.3, 0.2], [0.1, 0.6, -0.5]], [0.2, 0.3, 0.1])
    model.add(first)
    last = Dense(2, "sigmoid" if loss == "binary_crossentropy" else "linear", use_bias=bias)
    last.set_weights([[0.5, -0.4], [-0.2, 0.7], [0.3, 0.2]], [0.1, -0.2])
    model.add(last)
    model.compile(loss, optimizer, learning_rate=0.03)
    return model


class GradientTests(unittest.TestCase):
    def test_every_parameter_matches_finite_differences(self):
        x, y = [[0.3, -0.2], [0.8, 0.4]], [[0.1, 0.8], [0.7, 0.2]]
        for activation in ("linear", "sigmoid", "tanh", "relu"):
            for loss in ("mse", "binary_crossentropy"):
                with self.subTest(activation=activation, loss=loss):
                    model = network(activation, loss)
                    gradients = model.gradients(x, y)
                    epsilon = 1e-5
                    for layer, gradient in zip(model.layers, gradients):
                        pairs = list(zip(layer.weights, gradient["weights"])) + [(layer.bias, gradient["bias"])]
                        for parameters, analytical in pairs:
                            for j, expected in enumerate(analytical):
                                saved = parameters[j]
                                parameters[j] = saved + epsilon
                                plus = model.evaluate(x, y)["loss"]
                                parameters[j] = saved - epsilon
                                minus = model.evaluate(x, y)["loss"]
                                parameters[j] = saved
                                self.assertAlmostEqual(expected, (plus - minus) / (2 * epsilon), places=7)

    def test_saturated_crossentropy_retains_loss_and_gradient(self):
        model = Sequential(1)
        layer = Dense(1, "sigmoid")
        layer.set_weights([[1000]])
        model.add(layer)
        model.compile("binary_crossentropy")
        self.assertAlmostEqual(model.evaluate([[1]], [[0]])["loss"], 1000)
        self.assertAlmostEqual(model.gradients([[1]], [[0]])[0]["weights"][0][0], 1)
        self.assertEqual(model.predict([[-1000], [1000]]), [[0], [1]])

    def test_no_bias_stays_zero(self):
        model = network(bias=False, optimizer="adam")
        model.fit([[1, 2], [3, 4]], [[0, 1], [1, 0]], epochs=5, batch_size=1)
        self.assertTrue(all(value == 0 for layer in model.layers for value in layer.bias))


class TrainingTests(unittest.TestCase):
    def test_sgd_learns_regression_with_partial_final_batch(self):
        x, y = [[-1], [0], [1], [2], [3]], [[-1], [1], [3], [5], [7]]
        model = Sequential(1)
        model.add(Dense(1))
        model.compile(learning_rate=0.03)
        history = model.fit(x, y, epochs=250, batch_size=2)
        self.assertLess(history["loss"][-1], 1e-6)
        self.assertAlmostEqual(model.predict([[4]])[0][0], 9, places=3)

    def test_adam_learns_xor(self):
        x, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]]
        model = Sequential(2, seed=42)
        model.add(Dense(8, "tanh"))
        model.add(Dense(1, "sigmoid"))
        model.compile("binary_crossentropy", "adam", 0.03)
        model.fit(x, y, epochs=600, batch_size=4, shuffle=False)
        self.assertLess(model.evaluate(x, y)["loss"], 0.01)
        self.assertEqual(model.evaluate(x, y)["accuracy"], 1)

    def test_seed_reproduces_training_and_adam_persists_across_calls(self):
        x, y = classification("circles", samples=20)
        a, b = network(optimizer="adam"), network(optimizer="adam")
        a.fit(x, [[v[0], 1 - v[0]] for v in y], epochs=10, batch_size=20, shuffle=False)
        for _ in range(2):
            b.fit(x, [[v[0], 1 - v[0]] for v in y], epochs=5, batch_size=20, shuffle=False)
        self.assertEqual(a.predict(x), b.predict(x))
        self.assertEqual(classification("spiral", seed=12), classification("spiral", seed=12))

    def test_early_stopping_restores_best_validation_parameters(self):
        model = Sequential(1)
        layer = Dense(1)
        layer.set_weights([[0]])
        model.add(layer)
        model.compile(learning_rate=0.1)
        history = model.fit([[1]], [[1]], epochs=100, validation_data=([[1]], [[0]]), patience=3)
        self.assertEqual(len(history["loss"]), 4)
        self.assertAlmostEqual(model.evaluate([[1]], [[0]])["loss"], min(history["val_loss"]))

    def test_classification_dataset_can_be_learned(self):
        x, y = classification("circles", samples=64)
        model = Sequential(2)
        model.add(Dense(8, "tanh"))
        model.add(Dense(8, "tanh"))
        model.add(Dense(1, "sigmoid"))
        model.compile("binary_crossentropy", "adam", 0.02)
        model.fit(x, y, epochs=120, batch_size=16)
        self.assertGreaterEqual(model.evaluate(x, y)["accuracy"], 0.95)


class ContractTests(unittest.TestCase):
    def test_model_roundtrip_preserves_predictions(self):
        model = network(loss="binary_crossentropy", optimizer="adam")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.nn"
            model.save(path)
            restored = Sequential.load(path)
        self.assertEqual(model.predict([[0.2, 0.6]]), restored.predict([[0.2, 0.6]]))
        self.assertEqual(restored.optimizer, "adam")

    def test_rejects_invalid_models(self):
        for text in ("bad 1", "LECTURE_NN 2", "LECTURE_NN 1 0 2", "LECTURE_NN 1 2 1 mse sgd 0.1 1 tanh 1 0"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.nn"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    Sequential.load(path)

    def test_validation_happens_before_training(self):
        model = network(loss="binary_crossentropy")
        before = model.predict([[0, 0]])
        invalid = [([], []), ([[1]], [[0, 1]]), ([[1, 2]], [[2, 0]]),
                   ([[math.nan, 2]], [[0, 1]]), ([[1, 2]], [[0]])]
        for x, y in invalid:
            with self.subTest(x=x), self.assertRaises(ValueError):
                model.fit(x, y)
        self.assertEqual(model.predict([[0, 0]]), before)

    def test_layer_dimensions_and_compilation_lifecycle(self):
        model = Sequential(2)
        first = Dense(2)
        model.add(first)
        model.compile()
        with self.assertRaises(ValueError):
            model.add(first)
        model.add(Dense(1))
        with self.assertRaises(RuntimeError):
            model.predict([[1, 2]])
        model.compile()
        bad = Dense(1)
        bad.set_weights([[1]])
        model.add(bad)
        # The previous layer has one unit, so this final layer is valid.
        model.compile()
        mismatch = Dense(1)
        mismatch.set_weights([[1], [2]])
        model.add(mismatch)
        with self.assertRaises(ValueError):
            model.compile()

    def test_invalid_training_configuration(self):
        for units in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Dense(units)
        model = network()
        for options in ({"epochs": 0}, {"batch_size": -1}, {"patience": 2}):
            with self.assertRaises(ValueError):
                model.fit([[0, 0]], [[0, 0]], **options)
        self.assertEqual(model.predict([]), [])
