import math
import unittest

from lecture_nn import Dense, Sequential


class DenseTests(unittest.TestCase):
    def test_sigmoid_matches_reference_value(self):
        layer = Dense(1, activation="sigmoid")
        layer.set_weights([[1.0,],], [0.0])
        self.assertAlmostEqual(layer.forward([[0.0]])[0][0], 0.5)
        self.assertAlmostEqual(layer.forward([[2.0]])[0][0], 1.0 / (1.0 + math.exp(-2.0)))

    def test_rejects_wrong_sample_width(self):
        layer = Dense(2)
        layer.build(2)
        with self.assertRaises(ValueError):
            layer.forward([[1.0]])


class SequentialTests(unittest.TestCase):
    def test_layers_share_a_predict_interface(self):
        model = Sequential(input_shape=2)
        first = Dense(2, activation="linear")
        first.set_weights([[1.0, 0.0], [0.0, 1.0]])
        second = Dense(1, activation="sigmoid")
        second.set_weights([[1.0], [1.0]], [0.0])
        model.add(first)
        model.add(second)
        model.compile()

        output = model.predict([[1.0, 1.0]])
        self.assertAlmostEqual(output[0][0], 1.0 / (1.0 + math.exp(-2.0)))


if __name__ == "__main__":
    unittest.main()
