"""Run with LECTURE_NN_CPP pointing to the C++ model executable."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from lecture_nn import Sequential
from lecture_nn.lab import CppExperiment
from .test_training import network


@unittest.skipUnless(os.environ.get("LECTURE_NN_CPP"), "C++ executable not configured")
class InteroperabilityTests(unittest.TestCase):
    def test_cpp_visual_worker_preserves_training_state(self):
        experiment = CppExperiment({"dataset": "circles"}, os.environ["LECTURE_NN_CPP"])
        try:
            before = experiment.snapshot()
            experiment.step(10)
            after = experiment.step(10)
            self.assertEqual(after["epoch"], 20)
            self.assertEqual(len(after["history"]), 21)
            self.assertLess(after["metrics"]["loss"], before["metrics"]["loss"])
            self.assertEqual(len(after["surface"]), 1024)
        finally:
            experiment.close()

    def test_predictions_and_updates_match_for_both_optimizers(self):
        executable = os.environ["LECTURE_NN_CPP"]
        x, y = [[0.3, -0.2], [0.8, 0.4], [-0.4, 0.6]], [[0.1, 0.8], [0.7, 0.2], [0.2, 0.9]]
        for loss in ("mse", "binary_crossentropy"):
            for optimizer in ("sgd", "adam"):
                with self.subTest(loss=loss, optimizer=optimizer), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source, target = root / "source.nn", root / "target.nn"
                    inputs, targets = root / "x.csv", root / "y.csv"
                    for path, rows in ((inputs, x), (targets, y)):
                        path.write_text("\n".join(",".join(map(str, row)) for row in rows) + "\n")
                    model = network(loss=loss, optimizer=optimizer)
                    model.save(source)
                    output = subprocess.check_output([executable, "predict", str(source), str(inputs)], text=True)
                    prediction = [[float(v) for v in row.split(",")] for row in output.splitlines()]
                    for expected, actual in zip(model.predict(x), prediction):
                        for a, b in zip(expected, actual):
                            self.assertAlmostEqual(a, b, places=6)
                    subprocess.run([executable, "train", str(source), str(inputs), str(targets),
                                    str(target), "3", "2"], check=True)
                    model.fit(x, y, epochs=3, batch_size=2, shuffle=False)
                    restored = Sequential.load(target)
                    for expected, actual in zip(model.predict(x), restored.predict(x)):
                        for a, b in zip(expected, actual):
                            self.assertAlmostEqual(a, b, places=5)
