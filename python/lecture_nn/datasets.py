"""Reproducible two-dimensional classification datasets."""

from math import cos, pi, sin
from random import Random


def classification(name="xor", samples=96, seed=42):
    if name not in {"xor", "circles", "spiral"}:
        raise ValueError("dataset must be xor, circles, or spiral")
    if not isinstance(samples, int) or not 4 <= samples <= 10000:
        raise ValueError("samples must be between 4 and 10000")
    rng = Random(seed)
    x, y = [], []
    for i in range(samples):
        label = i % 2
        if name == "xor":
            a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
            label = int(a * b > 0)
        elif name == "circles":
            angle = rng.uniform(0, 2 * pi)
            radius = (0.35 if label == 0 else 0.85) + rng.gauss(0, 0.04)
            a, b = radius * cos(angle), radius * sin(angle)
        else:
            radius = 0.1 + 0.85 * (i // 2) / max(1, (samples - 1) // 2)
            angle = radius * 5 + label * pi + rng.gauss(0, 0.12)
            a, b = radius * cos(angle), radius * sin(angle)
        x.append([a, b])
        y.append([float(label)])
    return x, y
