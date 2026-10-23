"""Command line experiments and the local visual laboratory."""

import argparse
import json

from .core import Dense, Sequential
from .datasets import classification


def main():
    parser = argparse.ArgumentParser(description="Neural networks from first principles")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train a reproducible classification experiment")
    train.add_argument("--dataset", choices=["xor", "circles", "spiral"], default="xor")
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--save", metavar="MODEL.nn")
    lab = commands.add_parser("lab", help="open the local visual laboratory")
    lab.add_argument("--port", type=int, default=8765)
    lab.add_argument("--cpp", metavar="EXECUTABLE", help="enable the C++ engine in the laboratory")
    args = parser.parse_args()
    if args.command == "lab":
        from .lab import serve
        serve(args.port, args.cpp)
        return
    x, y = classification(args.dataset, seed=args.seed)
    model = Sequential(2, seed=args.seed)
    model.add(Dense(12, "tanh"))
    model.add(Dense(8, "tanh"))
    model.add(Dense(1, "sigmoid"))
    model.compile("binary_crossentropy", "adam", learning_rate=0.02)
    print(model.summary())
    initial = model.evaluate(x, y)["loss"]
    model.fit(x, y, epochs=args.epochs, batch_size=16)
    print(json.dumps({"initial_loss": initial, **model.evaluate(x, y)}, indent=2))
    if args.save:
        model.save(args.save)
        print(f"Saved {args.save}")


if __name__ == "__main__":
    main()
