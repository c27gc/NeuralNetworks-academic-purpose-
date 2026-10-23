"""A loopback-only HTTP interface to the Python training engine."""

import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from tempfile import TemporaryDirectory
from uuid import uuid4

from .core import Dense, Sequential
from .datasets import classification


class Experiment:
    def __init__(self, config):
        self.width = int(config.get("width", 8))
        self.depth = int(config.get("depth", 2))
        rate = float(config.get("rate", 0.02))
        if not 2 <= self.width <= 16 or not 1 <= self.depth <= 3 or not 0 < rate <= 0.5:
            raise ValueError("width, depth, or learning rate outside laboratory limits")
        self.x, self.y = classification(config.get("dataset", "xor"), samples=80, seed=int(config.get("seed", 42)))
        self.model = Sequential(2, seed=int(config.get("seed", 42)))
        for _ in range(self.depth):
            self.model.add(Dense(self.width, config.get("activation", "tanh")))
        self.model.add(Dense(1, "sigmoid"))
        self.model.compile("binary_crossentropy", config.get("optimizer", "adam"), rate)
        self.history = [self.model.evaluate(self.x, self.y)["loss"]]
        self.grid = [[-1 + 2 * col / 31, 1 - 2 * row / 31] for row in range(32) for col in range(32)]
        self.epoch = 0

    def step(self, epochs=10):
        if not 1 <= epochs <= 25 or self.epoch + epochs > 3000:
            raise ValueError("training limit reached; reset to start another experiment")
        result = self.model.fit(self.x, self.y, epochs=epochs, batch_size=16)
        self.history.extend(result["loss"])
        self.epoch += epochs
        return self.snapshot()

    def snapshot(self):
        return {"epoch": self.epoch, "history": self.history,
                "metrics": self.model.evaluate(self.x, self.y),
                "points": [row + target for row, target in zip(self.x, self.y)],
                "surface": [row[0] for row in self.model.predict(self.grid)],
                "layers": [2] + [self.width] * self.depth + [1],
                "summary": self.model.summary()}

    def close(self):
        pass


class CppExperiment(Experiment):
    """Keep a C++ worker alive so Adam moments survive between visual steps."""

    def __init__(self, config, executable):
        super().__init__(config)
        self._directory = TemporaryDirectory(prefix="lecture-nn-lab-")
        root = Path(self._directory.name)
        self.model.save(root / "model.nn")
        for name, rows in (("inputs", self.x), ("targets", self.y), ("grid", self.grid)):
            root.joinpath(name + ".csv").write_text("\n".join(",".join(map(str, row)) for row in rows) + "\n")
        try:
            self._process = subprocess.Popen(
                [str(executable), "worker", str(root / "model.nn"), str(root / "inputs.csv"),
                 str(root / "targets.csv"), str(root / "grid.csv")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
        except OSError:
            self._directory.cleanup()
            raise

    def _request(self, command):
        try:
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise ValueError("C++ worker stopped; reset the experiment")
            result = json.loads(line)
            self.epoch = result["epoch"]
            result.update(points=[row + target for row, target in zip(self.x, self.y)],
                          layers=[2] + [self.width] * self.depth + [1], summary=self.model.summary())
            return result
        except (OSError, BrokenPipeError) as error:
            raise ValueError("C++ worker is unavailable; reset the experiment") from error

    def snapshot(self):
        return self._request("snapshot")

    def step(self, epochs=10):
        if not 1 <= epochs <= 25 or self.epoch + epochs > 3000:
            raise ValueError("training limit reached; reset to start another experiment")
        return self._request(f"step {epochs}")

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.communicate("quit\n", timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.communicate()
        else:
            self._process.communicate()
        self._directory.cleanup()


def create_server(port=8765, cpp=None):
    executable = Path(cpp).resolve() if cpp else None
    if executable is not None and not executable.is_file():
        raise ValueError("C++ executable was not found")
    # One active experiment keeps CPU and memory bounded. Tokens reject stale UI requests.
    state = {"experiment": None, "token": None}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/config":
                self.respond(200, json.dumps({"cpp_available": executable is not None}).encode())
                return
            if self.path != "/":
                self.respond(404, b'{"error":"not found"}')
                return
            html = Path(__file__).with_name("static").joinpath("lab.html").read_bytes()
            self.respond(200, html, "text/html; charset=utf-8")

        def do_POST(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected or self.headers.get("Origin") not in {None, f"http://{expected}"}:
                self.respond(403, b'{"error":"local requests only"}')
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("expected JSON")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid request size")
                config = json.loads(self.rfile.read(length))
                if not isinstance(config, dict):
                    raise ValueError("expected a JSON object")
                with lock:
                    if self.path == "/api/reset":
                        backend = config.get("backend", "python")
                        if backend not in {"python", "cpp"} or (backend == "cpp" and executable is None):
                            raise ValueError("selected engine is unavailable")
                        experiment = CppExperiment(config, executable) if backend == "cpp" else Experiment(config)
                        if state["experiment"] is not None:
                            state["experiment"].close()
                        state.update(experiment=experiment, token=uuid4().hex)
                        result = experiment.snapshot()
                    elif self.path == "/api/step":
                        if state["experiment"] is None or config.get("token") != state["token"]:
                            raise ValueError("experiment expired; reset to continue")
                        result = state["experiment"].step(int(config.get("epochs", 10)))
                    else:
                        self.respond(404, b'{"error":"not found"}')
                        return
                    result["token"] = state["token"]
                self.respond(200, json.dumps(result, allow_nan=False).encode())
            except (ValueError, TypeError, OverflowError, FloatingPointError, OSError) as error:
                self.respond(400, json.dumps({"error": str(error)}).encode())

    class Server(ThreadingHTTPServer):
        def server_close(self):
            if state["experiment"] is not None:
                state["experiment"].close()
            super().server_close()

    return Server(("127.0.0.1", port), Handler)


def serve(port=8765, cpp=None):
    with create_server(port, cpp) as server:
        print(f"Neural Networks Lab: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLaboratory stopped.")
