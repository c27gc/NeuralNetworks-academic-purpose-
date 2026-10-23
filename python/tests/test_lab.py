import http.client
import json
import threading
import unittest

from lecture_nn.lab import Experiment, create_server


class LaboratoryTests(unittest.TestCase):
    def test_experiment_updates_real_training_metrics(self):
        experiment = Experiment({"dataset": "circles"})
        before = experiment.snapshot()
        after = experiment.step(20)
        self.assertLess(after["metrics"]["loss"], before["metrics"]["loss"])
        self.assertEqual(after["epoch"], 20)
        self.assertEqual(len(after["surface"]), 1024)
        self.assertEqual(after["layers"], [2, 8, 8, 1])

    def test_http_reset_step_errors_and_local_origin(self):
        with create_server(0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            connection = http.client.HTTPConnection("127.0.0.1", port)

            def request(path, data, origin=None):
                headers = {"Content-Type": "application/json"}
                if origin:
                    headers["Origin"] = origin
                connection.request("POST", path, json.dumps(data), headers)
                response = connection.getresponse()
                return response.status, json.loads(response.read())

            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"Start training", response.read())
                code, result = request("/api/reset", {})
                self.assertEqual(code, 200)
                code, trained = request("/api/step", {"token": result["token"], "epochs": 2})
                self.assertEqual((code, trained["epoch"]), (200, 2))
                self.assertEqual(request("/api/step", {"token": "expired"})[0], 400)
                self.assertEqual(request("/api/reset", {}, "https://example.com")[0], 403)
                self.assertEqual(request("/api/reset", {"width": 1000})[0], 400)
                self.assertEqual(request("/api/reset", [1, 2])[0], 400)
            finally:
                connection.close()
                server.shutdown()
                thread.join()
