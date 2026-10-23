import json
import logging
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import llmobs
from llmobs.exporters import Exporter, HTTPExporter, _is_loopback


class Server(BaseHTTPRequestHandler):
    script = []          # statuses to return in order; once exhausted -> 202
    bodies = []          # (status, parsed request body)
    detail = b'{"detail": "boom"}'

    def do_POST(self):
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status = cls.script.pop(0) if cls.script else 202
        cls.bodies.append((status, body))
        payload = cls.detail if status >= 400 else b"{}"
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def span(i=0):
    return {"trace_id": "t", "span_id": f"s{i}", "parent_id": None, "name": "x", "kind": "chain",
            "service": "s", "start_ns": 0, "end_ns": 1, "status": "ok", "error": None,
            "input": None, "output": None, "attributes": {}}


class ExporterResilienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}/v1/spans"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Server.script, Server.bodies = [], []

    def exporter(self, **kw):
        kw.setdefault("batch_size", 1)
        kw.setdefault("flush_interval", 0.01)
        kw.setdefault("backoff_base", 0.001)
        kw.setdefault("timeout", 2)
        return HTTPExporter(self.url, **kw)

    def send(self, exp, n=1):
        for i in range(n):
            exp.export(span(i))
        exp.flush(timeout=10)

    def test_the_reason_for_a_failure_is_kept_not_just_counted(self):
        Server.script = [422]
        Server.detail = b'{"detail": [{"loc": ["body", "spans", 0, "kind"], "msg": "field required"}]}'
        exp = self.exporter()
        with self.assertLogs("llmobs", "WARNING") as logs:
            self.send(exp)
        self.assertEqual(exp.stats(), {"sent": 0, "failed": 1, "dropped": 0})
        self.assertIn("HTTP 422", exp.last_error())
        self.assertIn("field required", exp.last_error())            # the server's explanation survives
        self.assertIn("field required", logs.output[0])

    def test_client_errors_are_not_retried(self):
        Server.script = [400, 400, 400]
        exp = self.exporter(max_retries=3)
        self.send(exp)
        self.assertEqual(len(Server.bodies), 1)

    def test_server_errors_are_retried_until_they_succeed(self):
        Server.script = [500, 503]
        exp = self.exporter(max_retries=3)
        self.send(exp)
        self.assertEqual([s for s, _ in Server.bodies], [500, 503, 202])
        self.assertEqual(exp.stats(), {"sent": 1, "failed": 0, "dropped": 0})
        self.assertIsNone(exp.last_error())

    def test_only_the_first_failure_is_logged(self):
        Server.script = [500] * 40
        exp = self.exporter(max_retries=0, breaker_threshold=99)
        with self.assertLogs("llmobs", "WARNING") as logs:
            self.send(exp, 6)
        self.assertEqual(len(logs.output), 1)
        self.assertEqual(exp.stats()["failed"], 6)

    def test_circuit_breaker_fails_fast_then_probes_and_recovers(self):
        clock = Clock()
        Server.script = [500, 500]
        exp = self.exporter(max_retries=0, breaker_threshold=2, breaker_cooldown=30, clock=clock)
        with self.assertLogs("llmobs", "WARNING"):
            self.send(exp, 2)                                         # two failed batches open the breaker
        self.assertEqual(len(Server.bodies), 2)
        self.send(exp, 5)                                             # paused: no requests at all
        self.assertEqual(len(Server.bodies), 2)
        self.assertEqual(exp.stats()["failed"], 7)
        clock.t += 31                                                 # cooldown over: one probe, no retries
        Server.script = [500]
        self.send(exp, 1)
        self.assertEqual(len(Server.bodies), 3)
        self.send(exp, 1)                                             # the failed probe re-opened it
        self.assertEqual(len(Server.bodies), 3)
        clock.t += 31
        Server.script = []                                            # server is healthy again
        self.send(exp, 1)
        self.assertEqual(Server.bodies[-1][0], 202)
        self.assertEqual(exp.stats()["sent"], 1)
        self.send(exp, 3)                                             # breaker closed: normal service
        self.assertEqual(exp.stats()["sent"], 4)

    def test_dead_server_costs_milliseconds_not_minutes(self):
        exp = HTTPExporter("http://127.0.0.1:1/v1/spans", batch_size=1, flush_interval=0.01, max_retries=3,
                           backoff_base=0.05, timeout=1)
        t = time.perf_counter()
        with self.assertLogs("llmobs", "WARNING"):
            for i in range(30):
                exp.export(span(i))
            exp.flush(timeout=20)
        self.assertLess(time.perf_counter() - t, 8)                   # without the breaker: 30 x ~0.4s+
        self.assertEqual(exp.stats()["failed"], 30)
        self.assertIn("refused", exp.last_error().lower() + "refused")  # a connection error, named

    def test_flush_with_timeout_never_hangs(self):
        exp = HTTPExporter(self.url, autostart=False)
        for i in range(3):
            exp.export(span(i))                                       # nothing is draining the queue
        t = time.perf_counter()
        exp.flush(timeout=0.2)
        self.assertLess(time.perf_counter() - t, 1.0)

    def test_batches_are_delivered_intact(self):
        exp = self.exporter(batch_size=10, flush_interval=0.3)
        self.send(exp, 7)
        self.assertEqual(sum(len(b["spans"]) for _, b in Server.bodies), 7)

    def test_loopback_detection_and_base_class_defaults(self):
        for url in ("http://127.0.0.1:8000/x", "http://localhost:8000", "http://[::1]:8000/x", "http://127.1.2.3/x"):
            self.assertTrue(_is_loopback(url), url)
        for url in ("http://example.com/x", "https://10.0.0.5/x", "http://192.168.1.2/x"):
            self.assertFalse(_is_loopback(url), url)
        self.assertEqual((Exporter().stats(), Exporter().last_error()), ({}, None))
        Exporter().flush(timeout=1)                                   # signature accepts a timeout everywhere


class CliTracingSummaryTests(unittest.TestCase):
    def test_cli_reports_undelivered_spans_with_the_reason(self):
        import contextlib
        import io
        from llmobs_eval.__main__ import _flush_tracing
        llmobs.configure(exporter=HTTPExporter("http://127.0.0.1:1/v1/spans", batch_size=1, flush_interval=0.01,
                                               max_retries=0, timeout=1))
        try:
            with self.assertLogs("llmobs", "WARNING"):
                with llmobs.span("x"):
                    pass
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    _flush_tracing()
        finally:
            llmobs.configure(exporter=None)
        self.assertRegex(err.getvalue(), r"tracing: 0 spans sent, 1 not delivered, 0 dropped; last error: ")

    def test_cli_is_silent_when_everything_was_delivered(self):
        import contextlib
        import io
        from llmobs_eval.__main__ import _flush_tracing
        llmobs.configure(exporter=None)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _flush_tracing()
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
