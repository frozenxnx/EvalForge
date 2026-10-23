import asyncio
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import llmobs
from llmobs import HTTPExporter, InMemoryExporter, Tracer
from llmobs.exporters import Exporter
from llmobs_server.store import SpanStore


def setup_mem():
    exp = InMemoryExporter()
    llmobs.configure(exporter=exp, service="test")
    return exp


class TracerTests(unittest.TestCase):
    def test_nesting_and_ids(self):
        exp = setup_mem()

        @llmobs.trace(kind="retrieval")
        def retrieve(q):
            return ["d1", "d2"]

        @llmobs.trace
        def answer(q):
            docs = retrieve(q)
            with llmobs.span("generate", kind="llm") as s:
                s.record_llm("gpt-x", input_tokens=10, output_tokens=5)
            return docs

        answer("hi")
        by_name = {s["name"]: s for s in exp.spans}
        root = by_name["answer"]
        self.assertIsNone(root["parent_id"])
        self.assertEqual(by_name["retrieve"]["parent_id"], root["span_id"])
        self.assertEqual(by_name["generate"]["parent_id"], root["span_id"])
        self.assertEqual({s["trace_id"] for s in exp.spans}, {root["trace_id"]})
        self.assertEqual(root["input"], {"q": "hi"})
        self.assertEqual(by_name["generate"]["attributes"]["gen_ai.usage.input_tokens"], 10)
        self.assertTrue(all(s["end_ns"] >= s["start_ns"] for s in exp.spans))

    def test_error_recorded_and_reraised(self):
        exp = setup_mem()

        @llmobs.trace
        def boom():
            raise ValueError("bad")

        with self.assertRaises(ValueError):
            boom()
        self.assertEqual(exp.spans[0]["status"], "error")
        self.assertIn("ValueError: bad", exp.spans[0]["error"])

    def test_async_traces_are_isolated(self):
        exp = setup_mem()

        @llmobs.trace
        async def child(i):
            await asyncio.sleep(0.01)
            return i

        @llmobs.trace
        async def parent(i):
            return await child(i)

        async def main():
            await asyncio.gather(parent(1), parent(2))

        asyncio.run(main())
        parents = [s for s in exp.spans if s["name"] == "parent"]
        self.assertEqual(len({p["trace_id"] for p in parents}), 2)
        for p in parents:
            kid = [s for s in exp.spans if s["name"] == "child" and s["trace_id"] == p["trace_id"]]
            self.assertEqual(kid[0]["parent_id"], p["span_id"])

    def test_exporter_failure_never_breaks_app(self):
        class Broken(Exporter):
            def export(self, span):
                raise RuntimeError("exporter down")

        llmobs.configure(exporter=Broken())

        @llmobs.trace
        def f():
            return 42

        self.assertEqual(f(), 42)

    def test_unserializable_and_huge_values_are_made_safe(self):
        exp = setup_mem()
        with llmobs.span("x") as s:
            s.input = {"obj": object(), "nan": float("nan"), "big": "a" * 20000}
        out = json.dumps(exp.spans[0])  # must not raise
        self.assertIn("truncated", out)
        self.assertIsNone(exp.spans[0]["input"]["nan"])


class StoreTests(unittest.TestCase):
    def _span(self, sid, parent, name, start, end, **kw):
        d = dict(trace_id="t1", span_id=sid, parent_id=parent, name=name, kind="chain",
                 service="svc", start_ns=start, end_ns=end, status="ok", error=None,
                 input=None, output=None, attributes={})
        d.update(kw)
        return d

    def test_ingest_is_idempotent_and_aggregates(self):
        st = SpanStore()
        spans = [
            self._span("a", None, "root", 0, 2_000_000),
            self._span("b", "a", "llm", 100, 1_000_000, status="error", error="x",
                       attributes={"gen_ai.usage.input_tokens": 7, "gen_ai.usage.output_tokens": 3}),
        ]
        st.ingest(spans)
        st.ingest(spans)  # retried batch
        self.assertEqual(len(st.get_trace("t1")), 2)
        row = st.list_traces()[0]
        self.assertEqual(row["span_count"], 2)
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["input_tokens"], 7)
        self.assertEqual(row["root_name"], "root")
        self.assertAlmostEqual(row["duration_ms"], 2.0)


class _Handler(BaseHTTPRequestHandler):
    store = None
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.store.ingest(body["spans"])
        self.send_response(202)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *a):
        pass


class EndToEndTests(unittest.TestCase):
    def test_sdk_to_http_to_store(self):
        store = SpanStore()
        _Handler.store = store
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        exp = HTTPExporter(f"http://127.0.0.1:{srv.server_port}/v1/spans",
                           batch_size=10, flush_interval=0.05)
        llmobs.configure(exporter=exp, service="e2e")

        @llmobs.trace
        def work(n):
            with llmobs.span("step"):
                pass
            return n

        for i in range(25):
            work(i)
        exp.flush()
        srv.shutdown()
        self.assertEqual(exp.stats()["sent"], 50)
        self.assertEqual(len(store.list_traces(limit=100)), 25)

    def test_unreachable_backend_does_not_block_and_counts_failures(self):
        exp = HTTPExporter("http://127.0.0.1:1/v1/spans", flush_interval=0.05,
                           max_retries=1, backoff_base=0.01, timeout=0.5)
        llmobs.configure(exporter=exp)
        t = time.perf_counter()
        for _ in range(3):
            with llmobs.span("x"):
                pass
        self.assertLess(time.perf_counter() - t, 0.1)  # app path stays fast
        exp.flush()
        self.assertEqual(exp.stats()["failed"], 3)

    def test_full_queue_drops_instead_of_blocking(self):
        exp = HTTPExporter("http://127.0.0.1:1/", max_queue=5, autostart=False)
        for i in range(12):
            exp.export({"i": i})
        self.assertEqual(exp.stats()["dropped"], 7)


if __name__ == "__main__":
    unittest.main()
