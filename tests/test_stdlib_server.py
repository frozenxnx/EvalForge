import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import llmobs
from llmobs.exporters import HTTPExporter
from llmobs_eval import ExperimentStore, evaluate_experiment
from llmobs_server.dev_server import serve
from tests.test_judge_reliability import FLAWED, judged, truth_labels
from tests.test_step3 import NO_SLEEP, ScriptedJudge


def request(base, path, method="GET", body=None, headers=None):
    data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw, status, hdrs = resp.read(), resp.status, resp.headers
    except urllib.error.HTTPError as exc:
        raw, status, hdrs = exc.read(), exc.code, exc.headers
    ctype = hdrs.get("Content-Type", "")
    return status, hdrs, (json.loads(raw) if "json" in ctype else raw.decode("utf-8"))


class ServerCase(unittest.TestCase):
    api_key = None

    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "t.db")
        self.store = ExperimentStore(self.db)
        self.srv = serve(self.db, "127.0.0.1", 0, self.api_key)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    def get(self, path, **kw):
        return request(self.base, path, **kw)


class BasicsTests(ServerCase):
    def test_health_page_and_errors(self):
        status, hdrs, body = self.get("/healthz")
        self.assertEqual((status, body), (200, {"ok": True}))
        self.assertIn("application/json", hdrs["Content-Type"])
        self.assertEqual(hdrs["Cache-Control"], "no-store")
        status, hdrs, html = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", hdrs["Content-Type"])
        self.assertIn("<title>llmobs", html)
        self.assertEqual(self.get("/nope")[0:3:2], (404, {"detail": "not found"}))
        self.assertEqual(self.get("/healthz", method="POST", body=b"{}")[0], 405)
        self.assertEqual(self.get("/v1/spans")[0], 405)

    def test_query_validation(self):
        for q in ("limit=abc", "limit=0", "limit=500", "offset=-1"):
            status, _, body = self.get(f"/v1/traces?{q}")
            self.assertEqual(status, 422, q)
            self.assertIn("query parameter", body["detail"])


class IngestionTests(ServerCase):
    def test_the_sdk_exporter_delivers_spans_to_the_server_and_they_can_be_read_back(self):
        exporter = HTTPExporter(self.base + "/v1/spans", batch_size=10, flush_interval=0.05)
        llmobs.configure(exporter=exporter, service="http-test")
        try:
            with llmobs.span("pipeline"):
                with llmobs.span("retrieve", kind="retrieval") as s:
                    s.record_retrieval("q", [{"id": "d1", "text": "t", "score": 1.0}])
                try:
                    with llmobs.span("generate", kind="llm"):
                        llmobs.record_llm("m", 10, 5)
                        raise RuntimeError("boom")
                except RuntimeError:
                    pass
            exporter.flush(timeout=10)
        finally:
            llmobs.configure(exporter=None)
        status, _, body = self.get("/v1/traces")
        self.assertEqual(status, 200)
        t = body["traces"][0]
        self.assertEqual((t["root_name"], t["span_count"], t["error_count"], t["input_tokens"], t["output_tokens"], t["service"]),
                         ("pipeline", 3, 1, 10, 5, "http-test"))
        status, _, detail = self.get(f"/v1/traces/{t['trace_id']}")
        names = {s["name"]: s for s in detail["spans"]}
        self.assertEqual(set(names), {"pipeline", "retrieve", "generate"})
        self.assertEqual(names["retrieve"]["attributes"]["retrieval.documents"][0]["id"], "d1")
        self.assertEqual((names["generate"]["status"], names["generate"]["error"]), ("error", "RuntimeError: boom"))
        self.assertEqual(self.get("/v1/traces/unknown")[0], 404)

    def test_bad_ingest_requests_are_rejected_with_a_reason(self):
        good = {"trace_id": "t", "span_id": "s", "name": "x", "start_ns": 1}
        self.assertEqual(self.get("/v1/spans", method="POST", body=b"{oops")[0], 400)
        self.assertEqual(self.get("/v1/spans", method="POST", body=b"[]")[0], 422)
        status, _, body = self.get("/v1/spans", method="POST", body={"spans": [good, {"trace_id": "t"}]})
        self.assertEqual(status, 422)
        self.assertIn("span 1", body["detail"])
        self.assertEqual(self.get("/v1/spans", method="POST", body={"spans": [good] * 1001})[0], 422)
        status, _, body = self.get("/v1/spans", method="POST", body={"spans": [good]})
        self.assertEqual((status, body), (202, {"accepted": 1}))
        self.assertEqual(self.get("/v1/spans", method="POST", body={"spans": [good]})[2], {"accepted": 1})   # idempotent upsert
        self.assertEqual(len(self.get("/v1/traces")[2]["traces"]), 1)


class AuthTests(ServerCase):
    api_key = "secret"

    def test_only_ingestion_requires_the_key(self):
        span = {"trace_id": "t", "span_id": "s", "name": "x", "start_ns": 1}
        self.assertEqual(self.get("/v1/spans", method="POST", body={"spans": [span]})[0], 401)
        self.assertEqual(self.get("/v1/spans", method="POST", body={"spans": [span]},
                                  headers={"Authorization": "Bearer wrong"})[0], 401)
        self.assertEqual(self.get("/v1/spans", method="POST", body={"spans": [span]},
                                  headers={"Authorization": "Bearer secret"})[0], 202)
        self.assertEqual(self.get("/v1/traces")[0], 200)                                # reading stays open


class ExperimentRoutesTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.exp = judged(self.store, FLAWED)
        evaluate_experiment(self.store, self.exp)
        truth_labels(self.store, self.exp)

    def test_list_and_overview(self):
        status, _, body = self.get("/v1/experiments")
        self.assertEqual((status, [e["id"] for e in body["experiments"]]), (200, [self.exp]))
        status, _, ov = self.get(f"/v1/experiments/{self.exp}")
        self.assertEqual(status, 200)
        self.assertEqual([v["label"] for v in ov["variants"]], ["base", "cand"])
        self.assertEqual((ov["dataset"]["n_examples"], ov["has_judge"], ov["n_labels"], ov["total_calls"]), (20, True, 40, 40))
        self.assertEqual({m["name"] for m in ov["metrics"]}, {"exact_match", "contains", "token_f1", "judge:judge-x"})
        self.store.hide_metric(self.exp, "judge:judge-x")
        hidden = {m["name"]: m["hidden"] for m in self.get(f"/v1/experiments/{self.exp}")[2]["metrics"]}
        self.assertEqual((hidden["judge:judge-x"], hidden["contains"]), (True, False))
        self.assertEqual(self.get("/v1/experiments/exp_nope")[0], 404)

    def test_compare_with_awkward_metric_names_and_bad_arguments(self):
        status, _, cmp = self.get(f"/v1/experiments/{self.exp}/compare?baseline=cand")
        self.assertEqual((status, cmp["baseline"]), (200, "cand"))
        metric = urllib.parse.quote("judge:judge-x", safe="")
        status, _, cmp = self.get(f"/v1/experiments/{self.exp}/compare?metric={metric}")
        self.assertEqual((status, cmp["primary_metric"]), (200, "judge:judge-x"))
        status, _, err = self.get(f"/v1/experiments/{self.exp}/compare?baseline=zzz")
        self.assertEqual(status, 400)
        self.assertIn("unknown baseline", err["detail"])
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/compare?top_k=0")[0], 422)
        self.assertEqual(self.get("/v1/experiments/exp_nope/compare")[0], 404)

    def test_examples_paging_and_sorting(self):
        status, _, page = self.get(f"/v1/experiments/{self.exp}/examples?limit=5&offset=5")
        self.assertEqual((status, page["total"], len(page["examples"])), (200, 20, 5))
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/examples?sort=dataset&limit=2")[2]["examples"][0]["id"], "q0")
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/examples?sort=random")[0], 400)
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/examples?limit=abc")[0], 422)

    def test_judge_routes(self):
        status, _, jc = self.get(f"/v1/experiments/{self.exp}/judge-check")
        self.assertEqual((status, jc["metric"], jc["totals"]["lenient"], jc["totals"]["strict"]), (200, "judge:judge-x", 2, 2))
        status, _, cal = self.get(f"/v1/experiments/{self.exp}/calibration")
        self.assertEqual((status, cal["all"]["n"]), (200, 40))
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/judge-check?metric=judge:nope")[0], 400)
        self.assertEqual(self.get("/v1/experiments/exp_nope/calibration")[0], 404)

    def test_failure_pattern_routes(self):
        status, _, p = self.get(f"/v1/experiments/{self.exp}/failure-patterns")
        self.assertEqual((status, p["variant"], p["view"]), (200, "cand", "failures"))
        status, _, reg = self.get(f"/v1/experiments/{self.exp}/failure-patterns?variant=cand&baseline=base&cluster=false")
        self.assertEqual((reg["view"], reg["cluster_mode"]), ("regressions", None))
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/failure-patterns?variant=zzz")[0], 400)
        self.assertEqual(self.get(f"/v1/experiments/{self.exp}/failure-patterns?threshold=7")[0], 422)
        self.assertEqual(self.get("/v1/experiments/exp_nope/failure-patterns")[0], 404)

    def test_concurrent_readers(self):
        with ThreadPoolExecutor(8) as pool:
            codes = list(pool.map(lambda _: self.get(f"/v1/experiments/{self.exp}/compare")[0], range(16)))
        self.assertEqual(set(codes), {200})


if __name__ == "__main__":
    unittest.main()
