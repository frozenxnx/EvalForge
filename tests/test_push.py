"""Moving experiments into the server: ExperimentStore export/import, the import route, and the CLI's --server."""
import contextlib
import gzip
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from llmobs_eval import ExperimentStore, evaluate_experiment
from llmobs_eval.__main__ import main as cli
from llmobs_eval.store import ImportConflict
from llmobs_server.auth import MAX_INFLATED
from llmobs_server.dev_server import serve
from tests.test_step3 import seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_CONFIG = {"name": "push-test", "providers": {"mock": {"type": "mock", "latency_ms": [1, 2], "error_rate": 0}},
               "variants": [{"label": "a", "provider": "mock", "model": "m"},
                            {"label": "b", "provider": "mock", "model": "m", "system": "Be brief."}],
               "concurrency": 2, "max_retries": 0}


def scored(store, **kw):
    exp = seed(store, **kw)
    evaluate_experiment(store, exp)
    return exp


class StoreRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.src, self.dst = ExperimentStore(), ExperimentStore()
        self.exp = scored(self.src)
        self.src.add_label(self.exp, "base", "q0", 1.0)
        self.src.hide_metric(self.exp, "exact_match")

    def test_import_reproduces_the_experiment_and_repeating_it_changes_nothing(self):
        first = self.dst.import_experiment(self.src.export_experiment(self.exp))
        self.assertTrue(first["created"])
        self.assertGreater(first["added"]["results"], 0)
        self.assertEqual(self.dst.export_experiment(self.exp), self.src.export_experiment(self.exp))
        again = self.dst.import_experiment(self.src.export_experiment(self.exp))
        self.assertFalse(again["created"])
        self.assertEqual(set(again["added"].values()), {0})
        self.assertFalse(again["hidden_metrics_changed"] or again["status_changed"])

    def test_later_work_is_added_and_mutable_parts_follow_the_source(self):
        self.dst.import_experiment(self.src.export_experiment(self.exp))
        self.src.add_scores([{"experiment_id": self.exp, "variant": "base", "example_id": "q0",
                              "metric": "extra", "value": 0.5, "created_at": 1.0}])
        self.src.add_label(self.exp, "base", "q0", 0.0, note="changed my mind")
        self.src.unhide_metric(self.exp, "exact_match")
        self.src.set_status(self.exp, "partial")
        res = self.dst.import_experiment(self.src.export_experiment(self.exp))
        self.assertEqual((res["added"]["scores"], res["added"]["labels"]), (1, 1))
        self.assertTrue(res["hidden_metrics_changed"] and res["status_changed"])
        self.assertEqual(self.dst.get_labels(self.exp)[0]["label"], 0.0)
        self.assertEqual(self.dst.hidden_metrics(self.exp), set())
        self.assertEqual(self.dst.get_experiment(self.exp)["status"], "partial")

    def test_a_different_history_is_refused_and_nothing_is_written(self):
        self.dst.import_experiment(self.src.export_experiment(self.exp))
        before = self.dst.export_experiment(self.exp)
        payload = self.src.export_experiment(self.exp)
        payload["tables"]["scores"][0]["value"] = 0.123            # same key, different content
        payload["tables"]["labels"][0]["label"] = 0.5               # would be applied if the import went through
        with self.assertRaisesRegex(ImportConflict, "different histories"):
            self.dst.import_experiment(payload)
        self.assertEqual(self.dst.export_experiment(self.exp), before)
        renamed = self.src.export_experiment(self.exp)
        renamed["experiment"]["name"] = "something else"
        with self.assertRaisesRegex(ImportConflict, "different name"):
            self.dst.import_experiment(renamed)

    def test_malformed_exports_are_rejected(self):
        good = self.src.export_experiment(self.exp)
        bad_cases = [None, {}, {**good, "format": "v0"}, {**good, "tables": {"users": []}},
                     {**good, "experiment": {**good["experiment"], "status": "done"}},
                     {**good, "experiment": {**good["experiment"], "extra": 1}}]
        stray = json.loads(json.dumps(good))
        stray["tables"]["results"][0]["experiment_id"] = "exp_other"
        missing = json.loads(json.dumps(good))
        del missing["tables"]["scores"][0]["detail"]
        for payload in bad_cases + [stray, missing]:
            with self.assertRaises(ValueError):
                self.dst.import_experiment(payload)
        self.assertEqual(self.dst.list_experiments(), [])

    def test_unknown_experiment(self):
        with self.assertRaises(KeyError):
            self.src.export_experiment("exp_nope")


class ServerCase(unittest.TestCase):
    api_key = "k3y"

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.server_db = os.path.join(self.dir, "server.db")
        self.srv = serve(self.server_db, "127.0.0.1", 0, self.api_key)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    def post(self, path, body, headers):
        req = urllib.request.Request(self.base + path, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}", **headers})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def server_store(self):
        return ExperimentStore(self.server_db)


class ImportRouteTests(ServerCase):
    def test_gzip_upload_conflict_and_bomb(self):
        src = ExperimentStore()
        exp = scored(src)
        body = gzip.compress(json.dumps(src.export_experiment(exp)).encode())
        status, out = self.post("/v1/experiments/import", body, {"Content-Encoding": "gzip"})
        self.assertEqual((status, out["created"]), (202, True))
        self.assertEqual(self.server_store().get_experiment(exp)["name"], src.get_experiment(exp)["name"])
        payload = src.export_experiment(exp)
        payload["tables"]["results"][0]["output"] = "rewritten"
        status, out = self.post("/v1/experiments/import", json.dumps(payload).encode(), {})
        self.assertEqual(status, 409)
        self.assertIn("different histories", out["detail"])
        bomb = gzip.compress(b" " * (MAX_INFLATED + 10))                # ~200 KB on the wire
        self.assertEqual(self.post("/v1/experiments/import", bomb, {"Content-Encoding": "gzip"})[0], 413)
        self.assertEqual(self.post("/v1/experiments/import", b"x", {"Content-Encoding": "br"})[0], 415)
        self.assertEqual(self.post("/v1/experiments/import", b"{oops", {})[0], 400)

    def test_import_needs_the_key(self):
        req = urllib.request.Request(self.base + "/v1/experiments/import", data=b"{}", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 401)


class CliServerTests(ServerCase):
    def setUp(self):
        super().setUp()
        self.local_db = os.path.join(self.dir, "local.db")
        self.cfg = os.path.join(self.dir, "cfg.json")
        with open(self.cfg, "w", encoding="utf-8") as fh:
            json.dump(MOCK_CONFIG, fh)
        os.environ["LLMOBS_API_KEY"] = self.api_key
        self.addCleanup(os.environ.pop, "LLMOBS_API_KEY", None)
        os.environ.pop("LLMOBS_SERVER", None)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli(["--db", self.local_db, *argv])
        return code, out.getvalue(), err.getvalue()

    def test_run_and_evaluate_land_on_the_server_with_their_traces(self):
        code, out, err = self.run_cli("--server", self.base, "run", "--dataset",
                                      os.path.join(ROOT, "examples", "dataset.jsonl"), "--config", self.cfg)
        self.assertEqual(code, 0, err)
        self.assertIn("on the server (", out.strip().splitlines()[-1])
        exp = ExperimentStore(self.local_db).list_experiments()[0]["id"]
        server = self.server_store()
        self.assertEqual(len(server.get_results(exp)), 10)
        self.assertEqual(server.get_scores(exp), [])
        traces = json.loads(urllib.request.urlopen(urllib.request.Request(
            self.base + "/v1/traces?limit=200", headers={"Authorization": f"Bearer {self.api_key}"}), timeout=10).read())
        self.assertEqual(len(traces["traces"]), 10)                    # traces followed --server too

        os.environ["LLMOBS_SERVER"] = self.base                         # the env var works like --server
        self.addCleanup(os.environ.pop, "LLMOBS_SERVER", None)
        code, out, err = self.run_cli("evaluate", exp)
        self.assertEqual(code, 0, err)
        self.assertIn(f"updated {exp} on the server", out)
        self.assertEqual(len(server.get_scores(exp)), len(ExperimentStore(self.local_db).get_scores(exp)))
        code, out, _ = self.run_cli("push", exp)                         # repeatable
        self.assertEqual(code, 0)
        self.assertIn("nothing new", out)

    def test_a_failed_upload_keeps_the_local_result_and_says_how_to_retry(self):
        os.environ["LLMOBS_API_KEY"] = "wrong"
        code, out, err = self.run_cli("--server", self.base, "run", "--dataset",
                                      os.path.join(ROOT, "examples", "dataset.jsonl"), "--config", self.cfg,
                                      "--trace-endpoint", "http://127.0.0.1:1/v1/spans")
        self.assertEqual(code, 2)
        self.assertIn("HTTP 401", err)
        self.assertIn("retry with: python -m llmobs_eval push exp_", err)
        exp = ExperimentStore(self.local_db).list_experiments()[0]["id"]
        self.assertEqual(len(ExperimentStore(self.local_db).get_results(exp)), 10)
        self.assertEqual(self.server_store().list_experiments(), [])
        os.environ["LLMOBS_API_KEY"] = self.api_key
        self.assertEqual(self.run_cli("--server", self.base, "push", exp)[0], 0)
        self.assertEqual(len(self.server_store().get_results(exp)), 10)

    def test_push_needs_a_server_and_a_known_experiment(self):
        code, _, err = self.run_cli("push", "exp_x")
        self.assertEqual(code, 2)
        self.assertIn("needs --server", err)
        self.assertEqual(self.run_cli("--server", self.base, "push", "exp_nope")[0], 2)
        self.assertEqual(self.run_cli("--server", "ftp://x", "list")[0], 2)


if __name__ == "__main__":
    unittest.main()
