import contextlib
import io
import os
import tempfile
import unittest

from llmobs_eval import ExperimentStore, calibration_payload, compare_payload, failure_patterns
from llmobs_eval.demo import build_demo, main
from llmobs_server.store import SpanStore


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = os.path.join(tempfile.mkdtemp(), "demo.db")
        cls.ids = build_demo(cls.db, seed=7, latency_scale=0.0)
        cls.store = ExperimentStore(cls.db)

    def test_two_complete_experiments_with_everything_the_ui_shows(self):
        a, b = self.ids["prompt-ab"], self.ids["rag-retrieval"]
        for exp in (a, b):
            self.assertEqual(self.store.get_experiment(exp)["status"], "complete")
        ca = compare_payload(self.store, a, n_boot=100)
        self.assertEqual([v["variant"] for v in ca["variants"]], ["baseline", "pr-one-sentence", "pr-step-by-step"])
        self.assertTrue({"contains", "judge:demo-judge"} <= set(ca["metrics"]))
        self.assertTrue(all(v["cost_usd"] is not None for v in ca["variants"]))
        cb = compare_payload(self.store, b, n_boot=100)
        self.assertTrue({"recall@3", "mrr", "contains", "judge:demo-judge"} <= set(cb["metrics"]))
        oracle = next(v for v in cb["variants"] if v["variant"] == "oracle")
        closed = next(v for v in cb["variants"] if v["variant"] == "closed-book")
        self.assertGreater(oracle["scores"]["contains"]["mean"], closed["scores"]["contains"]["mean"] + 0.4)

    def test_traces_include_retrieval_spans(self):
        spans = SpanStore(self.db)
        self.assertEqual(len(spans.list_traces(limit=200)), 200)                 # a full page ...
        self.assertTrue(spans.list_traces(limit=200, offset=200))               # ... and more behind it
        kinds = dict(spans._conn.execute("SELECT kind, COUNT(*) FROM spans GROUP BY kind").fetchall())
        self.assertGreaterEqual(kinds["retrieval"], 96)                          # bm25 + oracle, 48 questions each
        self.assertGreater(kinds["llm"], 400)
        roots = {r[0] for r in spans._conn.execute("SELECT DISTINCT name FROM spans WHERE parent_id IS NULL")}
        self.assertTrue({"baseline", "bm25-k3", "oracle", "judge"} <= roots)

    def test_labels_and_calibration_are_available(self):
        for exp in self.ids.values():
            cal = calibration_payload(self.store, exp, n_boot=100)
            self.assertEqual((cal["n_labels"], cal["metric"]), (28, "judge:demo-judge"))
            self.assertIsNotNone(cal["all"]["kappa"])

    def test_the_planted_weakness_is_findable(self):
        p = failure_patterns(self.store, self.ids["rag-retrieval"], "bm25-k3", metric="recall@3", cluster=False)
        self.assertEqual(p["categories"][0]["label"], "vocab_mismatch")
        self.assertTrue(p["categories"][0]["significant"])

    def test_deterministic_per_seed(self):
        def means(seed):
            path = os.path.join(tempfile.mkdtemp(), "d.db")
            ids = build_demo(path, seed=seed, latency_scale=0.0)
            c = compare_payload(ExperimentStore(path), ids["prompt-ab"], n_boot=50)
            return [round(v["scores"]["contains"]["mean"], 6) for v in c["variants"]]
        self.assertEqual(means(7), means(7))
        self.assertNotEqual(means(7), means(99))


class CliTests(unittest.TestCase):
    def test_refuses_to_overwrite_without_force(self):
        path = os.path.join(tempfile.mkdtemp(), "x.db")
        open(path, "w").close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--db", path]), 2)
        self.assertIn("already exists", out.getvalue())


if __name__ == "__main__":
    unittest.main()
