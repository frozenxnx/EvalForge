import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import llmobs
from llmobs import InMemoryExporter
from llmobs_eval import (ConfigError, DatasetError, ExperimentStore, create_experiment,
                         load_jsonl, normalize_config, run_experiment, summarize)
from llmobs_eval.providers import (AnthropicProvider, OpenAICompatProvider, ProviderError,
                                   build_provider)
from llmobs_eval.report import percentile
from llmobs_eval.resilience import RateLimiter, call_with_retries
from llmobs_eval.runner import compute_cost


def write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    f.close()
    return f.name


def mock_cfg(**over):
    cfg = {
        "providers": {"m": {"type": "mock", "latency_ms": [1, 2]}},
        "variants": [{"label": "a", "provider": "m", "model": "model-a"},
                     {"label": "b", "provider": "m", "model": "model-b", "system": "Be brief."}],
        "concurrency": 4, "max_retries": 1,
    }
    cfg.update(over)
    return cfg


def small_dataset(n=6):
    return load_jsonl(write_jsonl([{"id": f"q{i}", "input": f"question {i}"} for i in range(n)]))


class DatasetTests(unittest.TestCase):
    def test_loads_aliases_and_defaults(self):
        ds = load_jsonl(write_jsonl([
            {"prompt": "p1", "answer": "a1"},
            {"id": "x", "question": "p2", "relevant_doc_ids": ["d1"], "metadata": {"t": 1}},
        ]))
        self.assertEqual([e.id for e in ds.examples], ["1", "x"])
        self.assertEqual(ds.examples[0].reference, "a1")
        self.assertEqual(ds.examples[1].relevant_doc_ids, ("d1",))

    def test_errors_name_the_line(self):
        with self.assertRaisesRegex(DatasetError, r":2: missing required field"):
            load_jsonl(write_jsonl([{"input": "ok"}, {"reference": "no input"}]))
        with self.assertRaisesRegex(DatasetError, r":2: invalid JSON"):
            load_jsonl(write_jsonl([{"input": "ok"}, "{nope"]))
        with self.assertRaisesRegex(DatasetError, "duplicate id"):
            load_jsonl(write_jsonl([{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]))

    def test_hash_is_stable_and_content_sensitive(self):
        rows = [{"id": "a", "input": "x", "reference": "r"}]
        h1 = load_jsonl(write_jsonl(rows)).content_hash
        self.assertEqual(h1, load_jsonl(write_jsonl(rows)).content_hash)
        rows[0]["reference"] = "changed"
        self.assertNotEqual(h1, load_jsonl(write_jsonl(rows)).content_hash)


class ResilienceTests(unittest.TestCase):
    def test_rate_limiter_spaces_calls(self):
        rl = RateLimiter(20)  # 50 ms apart
        t = time.perf_counter()
        for _ in range(5):
            rl.acquire()
        self.assertGreaterEqual(time.perf_counter() - t, 0.19)

    def test_rate_limiter_is_shared_across_threads(self):
        rl = RateLimiter(50)
        t = time.perf_counter()
        ths = [threading.Thread(target=rl.acquire) for _ in range(6)]
        [x.start() for x in ths]
        [x.join() for x in ths]
        self.assertGreaterEqual(time.perf_counter() - t, 0.09)

    def test_retry_then_success_with_backoff(self):
        calls, sleeps = [], []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ProviderError("busy", status=503, retryable=True)
            return "ok"

        out, attempts = call_with_retries(fn, 5, base=1.0, sleep=sleeps.append, rng=lambda: 0.5)
        self.assertEqual((out, attempts), ("ok", 3))
        self.assertEqual(sleeps, [1.0, 2.0])  # base*2^n*(0.5+0.5)

    def test_non_retryable_raises_immediately(self):
        sleeps = []
        with self.assertRaises(ProviderError) as cm:
            call_with_retries(lambda: (_ for _ in ()).throw(ProviderError("bad key", 401)),
                              5, sleep=sleeps.append)
        self.assertEqual((cm.exception.attempts, sleeps), (1, []))

    def test_retry_after_wins_and_exhaustion_reports_attempts(self):
        sleeps = []

        def fn():
            raise ProviderError("slow down", 429, retry_after=7.0, retryable=True)

        with self.assertRaises(ProviderError) as cm:
            call_with_retries(fn, 2, sleep=sleeps.append)
        self.assertEqual(sleeps, [7.0, 7.0])
        self.assertEqual(cm.exception.attempts, 3)


class FakeLLM(BaseHTTPRequestHandler):
    script, seen = [], []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        cls.seen.append({"path": self.path, "body": body,
                         "headers": {k.lower(): v for k, v in self.headers.items()}})
        status, hdrs = cls.script.pop(0) if cls.script else (200, {})
        if status != 200:
            payload = b'{"error":"nope"}'
        elif self.path.endswith("/chat/completions"):
            payload = json.dumps({"choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 11, "completion_tokens": 3}}).encode()
        else:
            payload = json.dumps({"content": [{"type": "text", "text": "pong"}], "stop_reason": "end_turn",
                                  "usage": {"input_tokens": 12, "output_tokens": 4}}).encode()
        self.send_response(status)
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeLLM)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        FakeLLM.script, FakeLLM.seen = [], []

    def test_openai_compat(self):
        p = OpenAICompatProvider(self.base + "/v1", api_key="sk-test")
        c = p.complete("m1", [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}],
                       {"temperature": 0})
        self.assertEqual((c.text, c.input_tokens, c.output_tokens), ("pong", 11, 3))
        seen = FakeLLM.seen[0]
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(seen["body"]["temperature"], 0)

    def test_anthropic_lifts_system_prompt(self):
        p = AnthropicProvider("ak-test", base_url=self.base)
        c = p.complete("c1", [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], {})
        self.assertEqual((c.text, c.input_tokens, c.output_tokens), ("pong", 12, 4))
        seen = FakeLLM.seen[0]
        self.assertEqual(seen["headers"]["x-api-key"], "ak-test")
        self.assertIn("anthropic-version", seen["headers"])
        self.assertEqual(seen["body"]["system"], "be brief")
        self.assertEqual([m["role"] for m in seen["body"]["messages"]], ["user"])
        self.assertIn("max_tokens", seen["body"])

    def test_error_classification(self):
        p = OpenAICompatProvider(self.base + "/v1")
        FakeLLM.script = [(429, {"Retry-After": "3"}), (401, {}), (503, {})]
        with self.assertRaises(ProviderError) as e1:
            p.complete("m", [{"role": "user", "content": "x"}], {})
        self.assertEqual((e1.exception.status, e1.exception.retryable, e1.exception.retry_after), (429, True, 3.0))
        with self.assertRaises(ProviderError) as e2:
            p.complete("m", [{"role": "user", "content": "x"}], {})
        self.assertFalse(e2.exception.retryable)
        with self.assertRaises(ProviderError) as e3:
            p.complete("m", [{"role": "user", "content": "x"}], {})
        self.assertTrue(e3.exception.retryable)

    def test_unreachable_host_is_retryable(self):
        with self.assertRaises(ProviderError) as e:
            OpenAICompatProvider("http://127.0.0.1:1/v1", timeout=1).complete(
                "m", [{"role": "user", "content": "x"}], {})
        self.assertTrue(e.exception.retryable)

    def test_runner_recovers_from_429_over_real_http(self):
        FakeLLM.script = [(429, {"Retry-After": "0"}), (200, {})]
        cfg = {"providers": {"fake": {"type": "openai_compat", "base_url": self.base + "/v1"}},
               "variants": [{"label": "v", "provider": "fake", "model": "m"}],
               "max_retries": 2, "pricing": {"m": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}}
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(1), cfg)
        run_experiment(store, exp)
        r = store.get_results(exp)[0]
        self.assertEqual((r["status"], r["attempts"], r["output"]), ("ok", 2, "pong"))
        self.assertAlmostEqual(r["cost_usd"], (11 * 1.0 + 3 * 2.0) / 1e6)


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ConfigError):
            normalize_config({"providers": {}, "variants": []})
        with self.assertRaisesRegex(ConfigError, "unknown provider"):
            normalize_config({"providers": {"m": {"type": "mock"}},
                              "variants": [{"label": "a", "provider": "zzz", "model": "x"}]})
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            normalize_config(mock_cfg(variants=[{"label": "a", "provider": "m", "model": "x"}] * 2))
        with self.assertRaisesRegex(ConfigError, "must contain"):
            normalize_config(mock_cfg(variants=[{"label": "a", "provider": "m", "model": "x",
                                                 "prompt_template": "no placeholder"}]))

    def test_missing_api_key_fails_fast(self):
        with self.assertRaisesRegex(ConfigError, "NOPE_KEY is not set"):
            build_provider("p", {"type": "anthropic", "api_key_env": "NOPE_KEY"}, env={})

    def test_bad_config_creates_no_experiment(self):
        store = ExperimentStore()
        cfg = mock_cfg(providers={"m": {"type": "anthropic", "api_key_env": "SURELY_UNSET_KEY_123"}})
        with self.assertRaises(ConfigError):
            create_experiment(store, small_dataset(1), cfg)
        self.assertEqual(store.list_experiments(), [])

    def test_cost(self):
        pr = {"m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
        self.assertAlmostEqual(compute_cost(pr, "m", 1000, 500), (3000 + 7500) / 1e6)
        self.assertIsNone(compute_cost(pr, "other", 1000, 500))
        self.assertIsNone(compute_cost(pr, "m", None, 5))


class RunnerTests(unittest.TestCase):
    def test_runs_every_pair_and_summarizes(self):
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(6), mock_cfg())
        out = run_experiment(store, exp)
        self.assertEqual(out, {"ran": 12, "remaining": 0})
        self.assertEqual(store.get_experiment(exp)["status"], "complete")
        rows = {r["variant"]: r for r in summarize(store, exp)}
        self.assertEqual((rows["a"]["n"], rows["b"]["n"], rows["a"]["errors"]), (6, 6, 0))
        self.assertIsNotNone(rows["a"]["p95_ms"])
        # variant b's system prompt reached the model; variant a had none
        rb = [r for r in store.get_results(exp, "b")][0]
        self.assertTrue(rb["output"].startswith("[model-b]"))

    def test_failures_are_recorded_not_raised(self):
        cfg = mock_cfg(providers={"m": {"type": "mock", "latency_ms": [1, 1], "error_rate": 1.0}})
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(3), cfg)
        run_experiment(store, exp, sleep=lambda s: None)
        rs = store.get_results(exp)
        self.assertEqual(len(rs), 6)
        self.assertTrue(all(r["status"] == "error" and r["attempts"] == 2 for r in rs))  # 1 try + 1 retry
        self.assertEqual(store.get_experiment(exp)["status"], "complete")

    def test_resume_completes_without_touching_existing_results(self):
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(6), mock_cfg())
        first = run_experiment(store, exp, max_tasks=5)
        self.assertEqual(first, {"ran": 5, "remaining": 7})
        self.assertEqual(store.get_experiment(exp)["status"], "partial")
        before = {(r["variant"], r["example_id"]): r["created_at"] for r in store.get_results(exp)}
        second = run_experiment(store, exp)
        self.assertEqual(second, {"ran": 7, "remaining": 0})
        after = {(r["variant"], r["example_id"]): r["created_at"] for r in store.get_results(exp)}
        self.assertEqual(len(after), 12)
        for k, v in before.items():
            self.assertEqual(after[k], v)
        self.assertEqual(run_experiment(store, exp), {"ran": 0, "remaining": 0})  # idempotent

    def test_partial_runs_stay_balanced_across_variants(self):
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(6), mock_cfg())
        run_experiment(store, exp, max_tasks=6)
        counts = {}
        for r in store.get_results(exp):
            counts[r["variant"]] = counts.get(r["variant"], 0) + 1
        self.assertEqual(counts, {"a": 3, "b": 3})

    def test_results_and_definitions_are_immutable(self):
        store = ExperimentStore()
        exp = create_experiment(store, small_dataset(2), mock_cfg())
        run_experiment(store, exp)
        c = store._conn
        for sql in ("UPDATE results SET output = 'tampered'", "DELETE FROM results",
                    "UPDATE experiment_examples SET input = 'x'", "DELETE FROM experiment_examples",
                    "UPDATE experiments SET config = '{}'", "UPDATE experiments SET dataset_hash = 'x'",
                    "DELETE FROM experiments"):
            with self.assertRaises(sqlite3.IntegrityError, msg=sql):
                c.execute(sql)
        self.assertFalse(store.add_result({"experiment_id": exp, "variant": "a", "example_id": "q0",
                                           "status": "ok", "attempts": 1, "wall_ms": 1, "created_at": 0}))

    def test_dataset_is_snapshotted(self):
        path = write_jsonl([{"id": "q1", "input": "original"}])
        store = ExperimentStore()
        exp = create_experiment(store, load_jsonl(path), mock_cfg())
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": "q1", "input": "edited later"}) + "\n")
        run_experiment(store, exp)
        self.assertIn("original", store.get_results(exp, "a")[0]["output"])
        self.assertEqual(store.get_experiment(exp)["dataset_hash"], load_jsonl(write_jsonl(
            [{"id": "q1", "input": "original"}])).content_hash)

    def test_every_result_links_to_a_trace_with_llm_child(self):
        exp_mem = InMemoryExporter()
        llmobs.configure(exporter=exp_mem, service="eval-test")
        try:
            store = ExperimentStore()
            exp = create_experiment(store, small_dataset(3), mock_cfg())
            run_experiment(store, exp)
        finally:
            llmobs.configure(exporter=None)
        results = store.get_results(exp)
        self.assertEqual(len(results), 6)
        for r in results:
            spans = [s for s in exp_mem.spans if s["trace_id"] == r["trace_id"]]
            root = [s for s in spans if s["parent_id"] is None][0]
            self.assertEqual(root["attributes"]["experiment_id"], exp)
            self.assertEqual(root["attributes"]["variant"], r["variant"])
            llm = [s for s in spans if s["kind"] == "llm"]
            self.assertEqual(len(llm), r["attempts"])
            self.assertEqual(llm[-1]["attributes"]["gen_ai.usage.input_tokens"], r["input_tokens"])

    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50), 5)
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10)
        self.assertIsNone(percentile([], 50))


class CliTests(unittest.TestCase):
    def test_run_report_list(self):
        from llmobs_eval.__main__ import main
        d = tempfile.mkdtemp()
        db, cfgp = os.path.join(d, "t.db"), os.path.join(d, "c.json")
        json.dump(mock_cfg(), open(cfgp, "w"))
        ds = write_jsonl([{"id": f"q{i}", "input": f"q {i}"} for i in range(3)])
        self.assertEqual(main(["--db", db, "run", "--dataset", ds, "--config", cfgp]), 0)
        self.assertEqual(main(["--db", db, "list"]), 0)
        exp_id = ExperimentStore(db).list_experiments()[0]["id"]
        self.assertEqual(main(["--db", db, "report", exp_id]), 0)
        self.assertEqual(main(["--db", db, "report", "exp_missing"]), 2)


if __name__ == "__main__":
    unittest.main()
