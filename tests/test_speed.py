"""Making slow judging bearable: native Ollama provider, chunking, concurrency, a timing probe, progress."""
import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from llmobs_eval import (ConfigError, ExperimentStore, evaluate_experiment, format_probe, probe_judge)
from llmobs_eval.__main__ import _Progress, _dur, main
from llmobs_eval.providers import Completion, OllamaProvider, Provider, ProviderError, build_provider
from tests.test_step3 import NO_SLEEP, ScriptedJudge, seed

SPEC = {"provider": "m", "model": "judge-x"}
OK = '{"verdict": "correct", "reason": "ok"}'


class FakeOllama(BaseHTTPRequestHandler):
    seen, status, shape = [], 200, "ok"

    def do_POST(self):
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls.seen.append((self.path, body))
        if cls.status != 200:
            payload = b'{"error": "model not found"}'
        elif self.path == "/api/chat":
            payload = json.dumps({"model": body["model"], "message": {"role": "assistant", "content": "pong"},
                                  "done": True, "done_reason": "stop", "prompt_eval_count": 21, "eval_count": 5}
                                 if cls.shape == "ok" else {"model": "x", "done": True}).encode()
        else:
            n = len(body["input"]) - (1 if cls.shape == "short" else 0)
            payload = json.dumps({"embeddings": [[float(len(t)), 1.0] for t in body["input"][:n]]}).encode()
        self.send_response(cls.status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


class OllamaProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        FakeOllama.seen, FakeOllama.status, FakeOllama.shape = [], 200, "ok"

    MSGS = [{"role": "user", "content": "hi"}]

    def test_request_and_response_shape(self):
        p = OllamaProvider(self.base + "/", think=False, keep_alive="30m")            # trailing slash tolerated
        c = p.complete("qwen3:4b", self.MSGS, {"temperature": 0, "max_tokens": 50})
        self.assertEqual((c.text, c.input_tokens, c.output_tokens, c.finish_reason), ("pong", 21, 5, "stop"))
        path, body = FakeOllama.seen[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual((body["model"], body["stream"], body["think"], body["keep_alive"], body["messages"]),
                         ("qwen3:4b", False, False, "30m", self.MSGS))
        self.assertEqual(body["options"], {"temperature": 0, "num_predict": 50})       # max_tokens is an alias

    def test_think_is_only_sent_when_set_and_params_override_the_provider(self):
        OllamaProvider(self.base, keep_alive=None).complete("m", self.MSGS, {})
        body = FakeOllama.seen[-1][1]
        self.assertNotIn("think", body)
        self.assertNotIn("keep_alive", body)
        self.assertNotIn("options", body)                                              # nothing to put in it
        OllamaProvider(self.base, think=False).complete("m", self.MSGS, {"think": True, "keep_alive": "5m", "num_ctx": 2048})
        body = FakeOllama.seen[-1][1]
        self.assertEqual((body["think"], body["keep_alive"], body["options"]), (True, "5m", {"num_ctx": 2048}))

    def test_errors_are_classified_like_the_other_providers(self):
        FakeOllama.status = 404
        with self.assertRaises(ProviderError) as cm:
            OllamaProvider(self.base).complete("nope", self.MSGS, {})
        self.assertEqual((cm.exception.status, cm.exception.retryable), (404, False))
        FakeOllama.status = 500
        with self.assertRaises(ProviderError) as cm:
            OllamaProvider(self.base).complete("m", self.MSGS, {})
        self.assertTrue(cm.exception.retryable)
        FakeOllama.status, FakeOllama.shape = 200, "bad"
        with self.assertRaisesRegex(ProviderError, "unexpected response shape"):
            OllamaProvider(self.base).complete("m", self.MSGS, {})

    def test_embeddings(self):
        vecs = OllamaProvider(self.base).embed("nomic-embed-text", ["a", "bbb"])
        self.assertEqual(vecs, [[1.0, 1.0], [3.0, 1.0]])
        self.assertEqual(FakeOllama.seen[-1][0], "/api/embed")
        FakeOllama.shape = "short"
        with self.assertRaisesRegex(ProviderError, "unexpected embeddings response"):
            OllamaProvider(self.base).embed("m", ["a", "b"])

    def test_configured_through_a_provider_spec(self):
        p = build_provider("n", {"type": "ollama", "think": False, "keep_alive": "10m", "timeout": 300})
        self.assertEqual((type(p).__name__, p.base_url, p.think, p.keep_alive, p.timeout),
                         ("OllamaProvider", "http://localhost:11434", False, "10m", 300.0))
        d = build_provider("n", {"type": "ollama"})
        self.assertEqual((d.think, d.keep_alive), (None, "30m"))
        with self.assertRaisesRegex(Exception, "use mock, anthropic, openai_compat, ollama"):
            build_provider("n", {"type": "nope"})


class EvaluationOptionTests(unittest.TestCase):
    def test_extra_providers_serve_this_evaluation_only(self):
        store = ExperimentStore()
        exp = seed(store, n=4)
        before = store.get_experiment(exp)["config"]["providers"]
        with self.assertRaisesRegex(ConfigError, "add more with --extra-providers"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "extra", "model": "j"},
                                providers={"extra": ScriptedJudge()})
        stats = evaluate_experiment(store, exp, lexical=False, retrieval=False,
                                    judge={"provider": "extra", "model": "j"}, extra_providers={"extra": {"type": "mock"}},
                                    providers={"extra": ScriptedJudge()}, sleep=NO_SLEEP)
        self.assertEqual(stats["scored"], 8)
        self.assertEqual(store.get_experiment(exp)["config"]["providers"], before)       # nothing was stored

    def test_max_items_scores_in_chunks_and_resumes(self):
        store = ExperimentStore()
        exp = seed(store, n=10)                                                        # 20 answers to judge
        kw = dict(lexical=False, retrieval=False, judge=SPEC, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        first = evaluate_experiment(store, exp, max_items=7, **kw)
        self.assertEqual((first["scored"], first["deferred"]), (7, 13))
        second = evaluate_experiment(store, exp, max_items=7, **kw)
        self.assertEqual((second["scored"], second["deferred"]), (7, 6))
        third = evaluate_experiment(store, exp, max_items=7, **kw)
        self.assertEqual((third["scored"], third["deferred"]), (6, 0))
        self.assertEqual(len(store.get_scores(exp)), 20)

    def test_concurrency_overrides_the_experiments_setting(self):
        class Gauge(Provider):
            def __init__(self):
                self.lock, self.now, self.peak = threading.Lock(), 0, 0

            def complete(self, model, messages, params):
                with self.lock:
                    self.now += 1
                    self.peak = max(self.peak, self.now)
                time.sleep(0.02)
                with self.lock:
                    self.now -= 1
                return Completion(OK)

        def peak(workers):
            store = ExperimentStore()
            exp = seed(store, n=8)                                                     # the experiment itself says 4
            g = Gauge()
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": g},
                                concurrency=workers, sleep=NO_SLEEP)
            return g.peak
        self.assertEqual(peak(1), 1)
        self.assertGreaterEqual(peak(4), 2)


class Slow(Provider):
    def __init__(self, text=OK, out_tokens=40, delay=0.02, error=None):
        self.text, self.out_tokens, self.delay, self.error, self.calls = text, out_tokens, delay, error, []

    def complete(self, model, messages, params):
        self.calls.append((messages[-1]["content"], dict(params)))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return Completion(self.text, 300, self.out_tokens)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = seed(self.store, n=10)                                              # 20 answers, none judged

    def probe(self, provider, **judge):
        return probe_judge(self.store, self.exp, {**SPEC, **judge}, n=3, providers={"m": provider})

    def test_a_thinking_model_is_diagnosed_with_a_way_out(self):
        p = self.probe(Slow(text="<think>hmm, let me think</think>" + OK, out_tokens=420))
        self.assertEqual((p["sampled"], p["remaining"], p["workers"]), (3, 20, 4))
        self.assertGreaterEqual(p["mean_latency_s"], 0.02)
        self.assertAlmostEqual(p["estimate_s"], 20 * p["mean_latency_s"] / 4)
        text = " ".join(p["advice"])
        self.assertIn("about 420 tokens per call", text)
        self.assertIn("native Ollama provider", text)
        self.assertIn("--judge-suffix /no_think", text)
        self.assertEqual(self.store.get_scores(self.exp), [])                          # a probe stores nothing

    def test_a_healthy_fast_judge_gets_no_advice(self):
        p = self.probe(Slow(out_tokens=40))
        self.assertEqual(p["advice"], [])
        self.assertTrue(all(c["parsed"] and not c["thinking_tag"] for c in p["calls"]))

    def test_slow_calls_suggest_chunking_and_the_projection_uses_the_workers(self):
        import itertools
        with mock.patch("llmobs_eval.evaluate.time.perf_counter", side_effect=itertools.cycle([0.0, 12.0])):
            p = self.probe(Slow(delay=0.0))                                           # every call "takes" 12 s
        self.assertAlmostEqual(p["mean_latency_s"], 12.0)
        self.assertAlmostEqual(p["estimate_s"], 20 * 12.0 / 4)                        # 20 answers, 4 workers
        advice = " ".join(p["advice"])
        self.assertIn("each call takes 12.0s", advice)
        self.assertIn("--max-items N", advice)
        self.assertIn("mean 12.0s per call", format_probe(p))
        self.assertIn("about 1m00s", format_probe(p))
        quick = self.probe(Slow(delay=0.0))
        self.assertNotIn("--max-items", " ".join(quick["advice"]))                    # fast enough: no nagging

    def test_unparseable_replies_are_reported(self):
        p = self.probe(Slow(text="I think it is correct."))
        self.assertIn("3 of 3 replies could not be parsed", " ".join(p["advice"]))

    def test_a_server_that_is_down_stops_after_one_call(self):
        dead = Slow(error=ProviderError("cannot reach http://localhost:11434/v1: refused", retryable=True))
        p = self.probe(dead)
        self.assertEqual((len(dead.calls), len(p["calls"]), p["mean_latency_s"], p["estimate_s"]), (1, 1, None, None))
        self.assertIn("1 of 1 probe calls failed: cannot reach", " ".join(p["advice"]))
        self.assertIn("FAILED", format_probe(p))

    def test_settings_reach_the_provider_and_the_sample_is_reproducible(self):
        a, b = Slow(), Slow()
        self.probe(a, suffix="/no_think", params={"temperature": 0, "think": False})
        self.probe(b, suffix="/no_think", params={"temperature": 0, "think": False})
        self.assertTrue(all(c.endswith("\n\n/no_think") for c, _ in a.calls))
        # the probe also caps the output (a verdict needs ~40 tokens) so a reasoning model is caught quickly
        self.assertEqual({str(p) for _, p in a.calls}, {str({"temperature": 0, "think": False, "max_tokens": 300})})
        self.assertEqual([c for c, _ in a.calls], [c for c, _ in b.calls])

    def test_nothing_left_and_bad_input(self):
        evaluate_experiment(self.store, self.exp, lexical=False, retrieval=False, judge=SPEC,
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        p = self.probe(Slow())
        self.assertEqual((p["sampled"], p["remaining"]), (0, 0))
        self.assertIn("nothing is left to judge", " ".join(p["advice"]))
        with self.assertRaisesRegex(ConfigError, "--extra-providers"):
            probe_judge(self.store, self.exp, {"provider": "zzz", "model": "x"})
        with self.assertRaisesRegex(ConfigError, "unknown rubric"):
            probe_judge(self.store, self.exp, {**SPEC, "rubric": "v9"}, providers={"m": Slow()})

    def test_the_report_is_plain_ascii_and_readable(self):
        text = format_probe(self.probe(Slow(out_tokens=420, text="<think>x</think>" + OK)))
        self.assertTrue(text.isascii())
        for needle in ("judge speed probe: judge:judge-x", "call 1:", "generated tokens 420", "<think> seen",
                       "answers left to judge with 4 worker(s)"):
            self.assertIn(needle, text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        self.store = ExperimentStore(self.db)
        self.exp = seed(self.store, n=10)

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", self.db, *argv])
        return code, out.getvalue(), err.getvalue()

    def write(self, name, obj):
        path = os.path.join(self.dir, name)
        with open(path, "w") as fh:
            json.dump(obj, fh) if not isinstance(obj, str) else fh.write(obj)
        return path

    def test_probe_prints_a_report_and_stores_nothing(self):
        code, out, _ = self.cli("evaluate", self.exp, "--judge", "m:j", "--probe", "2")
        self.assertEqual(code, 0)
        self.assertIn("judge speed probe: judge:j", out)
        self.assertIn("NOT PARSED", out)                                              # the mock model is not a judge
        self.assertEqual(ExperimentStore(self.db).get_scores(self.exp), [])
        self.assertEqual(self.cli("evaluate", self.exp, "--probe")[0], 2)             # needs --judge

    def test_max_items_and_extra_providers(self):
        extra = self.write("extra.json", {"extra": {"type": "mock", "latency_ms": [0, 1]}})
        code, out, _ = self.cli("evaluate", self.exp, "--judge", "extra:j", "--extra-providers", extra,
                                "--no-lexical", "--no-retrieval", "--max-items", "5")
        self.assertEqual(code, 0)
        self.assertIn("deferred=15", out)
        self.assertEqual(len(ExperimentStore(self.db).get_scores(self.exp)), 5)
        wrapped = self.write("wrapped.json", {"providers": {"extra": {"type": "mock", "latency_ms": [0, 1]}}})
        code, out, _ = self.cli("evaluate", self.exp, "--judge", "extra:j", "--extra-providers", wrapped,
                                "--no-lexical", "--no-retrieval", "--max-items", "5", "--concurrency", "1")
        self.assertEqual(code, 0)
        self.assertEqual(len(ExperimentStore(self.db).get_scores(self.exp)), 10)

    def test_bad_extra_provider_files_exit_cleanly(self):
        for content in ("{oops", "[1, 2]", "{}", '{"p": "not an object"}'):
            path = self.write("bad.json", content)
            self.assertEqual(self.cli("evaluate", self.exp, "--judge", "m:j", "--extra-providers", path)[0], 2, content)
        self.assertEqual(self.cli("evaluate", self.exp, "--judge", "m:j", "--extra-providers", "nope.json")[0], 2)


class ProgressTests(unittest.TestCase):
    def test_durations(self):
        self.assertEqual([_dur(x) for x in (0, 59, 65, 600, 3725)], ["0s", "59s", "1m05s", "10m00s", "1h02m"])

    def test_progress_line_has_elapsed_time_and_an_estimate(self):
        err = io.StringIO()
        with mock.patch("llmobs_eval.__main__.time.monotonic", side_effect=[0.0, 10.0, 40.0]):
            progress = _Progress()
            with contextlib.redirect_stderr(err):
                progress(1, 4)
                progress(4, 4)
        text = err.getvalue()
        self.assertIn("1/4 calls  (10s elapsed, ~30s left)", text)
        self.assertIn("4/4 calls  (40s elapsed)", text)
        self.assertTrue(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
