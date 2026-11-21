"""A server that accepts the request but never answers (the 302-second probe): timeouts are their own thing."""
import contextlib
import io
import socket
import threading
import time
import unittest

from llmobs_eval import (ConfigError, ExperimentStore, create_experiment, evaluate_experiment, format_probe, load_jsonl,
                         probe_judge, run_experiment)
from llmobs_eval.__main__ import main
from llmobs_eval.providers import Completion, OllamaProvider, OpenAICompatProvider, Provider, ProviderError
from llmobs_eval.resilience import MAX_TIMEOUT_RETRIES, call_with_retries
from tests.test_eval import mock_cfg, write_jsonl
from tests.test_step3 import NO_SLEEP, seed

SPEC = {"provider": "m", "model": "judge-x"}
OK = '{"verdict": "correct", "reason": "ok"}'
MSGS = [{"role": "user", "content": "hi"}]


class Hang:
    """A TCP server that accepts connections and then says nothing: a model stuck reasoning or a backlogged queue."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.held = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                self.held.append(self.sock.accept()[0])
            except OSError:
                return

    def close(self):
        self.sock.close()
        for c in self.held:
            c.close()


def timeout_error(kind="timeout"):
    return ProviderError("timed out after 300s waiting for http://localhost:11434/v1/chat/completions: ...",
                         retryable=True, kind=kind)


class ErrorKindTests(unittest.TestCase):
    def setUp(self):
        self.hang = Hang()
        self.addCleanup(self.hang.close)

    def test_a_silent_server_is_a_timeout_not_a_connection_failure(self):
        started = time.perf_counter()
        with self.assertRaises(ProviderError) as cm:
            OpenAICompatProvider(f"http://127.0.0.1:{self.hang.port}/v1", timeout=0.3).complete("m", MSGS, {})
        e = cm.exception
        self.assertEqual((e.kind, e.retryable, e.status), ("timeout", True, None))
        self.assertRegex(str(e), rf"^timed out after 0\.3s waiting for http://127\.0\.0\.1:{self.hang.port}/v1/chat/completions")
        self.assertIn("busy with earlier requests", str(e))
        self.assertLess(time.perf_counter() - started, 5)

    def test_the_native_provider_classifies_the_same_way(self):
        with self.assertRaises(ProviderError) as cm:
            OllamaProvider(f"http://127.0.0.1:{self.hang.port}", timeout=0.3).complete("m", MSGS, {})
        self.assertEqual(cm.exception.kind, "timeout")

    def test_a_refused_connection_is_a_connection_failure(self):
        with self.assertRaises(ProviderError) as cm:
            OpenAICompatProvider("http://127.0.0.1:1/v1", timeout=1).complete("m", MSGS, {})
        self.assertEqual((cm.exception.kind, cm.exception.retryable), ("connection", True))
        self.assertTrue(str(cm.exception).startswith("cannot reach "))


class RetryPolicyTests(unittest.TestCase):
    @staticmethod
    def counting(error):
        calls = []

        def fn():
            calls.append(1)
            raise error
        return fn, calls

    def test_a_timeout_gets_one_more_try_not_five(self):
        fn, calls = self.counting(timeout_error())
        with self.assertRaises(ProviderError) as cm:
            call_with_retries(fn, 5, sleep=NO_SLEEP)
        self.assertEqual((len(calls), MAX_TIMEOUT_RETRIES, cm.exception.attempts), (2, 1, 2))

    def test_other_transient_failures_still_use_the_full_budget(self):
        for error in (ProviderError("refused", retryable=True, kind="connection"),
                      ProviderError("HTTP 503", status=503, retryable=True)):
            fn, calls = self.counting(error)
            with self.assertRaises(ProviderError):
                call_with_retries(fn, 5, sleep=NO_SLEEP)
            self.assertEqual(len(calls), 6, error)

    def test_a_timeout_followed_by_success_is_still_recovered(self):
        state = {"n": 0}

        def fn():
            state["n"] += 1
            if state["n"] == 1:
                raise timeout_error()                       # the cold-start case: the model finished loading meanwhile
            return "answer"
        self.assertEqual(call_with_retries(fn, 4, sleep=NO_SLEEP), ("answer", 2))


class Slow(Provider):
    def __init__(self, error=None, delay=0.02, text=OK, out_tokens=40):
        self.error, self.delay, self.text, self.out_tokens = error, delay, text, out_tokens
        self.calls, self.params = 0, []

    def complete(self, model, messages, params):
        self.calls += 1
        self.params.append(dict(params))
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return Completion(self.text, 300, self.out_tokens)


class AbortTests(unittest.TestCase):
    def test_evaluation_stops_after_three_timeouts_not_five(self):
        store = ExperimentStore()
        exp = seed(store, n=20)
        slow = Slow(error=timeout_error())
        with self.assertRaisesRegex(ProviderError, r"stopped after 3 consecutive timeouts.*re-run"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": slow}, sleep=NO_SLEEP)
        self.assertLess(slow.calls, 20)                                           # 40 answers x 2 attempts = 80 without it
        self.assertEqual(store.get_scores(exp), [])

    def test_connection_failures_keep_their_higher_threshold(self):
        store = ExperimentStore()
        exp = seed(store, n=20)
        slow = Slow(error=ProviderError("cannot reach x", retryable=True, kind="connection"))
        with self.assertRaisesRegex(ProviderError, r"stopped after 5 consecutive connection failures"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": slow}, sleep=NO_SLEEP)

    def test_a_run_records_nothing_for_timeouts_and_resume_completes_it(self):
        store = ExperimentStore()
        cfg = mock_cfg()
        cfg["concurrency"] = 4
        exp = create_experiment(store, load_jsonl(write_jsonl(
            [{"id": f"q{i}", "input": f"question {i}", "reference": f"a{i}"} for i in range(8)])), cfg)

        class Stuck(Provider):
            def __init__(self):
                self.stuck, self.calls = True, 0

            def complete(self, model, messages, params):
                self.calls += 1
                time.sleep(0.02)
                if self.stuck:
                    raise timeout_error()
                return Completion("fine", 5, 3)
        server = Stuck()
        with self.assertRaisesRegex(ProviderError, r"could not reach the provider and were not recorded.*resume " + exp):
            run_experiment(store, exp, providers={"m": server}, sleep=NO_SLEEP)
        self.assertEqual(store.get_results(exp), [])
        self.assertLess(server.calls, 16)                                         # 16 tasks x 2 attempts = 32 otherwise
        server.stuck = False
        self.assertEqual(run_experiment(store, exp, providers={"m": server}, sleep=NO_SLEEP), {"ran": 16, "remaining": 0})
        self.assertEqual({r["status"] for r in store.get_results(exp)}, {"ok"})


class JudgeTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.hang = Hang()
        self.addCleanup(self.hang.close)
        self.extra = {"hang": {"type": "ollama", "base_url": f"http://127.0.0.1:{self.hang.port}", "timeout": 300}}

    def test_judge_timeout_overrides_the_300s_in_the_config(self):
        store = ExperimentStore()
        exp = seed(store, n=10)
        started = time.perf_counter()
        with self.assertRaisesRegex(ProviderError, r"stopped after 3 consecutive timeouts \(timed out after 0\.2s"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "hang", "model": "x"},
                                extra_providers=self.extra, timeout=0.2, sleep=NO_SLEEP)
        self.assertLess(time.perf_counter() - started, 15)                        # not 3 x 2 x 300 s
        self.assertEqual(store.get_scores(exp), [])


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = seed(self.store, n=10)

    def probe(self, provider, **kw):
        judge = {**SPEC, **kw.pop("judge", {})}
        return probe_judge(self.store, self.exp, judge, n=3, providers={"m": provider}, **kw)

    def test_a_stuck_server_is_diagnosed_in_seconds_with_the_likely_causes(self):
        hang = Hang()
        self.addCleanup(hang.close)
        extra = {"hang": {"type": "ollama", "base_url": f"http://127.0.0.1:{hang.port}", "timeout": 300}}
        started = time.perf_counter()
        p = probe_judge(self.store, self.exp, {"provider": "hang", "model": "x"}, extra_providers=extra, timeout=0.3)
        self.assertLess(time.perf_counter() - started, 5)                         # the real probe waited 302 s
        self.assertEqual((len(p["calls"]), p["calls"][0]["kind"], p["mean_latency_s"], p["timeout"]), (1, "timeout", None, 0.3))
        advice = " ".join(p["advice"])
        for needle in ("did not finish within 0.3s", "think=false", "restart Ollama", "`ollama ps`", "task manager"):
            self.assertIn(needle, advice)
        text = format_probe(p)
        self.assertTrue(text.isascii())
        self.assertIn("timeout 0.3s", text)

    def test_the_probe_uses_the_smaller_of_its_own_and_the_configured_timeout(self):
        hang = Hang()
        self.addCleanup(hang.close)
        extra = {"hang": {"type": "ollama", "base_url": f"http://127.0.0.1:{hang.port}", "timeout": 0.4}}
        started = time.perf_counter()
        p = probe_judge(self.store, self.exp, {"provider": "hang", "model": "x"}, extra_providers=extra, timeout=60)
        self.assertEqual(p["timeout"], 0.4)                                       # min(configured 0.4 s, probe 60 s)
        self.assertLess(time.perf_counter() - started, 5)

    def test_output_is_capped_by_default_but_the_users_setting_wins(self):
        a, b, c, d = Slow(), Slow(), Slow(), Slow()
        self.probe(a)
        self.probe(b, judge={"params": {"temperature": 0, "max_tokens": 50}})
        self.probe(c, judge={"params": {"temperature": 0, "num_predict": 80}})
        self.probe(d, max_tokens=None)
        self.assertEqual(a.params[0]["max_tokens"], 300)
        self.assertEqual(b.params[0]["max_tokens"], 50)
        self.assertEqual((c.params[0].get("num_predict"), "max_tokens" in c.params[0]), (80, False))
        self.assertNotIn("max_tokens", d.params[0])

    def test_hitting_the_cap_without_a_verdict_is_diagnosed_as_reasoning(self):
        p = self.probe(Slow(text="<think>Let me consider the reference carefully. The question asks", out_tokens=300))
        advice = " ".join(p["advice"])
        self.assertIn("reached the 300-token limit without giving a verdict", advice)
        self.assertIn("think=false", advice)
        self.assertNotIn("could not be parsed", advice)                           # one diagnosis, not two
        self.assertEqual(p["max_tokens"], 300)

    def test_tokens_per_second_is_reported(self):
        p = self.probe(Slow(delay=0.05, out_tokens=40))
        self.assertGreater(p["calls"][0]["tokens_per_s"], 0)
        self.assertIn("tok/s", format_probe(p))


class CliTests(unittest.TestCase):
    def test_the_new_flags_are_accepted(self):
        import os
        import tempfile
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        store = ExperimentStore(db)
        exp = seed(store, n=4)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", db, "evaluate", exp, "--judge", "m:j", "--probe", "1", "--probe-timeout", "5"])
            code2 = main(["--db", db, "evaluate", exp, "--judge", "m:j", "--no-lexical", "--no-retrieval",
                          "--judge-timeout", "30", "--max-items", "2"])
        self.assertEqual((code, code2), (0, 0))
        self.assertIn("output capped at 300 tokens; timeout 5s", out.getvalue())


if __name__ == "__main__":
    unittest.main()
