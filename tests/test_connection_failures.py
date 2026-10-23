"""What happens when the model server is not running (the WinError 10061 situation)."""
import contextlib
import io
import json
import os
import tempfile
import time
import unittest

from llmobs_eval import (ExperimentStore, create_experiment, evaluate_experiment, load_jsonl, run_experiment)
from llmobs_eval.__main__ import main
from llmobs_eval.providers import Completion, OpenAICompatProvider, Provider, ProviderError
from tests.test_eval import mock_cfg, write_jsonl
from tests.test_step3 import NO_SLEEP, ScriptedJudge, seed

SPEC = {"provider": "m", "model": "judge-x"}
DOWN = "cannot reach http://localhost:11434/v1/chat/completions: [WinError 10061] connection refused"


def down():
    return ProviderError(DOWN, retryable=True)                 # no HTTP status: the server never answered


class Dead(Provider):
    """A server that is not there. Each attempt takes a moment, as a real refused/timed-out connection plus
    retries does; an instant failure would finish all the work before any breaker could react."""

    def __init__(self, delay=0.03):
        self.calls, self.delay = 0, delay

    def complete(self, model, messages, params):
        self.calls += 1
        time.sleep(self.delay)
        raise down()


class Outage(Provider):
    """Answers normally unless `down` is set; `fail_when` can single out particular requests."""

    def __init__(self, is_down=True, fail_when=None):
        self.down, self.fail_when, self.calls = is_down, fail_when, 0

    def complete(self, model, messages, params):
        self.calls += 1
        if self.down or (self.fail_when and self.fail_when(model, messages[-1]["content"])):
            raise down()
        return Completion(f"answer from {model}", 5, 3)


def rows(n):
    return [{"id": f"q{i}", "input": f"question {i}", "reference": f"answer {i}"} for i in range(n)]


class ProviderMessageTests(unittest.TestCase):
    def test_the_error_names_what_could_not_be_reached(self):
        with self.assertRaises(ProviderError) as cm:
            OpenAICompatProvider("http://127.0.0.1:1/v1", timeout=1).complete("m", [{"role": "user", "content": "x"}], {})
        e = cm.exception
        self.assertTrue(e.retryable)
        self.assertIsNone(e.status)
        self.assertRegex(str(e), r"^cannot reach http://127\.0\.0\.1:1/v1/chat/completions: ")


class EvaluateOutageTests(unittest.TestCase):
    def test_a_dead_provider_stops_the_evaluation_after_a_few_failures(self):
        store = ExperimentStore()
        exp = seed(store, n=20)                                                    # 40 answers to judge
        dead = Dead()
        with self.assertRaisesRegex(ProviderError, r"stopped after 5 consecutive connection failures.*re-run"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": dead},
                                sleep=NO_SLEEP)
        self.assertLess(dead.calls, 40)                                            # not 40 items x 2 attempts = 80
        self.assertEqual(store.get_scores(exp), [])                                # nothing recorded: a re-run redoes it

    def test_scattered_connection_failures_do_not_abort(self):
        store = ExperimentStore()
        exp = seed(store, n=20)

        class Spotty(ScriptedJudge):
            def complete(self, model, messages, params):
                if any(f"<candidate_answer>\nanswer {i}\n" in messages[0]["content"] for i in (3, 7, 11)):
                    raise down()
                return super().complete(model, messages, params)

        stats = evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC,
                                    providers={"m": Spotty()}, sleep=NO_SLEEP)
        # "answer 3" and "answer 7" are only base's output (cand says "wrong 3/7"); "answer 11" is both variants'
        self.assertEqual(stats["judge_failed"], 4)                                 # never 5 in a row, so no abort
        self.assertEqual(stats["scored"], 36)
        again = evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC,
                                    providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        self.assertEqual(again["scored"], 4)                                       # only the failed ones are redone

    def test_http_errors_are_per_item_and_never_trip_the_breaker(self):
        store = ExperimentStore()
        exp = seed(store, n=20)

        class Busy(Provider):
            def complete(self, model, messages, params):
                raise ProviderError("HTTP 503: busy", status=503, retryable=True)

        stats = evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": Busy()},
                                    sleep=NO_SLEEP)
        self.assertEqual((stats["judge_failed"], stats["scored"]), (40, 0))        # all attempted, none aborted
        self.assertIn("HTTP 503", stats["error_samples"][0])


class RunOutageTests(unittest.TestCase):
    def experiment(self, n=6):
        store = ExperimentStore()
        cfg = mock_cfg()
        cfg["concurrency"] = 4
        return store, create_experiment(store, load_jsonl(write_jsonl(rows(n))), cfg)

    def test_an_outage_records_nothing_and_resume_completes_the_experiment(self):
        store, exp = self.experiment()                                             # 6 examples x 2 variants = 12 calls
        server = Outage(is_down=True)
        with self.assertRaisesRegex(ProviderError, r"could not reach the provider and were not recorded.*resume " + exp):
            run_experiment(store, exp, providers={"m": server}, sleep=NO_SLEEP)
        self.assertEqual(store.get_results(exp), [])                               # no immutable error results
        self.assertEqual(store.get_experiment(exp)["status"], "partial")
        self.assertLess(server.calls, 20)                                          # stopped early (12 tasks x 2 attempts = 24)
        server.down = False                                                        # "start Ollama"
        out = run_experiment(store, exp, providers={"m": server}, sleep=NO_SLEEP)
        self.assertEqual(out, {"ran": 12, "remaining": 0})
        results = store.get_results(exp)
        self.assertEqual((len(results), {r["status"] for r in results}), (12, {"ok"}))   # no outage damage
        self.assertEqual(store.get_experiment(exp)["status"], "complete")

    def test_one_dropped_request_is_left_for_resume_and_everything_else_is_kept(self):
        store, exp = self.experiment()
        flaky = Outage(is_down=False, fail_when=lambda model, content: model == "model-b" and "question 2" in content)
        with self.assertRaisesRegex(ProviderError, r"1 call\(s\) could not reach the provider"):
            run_experiment(store, exp, providers={"m": flaky}, sleep=NO_SLEEP)
        self.assertEqual(len(store.get_results(exp)), 11)
        self.assertNotIn(("b", "q2"), store.done_keys(exp))
        self.assertEqual(store.get_experiment(exp)["status"], "partial")
        before = {(r["variant"], r["example_id"]): r["created_at"] for r in store.get_results(exp)}
        flaky.fail_when = None
        self.assertEqual(run_experiment(store, exp, providers={"m": flaky}, sleep=NO_SLEEP), {"ran": 1, "remaining": 0})
        after = {(r["variant"], r["example_id"]): r["created_at"] for r in store.get_results(exp)}
        self.assertEqual({k: after[k] for k in before}, before)                    # existing results untouched

    def test_http_failures_are_still_recorded_as_results(self):
        store, exp = self.experiment(3)

        class Rejects(Provider):
            def complete(self, model, messages, params):
                raise ProviderError("HTTP 400: context too long", status=400)      # the provider answered: that IS a result

        out = run_experiment(store, exp, providers={"m": Rejects()}, sleep=NO_SLEEP)
        self.assertEqual(out["remaining"], 0)
        results = store.get_results(exp)
        self.assertEqual((len(results), {r["status"] for r in results}), (6, {"error"}))
        self.assertIn("context too long", results[0]["error"])


class CliOutageTests(unittest.TestCase):
    def test_run_against_a_server_that_is_not_running(self):
        d = tempfile.mkdtemp()
        db, cfgp = os.path.join(d, "t.db"), os.path.join(d, "c.json")
        with open(cfgp, "w") as fh:
            json.dump({"providers": {"ollama": {"type": "openai_compat", "base_url": "http://127.0.0.1:1/v1", "timeout": 1}},
                       "variants": [{"label": "v", "provider": "ollama", "model": "llama3.2"}],
                       "concurrency": 2, "max_retries": 0}, fh)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", db, "run", "--dataset", write_jsonl(rows(12)), "--config", cfgp])
        self.assertEqual(code, 2)
        self.assertIn("cannot reach http://127.0.0.1:1/v1/chat/completions", err.getvalue())
        self.assertIn("resume exp_", err.getvalue())
        store = ExperimentStore(db)
        exp = store.list_experiments()[0]
        self.assertEqual((exp["status"], store.get_results(exp["id"])), ("partial", []))     # nothing was ruined


if __name__ == "__main__":
    unittest.main()
