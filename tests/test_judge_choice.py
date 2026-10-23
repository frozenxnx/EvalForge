import argparse
import contextlib
import io
import os
import shutil
import tempfile
import unittest

from llmobs_eval import (ConfigError, ExperimentStore, compare_payload, evaluate_experiment, format_compare,
                         format_judge_table, judge_table_payload)
from llmobs_eval import faithfulness as faith
from llmobs_eval.__main__ import _judge_spec, main
from llmobs_eval.judge import _STRICT_RULES, build_prompt, judge_metric_name, parse_verdict
from llmobs_eval.providers import Completion, Provider
from llmobs_eval.textutil import strip_reasoning
from tests.test_examples import node_render
from tests.test_judge_reliability import FLAWED, judged, truth_labels
from tests.test_step3 import NO_SLEEP, ScriptedJudge, seed

SPEC = {"provider": "m", "model": "judge-x"}
OK = '{"verdict": "correct", "reason": "ok"}'


class Recorder(Provider):
    """Records every request; optionally answers garbage first so the retry nudge is exercised."""

    def __init__(self, garbage_first=False):
        self.calls, self.garbage_first = [], garbage_first

    def complete(self, model, messages, params):
        self.calls.append({"messages": messages, "params": params})
        if self.garbage_first and len(messages) == 1:
            return Completion("hmm")
        return Completion(OK)


class ReasoningBlockTests(unittest.TestCase):
    def test_strip_reasoning_shapes(self):
        self.assertEqual(strip_reasoning("<think>pondering</think>answer").strip(), "answer")
        self.assertEqual(strip_reasoning("<THINK>a\nb</Think>\n{\"x\": 1}").strip(), '{"x": 1}')
        self.assertEqual(strip_reasoning("pondering</think>answer").strip(), "answer")       # opening tag removed by server
        self.assertEqual(strip_reasoning("<think>cut off mid thought").strip(), "")          # nothing to parse
        self.assertEqual(strip_reasoning("no reasoning here"), "no reasoning here")
        self.assertEqual(strip_reasoning(None), "")

    def test_a_verdict_inside_the_thinking_is_not_the_answer(self):
        text = ('<think>Maybe {"verdict": "correct", "reason": "x"} ... no, the value differs.</think>'
                '{"verdict": "incorrect", "reason": "different value"}')
        self.assertEqual(parse_verdict(text), {"verdict": "incorrect", "reason": "different value"})
        self.assertIsNone(parse_verdict('<think>{"verdict": "correct", "reason": "x"}'))      # never finished
        self.assertEqual(parse_verdict('thoughts</think>{"verdict": "correct", "reason": "r"}')["verdict"], "correct")

    def test_faithfulness_parsers_ignore_thinking_too(self):
        self.assertEqual(faith.parse_claims('<think>{"claims": ["wrong"]}</think>{"claims": ["right"]}'), ["right"])
        self.assertEqual(faith.parse_verdicts('<think>hm</think>{"verdicts": [{"id": 1, "verdict": "supported"}]}', 1),
                         ["supported"])


class RubricTests(unittest.TestCase):
    def test_v2_only_adds_rules_and_v1_is_unchanged(self):
        v1, v2 = build_prompt("Q?", "24", "I don't know"), build_prompt("Q?", "24", "I don't know", rubric="v2")
        self.assertEqual(v2.replace(_STRICT_RULES + "\n\n", ""), v1)
        self.assertNotIn("does not know", v1)
        for needle in ("take precedence", "a specific value", "says it does not know", "approximate"):
            self.assertIn(needle, v2)
        self.assertIn("<candidate_answer>\nI don't know\n</candidate_answer>", v2)
        with self.assertRaisesRegex(ValueError, "unknown rubric"):
            build_prompt("q", "r", "c", rubric="v9")

    def test_metric_names_keep_rubrics_apart(self):
        self.assertEqual(judge_metric_name("qwen3:4b"), "judge:qwen3:4b")
        self.assertEqual(judge_metric_name("qwen3:4b", "v2"), "judge:qwen3:4b#v2")

    def test_v1_and_v2_scores_coexist_on_one_experiment(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)                                       # v1 judge-x
        prov = Recorder()
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "rubric": "v2"},
                            providers={"m": prov}, sleep=NO_SLEEP)
        metrics = {s["metric"] for s in store.get_scores(exp)}
        self.assertEqual(metrics, {"judge:judge-x", "judge:judge-x#v2"})
        v2 = [s for s in store.get_scores(exp) if s["metric"] == "judge:judge-x#v2"]
        self.assertEqual({s["detail"]["rubric"] for s in v2}, {"v2"})
        self.assertTrue(all("take precedence" in c["messages"][-1]["content"] for c in prov.calls))
        self.assertEqual({s["detail"]["rubric"] for s in store.get_scores(exp) if s["metric"] == "judge:judge-x"}, {"v1"})
        with self.assertRaisesRegex(ConfigError, "unknown rubric"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "rubric": "v9"},
                                providers={"m": prov})

    def test_self_judging_warning_sees_through_the_rubric_tag(self):
        store = ExperimentStore()
        exp = seed(store, n=6)
        evaluate_experiment(store, exp, lexical=False, retrieval=False,
                            judge={"provider": "m", "model": "model-a", "rubric": "v2"},
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        text = " ".join(compare_payload(store, exp, n_boot=50)["warnings"])
        self.assertIn("judge:model-a#v2: the judge model is also a candidate", text)


class RequestShapingTests(unittest.TestCase):
    def test_suffix_is_appended_to_every_request_including_the_retry_nudge(self):
        store = ExperimentStore()
        exp = seed(store, n=2, n_wrong=0)
        prov = Recorder(garbage_first=True)
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "suffix": "/no_think"},
                            providers={"m": prov}, sleep=NO_SLEEP)
        self.assertEqual(len(prov.calls), 8)                              # 4 items x (ask + nudge)
        for c in prov.calls:
            self.assertTrue(c["messages"][-1]["content"].endswith("\n\n/no_think"))
        self.assertEqual(sorted({len(c["messages"]) for c in prov.calls}), [1, 3])
        first = prov.calls[0]["messages"][0]["content"]
        self.assertIn("<candidate_answer>", first)                        # the original prompt is intact

    def test_no_suffix_means_the_prompt_is_untouched(self):
        store = ExperimentStore()
        prov = Recorder()
        evaluate_experiment(store, seed(store, n=2), lexical=False, retrieval=False, judge=SPEC,
                            providers={"m": prov}, sleep=NO_SLEEP)
        self.assertTrue(all(c["messages"][-1]["content"].endswith("</candidate_answer>\n\nJudge ONLY") is False and
                            "/no_think" not in c["messages"][-1]["content"] for c in prov.calls))

    def test_params_default_to_temperature_zero_and_can_be_extended(self):
        store = ExperimentStore()
        exp = seed(store, n=2)
        a, b = Recorder(), Recorder()
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, providers={"m": a}, sleep=NO_SLEEP)
        self.assertEqual({str(c["params"]) for c in a.calls}, {str({"temperature": 0})})
        evaluate_experiment(store, exp, lexical=False, retrieval=False,
                            judge={"provider": "m", "model": "judge-y", "params": {"temperature": 0, "think": False}},
                            providers={"m": b}, sleep=NO_SLEEP)
        self.assertEqual(b.calls[0]["params"], {"temperature": 0, "think": False})

    def test_cli_builds_the_judge_spec(self):
        ns = argparse.Namespace(judge="ollama:qwen3:4b", rubric="v2", judge_suffix="/no_think",
                                judge_param=["think=false", "reasoning_effort=none", "n=2", "temperature=0.2"])
        self.assertEqual(_judge_spec(ns), {
            "provider": "ollama", "model": "qwen3:4b", "suffix": "/no_think", "rubric": "v2",
            "params": {"temperature": 0.2, "think": False, "reasoning_effort": "none", "n": 2}})
        plain = argparse.Namespace(judge="ollama:qwen2.5:3b", rubric="v1", judge_suffix=None, judge_param=None)
        self.assertEqual(_judge_spec(plain), {"provider": "ollama", "model": "qwen2.5:3b"})
        self.assertIsNone(_judge_spec(argparse.Namespace(judge=None, rubric="v1", judge_suffix=None, judge_param=None)))
        with self.assertRaisesRegex(ConfigError, "KEY=VALUE"):
            _judge_spec(argparse.Namespace(judge="a:b", rubric="v1", judge_suffix=None, judge_param=["oops"]))


class OnlyLabeledTests(unittest.TestCase):
    def test_judges_only_what_a_human_labeled(self):
        store = ExperimentStore()
        exp = seed(store, n=10)
        keys = [("base", "q0"), ("cand", "q1"), ("cand", "q2"), ("base", "q9"), ("cand", "q9")]
        for v, e in keys:
            store.add_label(exp, v, e, 1.0, "me", "random")
        prov = Recorder()
        stats = evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, only_labeled=True,
                                    providers={"m": prov}, sleep=NO_SLEEP)
        self.assertEqual(stats["scored"], 5)
        self.assertEqual(len(prov.calls), 5)
        self.assertEqual({(s["variant"], s["example_id"]) for s in store.get_scores(exp)}, set(keys))
        again = evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, only_labeled=True,
                                    providers={"m": prov}, sleep=NO_SLEEP)
        self.assertEqual(again["scored"], 0)

    def test_needs_labels(self):
        store = ExperimentStore()
        exp = seed(store, n=4)
        with self.assertRaisesRegex(ConfigError, "needs human labels"):
            evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, only_labeled=True,
                                providers={"m": Recorder()})


class JudgeTableTests(unittest.TestCase):
    def build(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED, model="judge-x")
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "model": "judge-y"},
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        truth_labels(store, exp)
        return store, exp

    def test_two_judges_on_the_same_labels(self):
        store, exp = self.build()
        p = judge_table_payload(store, exp, n_boot=200)
        rows = {r["metric"]: r for r in p["judges"]}
        self.assertEqual(p["n_labels"], 40)
        self.assertEqual((rows["judge:judge-x"]["n"], rows["judge:judge-y"]["n"]), (40, 40))
        self.assertAlmostEqual(rows["judge:judge-x"]["binary_kappa"], 0.6875)
        self.assertAlmostEqual(rows["judge:judge-y"]["binary_kappa"], 1.0)
        self.assertAlmostEqual(rows["judge:judge-x"]["bias_by_variant"]["cand"], 0.05)
        self.assertEqual((rows["judge:judge-x"]["lenient"], rows["judge:judge-x"]["strict"]), (2, 2))
        self.assertEqual((rows["judge:judge-y"]["lenient"], rows["judge:judge-y"]["strict"]), (0, 0))

    def test_terminal_table(self):
        store, exp = self.build()
        text = format_judge_table(judge_table_payload(store, exp, n_boot=100))
        self.assertTrue(text.isascii())
        for needle in ("judges compared on the same 40 human labels", "judge:judge-x", "judge:judge-y", "0.69", "1.00",
                       "2/2 of 40", "0/0 of 40", "bias by variant", "judge:judge-x: base -0.05, cand +0.05"):
            self.assertIn(needle, text)

    def test_a_judge_scored_on_fewer_labels_shows_its_own_n(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        truth_labels(store, exp)
        small = Recorder()
        store.add_label(exp, "base", "q3", 1.0, "me2", "random")          # already labeled; unaffected
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "model": "judge-z"},
                            only_labeled=True, providers={"m": small}, sleep=NO_SLEEP)
        rows = {r["metric"]: r for r in judge_table_payload(store, exp, n_boot=50)["judges"]}
        self.assertEqual((rows["judge:judge-x"]["n"], rows["judge:judge-z"]["n"]), (40, 40))

    def test_no_judges(self):
        store = ExperimentStore()
        text = format_judge_table(judge_table_payload(store, seed(store, n=3)))
        self.assertIn("no judge scores yet", text)


class PartialScoringTests(unittest.TestCase):
    """A judge run with --only-labeled covers only a few answers; the output must say so, never show a
    dramatic delta from a handful of pairs, and judges must be compared on the answers they share."""

    def partial_experiment(self):
        store = ExperimentStore()
        exp = seed(store, n=20, n_wrong=8)
        evaluate_experiment(store, exp)                                         # lexical metrics: all 40 answers
        for v in ("base", "cand"):
            for e in ("q0", "q1", "q2"):
                store.add_label(exp, v, e, 1.0, "me", "random")
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge=SPEC, only_labeled=True,
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        return store, exp

    def test_terminal_table_shows_partial_coverage_and_refuses_tiny_deltas(self):
        store, exp = self.partial_experiment()
        text = format_compare(compare_payload(store, exp, n_boot=100))
        self.assertIn("(3/20)", text)                                           # judge scored 3 of 20 answers
        self.assertNotIn("(20/20)", text)                                       # fully scored metrics stay clean
        line = next(l for l in text.splitlines() if "judge-x" in l and "too few" in l)
        self.assertIn("too few pairs to compare (n=3)", line)
        self.assertFalse(any("judge-x" in l and "-1.00 [" in l for l in text.splitlines()))   # the old misleading line
        self.assertTrue(any(l.strip().startswith("cand") and "contains" in l and "n=20" in l for l in text.splitlines()))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_marks_partial_coverage_and_tiny_deltas(self):
        store, exp = self.partial_experiment()
        h = node_render("renderCompare", compare_payload(store, exp, n_boot=100))
        self.assertIn("n=3/20", h)
        self.assertIn("n=3: too few pairs to compare", h)
        self.assertNotIn("undefined", h)
        self.assertNotIn("NaN", h)

    def test_judges_are_compared_on_the_answers_they_share(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED, model="judge-x")                            # judge-x scored all 40 answers
        keys = {("cand", "q0"), ("cand", "q5"), ("base", "q10"), ("base", "q3"), ("cand", "q10")}
        truth_labels(store, exp, only=keys)
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={**SPEC, "model": "judge-y"},
                            only_labeled=True, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)   # judge-y: 5 answers
        p = judge_table_payload(store, exp, n_boot=100)
        rows = {r["metric"]: r for r in p["judges"]}
        self.assertEqual(p["common_items"], 5)
        # judge-x's mistakes on the 5 shared answers: lenient on cand/q0, strict on base/q10 and cand/q10
        self.assertEqual((rows["judge:judge-x"]["checkable"], rows["judge:judge-x"]["lenient"], rows["judge:judge-x"]["strict"]),
                         (5, 1, 2))
        self.assertEqual((rows["judge:judge-y"]["checkable"], rows["judge:judge-y"]["lenient"], rows["judge:judge-y"]["strict"]),
                         (5, 0, 0))
        text = format_judge_table(p)
        self.assertIn("1/2 of 5", text)
        self.assertIn("0/0 of 5", text)
        self.assertIn("counted on the 5 answers every judge scored", text)

    def test_a_single_judge_is_checked_on_everything_it_scored(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        truth_labels(store, exp)
        p = judge_table_payload(store, exp, n_boot=50)
        self.assertIsNone(p["common_items"])
        self.assertEqual(p["judges"][0]["checkable"], 40)
        self.assertNotIn("every judge scored", format_judge_table(p))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        store = ExperimentStore(self.db)
        self.exp = judged(store, FLAWED)
        truth_labels(store, self.exp)

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--db", self.db, *argv])
        return code, out.getvalue()

    def test_evaluate_with_all_the_new_flags_and_compare_judges(self):
        code, out = self.run_cli("evaluate", self.exp, "--judge", "m:judge-z", "--rubric", "v2",
                                 "--judge-param", "temperature=0.3", "--judge-param", "think=false",
                                 "--judge-suffix", "/no_think", "--only-labeled", "--no-lexical", "--no-retrieval")
        self.assertEqual(code, 0)
        metrics = {s["metric"] for s in ExperimentStore(self.db).get_scores(self.exp)}
        self.assertIn("judge:judge-z#v2", metrics)                         # its own metric, apart from v1
        code, out = self.run_cli("calibrate", self.exp, "--all")
        self.assertEqual(code, 0)
        self.assertIn("judge:judge-x", out)
        self.assertIn("judge:judge-z#v2", out)
        self.assertIn("judges compared on the same 40 human labels", out)

    def test_bad_arguments_exit_cleanly(self):
        self.assertEqual(self.run_cli("evaluate", self.exp, "--judge", "m:j", "--judge-param", "oops")[0], 2)
        self.assertEqual(self.run_cli("evaluate", self.exp, "--judge", "m:j", "--rubric", "v1", "--only-labeled",
                                      "--no-correctness")[0], 2)           # nothing to judge
        plain = ExperimentStore(self.db)
        exp2 = seed(plain, n=3)
        self.assertEqual(self.run_cli("evaluate", exp2, "--judge", "m:j", "--only-labeled")[0], 2)   # no labels


if __name__ == "__main__":
    unittest.main()
