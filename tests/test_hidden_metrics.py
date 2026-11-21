import contextlib
import io
import os
import tempfile
import unittest

from llmobs_eval import (ExperimentStore, calibration_payload, compare_payload, evaluate_experiment, examples_payload,
                         format_compare, gate_experiment, judge_check_payload, judge_table_payload, parse_rules)
from llmobs_eval.__main__ import main
from tests.test_judge_reliability import FLAWED, judged, truth_labels
from tests.test_step3 import NO_SLEEP, ScriptedJudge

KEYS = {("cand", "q0"), ("cand", "q5"), ("base", "q10"), ("base", "q3"), ("cand", "q10")}


def two_judges():
    """judge-x scored everything; judge-y (the half-finished one we want to retire) only 5 labeled answers."""
    store = ExperimentStore()
    exp = judged(store, FLAWED, model="judge-x")
    evaluate_experiment(store, exp)                                                  # lexical metrics too
    truth_labels(store, exp, only=KEYS)
    evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "m", "model": "judge-y"},
                        only_labeled=True, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
    return store, exp


class HiddenMetricTests(unittest.TestCase):
    def setUp(self):
        self.store, self.exp = two_judges()

    def metrics(self, **kw):
        return compare_payload(self.store, self.exp, n_boot=50, **kw)["metrics"]

    def test_hiding_removes_a_metric_from_views_but_not_from_the_database(self):
        self.assertIn("judge:judge-y", self.metrics())
        self.store.hide_metric(self.exp, "judge:judge-y")
        self.assertNotIn("judge:judge-y", self.metrics())
        self.assertIn("judge:judge-x", self.metrics())
        self.assertNotIn("judge-y", format_compare(compare_payload(self.store, self.exp, n_boot=50)))
        self.assertEqual(sum(1 for s in self.store.get_scores(self.exp) if s["metric"] == "judge:judge-y"), 5)   # kept
        self.store.unhide_metric(self.exp, "judge:judge-y")
        self.assertIn("judge:judge-y", self.metrics())

    def test_naming_a_hidden_metric_overrides_the_hiding(self):
        self.store.hide_metric(self.exp, "judge:judge-y")
        self.assertIn("judge:judge-y", self.metrics(metric="judge:judge-y"))
        self.assertEqual(compare_payload(self.store, self.exp, metric="judge:judge-y", n_boot=50)["primary_metric"],
                         "judge:judge-y")

    def test_warnings_about_a_hidden_metric_disappear_with_it(self):
        before = " ".join(compare_payload(self.store, self.exp, n_boot=50)["warnings"])
        self.assertIn("judge:judge-y", before)                                       # partial coverage warning
        self.store.hide_metric(self.exp, "judge:judge-y")
        self.assertNotIn("judge-y", " ".join(compare_payload(self.store, self.exp, n_boot=50)["warnings"]))

    def test_example_view_skips_it_too(self):
        self.store.hide_metric(self.exp, "judge:judge-y")
        self.assertNotIn("judge:judge-y", examples_payload(self.store, self.exp)["metrics"])

    def test_judge_tools_default_to_the_visible_judge_and_still_accept_the_hidden_one_by_name(self):
        self.store.hide_metric(self.exp, "judge:judge-x")                            # retire the OTHER way round
        p = judge_check_payload(self.store, self.exp)
        self.assertEqual((p["metric"], p["available"]), ("judge:judge-y", ["judge:judge-y"]))
        named = judge_check_payload(self.store, self.exp, metric="judge:judge-x")
        self.assertEqual(named["metric"], "judge:judge-x")
        self.assertEqual(calibration_payload(self.store, self.exp, metric="judge:judge-x", n_boot=50)["metric"], "judge:judge-x")
        self.assertEqual([r["metric"] for r in judge_table_payload(self.store, self.exp, n_boot=50)["judges"]], ["judge:judge-y"])

    def test_a_gate_can_still_use_a_hidden_metric(self):
        self.store.hide_metric(self.exp, "judge:judge-x")
        rules = parse_rules({"metrics": {"judge:judge-x": 0.9}, "n_boot": 100})
        self.assertEqual(gate_experiment(self.store, self.exp, rules)["status"], "pass")

    def test_hiding_is_per_experiment(self):
        other = judged(self.store, FLAWED, model="judge-x")
        self.store.hide_metric(self.exp, "judge:judge-x")
        self.assertEqual(self.store.hidden_metrics(other), set())


class CliTests(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "t.db")
        self.store = ExperimentStore(self.db)
        self.exp = judged(self.store, FLAWED, model="judge-x")
        truth_labels(self.store, self.exp, only=KEYS)
        evaluate_experiment(self.store, self.exp, lexical=False, retrieval=False, judge={"provider": "m", "model": "judge-y"},
                            only_labeled=True, providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", self.db, *argv])
        return code, out.getvalue()

    def test_list_hide_unhide(self):
        code, out = self.cli("metrics", self.exp, "list")
        self.assertEqual(code, 0)
        self.assertIn("judge:judge-y", out)
        self.assertIn("5 of 40 answers", out)                                        # coverage is visible
        code, out = self.cli("metrics", self.exp, "hide", "judge:judge-y")
        self.assertEqual(code, 0)
        self.assertRegex(out, r"judge:judge-y\s+hidden")
        self.assertIn("kept in the database", out)
        code, out = self.cli("compare", self.exp)
        self.assertNotIn("judge-y", out)
        self.assertIn("judge:judge-x", out)
        self.assertRegex(self.cli("metrics", self.exp, "unhide", "judge:judge-y")[1], r"judge:judge-y\s+shown")

    def test_errors_exit_cleanly(self):
        self.assertEqual(self.cli("metrics", self.exp, "hide")[0], 2)
        self.assertEqual(self.cli("metrics", self.exp, "hide", "judge:nope")[0], 2)
        self.assertEqual(self.cli("metrics", "exp_missing", "list")[0], 2)


if __name__ == "__main__":
    unittest.main()
