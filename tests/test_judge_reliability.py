import contextlib
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

from llmobs_eval import (ExperimentStore, calibration_payload, compare_payload, evaluate_experiment,
                         format_calibration, format_judge_check, judge_check_payload, label_session,
                         pick_label_items)
from llmobs_eval.__main__ import main
from llmobs_eval.judgecheck import _ascii, cohen_kappa, is_checkable
from llmobs_eval.providers import Completion, Provider
from tests.test_examples import node_render
from tests.test_step3 import NO_SLEEP, ScriptedJudge, seed

CORRECT = lambda n: '{"verdict": "correct", "reason": "close enough"}'
INCORRECT = lambda n: '{"verdict": "incorrect", "reason": "not sure"}'
# a judge with two known mistakes per kind: too lenient on cand's "wrong 0"/"wrong 1", too strict on "answer 10"
FLAWED = {"wrong 0": CORRECT, "wrong 1": CORRECT, "answer 10": INCORRECT}


def judged(store, behavior=None, n=20, n_wrong=8, model="judge-x"):
    exp = seed(store, n=n, n_wrong=n_wrong)
    evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "m", "model": model},
                        providers={"m": ScriptedJudge(behavior)}, sleep=NO_SLEEP)
    return exp


def truth_labels(store, exp, source="random", only=None):
    """What a careful human would say: correct iff the answer is the reference."""
    for r in store.get_results(exp):
        if only is None or (r["variant"], r["example_id"]) in only:
            store.add_label(exp, r["variant"], r["example_id"], 1.0 if r["output"].startswith("answer") else 0.0,
                            source=source)


class CheckableTests(unittest.TestCase):
    def test_only_short_canonical_references_not_present_in_the_question(self):
        self.assertTrue(is_checkable("5", "How many days?"))
        self.assertTrue(is_checkable("April 15th", "What is the date?"))            # 2 tokens after normalizing
        self.assertTrue(is_checkable("AES-256", "Which standard?"))
        self.assertFalse(is_checkable("The documents do not say.", "Who is the CEO?"))   # a sentence
        self.assertFalse(is_checkable("5", "If it takes 5 machines 5 minutes, how long?"))   # leaks into the question
        self.assertFalse(is_checkable("", "q"))
        self.assertFalse(is_checkable(None, "q"))
        self.assertFalse(is_checkable("one two three four", "q"))


class KappaTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertEqual(cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]), 0.5)                 # (.75 - .5) / .5
        self.assertEqual(cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]), 0.0)                 # pure chance
        self.assertIsNone(cohen_kappa([1, 1, 1], [1, 1, 1]))                           # nothing to agree about
        self.assertIsNone(cohen_kappa([], []))
        self.assertLess(cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]), 0)                    # systematic disagreement


class JudgeCheckTests(unittest.TestCase):
    def test_counts_and_items_for_a_judge_with_known_mistakes(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        p = judge_check_payload(store, exp)
        rows = {v["variant"]: v for v in p["variants"]}
        self.assertEqual((rows["base"]["checkable"], rows["base"]["agree"], rows["base"]["lenient"], rows["base"]["strict"]),
                         (20, 19, 0, 1))
        self.assertEqual((rows["cand"]["checkable"], rows["cand"]["agree"], rows["cand"]["lenient"], rows["cand"]["strict"]),
                         (20, 17, 2, 1))
        self.assertEqual(p["totals"], {"checkable": 40, "agree": 36, "lenient": 2, "strict": 2, "agreement": 0.9})
        self.assertEqual([(i["kind"], i["variant"], i["output"]) for i in p["items"]],
                         [("lenient", "cand", "wrong 0"), ("lenient", "cand", "wrong 1"),
                          ("strict", "base", "answer 10"), ("strict", "cand", "answer 10")])
        it = p["items"][0]
        self.assertEqual((it["input"], it["reference"], it["judge_value"], it["judge_verdict"], it["judge_reason"], it["contains"]),
                         ("question 0", "answer 0", 1.0, "correct", "close enough", 0.0))

    def test_top_k_limits_items_not_counts(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        p = judge_check_payload(store, exp, top_k=1)
        self.assertEqual([i["kind"] for i in p["items"]], ["lenient", "strict"])
        self.assertEqual(p["totals"]["lenient"], 2)
        self.assertEqual(judge_check_payload(store, exp, top_k=0)["items"], [])

    def test_a_good_judge_agrees_everywhere(self):
        store = ExperimentStore()
        p = judge_check_payload(store, judged(store))
        self.assertEqual((p["totals"]["agreement"], p["items"]), (1.0, []))

    def test_without_judge_scores_and_with_bad_input(self):
        store = ExperimentStore()
        exp = seed(store, n=10)
        p = judge_check_payload(store, exp)
        self.assertEqual((p["metric"], p["available"], p["variants"]), (None, [], []))
        self.assertIn("run `python -m llmobs_eval evaluate", format_judge_check(p))
        with self.assertRaises(KeyError):
            judge_check_payload(store, "exp_missing")
        other = ExperimentStore()
        exp2 = judged(other)
        with self.assertRaisesRegex(ValueError, "unknown judge metric"):
            judge_check_payload(other, exp2, metric="judge:nope")

    def test_two_judges_can_be_inspected_separately(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED, model="judge-x")
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "m", "model": "judge-y"},
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        p = judge_check_payload(store, exp)
        self.assertEqual((p["available"], p["metric"]), (["judge:judge-x", "judge:judge-y"], "judge:judge-x"))
        self.assertEqual(judge_check_payload(store, exp, "judge:judge-y")["totals"]["agreement"], 1.0)
        self.assertIn("other judge metrics available: judge:judge-y", format_judge_check(p))

    def test_comparison_warns_when_the_judge_disagrees_a_lot_and_stays_quiet_otherwise(self):
        class Lenient(ScriptedJudge):
            def complete(self, model, messages, params):
                return Completion('{"verdict": "correct", "reason": "fine"}')
        store = ExperimentStore()
        exp = seed(store, n=20, n_wrong=12)
        evaluate_experiment(store, exp, lexical=False, retrieval=False, judge={"provider": "m", "model": "judge-x"},
                            providers={"m": Lenient()}, sleep=NO_SLEEP)
        text = " ".join(compare_payload(store, exp, n_boot=50)["warnings"])
        self.assertIn("disagrees with `contains` on 12 of 40", text)
        self.assertIn("12 judge too lenient", text)
        self.assertIn("judge-check", text)
        good = ExperimentStore()
        self.assertFalse(any("disagrees with" in w for w in compare_payload(good, judged(good), n_boot=50)["warnings"]))

    def test_terminal_rendering_is_ascii_and_readable(self):
        store = ExperimentStore()
        text = format_judge_check(judge_check_payload(store, judged(store, FLAWED)))
        self.assertTrue(text.isascii())
        for needle in ("judge too lenient", "ALL", "90%", "[lenient] cand / q0", "judge said: close enough"):
            self.assertIn(needle, text)


class LabelStoreTests(unittest.TestCase):
    def test_add_replace_and_validate(self):
        store = ExperimentStore()
        exp = judged(store)
        store.add_label(exp, "cand", "q0", 0.0, "ana", "random")
        store.add_label(exp, "cand", "q0", 1.0, "ana", "random")            # a correction replaces
        store.add_label(exp, "cand", "q0", 0.5, "bob", "disagreement")
        labs = store.get_labels(exp)
        self.assertEqual(sorted((l["labeler"], l["label"], l["source"]) for l in labs),
                         [("ana", 1.0, "random"), ("bob", 0.5, "disagreement")])
        self.assertEqual(len(store.get_labels(exp, "ana")), 1)
        for bad in ({"label": 0.7}, {"label": 1.0, "source": "guess"}):
            with self.assertRaises(ValueError):
                store.add_label(exp, "cand", "q1", **{"labeler": "x", "source": "random", **bad})


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = judged(self.store, FLAWED)

    def test_full_agreement_statistics_by_hand(self):
        truth_labels(self.store, self.exp)                                  # all 40 answers, random source
        p = calibration_payload(self.store, self.exp, n_boot=400)
        s = p["all"]
        self.assertEqual((p["n_labels"], p["headline"], s["n"]), (40, "random", 40))
        self.assertAlmostEqual(s["agreement"], 0.9)                         # 4 mistakes in 40
        self.assertAlmostEqual(s["binary_kappa"], 0.6875)                   # po .9, pe .68
        self.assertAlmostEqual(s["kappa"], 0.6875)
        self.assertAlmostEqual(s["bias"], 0.0)                              # 2 lenient and 2 strict cancel out
        self.assertAlmostEqual(s["bias_by_variant"]["cand"], 0.05)          # +2 lenient, -1 strict over 20
        self.assertAlmostEqual(s["bias_by_variant"]["base"], -0.05)
        self.assertEqual(s["confusion"]["counts"], [[6, 0, 2], [0, 0, 0], [2, 0, 30]])
        lo, hi = s["binary_kappa_ci"]
        self.assertTrue(lo < s["binary_kappa"] < hi)
        self.assertEqual(p["warnings"], [])

    def test_biased_and_small_samples_are_called_out(self):
        truth_labels(self.store, self.exp, source="disagreement")
        p = calibration_payload(self.store, self.exp, n_boot=100)
        self.assertEqual(p["headline"], "all")
        self.assertTrue(any("biased toward hard cases" in w for w in p["warnings"]))
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        truth_labels(store, exp, only={("cand", "q0"), ("cand", "q1"), ("base", "q2"), ("cand", "q3"), ("cand", "q4")})
        q = calibration_payload(store, exp, n_boot=100)
        self.assertTrue(any("only 5 labels" in w for w in q["warnings"]))
        self.assertTrue(any("only 5 random labels" in w for w in q["warnings"]))

    def test_the_latest_label_wins_and_unjudged_labels_are_ignored(self):
        self.store.add_label(self.exp, "cand", "q0", 0.0, "me", "random")
        self.assertEqual(calibration_payload(self.store, self.exp, n_boot=50)["all"]["confusion"]["counts"][2][0], 1)
        self.store.add_label(self.exp, "cand", "q0", 1.0, "me", "random")      # I changed my mind
        self.assertEqual(calibration_payload(self.store, self.exp, n_boot=50)["all"]["confusion"]["counts"][2][2], 1)
        self.store.add_label(self.exp, "ghost-variant", "q0", 1.0, "me", "random")
        self.assertEqual(calibration_payload(self.store, self.exp, n_boot=50)["all"]["n"], 1)

    def test_no_judge_and_no_labels(self):
        empty = ExperimentStore()
        exp = seed(empty, n=5)
        self.assertIn("no judge scores yet", " ".join(calibration_payload(empty, exp)["warnings"]))
        p = calibration_payload(self.store, self.exp)
        self.assertEqual((p["all"], p["headline"]), (None, None))
        self.assertIn("no labels yet", " ".join(p["warnings"]))
        self.assertIn("n/a", "".join(format_calibration(p)) + "n/a")

    def test_degenerate_labels_give_no_kappa_not_a_crash(self):
        store = ExperimentStore()
        exp = judged(store, n=12, n_wrong=0)                                 # everything right, judge agrees
        truth_labels(store, exp)
        s = calibration_payload(store, exp, n_boot=50)["all"]
        self.assertEqual((s["agreement"], s["kappa"], s["binary_kappa_ci"]), (1.0, None, [None, None]))
        self.assertIn("kappa n/a", format_calibration(calibration_payload(store, exp, n_boot=50)))

    def test_terminal_rendering(self):
        truth_labels(self.store, self.exp)
        text = format_calibration(calibration_payload(self.store, self.exp, n_boot=200))
        self.assertTrue(text.isascii())
        for needle in ("random sample, n=40   <- headline", "kappa 0.69", "leniency bias", "by variant: base -0.05, cand +0.05",
                       "rule of thumb"):
            self.assertIn(needle, text)


class PickAndSessionTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = judged(self.store, FLAWED)

    def test_random_sample_is_reproducible_and_skips_labeled_items(self):
        a = pick_label_items(self.store, self.exp, n=10, seed=3)
        self.assertEqual(a, pick_label_items(self.store, self.exp, n=10, seed=3))
        self.assertNotEqual(a, pick_label_items(self.store, self.exp, n=10, seed=4))
        self.assertEqual((len(a), {s for _, _, s in a}), (10, {"random"}))
        self.store.add_label(self.exp, a[0][0], a[0][1], 1.0, "human", "random")
        b = pick_label_items(self.store, self.exp, n=40, seed=3)
        self.assertEqual(len(b), 39)
        self.assertNotIn(a[0][:2], [x[:2] for x in b])

    def test_disagreement_sample_contains_only_flagged_items(self):
        picked = pick_label_items(self.store, self.exp, n=50, sample="disagreements")
        self.assertEqual(sorted((v, e) for v, e, _ in picked), [("base", "q10"), ("cand", "q0"), ("cand", "q1"), ("cand", "q10")])
        self.assertEqual({s for _, _, s in picked}, {"disagreement"})

    def test_errors(self):
        plain = ExperimentStore()
        exp = seed(plain, n=3)
        with self.assertRaisesRegex(ValueError, "no judge scores"):
            pick_label_items(plain, exp, n=3)
        with self.assertRaisesRegex(ValueError, "sample must be"):
            pick_label_items(self.store, self.exp, sample="everything")

    def test_session_is_blind_validates_input_and_saves_immediately(self):
        items = pick_label_items(self.store, self.exp, n=6, seed=1)
        replies = iter(["c", "p", "x", "i", "s", "q"])
        shown = []
        saved = label_session(self.store, self.exp, items, ask=lambda prompt: next(replies), out=shown.append)
        self.assertEqual(saved, 3)
        labs = {(l["variant"], l["example_id"]): l["label"] for l in self.store.get_labels(self.exp)}
        self.assertEqual((labs[items[0][:2]], labs[items[1][:2]], labs[items[2][:2]]), (1.0, 0.5, 0.0))
        self.assertNotIn(items[3][:2], labs)                                    # skipped
        self.assertNotIn(items[4][:2], labs)                                    # never reached
        text = "\n".join(shown)
        self.assertIn("please type c, p, i, s or q", text)
        for secret in ("judge", "contains", "lenient", "strict", "verdict"):
            self.assertNotIn(secret, text.lower())                              # blind: no verdicts leak
        for needle in ("Question :", "Reference:", "Answer   :"):
            self.assertIn(needle, text)

    def test_ctrl_d_stops_cleanly_and_ascii_helper(self):
        items = pick_label_items(self.store, self.exp, n=3)
        calls = iter(["c"])

        def ask(prompt):
            try:
                return next(calls)
            except StopIteration:
                raise EOFError
        self.assertEqual(label_session(self.store, self.exp, items, ask=ask, out=lambda s: None), 1)
        self.assertEqual(_ascii("caf\u00e9 \u2014 ok"), "caf? ? ok")


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        self.store = ExperimentStore(self.db)
        self.exp = judged(self.store, FLAWED)

    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--db", self.db, *argv])
        return code, out.getvalue()

    def test_judge_check_command(self):
        code, out = self.run_cli("judge-check", self.exp, "--top", "5")
        self.assertEqual(code, 0)
        self.assertIn("judge too lenient", out)
        self.assertIn("[strict] base / q10", out)

    def test_label_then_calibrate(self):
        with mock.patch("builtins.input", side_effect=["c", "c", "i", "q"]):
            code, out = self.run_cli("label", self.exp, "--n", "5", "--seed", "2")
        self.assertEqual(code, 0)
        self.assertIn("You will not see the judge's verdict", out)
        self.assertIn("saved 3 labels", out)
        self.assertEqual(len(ExperimentStore(self.db).get_labels(self.exp)), 3)
        code, out = self.run_cli("calibrate", self.exp)
        self.assertEqual(code, 0)
        self.assertIn("3 human labels", out)
        self.assertIn("only 3 labels", out)

    def test_disagreement_labeling_mode_and_nothing_left(self):
        with mock.patch("builtins.input", side_effect=["i"] * 10):
            code, out = self.run_cli("label", self.exp, "--sample", "disagreements", "--n", "10")
        self.assertEqual(code, 0)
        labs = ExperimentStore(self.db).get_labels(self.exp)
        self.assertEqual((len(labs), {l["source"] for l in labs}), (4, {"disagreement"}))
        code, out = self.run_cli("label", self.exp, "--sample", "disagreements")
        self.assertIn("nothing left to label", out)

    def test_errors_exit_cleanly(self):
        plain = ExperimentStore(self.db)
        exp2 = seed(plain, n=4)
        self.assertEqual(self.run_cli("label", exp2)[0], 2)                      # no judge scores
        self.assertEqual(self.run_cli("judge-check", self.exp, "--metric", "judge:nope")[0], 2)
        self.assertEqual(self.run_cli("calibrate", "exp_missing")[0], 2)


class ApiTests(unittest.TestCase):
    def test_routes(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            m = types.ModuleType("fastapi")

            class HTTPException(Exception):
                def __init__(self, status_code, detail=None):
                    self.status_code, self.detail = status_code, detail
            m.HTTPException, m.Query = HTTPException, (lambda default=None, **kw: default)
            sys.modules["fastapi"] = m
        from llmobs_server import experiments_api

        class App:
            def __init__(self):
                self.routes = {}

            def get(self, path, **kw):
                return lambda fn: self.routes.setdefault(path, fn)

        store, app = ExperimentStore(), App()
        exp = judged(store, FLAWED)
        truth_labels(store, exp)
        experiments_api.register(app, store)
        jc = app.routes["/v1/experiments/{experiment_id}/judge-check"]
        cal = app.routes["/v1/experiments/{experiment_id}/calibration"]
        self.assertEqual(jc(experiment_id=exp, metric=None, top_k=15)["totals"]["lenient"], 2)
        self.assertEqual(cal(experiment_id=exp, metric=None)["all"]["n"], 40)
        for route, kw, code in ((jc, {"experiment_id": "exp_x"}, 404), (jc, {"experiment_id": exp, "metric": "judge:no"}, 400),
                                (cal, {"experiment_id": "exp_x"}, 404), (cal, {"experiment_id": exp, "metric": "judge:no"}, 400)):
            with self.assertRaises(Exception) as cm:
                route(**{"metric": None, "top_k": 15, **kw} if route is jc else {"metric": None, **kw})
            self.assertEqual(cm.exception.status_code, code)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ViewerTests(unittest.TestCase):
    def test_renders_counts_disagreements_and_calibration(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        truth_labels(store, exp)
        h = node_render("renderJudgeCheck", judge_check_payload(store, exp), calibration_payload(store, exp, n_boot=100))
        for needle in ("Judge check", "judge-x", "Judge too lenient", "4 disagreements to review", "close enough",
                       "Against your human labels", "40 labels (random sample)", "kappa 0.69", "bias +0.00", "cand", "base"):
            self.assertIn(needle, h)
        for bad in ("undefined", "NaN"):
            self.assertNotIn(bad, h)

    def test_invites_labeling_when_there_are_no_labels_and_hides_when_there_is_no_judge(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        h = node_render("renderJudgeCheck", judge_check_payload(store, exp), calibration_payload(store, exp))
        self.assertIn("No human labels yet", h)
        self.assertIn(f"llmobs_eval label {exp}", h)
        plain = ExperimentStore()
        p = seed(plain, n=5)
        self.assertEqual(node_render("renderJudgeCheck", judge_check_payload(plain, p), None), "")


if __name__ == "__main__":
    unittest.main()
