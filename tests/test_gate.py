import contextlib
import io
import json
import os
import random
import tempfile
import unittest

from llmobs_eval import (ConfigError, ExperimentStore, evaluate_experiment, export_snapshot, format_gate,
                         gate_experiment, gate_markdown, parse_rules, record_calibration, run_gate,
                         side_from_snapshot)
from llmobs_eval.__main__ import main
from llmobs_eval.gate import Side, judge_trust
from llmobs_eval import calibration_payload
from tests.test_eval import mock_cfg, write_jsonl
from tests.test_judge_reliability import FLAWED, judged, truth_labels
from tests.test_step3 import NO_SLEEP, seed


def side(label, vals, metric="m", failed=(), p95=150.0, n=None):
    return Side(label, {metric: {f"q{i}": v for i, v in enumerate(vals)}}, set(failed), n or len(vals) + len(failed),
                p95 / 2 if p95 else None, p95, {f"q{i}": f"out {i}" for i in range(len(vals))}, {})


def rules(metrics=None, **kw):
    return parse_rules({"metrics": metrics or {"m": {"max_drop": 0.03}}, "n_boot": 500, **kw})


def check(report, metric="m"):
    return next(c for c in report["checks"] if c["metric"] == metric)


class RuleParsingTests(unittest.TestCase):
    def test_valid_forms_and_defaults(self):
        r = parse_rules({"metrics": {"a": 0.05, "b": {"max_drop": 0.1, "min": 0.6}, "c": {}}})
        self.assertEqual([(m.name, m.max_drop, m.min) for m in r.metrics], [("a", 0.05, None), ("b", 0.1, 0.6), ("c", 0.0, None)])
        self.assertEqual((r.on_inconclusive, r.alpha, r.n_boot, r.min_pairs, r.errors_as_zero, r.bonferroni),
                         ("warn", 0.05, 2000, 10, True, False))

    def test_typos_and_nonsense_are_errors_not_silently_ignored(self):
        bad = [
            ({}, "non-empty 'metrics'"), ({"metrics": {}}, "non-empty 'metrics'"),
            ({"metrics": {"a": 0.1}, "max_drops": 1}, "unknown gate setting"),
            ({"metrics": {"a": {"max_drops": 0.1}}}, "unknown setting"),
            ({"metrics": {"a": {"max_drop": -0.1}}}, "between"), ({"metrics": {"a": {"max_drop": True}}}, "must be a number"),
            ({"metrics": {"a": 0.1}, "on_inconclusive": "ignore"}, "'warn' or 'fail'"),
            ({"metrics": {"a": 0.1}, "alpha": 0.9}, "alpha"), ({"metrics": {"a": 0.1}, "max_error_rate": 2}, "max_error_rate"),
            ({"metrics": {"a": 0.1}, "judge_trust": {"min_kapa": 0.5}}, "judge_trust"),
            ({"metrics": {"a": 0.1}, "judge_calibration": {"judge:x": 0.5}}, "judge_calibration"),
            ({"metrics": {"a": "x"}}, "must be an object"),
        ]
        for cfg, msg in bad:
            with self.assertRaisesRegex(ConfigError, msg, msg=str(cfg)):
                parse_rules(cfg)


class DecisionTests(unittest.TestCase):
    def test_no_change_passes(self):
        r = run_gate(side("a", [1.0, 0.0] * 20), side("b", [1.0, 0.0] * 20), rules())
        self.assertEqual((r["status"], r["exit_code"], check(r)["delta"]), ("pass", 0, 0.0))

    def test_a_significant_drop_beyond_tolerance_fails_and_lists_the_worst_examples(self):
        r = run_gate(side("a", [1.0] * 40), side("b", [1.0] * 28 + [0.0] * 12), rules(), {"q28": "why?"})
        c = check(r)
        self.assertEqual((r["status"], r["exit_code"], c["verdict"]), ("fail", 1, "fail"))
        self.assertAlmostEqual(c["delta"], -0.3)
        self.assertLess(c["ci_high"], 0)
        self.assertIn("statistically significant", c["reasons"][0])
        first = r["regressions"]["m"][0]
        self.assertEqual((first["example_id"], first["baseline"], first["candidate"], first["question"],
                          first["candidate_output"]), ("q28", 1.0, 0.0, "why?", "out 28"))
        self.assertEqual((first["baseline_output"], len(r["regressions"]["m"])), ("out 28", 5))

    def test_missing_outputs_are_labeled_honestly(self):
        bare = lambda label, vals: Side(label, {"m": {f"q{i}": v for i, v in enumerate(vals)}}, set(), len(vals), 1.0, 1.0, {}, {})
        first = run_gate(bare("a", [1.0] * 40), bare("b", [1.0] * 28 + [0.0] * 12), rules())["regressions"]["m"][0]
        self.assertEqual((first["baseline_output"], first["candidate_output"]), ("(output not recorded)", "(output not recorded)"))

    def test_a_significant_drop_inside_the_tolerance_passes(self):
        a, b = side("a", [1.0] * 200), side("b", [0.0] * 4 + [1.0] * 196)
        self.assertEqual(check(run_gate(a, b, rules({"m": {"max_drop": 0.03}})))["verdict"], "pass")
        r = run_gate(a, b, rules({"m": {"max_drop": 0.0}}))                    # same data, no tolerance
        self.assertEqual((check(r)["delta"], r["status"]), (-0.02, "fail"))

    def test_a_drop_that_could_be_noise_is_inconclusive_and_says_how_much_it_can_resolve(self):
        r = run_gate(side("a", [1.0] * 12), side("b", [1.0] * 10 + [0.0] * 2), rules())
        c = check(r)
        self.assertEqual((c["verdict"], c["effective"], r["status"], r["exit_code"]), ("inconclusive", "warn", "warn", 0))
        self.assertIn("cannot be told from noise", c["reasons"][0])
        self.assertRegex(c["reasons"][0], r"resolve differences down to about \+/-0\.\d\d")
        strict = run_gate(side("a", [1.0] * 12), side("b", [1.0] * 10 + [0.0] * 2), rules(on_inconclusive="fail"))
        self.assertEqual((strict["status"], strict["exit_code"]), ("fail", 1))

    def test_too_few_pairs_is_never_significant(self):
        c = check(run_gate(side("a", [1.0] * 5), side("b", [0.0] * 5), rules()))
        self.assertEqual(c["verdict"], "inconclusive")
        self.assertIn("only 5 paired examples (need 10)", c["reasons"][0])
        self.assertNotIn("resolve differences", c["reasons"][0])
        lenient = rules(min_pairs=5)                                          # the team decides what is enough
        self.assertEqual(check(run_gate(side("a", [1.0] * 5), side("b", [0.0] * 5), lenient))["verdict"], "fail")

    def test_improvements_pass(self):
        r = run_gate(side("a", [0.0] * 20 + [1.0] * 20), side("b", [1.0] * 40), rules())
        self.assertEqual((r["status"], check(r)["delta"] > 0), ("pass", True))
        self.assertEqual(r["regressions"], {})

    def test_an_absolute_floor_fails_even_without_any_drop(self):
        r = run_gate(side("a", [0.5] * 20), side("b", [0.5] * 20), rules({"m": {"max_drop": 0.1, "min": 0.8}}))
        self.assertEqual(r["status"], "fail")
        self.assertIn("below the required minimum 0.8", check(r)["reasons"][0])

    def test_failed_generations_count_as_zero_so_they_cannot_hide(self):
        base = side("a", [1.0] * 20)
        cand = side("b", [1.0] * 14, failed={f"q{i}" for i in range(14, 20)})
        r = run_gate(base, cand, rules())
        c = check(r)
        self.assertEqual((r["status"], c["n_filled"], c["n_pairs"]), ("fail", 6, 20))
        self.assertAlmostEqual(c["delta"], -0.3)
        self.assertIn("6 failed generations counted as 0", c["reasons"])
        self.assertEqual(r["regressions"]["m"][0]["candidate_output"], "(generation failed)")
        off = run_gate(base, cand, rules(errors_as_zero=False))
        self.assertEqual((off["status"], check(off)["n_pairs"]), ("pass", 14))

    def test_error_rate_rule(self):
        base, cand = side("a", [1.0] * 17), side("b", [1.0] * 17, failed={"x1", "x2", "x3"})
        cand.scores["m"].update({})
        r = run_gate(side("a", [1.0] * 20), side("b", [1.0] * 17, failed={"q17", "q18", "q19"}),
                     rules(max_error_rate=0.1, errors_as_zero=False))
        c = check(r, "generation errors")
        self.assertEqual((c["verdict"], r["status"]), ("fail", "fail"))
        self.assertIn("3 of 20 generations failed", c["reasons"][0])
        ok = run_gate(side("a", [1.0] * 20), side("b", [1.0] * 17, failed={"q17", "q18", "q19"}),
                      rules(max_error_rate=0.2, errors_as_zero=False))
        self.assertEqual(ok["status"], "pass")

    def test_latency_rule(self):
        a = side("a", [1.0] * 20, p95=100)
        self.assertEqual(run_gate(a, side("b", [1.0] * 20, p95=200), rules(max_p95_ratio=1.5))["status"], "fail")
        self.assertEqual(run_gate(a, side("b", [1.0] * 20, p95=140), rules(max_p95_ratio=1.5))["status"], "pass")
        r = run_gate(a, side("b", [1.0] * 20, p95=None), rules(max_p95_ratio=1.5))
        self.assertEqual((r["status"], r["exit_code"]), ("error", 2))

    def test_a_gate_that_cannot_decide_never_passes(self):
        both = side("a", [1.0] * 20)
        cases = [
            (rules({"nope": 0.1}), both, side("b", [1.0] * 20), "the baseline has no scores for 'nope'"),
            (rules(), side("a", []), side("b", [1.0] * 20), "the baseline has no scores"),
            (rules(), both, side("b", [], metric="other"), "the candidate has no scores"),
            (rules(), side("a", [1.0] * 3), Side("b", {"m": {"zz": 1.0}}, set(), 1, 1.0, 1.0, {}, {}), "share no scored examples"),
        ]
        for rl, a, b, msg in cases:
            r = run_gate(a, b, rl)
            self.assertEqual((r["status"], r["exit_code"]), ("error", 2), msg)
            self.assertIn(msg, check(r, next(iter(c.name for c in rl.metrics)))["reasons"][0])

    def test_an_error_outranks_a_failure_and_a_failure_outranks_a_warning(self):
        a = side("a", [1.0] * 40)
        a.scores["warn"] = {f"q{i}": 1.0 for i in range(12)}
        b = side("b", [1.0] * 28 + [0.0] * 12)
        b.scores["warn"] = {f"q{i}": 1.0 if i < 10 else 0.0 for i in range(12)}
        both = run_gate(a, b, rules({"m": 0.03, "warn": 0.03}))
        self.assertEqual([c["effective"] for c in both["checks"]], ["fail", "warn"])
        self.assertEqual(both["status"], "fail")
        broken = run_gate(a, b, rules({"m": 0.03, "missing": 0.03}))
        self.assertEqual(broken["status"], "error")

    def test_bonferroni_tightens_alpha_and_results_are_deterministic(self):
        a, b = side("a", [1.0] * 40), side("b", [1.0] * 30 + [0.0] * 10)
        a.scores["n"], b.scores["n"] = a.scores["m"], b.scores["m"]
        rl = rules({"m": 0.0, "n": 0.0}, bonferroni=True)
        r1, r2 = run_gate(a, b, rl), run_gate(a, b, rl)
        self.assertEqual(r1["settings"]["alpha"], 0.025)
        self.assertEqual(r1["checks"], r2["checks"])

    def test_judge_trust_blocks_a_clean_pass_but_not_a_real_failure(self):
        untrusted = {"m": {"trusted": False, "reason": "m is not trusted: kappa 0.5 is below 0.6", "kappa": 0.5}}
        trusted = {"m": {"trusted": True, "reason": "calibrated", "kappa": 0.9}}
        same = (side("a", [1.0] * 20), side("b", [1.0] * 20))
        r = run_gate(*same, rules(), trust=untrusted)
        self.assertEqual((check(r)["verdict"], r["status"], r["exit_code"]), ("untrusted", "warn", 0))
        self.assertEqual(run_gate(*same, rules(on_inconclusive="fail"), trust=untrusted)["exit_code"], 1)
        self.assertEqual(run_gate(*same, rules(), trust=trusted)["status"], "pass")
        bad = run_gate(side("a", [1.0] * 40), side("b", [1.0] * 25 + [0.0] * 15), rules(), trust=untrusted)
        self.assertEqual(check(bad)["verdict"], "fail")
        self.assertIn("not trusted", " ".join(check(bad)["reasons"]))


class StatisticalBehaviourTests(unittest.TestCase):
    """Pins the error rates measured by simulation: the gate must not cry wolf and must catch real regressions."""

    @staticmethod
    def rates(n, drop, trials, max_drop, seed_=11):
        rng = random.Random(seed_)
        rl = parse_rules({"metrics": {"m": {"max_drop": max_drop}}, "n_boot": 200})
        fail = warn = 0
        for t in range(trials):
            p = [rng.uniform(0.2, 1.0) for _ in range(n)]
            a = [float(rng.random() < x) for x in p]
            b = [float(rng.random() < max(0.0, x - drop)) for x in p]
            status = run_gate(side(f"a{t}", a), side(f"b{t}", b), rl)["status"]
            fail += status == "fail"
            warn += status == "warn"
        return fail / trials, warn / trials

    def test_it_rarely_fails_a_build_when_nothing_changed(self):
        fail, _ = self.rates(48, 0.0, 120, 0.05)
        self.assertLessEqual(fail, 0.12)                                      # nominal 5%

    def test_it_catches_a_large_regression_with_enough_examples(self):
        fail, _ = self.rates(100, 0.20, 100, 0.05)
        self.assertGreaterEqual(fail, 0.75)

    def test_the_tolerance_is_what_keeps_noise_warnings_rare(self):
        _, noisy = self.rates(100, 0.0, 100, 0.02)
        _, quiet = self.rates(100, 0.0, 100, 0.10)
        self.assertLess(quiet, 0.12)
        self.assertGreater(noisy, quiet + 0.10)

    def test_small_datasets_cannot_catch_small_regressions(self):
        """The honest limit: at n=20 a true 0.10 drop is mostly missed. The docs say so; this keeps us honest."""
        fail, _ = self.rates(20, 0.10, 100, 0.05)
        self.assertLess(fail, 0.35)


class ExperimentGateTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = seed(self.store, n=20, n_wrong=8)                          # cand is wrong on 8 of 20
        evaluate_experiment(self.store, self.exp)
        self.rules = parse_rules({"metrics": {"contains": 0.03, "exact_match": 0.03, "token_f1": 0.03}, "n_boot": 500})

    def test_a_worse_candidate_fails_with_evidence(self):
        res = gate_experiment(self.store, self.exp, self.rules)
        self.assertEqual((res["status"], res["exit_code"], len(res["reports"])), ("fail", 1, 1))
        rep = res["reports"][0]
        self.assertEqual((rep["baseline"]["label"], rep["candidate"]["label"]), ("base", "cand"))
        c = check(rep, "contains")
        self.assertAlmostEqual(c["delta"], -0.4)
        reg = rep["regressions"]["contains"][0]
        self.assertEqual((reg["question"], reg["baseline_output"], reg["candidate_output"]), ("question 0", "answer 0", "wrong 0"))
        json.dumps(res)                                                       # machine-readable

    def test_an_unchanged_candidate_passes_and_the_baseline_choice_matters(self):
        store = ExperimentStore()
        same = seed(store, n=20, identical=True)
        evaluate_experiment(store, same)
        self.assertEqual(gate_experiment(store, same, self.rules)["status"], "pass")
        flipped = gate_experiment(self.store, self.exp, self.rules, baseline_label="cand")   # "base" is now the improvement
        self.assertEqual(flipped["status"], "pass")

    def test_refuses_partial_experiments_and_unknown_variants(self):
        store = ExperimentStore()
        partial = seed(store, n=10, status="partial")
        with self.assertRaisesRegex(ConfigError, "partial.*proves nothing"):
            gate_experiment(store, partial, self.rules)
        with self.assertRaisesRegex(ConfigError, "unknown variant 'zzz'"):
            gate_experiment(self.store, self.exp, self.rules, baseline_label="zzz")
        from llmobs_eval import create_experiment, load_jsonl
        solo = create_experiment(self.store, load_jsonl(write_jsonl([{"id": "a", "input": "q", "reference": "r"}])),
                                 {**mock_cfg(), "variants": [mock_cfg()["variants"][0]]})
        self.store.set_status(solo, "complete")
        with self.assertRaisesRegex(ConfigError, "no candidate variant"):
            gate_experiment(self.store, solo, self.rules)

    def test_renders_ascii_text_and_markdown(self):
        res = gate_experiment(self.store, self.exp, self.rules)
        text = format_gate(res)
        self.assertTrue(text.isascii())
        for needle in ("GATE FAIL", "baseline = base", "candidate = cand", "contains", "-0.400", "FAIL",
                       "worst regressions on contains", "q0", "result: FAIL"):
            self.assertIn(needle, text)
        md = gate_markdown(res)
        for needle in ("### LLM eval gate: FAIL", "| check | baseline | candidate |", "| contains |", "<details>"):
            self.assertIn(needle, md)


class JudgeTrustTests(unittest.TestCase):
    def rules(self, **kw):
        return parse_rules({"metrics": {"judge:judge-x": {"max_drop": 0.5}}, "judge_trust": {"min_kappa": 0.8, "min_labels": 20}, **kw})

    def test_local_labels_decide(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        truth_labels(store, exp)                                              # kappa 0.6875 on 40 labels
        t = judge_trust(store, exp, self.rules())["judge:judge-x"]
        self.assertFalse(t["trusted"])
        self.assertEqual((t["source"], t["labels"]), ("local human labels", 40))
        self.assertIn("kappa 0.69 is below 0.80", t["reason"])
        loose = parse_rules({"metrics": {"judge:judge-x": 0.5}, "judge_trust": {"min_kappa": 0.6, "min_labels": 20}})
        self.assertTrue(judge_trust(store, exp, loose)["judge:judge-x"]["trusted"])
        res = gate_experiment(store, exp, loose)
        self.assertEqual(check(res["reports"][0], "judge:judge-x")["trust"]["trusted"], True)

    def test_the_ledger_in_the_gate_config_is_used_when_there_are_no_labels(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        base = {"metrics": {"judge:judge-x": 0.5}, "judge_trust": {"min_kappa": 0.7, "min_labels": 20}}
        good = parse_rules({**base, "judge_calibration": {"judge:judge-x": {"kappa": 0.9, "labels": 30}}})
        t = judge_trust(store, exp, good)["judge:judge-x"]
        self.assertEqual((t["trusted"], t["source"]), (True, "judge_calibration in the gate config"))
        weak = parse_rules({**base, "judge_calibration": {"judge:judge-x": {"kappa": 0.5, "labels": 10}}})
        reason = judge_trust(store, exp, weak)["judge:judge-x"]["reason"]
        self.assertIn("only 10 labels (need 20)", reason)
        self.assertIn("kappa 0.50 is below 0.70", reason)
        none = judge_trust(store, exp, parse_rules(base))["judge:judge-x"]
        self.assertFalse(none["trusted"])
        self.assertIn("no calibration on record", none["reason"])
        undefined = parse_rules({**base, "judge_calibration": {"judge:judge-x": {"kappa": None, "labels": 30}}})
        self.assertIn("kappa is undefined", judge_trust(store, exp, undefined)["judge:judge-x"]["reason"])

    def test_nothing_is_checked_unless_a_policy_is_set_or_the_metric_is_not_a_judge(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        self.assertEqual(judge_trust(store, exp, parse_rules({"metrics": {"judge:judge-x": 0.5}})), {})
        self.assertEqual(judge_trust(store, exp, parse_rules({"metrics": {"contains": 0.5}, "judge_trust": {}})), {})

    def test_an_untrusted_judge_cannot_give_a_clean_pass_in_a_full_gate(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        res = gate_experiment(store, exp, self.rules())
        c = check(res["reports"][0], "judge:judge-x")
        self.assertEqual((c["verdict"], res["status"]), ("untrusted", "warn"))
        self.assertEqual(gate_experiment(store, exp, self.rules(on_inconclusive="fail"))["exit_code"], 1)

    def test_record_calibration_writes_and_merges_the_ledger(self):
        store = ExperimentStore()
        exp = judged(store, FLAWED)
        path = os.path.join(tempfile.mkdtemp(), "gate.json")
        with self.assertRaisesRegex(ConfigError, "no human labels"):
            record_calibration(path, "judge:judge-x", calibration_payload(store, exp, n_boot=50))
        truth_labels(store, exp)
        json.dump({"metrics": {"contains": 0.05}, "on_inconclusive": "fail"}, open(path, "w"))
        entry = record_calibration(path, "judge:judge-x", calibration_payload(store, exp, n_boot=50))
        cfg = json.load(open(path))
        self.assertEqual(cfg["metrics"], {"contains": 0.05})                  # existing settings survive
        self.assertEqual(cfg["on_inconclusive"], "fail")
        self.assertAlmostEqual(cfg["judge_calibration"]["judge:judge-x"]["kappa"], 0.6875, places=2)
        self.assertEqual(cfg["judge_calibration"]["judge:judge-x"]["labels"], 40)
        self.assertEqual(entry["sample"], "random")
        parse_rules(cfg)                                                      # the result is still a valid gate


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.a = seed(self.store, n=20, n_wrong=8)
        evaluate_experiment(self.store, self.a)
        self.path = os.path.join(tempfile.mkdtemp(), "baseline.json")
        self.rules = parse_rules({"metrics": {"contains": 0.03}, "n_boot": 500})

    def test_export_and_roundtrip(self):
        snap = export_snapshot(self.store, self.a, "base", self.path)
        on_disk = json.load(open(self.path))
        self.assertEqual((on_disk["format"], on_disk["variant"], on_disk["dataset"]["n_examples"]), (1, "base", 20))
        self.assertEqual(set(on_disk["scores"]), {"exact_match", "contains", "token_f1"})
        self.assertEqual(on_disk["scores"]["contains"]["q0"], 1.0)
        self.assertEqual(snap["dataset"]["hash"], on_disk["dataset"]["hash"])
        s = side_from_snapshot(self.path)
        self.assertTrue(s.label.startswith("base (snapshot"))
        self.assertEqual((len(s.scores["contains"]), s.failed, s.p95_ms is not None), (20, set(), True))

    def test_gating_against_a_snapshot_equals_gating_against_the_variant(self):
        export_snapshot(self.store, self.a, "base", self.path)
        b = seed(self.store, n=20, n_wrong=8)                                  # a new run of the same dataset
        evaluate_experiment(self.store, b)
        via_file = gate_experiment(self.store, b, self.rules, baseline_file=self.path, candidates=["cand"])
        via_variant = gate_experiment(self.store, b, self.rules, baseline_label="base", candidates=["cand"])
        f, v = check(via_file["reports"][0], "contains"), check(via_variant["reports"][0], "contains")
        for key in ("baseline_mean", "candidate_mean", "delta", "ci_low", "ci_high", "n_pairs", "verdict"):
            self.assertEqual(f[key], v[key], key)
        self.assertEqual(via_file["status"], "fail")

    def test_a_different_dataset_is_refused_unless_explicitly_allowed(self):
        export_snapshot(self.store, self.a, "base", self.path)
        other = seed(self.store, n=12, n_wrong=5)
        evaluate_experiment(self.store, other)
        with self.assertRaisesRegex(ConfigError, "different dataset.*regenerate the baseline"):
            gate_experiment(self.store, other, self.rules, baseline_file=self.path, candidates=["cand"])
        res = gate_experiment(self.store, other, self.rules, baseline_file=self.path, candidates=["cand"],
                              allow_dataset_mismatch=True)
        rep = res["reports"][0]
        self.assertEqual(check(rep, "contains")["n_pairs"], 12)               # only the examples both runs contain
        self.assertTrue(any("comparing only the examples both contain" in w for w in rep["warnings"]))

    def test_bad_snapshot_files(self):
        with self.assertRaisesRegex(ConfigError, "not found"):
            side_from_snapshot(os.path.join(tempfile.mkdtemp(), "nope.json"))
        junk = os.path.join(tempfile.mkdtemp(), "j.json")
        open(junk, "w").write("{oops")
        with self.assertRaisesRegex(ConfigError, "invalid JSON"):
            side_from_snapshot(junk)
        json.dump({"format": 99}, open(junk, "w"))
        with self.assertRaisesRegex(ConfigError, "not a baseline snapshot"):
            side_from_snapshot(junk)
        partial = seed(self.store, n=5, status="partial")
        with self.assertRaisesRegex(ConfigError, "finish it"):
            export_snapshot(self.store, partial, "base", self.path)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "t.db")
        self.store = ExperimentStore(self.db)
        self.exp = seed(self.store, n=20, n_wrong=8)
        evaluate_experiment(self.store, self.exp)

    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", self.db, *argv])
        return code, out.getvalue(), err.getvalue()

    def path(self, name):
        return os.path.join(self.dir, name)

    def test_gate_exit_codes_and_outputs(self):
        code, out, _ = self.cli("gate", self.exp, "--metric", "contains=0.03", "--json", self.path("r.json"),
                                "--markdown", self.path("r.md"))
        self.assertEqual(code, 1)
        self.assertIn("GATE FAIL", out)
        result = json.load(open(self.path("r.json")))
        self.assertEqual((result["status"], result["exit_code"]), ("fail", 1))
        self.assertIn("LLM eval gate: FAIL", open(self.path("r.md"), encoding="utf-8").read())
        self.assertEqual(self.cli("gate", self.exp, "--metric", "contains=0.03", "--baseline", "cand")[0], 0)

    def test_config_file_and_flags_combine_and_flags_win(self):
        json.dump({"metrics": {"contains": {"max_drop": 0.9}}, "on_inconclusive": "fail"}, open(self.path("g.json"), "w"))
        self.assertEqual(self.cli("gate", self.exp, "--config", self.path("g.json"))[0], 0)          # 0.4 drop < 0.9 allowed
        self.assertEqual(self.cli("gate", self.exp, "--config", self.path("g.json"), "--metric", "contains=0.03")[0], 1)

    def test_misconfiguration_exits_2_not_0(self):
        for argv in (("gate", self.exp),                                           # no metrics at all
                     ("gate", self.exp, "--metric", "oops"), ("gate", self.exp, "--metric", "contains=abc"),
                     ("gate", self.exp, "--metric", "nonexistent=0.1"),            # metric that was never scored
                     ("gate", self.exp, "--config", self.path("missing.json")),
                     ("gate", "exp_missing", "--metric", "contains=0.1")):
            self.assertEqual(self.cli(*argv)[0], 2, argv)

    def test_baseline_export_then_gate_against_the_file(self):
        code, out, _ = self.cli("baseline-export", self.exp, "--variant", "base", "--out", self.path("b.json"))
        self.assertEqual(code, 0)
        self.assertIn("wrote", out)
        b = seed(self.store, n=20, n_wrong=8)
        evaluate_experiment(self.store, b)
        code, out, _ = self.cli("gate", b, "--baseline-file", self.path("b.json"), "--candidate", "cand",
                                "--metric", "contains=0.03")
        self.assertEqual(code, 1)
        self.assertIn("base (snapshot", out)

    def test_calibrate_record_updates_the_gate_config(self):
        judged_store = ExperimentStore(self.db)
        exp = judged(judged_store, FLAWED)
        truth_labels(judged_store, exp)
        code, out, _ = self.cli("calibrate", exp, "--record", self.path("g.json"))
        self.assertEqual(code, 0)
        self.assertIn("recorded in", out)
        self.assertEqual(json.load(open(self.path("g.json")))["judge_calibration"]["judge:judge-x"]["labels"], 40)
        self.assertEqual(self.cli("calibrate", exp, "--all", "--record", self.path("g.json"))[0], 2)


class ShippedExamplesTests(unittest.TestCase):
    def test_every_example_gate_and_experiment_config_is_valid(self):
        from llmobs_eval import normalize_config
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "ci")
        gates = [f for f in os.listdir(root) if f.startswith("gate")]
        exps = [f for f in os.listdir(root) if f.startswith("experiment")]
        self.assertGreaterEqual((len(gates), len(exps)), (2, 2))
        for name in gates:
            parse_rules(json.load(open(os.path.join(root, name), encoding="utf-8")))
        for name in exps:
            normalize_config(json.load(open(os.path.join(root, name), encoding="utf-8")))
        self.assertTrue(os.path.exists(os.path.join(root, "github-actions.yml")))


class CiCommandTests(unittest.TestCase):
    """The one-step command, end to end, with mock models (so no network, no keys)."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "ci.db")
        rows = [{"id": f"q{i}", "input": f"question {i}", "reference": f"answer {i}"} for i in range(12)]
        self.dataset = write_jsonl(rows)
        self.config = os.path.join(self.dir, "exp.json")
        cfg = mock_cfg()
        cfg["providers"]["m"]["latency_ms"] = [0, 1]
        json.dump(cfg, open(self.config, "w"))

    def gate(self, cfg):
        path = os.path.join(self.dir, "gate.json")
        json.dump(cfg, open(path, "w"))
        return path

    def ci(self, gate_cfg, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--db", self.db, "ci", "--dataset", self.dataset, "--config", self.config,
                         "--gate", self.gate(gate_cfg), *extra])
        return code, out.getvalue(), err.getvalue()

    def test_runs_evaluates_gates_and_writes_reports(self):
        code, out, _ = self.ci({"metrics": {"contains": 0.05}}, "--json", os.path.join(self.dir, "r.json"),
                               "--markdown", os.path.join(self.dir, "r.md"))
        self.assertEqual(code, 0)                                              # both variants score 0 on contains: no change
        for needle in ("created exp_", "ran 24 calls", "evaluated:", "GATE PASS"):
            self.assertIn(needle, out)
        self.assertEqual(json.load(open(os.path.join(self.dir, "r.json")))["status"], "pass")
        self.assertTrue(os.path.exists(os.path.join(self.dir, "r.md")))

    def test_an_absolute_floor_fails_the_build(self):
        code, out, _ = self.ci({"metrics": {"contains": {"max_drop": 0.1, "min": 0.5}}})
        self.assertEqual(code, 1)
        self.assertIn("below the required minimum", out)

    def test_preflight_refuses_before_any_experiment_is_created(self):
        code, _, err = self.ci({"metrics": {"judge:some-model": 0.1}})
        self.assertEqual(code, 2)
        self.assertIn("will not produce it", err)
        self.assertIn("--judge", err)
        self.assertEqual(ExperimentStore(self.db).list_experiments(), [])
        code, _, err = self.ci({"metrics": {"contains": 0.1}}, "--no-lexical")
        self.assertEqual(code, 2)
        self.assertEqual(ExperimentStore(self.db).list_experiments(), [])

    def test_bad_gate_config_and_dataset_mismatch_fail_fast(self):
        self.assertEqual(self.ci({"metrics": {"contains": 0.1}, "max_drops": 1})[0], 2)
        self.assertEqual(ExperimentStore(self.db).list_experiments(), [])
        other = ExperimentStore(os.path.join(self.dir, "other.db"))
        exp = seed(other, n=7)
        evaluate_experiment(other, exp)
        snap = os.path.join(self.dir, "b.json")
        export_snapshot(other, exp, "base", snap)
        code, _, err = self.ci({"metrics": {"contains": 0.1}}, "--baseline-file", snap)
        self.assertEqual(code, 2)
        self.assertIn("different dataset", err)
        self.assertEqual(ExperimentStore(self.db).list_experiments(), [])


if __name__ == "__main__":
    unittest.main()
