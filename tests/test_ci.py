import unittest

from llmobs_eval import ExperimentStore, compare_payload, evaluate_experiment
from llmobs_eval.stats import mean_ci, wilson_ci
from tests.test_step3 import NO_SLEEP, ScriptedJudge, seed


class IntervalTests(unittest.TestCase):
    def test_wilson_matches_known_values(self):
        lo, hi = wilson_ci(1.0, 20)
        self.assertAlmostEqual(lo, 0.8389, places=3)      # 20/20
        self.assertEqual(hi, 1.0)
        lo, hi = wilson_ci(0.0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1611, places=3)      # 0/20
        lo, hi = wilson_ci(0.5, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)

    def test_perfect_scores_no_longer_claim_certainty(self):
        r = mean_ci([1.0] * 20)
        self.assertEqual((r["mean"], r["method"]), (1.0, "wilson"))
        self.assertLess(r["ci_low"], 0.9)
        self.assertEqual(r["ci_high"], 1.0)

    def test_binary_metrics_use_wilson_and_stay_inside_0_1(self):
        r = mean_ci([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], seed=1)
        self.assertEqual(r["method"], "wilson")
        self.assertTrue(0.0 <= r["ci_low"] <= r["mean"] <= r["ci_high"] <= 1.0)

    def test_graded_metrics_still_use_the_bootstrap(self):
        r = mean_ci([0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0])
        self.assertEqual(r["method"], "bootstrap")
        self.assertLess(r["ci_low"], r["mean"])

    def test_constant_non_binary_is_also_not_a_point(self):
        r = mean_ci([0.5] * 10)
        self.assertEqual(r["method"], "wilson")
        self.assertLess(r["ci_low"], 0.5 < r["ci_high"])

    def test_values_outside_unit_interval_never_use_wilson(self):
        self.assertEqual(mean_ci([3.0] * 6)["method"], "bootstrap")

    def test_tiny_samples_still_get_no_interval(self):
        self.assertIsNone(mean_ci([1.0])["ci_low"])
        self.assertIsNone(mean_ci([])["mean"])

    def test_compare_payload_uses_it_for_a_perfect_variant(self):
        store = ExperimentStore()
        exp = seed(store, n=20, n_wrong=0)                # both variants perfect
        evaluate_experiment(store, exp, judge={"provider": "m", "model": "judge-x"},
                            providers={"m": ScriptedJudge()}, sleep=NO_SLEEP)
        for v in compare_payload(store, exp, n_boot=200)["variants"]:
            s = v["scores"]["judge:judge-x"]
            self.assertEqual((s["mean"], s["method"]), (1.0, "wilson"))
            self.assertLess(s["ci_low"], 0.9)


if __name__ == "__main__":
    unittest.main()
