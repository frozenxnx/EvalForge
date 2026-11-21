import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

from llmobs_eval import (ConfigError, ExperimentStore, create_experiment, evaluate_experiment, failure_patterns,
                         format_patterns, load_jsonl, run_experiment)
from llmobs_eval.__main__ import main
from llmobs_eval.patterns import hypergeom_upper_tail, is_refusal, kmeans, tfidf_vectors
from llmobs_eval.providers import MockProvider, ProviderError
from tests.test_eval import write_jsonl
from tests.test_examples import node_render
from tests.test_rag import Capture, tiny_experiment
from tests.test_step3 import cfg2


def rows(n=12):
    """Three question families with disjoint vocabulary; the zqNx names are unique so they carry no signal."""
    out = []
    for i in range(n):
        out.append({"id": f"cal{i}", "input": f"On which day of the week does the date zq{i}a fall after Monday?",
                    "reference": "yes", "metadata": {"category": "calendar", "kind": "time"}})
        out.append({"id": f"ari{i}", "input": f"What is the total sum when the number zq{i}b is multiplied?",
                    "reference": "yes", "metadata": {"category": "arithmetic", "kind": "math"}})
        out.append({"id": f"geo{i}", "input": f"Which country has the capital city zq{i}c on the list?",
                    "reference": "yes", "metadata": {"category": "geography", "kind": "facts"}})
    return out


FAIL_TEXT = {"cal": "I don't know", "ari": "no", "geo": "no"}


def build(store, cand_fails=(), base_fails=(), errors=(), data=None, texts=None):
    ds = load_jsonl(write_jsonl(data or rows()))
    exp = create_experiment(store, ds, cfg2())
    for e in ds.examples:
        for variant, fails in (("base", base_fails), ("cand", cand_fails)):
            rec = dict(experiment_id=exp, variant=variant, example_id=e.id, attempts=1, wall_ms=1, created_at=1)
            if variant == "cand" and e.id in errors:
                store.add_result({**rec, "status": "error", "error": "HTTP 500: boom"})
            else:
                store.add_result({**rec, "status": "ok", "output": (texts or FAIL_TEXT)[e.id[:3]] if e.id in fails else "yes"})
    store.set_status(exp, "complete")
    evaluate_experiment(store, exp)
    return exp


CAND = [f"cal{i}" for i in range(12)] + ["ari0", "ari1"]            # candidate: all of calendar, plus two arithmetic
BASE = ["geo0"]


class StatisticsTests(unittest.TestCase):
    def test_hypergeometric_tail_by_hand(self):
        self.assertAlmostEqual(hypergeom_upper_tail(3, 10, 4, 3), 4 / 120)           # C(4,3)C(6,0)/C(10,3)
        self.assertAlmostEqual(hypergeom_upper_tail(2, 10, 4, 3), 40 / 120)          # (36 + 4) / 120
        self.assertEqual(hypergeom_upper_tail(0, 10, 4, 3), 1.0)
        self.assertEqual(hypergeom_upper_tail(4, 10, 4, 3), 0.0)                     # impossible
        self.assertAlmostEqual(hypergeom_upper_tail(12, 36, 14, 12), 91 / math.comb(36, 12))
        self.assertAlmostEqual(sum(hypergeom_upper_tail(k, 20, 7, 5) - hypergeom_upper_tail(k + 1, 20, 7, 5)
                                   for k in range(0, 6)), 1.0)                       # a proper distribution

    def test_refusal_detector(self):
        for yes in ("I don't know.", "I do not know", "The documents do not say.", "That is not mentioned in the context",
                    "I'm unable to help with that", "I can't answer that", "Unknown", "I don\u2019t know"):
            self.assertTrue(is_refusal(yes), yes)
        for no in ("yes", "The answer is 5", "no", "", None, "Knowledge is power"):
            self.assertFalse(is_refusal(no), no)

    def test_clustering_separates_obvious_groups_and_is_deterministic(self):
        qs = [r["input"] for r in rows(8)]
        from llmobs_eval.retrieval import tokenize
        vecs = tfidf_vectors([tokenize(q) for q in qs])
        a = kmeans(vecs, 3, seed=1)
        self.assertEqual(a, kmeans(vecs, 3, seed=1))
        family = [i % 3 for i in range(len(qs))]                                     # cal, ari, geo interleaved
        for f in range(3):
            self.assertEqual(len({a[i] for i in range(len(qs)) if family[i] == f}), 1)     # each family in one cluster
        self.assertEqual(len(set(a)), 3)
        self.assertEqual(kmeans([[0.0, 0.0], [0.0, 0.0]], 2, seed=1), [-1, -1])    # nothing to group


class CategoryTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = build(self.store, CAND, BASE)

    def test_the_calendar_pattern_is_found_and_everything_else_is_not(self):
        p = failure_patterns(self.store, self.exp, "cand", cluster=False)
        self.assertEqual((p["metric"], p["view"], p["population"], p["failures"]), ("contains", "failures", 36, 14))
        cats = {g["label"]: g for g in p["categories"]}
        self.assertEqual(p["categories"][0]["label"], "calendar")                    # most significant first
        cal = cats["calendar"]
        self.assertEqual((cal["n"], cal["failing"], cal["rate"]), (12, 12, 1.0))
        self.assertAlmostEqual(cal["enrichment"], 1.0 / (14 / 36))
        self.assertAlmostEqual(cal["p_value"], 91 / math.comb(36, 12))
        self.assertAlmostEqual(cal["p_adjusted"], cal["p_value"] * 3)               # Bonferroni over 3 categories
        self.assertTrue(cal["significant"])
        self.assertFalse(cats["arithmetic"]["significant"])                          # 2 of 12 is chance
        self.assertEqual((cats["geography"]["failing"], cats["geography"]["significant"]), (0, False))
        self.assertEqual((p["cluster_mode"], p["clusters"]), (None, []))

    def test_refusals_are_counted_where_they_happen(self):
        p = failure_patterns(self.store, self.exp, "cand", cluster=False)
        cats = {g["label"]: g for g in p["categories"]}
        self.assertEqual((cats["calendar"]["refusals"], cats["arithmetic"]["refusals"], p["refusals"]), (12, 0, 12))
        self.assertIsNone(p["no_relevant_retrieved"])                                # no retriever in this experiment

    def test_failing_examples_are_listed_with_their_answers(self):
        p = failure_patterns(self.store, self.exp, "cand", cluster=False, top_examples=5)
        ex = p["categories"][0]["examples"]
        self.assertEqual(len(ex), 5)
        self.assertEqual((ex[0]["id"], ex[0]["score"], ex[0]["output"], ex[0]["refused"]), ("cal0", 0.0, "I don't know", True))
        self.assertIn("date zq0a", ex[0]["question"])

    def test_regressions_view_only_considers_what_the_baseline_got_right(self):
        p = failure_patterns(self.store, self.exp, "cand", baseline="base", cluster=False)
        self.assertEqual((p["view"], p["baseline"], p["population"], p["failures"]), ("regressions", "base", 35, 14))
        cats = {g["label"]: g for g in p["categories"]}
        self.assertEqual(cats["geography"]["n"], 11)                                 # geo0 failed for the baseline: excluded
        self.assertTrue(cats["calendar"]["significant"])

    def test_failed_generations_count_as_failures(self):
        store = ExperimentStore()
        exp = build(store, cand_fails=["cal0"], errors=["geo0", "geo1", "geo2"])
        p = failure_patterns(store, exp, "cand", cluster=False)
        self.assertEqual(p["failures"], 4)
        geo = {g["label"]: g for g in p["categories"]}["geography"]
        self.assertEqual(geo["failing"], 3)
        self.assertEqual(geo["examples"][0]["output"], "(generation failed)")

    def test_a_different_threshold_and_group_key(self):
        p = failure_patterns(self.store, self.exp, "cand", cluster=False, group_by="kind")
        self.assertEqual({g["label"] for g in p["categories"]}, {"time", "math", "facts"})
        self.assertEqual(p["group_by"], "kind")
        graded = failure_patterns(self.store, self.exp, "cand", cluster=False, metric="token_f1", threshold=0.9)
        self.assertEqual(graded["metric"], "token_f1")
        self.assertGreaterEqual(graded["failures"], 14)

    def test_results_are_deterministic(self):
        a = failure_patterns(self.store, self.exp, "cand")
        b = failure_patterns(self.store, self.exp, "cand")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = build(self.store, CAND, BASE)

    def test_tfidf_clusters_rediscover_the_pattern_without_any_labels(self):
        p = failure_patterns(self.store, self.exp, "cand")
        self.assertEqual(p["cluster_mode"], "tfidf")
        sig = [g for g in p["clusters"] if g["significant"]]
        self.assertTrue(sig)
        for g in sig:
            self.assertTrue(all(m.startswith("cal") for m in g["member_ids"]), g["label"])        # only calendar questions
            self.assertTrue(set(g["terms"]) <= {"day", "week", "date", "fall", "after", "monday"}, g["terms"])
        self.assertEqual(p["clusters"][0]["failing"] > 0, True)
        self.assertIn("clusters are hints", " ".join(p["warnings"]))

    def test_the_number_of_clusters_can_be_chosen(self):
        p = failure_patterns(self.store, self.exp, "cand", cluster=True, k=3)
        self.assertEqual(len(p["clusters"]), 3)
        self.assertEqual(sorted(g["n"] for g in p["clusters"]), [12, 12, 12])

    def test_embedding_mode_finds_it_too(self):
        p = failure_patterns(self.store, self.exp, "cand", embed={"provider": "m", "model": "e"},
                             providers={"m": MockProvider(latency_ms=(0, 0))}, k=3)
        self.assertEqual(p["cluster_mode"], "embedding")
        top = p["clusters"][0]
        self.assertTrue(top["significant"])
        self.assertTrue(all(m.startswith("cal") for m in top["member_ids"]))

    def test_embedding_failures_and_unknown_providers_are_reported(self):
        class Down(MockProvider):
            def embed(self, model, texts):
                raise ProviderError("no embedding model", 404)
        with self.assertRaisesRegex(ProviderError, "no embedding model"):
            failure_patterns(self.store, self.exp, "cand", embed={"provider": "m", "model": "e"}, providers={"m": Down()})
        with self.assertRaisesRegex(ConfigError, "add more with --extra-providers"):
            failure_patterns(self.store, self.exp, "cand", embed={"provider": "zzz", "model": "e"})


class EdgeCaseTests(unittest.TestCase):
    def test_nothing_to_explain(self):
        store = ExperimentStore()
        exp = build(store, cand_fails=[])
        p = failure_patterns(store, exp, "cand")
        self.assertEqual((p["failures"], p["categories"], p["clusters"]), (0, [], []))
        self.assertIn("no failures to explain", p["message"])
        everything = build(ExperimentStore(), cand_fails=[r["id"] for r in rows()])
        self.assertIsNotNone(everything)
        store2 = ExperimentStore()
        exp2 = build(store2, cand_fails=[r["id"] for r in rows()])
        self.assertIn("every example fails", failure_patterns(store2, exp2, "cand")["message"])

    def test_small_datasets_and_missing_categories_explain_themselves(self):
        store = ExperimentStore()
        exp = build(store, cand_fails=["cal0", "cal1"], data=rows(1) + rows(1)[:0])       # 3 examples
        p = failure_patterns(store, exp, "cand")
        self.assertIn("too few examples to cluster", p["warnings"])
        store2 = ExperimentStore()
        plain = [{k: v for k, v in r.items() if k != "metadata"} for r in rows(6)]
        exp2 = build(store2, cand_fails=[f"cal{i}" for i in range(6)], data=plain)
        p2 = failure_patterns(store2, exp2, "cand")
        self.assertEqual(p2["categories"], [])
        self.assertIn("no metadata key 'category'", " ".join(p2["warnings"]))
        self.assertTrue(p2["clusters"])                                                    # clustering needs no labels

    def test_bad_input_is_a_clear_error(self):
        store = ExperimentStore()
        exp = build(store, CAND, BASE)
        for kwargs, msg in (({"variant": "zzz"}, "unknown variant"), ({"variant": "cand", "baseline": "cand"}, "different variant"),
                            ({"variant": "cand", "baseline": "zzz"}, "different variant"),
                            ({"variant": "cand", "metric": "nope"}, "available:")):
            with self.assertRaisesRegex(ConfigError, msg):
                failure_patterns(store, exp, **kwargs)
        with self.assertRaises(KeyError):
            failure_patterns(store, "exp_missing", "cand")
        bare = ExperimentStore()
        unscored = create_experiment(bare, load_jsonl(write_jsonl(rows(2))), cfg2())
        with self.assertRaisesRegex(ConfigError, "run `evaluate` first"):
            failure_patterns(bare, unscored, "cand")

    def test_hidden_metrics_are_not_chosen_by_default_but_can_be_named(self):
        store = ExperimentStore()
        exp = build(store, CAND, BASE)
        self.assertEqual(failure_patterns(store, exp, "cand", cluster=False)["metric"], "contains")
        store.hide_metric(exp, "contains")
        self.assertEqual(failure_patterns(store, exp, "cand", cluster=False)["metric"], "token_f1")
        self.assertEqual(failure_patterns(store, exp, "cand", cluster=False, metric="contains")["metric"], "contains")


class RagCounterTests(unittest.TestCase):
    def test_retrieval_misses_among_the_failures(self):
        store = ExperimentStore()
        exp = tiny_experiment(store)
        run_experiment(store, exp, providers={"m": Capture()})
        evaluate_experiment(store, exp)
        bm25 = failure_patterns(store, exp, "bm25", cluster=False)
        self.assertEqual(bm25["failures"], bm25["population"])                       # the fake model answers nothing right
        self.assertEqual(bm25["no_relevant_retrieved"], 1)                           # q4: labeled, but BM25 found nothing
        self.assertIsNone(failure_patterns(store, exp, "closed", cluster=False)["no_relevant_retrieved"])
        oracle = failure_patterns(store, exp, "oracle", cluster=False)
        self.assertEqual(oracle["no_relevant_retrieved"], 0)                         # the oracle always finds them


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.store = ExperimentStore()
        self.exp = build(self.store, CAND, BASE)

    def test_terminal_report(self):
        text = format_patterns(failure_patterns(self.store, self.exp, "cand"))
        self.assertTrue(text.isascii())
        for needle in ("FAILURE PATTERNS: cand on contains", "36 examples, 14 fail (39%)", "12 of the failing answers are refusals",
                       "by category (metadata key 'category')", "calendar", "p (adj)", "by similar questions (tfidf clusters)",
                       "cal0", "answer: I don't know", "more failures than chance explains"):
            self.assertIn(needle, text)
        reg = format_patterns(failure_patterns(self.store, self.exp, "cand", baseline="base", cluster=False))
        self.assertIn("new failures: the baseline 'base' passes, 'cand' fails", reg)

    def test_the_default_variant_is_the_candidate_everywhere(self):
        """CLI, API and viewer must agree: the first variant after the baseline (which defaults to the first)."""
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        exp = build(ExperimentStore(db), CAND, BASE)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            main(["--db", db, "failures", exp, "--no-cluster"])
            main(["--db", db, "failures", exp, "--no-cluster", "--baseline", "cand"])        # baseline cand -> analyse base
        self.assertIn("FAILURE PATTERNS: cand on contains", out.getvalue())
        self.assertIn("FAILURE PATTERNS: base on contains", out.getvalue())

    def test_tiny_p_values_and_refusal_noise(self):
        text = format_patterns(failure_patterns(self.store, self.exp, "cand", cluster=False))
        self.assertIn("<0.001 *", text)                                              # 4e-7 is not "0.000"
        self.assertIn("p < 0.001 after correction", text)
        store = ExperimentStore()
        exp = build(store, cand_fails=[f"cal{i}" for i in range(12)], texts={"cal": "no", "ari": "no", "geo": "no"})
        quiet = format_patterns(failure_patterns(store, exp, "cand", cluster=False))
        self.assertNotIn("refusals", quiet)                                          # wrong answers, not refusals: no noise
        self.assertIn("calendar", quiet)

    def test_cli(self):
        db = os.path.join(tempfile.mkdtemp(), "t.db")
        store = ExperimentStore(db)
        exp = build(store, CAND, BASE)
        path = os.path.join(os.path.dirname(db), "out.json")

        def run(*argv):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["--db", db, *argv])
            return code, out.getvalue()
        code, out = run("failures", exp, "--json", path)
        self.assertEqual(code, 0)
        self.assertIn("FAILURE PATTERNS: cand on contains", out)                        # default: first non-baseline variant
        self.assertEqual(json.load(open(path))["failures"], 14)
        code, out = run("failures", exp, "--baseline", "base", "--no-cluster", "--metric", "contains")
        self.assertIn("new failures", out)
        self.assertNotIn("by similar questions", out)
        for argv in (("failures", exp, "--variant", "zzz"), ("failures", "exp_missing"), ("failures", exp, "--metric", "nope")):
            self.assertEqual(run(*argv)[0], 2, argv)

    def test_api_route(self):
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
        app = App()
        experiments_api.register(app, self.store)
        route = app.routes["/v1/experiments/{experiment_id}/failure-patterns"]
        base = dict(variant=None, baseline=None, metric=None, threshold=0.5, group_by="category", cluster=True, k=None)
        out = route(experiment_id=self.exp, **base)
        self.assertEqual((out["variant"], out["failures"]), ("cand", 14))
        self.assertEqual(route(experiment_id=self.exp, **{**base, "baseline": "base", "variant": "cand"})["view"], "regressions")
        for kw, code in (({"experiment_id": "exp_x"}, 404), ({"experiment_id": self.exp, "variant": "zzz"}, 400)):
            with self.assertRaises(Exception) as cm:
                route(**{**base, **kw})
            self.assertEqual(cm.exception.status_code, code)

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_viewer_panel(self):
        p = failure_patterns(self.store, self.exp, "cand")
        h = node_render("renderFailurePatterns", p, ["base", "cand"], "base")
        for needle in ("By category", "By similar questions", "calendar", "<details>",
                       'id="fp-variant"', 'id="fp-view"', "new failures vs base", "14 of 36 examples fail (39%)",
                       "12 of the failing answers are refusals", "more failures than chance explains"):
            self.assertIn(needle, h)
        for bad in ("undefined", "NaN"):
            self.assertNotIn(bad, h)
        clean_store = ExperimentStore()
        clean = failure_patterns(clean_store, build(clean_store, cand_fails=[]), "cand", cluster=False)
        h2 = node_render("renderFailurePatterns", clean, ["base", "cand"], "base")
        self.assertIn("no failures to explain", h2)
        self.assertNotIn("undefined", h2)


if __name__ == "__main__":
    unittest.main()
