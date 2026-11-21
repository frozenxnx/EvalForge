import json
import re
import shutil
import subprocess
import tempfile
import unittest

from tests.test_examples import ROOT, node_render
import os


def node_eval(expr: str):
    """Evaluate a JS expression against the viewer's exported functions and return the JSON result."""
    with open(os.path.join(ROOT, "llmobs_server", "static", "viewer.html"), encoding="utf-8") as fh:
        script = re.search(r"<script>([\s\S]*)</script>", fh.read()).group(1)
    harness = f"""
const mod = {{ exports: {{}} }};
new Function("module", {json.dumps(script)})(mod);
const m = mod.exports;
process.stdout.write(JSON.stringify({expr}));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
    out = subprocess.run(["node", fh.name], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def delta(d, lo, hi, n=40, sig=None, enough=True):
    sig = (lo > 0 or hi < 0) if sig is None else sig
    return {"delta": d, "ci_low": lo, "ci_high": hi, "n_paired": n, "significant": sig, "enough_pairs": enough}


def cmp_payload(vs_a, vs_b=None):
    variants = [{"variant": "base", "model": "m", "vs_baseline": {}, "scores": {}}]
    for name, vs in (("a", vs_a), ("b", vs_b)):
        if vs is not None:
            variants.append({"variant": name, "model": "m", "vs_baseline": vs, "scores": {}})
    return {"baseline": "base", "metrics": sorted({m for v in variants for m in v["vs_baseline"]}), "variants": variants}


@unittest.skipUnless(shutil.which("node"), "node not installed")
class VerdictTests(unittest.TestCase):
    def verdict(self, vs):
        v = cmp_payload(vs)["variants"][1]
        return node_eval(f"m.verdictFor({json.dumps(v)}, {json.dumps(sorted(vs))})")

    def test_significant_gains_and_losses_are_counted_separately(self):
        r = self.verdict({"x": delta(.2, .1, .3), "y": delta(-.2, -.3, -.1), "z": delta(.01, -.1, .1)})
        self.assertEqual((r["better"], r["worse"], r["flat"], r["tone"]), (["x"], ["y"], ["z"], "bad"))

    def test_any_significant_loss_outweighs_gains_in_the_tone(self):
        self.assertEqual(self.verdict({"x": delta(.2, .1, .3), "y": delta(-.2, -.3, -.1)})["tone"], "bad")
        self.assertEqual(self.verdict({"x": delta(.2, .1, .3), "y": delta(0, -.1, .1)})["tone"], "good")
        self.assertEqual(self.verdict({"x": delta(0, -.1, .1)})["tone"], "flat")

    def test_too_few_pairs_is_never_called_better_or_worse(self):
        r = self.verdict({"x": delta(1.0, 1.0, 1.0, n=3, sig=False, enough=False)})
        self.assertEqual((r["better"], r["worse"], r["flat"], r["thin"]), ([], [], [], ["x"]))

    def test_metrics_without_common_examples_are_skipped(self):
        r = self.verdict({"x": delta(0, 0, 0, n=0, sig=False)})
        self.assertEqual((r["better"], r["worse"], r["flat"], r["thin"]), ([], [], [], []))

    def test_cards_say_so_in_words(self):
        h = node_render("verdictCards", cmp_payload({"x": delta(.2, .1, .3)}, {"x": delta(0, -.1, .1)}))
        self.assertIn("Better on 1 metric", h)
        self.assertIn("No significant difference", h)
        for bad in ("undefined", "NaN"):
            self.assertNotIn(bad, h)
        thin = node_render("verdictCards", cmp_payload({"x": delta(1, 1, 1, n=3, sig=False, enough=False)}))
        self.assertIn("too few pairs to judge", thin)
        self.assertNotIn("Better on", thin)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class ForestPlotTests(unittest.TestCase):
    def test_marks_significance_and_clamps_nothing_out_of_view(self):
        h = node_render("forestPlot", cmp_payload({"x": delta(.4, .3, .5), "y": delta(-.2, -.3, -.1), "z": delta(0, -.1, .1)}))
        self.assertIn('<svg class="forest"', h)
        self.assertEqual(h.count('<g class="good">'), 1)
        self.assertEqual(h.count('<g class="bad">'), 1)
        self.assertEqual(h.count('<g class="flat">'), 1)
        self.assertIn("(significant)", h)
        self.assertIn("(not significant)", h)
        for bad in ("undefined", "NaN", "Infinity"):
            self.assertNotIn(bad, h)
        xs = [float(x) for x in re.findall(r'(?:cx|x1|x2)="(-?[0-9.]+)"', h)]
        self.assertTrue(all(0 <= x <= 880 for x in xs), "every mark stays inside the plot")

    def test_thin_pairs_are_drawn_dashed_and_labelled(self):
        h = node_render("forestPlot", cmp_payload({"x": delta(.9, .9, .9, n=3, sig=False, enough=False)}))
        self.assertIn("ci thin", h)
        self.assertIn("n=3", h)
        self.assertIn("too few pairs", h)

    def test_nothing_to_draw_when_there_are_no_candidates_or_no_pairs(self):
        self.assertEqual(node_render("forestPlot", {"baseline": "base", "metrics": [], "variants": cmp_payload({})["variants"][:1]}), "")
        self.assertEqual(node_render("forestPlot", cmp_payload({"x": delta(0, 0, 0, n=0, sig=False)})), "")

    def test_text_from_the_data_is_escaped(self):
        p = cmp_payload({"<img src=x onerror=alert(1)>": delta(.2, .1, .3)})
        p["variants"][1]["variant"] = "<script>alert(1)</script>"
        h = node_render("forestPlot", p) + node_render("verdictCards", p)
        self.assertNotIn("<script>", h)
        self.assertNotIn("<img", h)


if __name__ == "__main__":
    unittest.main()
