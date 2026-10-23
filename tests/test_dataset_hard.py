"""Guards the hard dataset: every answer that can be computed is recomputed here from scratch.
A wrong reference in an eval set silently punishes correct models, so these must stay green."""
import calendar
import collections
import os
import unittest
from datetime import date, timedelta

from llmobs_eval import load_jsonl
from llmobs_eval.metrics import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "examples", "dataset_hard.jsonl")


def primes_below(n):
    return [p for p in range(2, n) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


d_100 = date(2023, 1, 1) + timedelta(days=100)
COMPUTED = {
    "h01": str(round(18 / 3 * 1.50)),
    "h02": str(round(60 * 2.5 + 40 * 3)),
    "h03": str(4 * 12 - (4 * 12) // 4 - 5),
    "h04": str(23 * 47),
    "h05": str(round(80 * (1 - 0.25) * (1 - 0.10))),
    "h06": str(sum(range(1, 101))),
    "h07": f"{d_100:%B} {d_100.day}",
    "h08": date(2000, 1, 1).strftime("%A"),
    "h09": str((date(2023, 6, 1) - date(2023, 3, 1)).days),
    "h10": calendar.day_name[(calendar.WEDNESDAY + 100) % 7],
    "h11": str(calendar.monthrange(2100, 2)[1]),
    "h12": str("strawberry".count("r")),
    "h13": str(len("encyclopedia")),
    "h14": "elephant"[2],
    "h15": str(sum(c in "aeiou" for c in "education")),
    "h16": str(len("The quick brown fox jumps over the lazy dog".split())),
    "h17": str(round((1.10 - 1.00) / 2 * 100)),                       # ball + (ball + 1.00) = 1.10
    "h18": str(round(100 / (100 * (5 / (5 * 5))))),                    # 0.2 widgets per machine-minute
    "h19": str(48 - 1),
    "h21": str(sum(1 for m in range(1, 13) if calendar.monthrange(2023, m)[1] >= 28)),
    "h22": str(9),
    "h23": str(3 * 3600),
    "h24": str(round(2.5 * 1000 * 100)),
    "h25": str(round(3 * 36 / 12)),
    "h26": str(6 * 7),
    "h27": str(6 * 6 + 2),
    "h28": str(len(primes_below(30))),
}


class HardDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_jsonl(PATH)
        cls.by_id = {e.id: e for e in cls.ds.examples}

    def test_shape(self):
        self.assertEqual(len(self.ds.examples), 44)
        cats = collections.Counter(e.metadata["category"] for e in self.ds.examples)
        self.assertEqual(cats, {"arithmetic": 6, "calendar": 5, "letters": 5, "logic": 6, "units": 3,
                                "sequence": 3, "facts": 12, "false_premise": 4})

    def test_computed_references_are_correct(self):
        for ex_id, expected in COMPUTED.items():
            self.assertEqual(normalize(self.by_id[ex_id].reference), normalize(expected), ex_id)

    def test_sequence_questions_show_the_terms_the_rule_generates(self):
        self.assertEqual([n * (n + 1) for n in range(1, 6)], [2, 6, 12, 20, 30])
        self.assertEqual([n * n + 2 for n in range(1, 6)], [3, 6, 11, 18, 27])
        self.assertIn("2, 6, 12, 20, 30", self.by_id["h26"].input)
        self.assertIn("3, 6, 11, 18, 27", self.by_id["h27"].input)

    def test_classic_riddle_wording(self):
        ex = self.by_id["h20"]
        self.assertTrue(ex.input.startswith("Mary's father"))
        self.assertEqual(ex.reference, "Mary")

    def test_check_flags_match_what_the_lexical_cross_check_can_really_do(self):
        """contains / contains_leaky / judge_only must describe each item truthfully:
        - contains: short canonical reference that does NOT appear in the question
        - contains_leaky: short reference that DOES appear in the question (a wrong answer that restates
          the question would pass `contains`, so it cannot validate the judge)
        - judge_only: a sentence-length reference; only a judge can score it."""
        for e in self.ds.examples:
            ref = normalize(e.reference)
            toks, check = ref.split(), e.metadata["check"]
            leaks = f" {ref} " in f" {normalize(e.input)} "
            if check == "contains":
                self.assertLessEqual(len(toks), 2, e.id)
                self.assertFalse(leaks, f"{e.id}: reference appears in the question; mark it contains_leaky")
            elif check == "contains_leaky":
                self.assertLessEqual(len(toks), 2, e.id)
                self.assertTrue(leaks, f"{e.id}: not actually leaky; mark it contains")
            else:
                self.assertEqual(check, "judge_only", e.id)
                self.assertEqual(e.metadata["category"], "false_premise", e.id)
                self.assertGreater(len(toks), 3, e.id)

    def test_exactly_two_items_are_leaky(self):
        leaky = sorted(e.id for e in self.ds.examples if e.metadata["check"] == "contains_leaky")
        self.assertEqual(leaky, ["h18", "h22"])


if __name__ == "__main__":
    unittest.main()
