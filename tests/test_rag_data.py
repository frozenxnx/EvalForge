"""Guards the RAG example data: labels point at real documents and every checkable answer is in them."""
import json
import os
import re
import unittest

from llmobs_eval import load_corpus_jsonl, load_jsonl
from llmobs_eval.metrics import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
DERIVED = {"q30", "q31"}      # answers computed from two documents, not quoted from either


def phrase_in(phrase, text):
    return f" {normalize(phrase)} " in f" {normalize(text)} "


class RagDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus_jsonl(os.path.join(EX, "corpus_lumen.jsonl"))
        cls.docs = {d.id: d for d in cls.corpus.docs}
        cls.ds = load_jsonl(os.path.join(EX, "dataset_rag.jsonl"))

    def test_shape(self):
        self.assertEqual((len(self.corpus.docs), len(self.ds.examples)), (31, 48))
        self.assertTrue(all(d.title for d in self.corpus.docs))

    def test_every_relevant_id_exists(self):
        for e in self.ds.examples:
            for i in e.relevant_doc_ids:
                self.assertIn(i, self.docs, f"{e.id} -> {i}")

    def test_unanswerable_questions_are_exactly_the_unlabeled_ones(self):
        for e in self.ds.examples:
            unanswerable = e.metadata["category"] == "unanswerable"
            self.assertEqual(unanswerable, not e.relevant_doc_ids, e.id)
            if unanswerable:
                self.assertEqual((e.reference, e.metadata["check"]), ("The documents do not say.", "judge_only"))

    def test_quoted_answers_really_appear_in_a_relevant_document(self):
        for e in self.ds.examples:
            if e.metadata["check"] != "contains" or e.id in DERIVED:
                continue
            text = " ".join(f"{self.docs[i].title}. {self.docs[i].text}" for i in e.relevant_doc_ids)
            self.assertTrue(phrase_in(e.reference, text), f"{e.id}: {e.reference!r} not in {e.relevant_doc_ids}")

    def test_derived_answers_are_recomputed_from_the_documents(self):
        prices = [int(x) for x in re.findall(r"costs \$(\d+)", self.docs["d11"].text)]
        minimums = [int(x) for x in re.findall(r"minimum of (\d+) users", self.docs["d19"].text)]
        by_id = {e.id: e for e in self.ds.examples}
        self.assertEqual(normalize(by_id["q30"].reference), str(prices[0] * minimums[0]))   # Standard
        self.assertEqual(normalize(by_id["q31"].reference), str(prices[1] * minimums[1]))   # Premium
        self.assertEqual(sorted(by_id["q30"].relevant_doc_ids), ["d11", "d19"])

    def test_references_do_not_leak_into_their_question(self):
        for e in self.ds.examples:
            if e.metadata["check"] == "contains":
                self.assertFalse(phrase_in(e.reference, e.input), e.id)

    def test_corpus_has_real_confusers(self):
        """The point of the corpus: the same numbers and topics recur, so retrieval can go wrong."""
        def count(n):
            return sum(phrase_in(n, d.text) for d in self.corpus.docs)
        self.assertGreaterEqual(count("30"), 4)
        self.assertGreaterEqual(count("14"), 3)
        self.assertGreaterEqual(sum("refund" in d.text.lower() for d in self.corpus.docs), 4)

    def test_categories(self):
        import collections
        cats = collections.Counter(e.metadata["category"] for e in self.ds.examples)
        self.assertEqual(cats, {"lookup": 18, "confuser": 10, "multi_doc": 1, "multi_hop": 2,
                                "vocab_mismatch": 12, "unanswerable": 5})

    def test_dataset_still_discriminates_between_retrievers(self):
        """BM25 must keep struggling on the paraphrase questions, or there is nothing for a better
        retriever to win. If you improve the tokenizer enough to fail this, add harder questions."""
        from llmobs_eval.retrieval import BM25Retriever, retrieval_metrics
        bm25 = BM25Retriever(self.corpus.docs)
        recalls = {}
        for e in self.ds.examples:
            if e.relevant_doc_ids:
                m = retrieval_metrics([h.doc_id for h in bm25.retrieve(e, 3)], e.relevant_doc_ids, 3)
                recalls.setdefault(e.metadata["category"], []).append(m["recall@3"])
        mean = lambda xs: sum(xs) / len(xs)
        self.assertLessEqual(mean(recalls["vocab_mismatch"]), 0.6)
        self.assertGreaterEqual(mean(recalls["lookup"]), 0.9)          # plain lookups should still work
        overall = mean([x for v in recalls.values() for x in v])
        self.assertTrue(0.7 <= overall <= 0.95, overall)


if __name__ == "__main__":
    unittest.main()
