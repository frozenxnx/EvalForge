import unittest

from llmobs_eval.metrics import contains, exact_match, normalize


class NormalizeTests(unittest.TestCase):
    def test_number_formatting_is_smoothed(self):
        self.assertEqual(normalize("$9.00"), "9 00")
        self.assertEqual(normalize("10,800 seconds"), "10800 seconds")
        self.assertEqual(normalize("1,000,000"), "1000000")
        self.assertEqual(normalize("April 11th, 2023"), "april 11 2023")
        self.assertEqual(normalize("the 1st, 2nd, 3rd and 21st"), "1 2 3 and 21")

    def test_things_that_must_not_change(self):
        self.assertEqual(normalize("3-7 business days"), "3 7 business days")   # ranges stay apart
        self.assertEqual(normalize("first and fourth"), "first and fourth")      # words are not ordinals
        self.assertEqual(normalize("1,5 and 2,34"), "1 5 and 2 34")              # not thousands groups
        self.assertEqual(normalize("The Bank's rule"), "banks rule")

    def test_equivalent_answers_now_match(self):
        self.assertEqual(contains("It costs $9.", "9"), 1.0)
        self.assertEqual(contains("There are 10,800 seconds.", "10800"), 1.0)
        self.assertEqual(contains("It will be April 11th.", "April 11"), 1.0)
        self.assertEqual(exact_match("April 15th", "April 15"), 1.0)

    def test_wrong_answers_still_do_not_match(self):
        self.assertEqual(contains("It will be May 15th.", "April 15"), 0.0)
        self.assertEqual(contains("It costs $19.", "9"), 0.0)
        self.assertEqual(contains("About 15 cents", "5"), 0.0)


if __name__ == "__main__":
    unittest.main()
