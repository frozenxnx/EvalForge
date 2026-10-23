"""Reference-based lexical metrics. Cheap and deterministic, but blind to paraphrase:
use them as a floor and a sanity check next to the LLM judge, not instead of it."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Callable, Dict, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes, turn other punctuation into spaces, drop articles.

    Punctuation becomes a space (not nothing) so "3-7 days" stays "3 7 days" instead of "37 days".
    Number formatting is smoothed so equal answers compare equal: "$9" ~ "9", "10,800" ~ "10800",
    "April 11th" ~ "April 11".
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)    # 10,800 -> 10800
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text)    # 11th -> 11
    chars = []
    for ch in text:
        if ch in "'\u2019":
            continue
        cat = unicodedata.category(ch)
        chars.append(" " if cat.startswith("P") or cat == "Sc" else ch)   # punctuation and $ become spaces
    return " ".join(_ARTICLES.sub(" ", "".join(chars)).split())


def exact_match(output: str, reference: str) -> float:
    return float(normalize(output) == normalize(reference))


def contains(output: str, reference: str) -> float:
    """1.0 if the normalized reference appears as whole words inside the normalized output."""
    ref = normalize(reference)
    return float(bool(ref) and f" {ref} " in f" {normalize(output)} ")


def token_f1(output: str, reference: str) -> float:
    out, ref = normalize(output).split(), normalize(reference).split()
    if not out and not ref:
        return 1.0
    common = sum((Counter(out) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(out), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


LEXICAL: Dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains": contains,
    "token_f1": token_f1,
}
