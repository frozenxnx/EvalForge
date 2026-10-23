"""Small text helpers shared by the judge parsers."""
from __future__ import annotations

import re
from typing import Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: Optional[str]) -> str:
    """Drop a reasoning model's <think>...</think> block so only its final answer is parsed.

    Handles three shapes: a complete block, a reply where the server already removed the opening tag
    (only '</think>' remains), and a reply cut off mid-thought (an unterminated '<think>', which has
    no answer to parse).
    """
    text = _THINK_BLOCK.sub("", text or "")
    low = text.lower()
    if "</think>" in low:
        text = text[low.rindex("</think>") + len("</think>"):]
        low = text.lower()
    if "<think>" in low:
        text = text[:low.index("<think>")]
    return text
