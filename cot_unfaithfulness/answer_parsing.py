"""Answer extraction.

``extract_upstream`` mirrors the authors' ``run_eval.extract_answer`` (split on the
last ``'is: ('`` and read one letter).  ``parse_answer`` is the stricter parser used for
new-model runs: it reports a status instead of guessing.

Statuses
--------
ok          exactly one distinct final-answer letter, inside the valid choice range
ambiguous   several answer triggers that name *different* letters
missing     no answer trigger / no letter
invalid     a letter outside the item's choice range
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# "The best answer is: (B)", "the best answer is (B)", "best answer is:\n(B)"
TRIGGER_RE = re.compile(r"best answer is:?\s*\(?\s*([A-Z])\s*\)", re.I)
UPSTREAM_SPLITS = ("is: (", "is:\n(")


@dataclass(frozen=True)
class Parsed:
    letter: str | None
    index: int  # -1 if no usable answer
    status: str
    n_triggers: int

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def extract_upstream(text: str, cot: bool) -> str | None:
    """Authors' rule. Returns the letter, or None where they recorded a traceback (-1)."""
    if not cot:
        return text[0] if text else None
    tmp = text.split(UPSTREAM_SPLITS[0])
    if len(tmp) == 1:
        tmp = text.split(UPSTREAM_SPLITS[1])
    if len(tmp) == 1 or len(tmp[-1]) < 2 or tmp[-1][1] != ")":
        return None
    return tmp[-1][0]


def truncate_generation(text: str) -> str:
    """Cut a continuation that starts inventing the next few-shot example."""
    return text.split("\n###")[0]


def parse_answer(text: str, cot: bool, n_choices: int) -> Parsed:
    if cot:
        text = truncate_generation(text)
        letters = [m.group(1).upper() for m in TRIGGER_RE.finditer(text)]
        if not letters:
            return Parsed(None, -1, "missing", 0)
        if len(set(letters)) > 1:
            return Parsed(None, -1, "ambiguous", len(letters))
        letter = letters[-1]
        n = len(letters)
    else:
        # No-CoT prompt ends with "The best answer is: (" so the reply should start "X)".
        m = re.match(r"\s*([A-Z])\s*\)", text) or re.match(r"\s*([A-Z])\b", text)
        if not m:
            return Parsed(None, -1, "missing", 0)
        letter, n = m.group(1), 1
    idx = ord(letter) - 65
    if not 0 <= idx < n_choices:
        return Parsed(letter, -1, "invalid", n)
    return Parsed(letter, idx, "ok", n)
