"""Small eval set for retrieval/synthesis quality (README roadmap item).

Needs a live LLM and a real db, so this is *not* part of the fast unit-test
suite - run it with `yaad eval --db <db>`.

Grading is substring-based, not exact-match: the synthesizer is free to
phrase answers however it likes (ANSWER_SYSTEM only requires that claims
are grounded and cited), so a case passes when the required facts show up
as case-insensitive substrings of the answer, not when the wording lines
up exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Engine

# AND of OR-groups: every inner list must have at least one substring hit.
IncludeSpec = list[list[str]]


@dataclass
class EvalCase:
    id: str
    category: str
    question: str
    must_include: IncludeSpec = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class EvalResult:
    case: EvalCase
    answer: str
    intent: str
    passed: bool
    reason: str


EVAL_CASES: list[EvalCase] = [
    EvalCase("s1", "single_hop", "Who found the villa in Anjuna?",
             must_include=[["Priya"]]),
    EvalCase("s2", "single_hop", "What dates did we finalize for the trip?",
             must_include=[["29"]]),
    EvalCase("s3", "single_hop", "How much was the flight per person?",
             must_include=[["4.2", "4,200", "4200"]]),
    EvalCase("s4", "single_hop", "What's the total cost per head for the trip?",
             must_include=[["14.2", "14,200", "14200"]]),

    EvalCase(
        "c1", "cross_chunk",
        "The person who complained the train takes too long — "
        "did they end up going on the trip?",
        must_include=[["Dev"], ["yes", "went", "all 5", "booked"]],
    ),
    EvalCase(
        "c2", "cross_chunk",
        "Who wanted to swap out the church visit, and what did they suggest instead?",
        must_include=[["Priya"], ["thalassa"]],
    ),
    EvalCase(
        "c3", "cross_chunk",
        "Who suggested taking the train, and how did the group actually travel?",
        must_include=[["Sameer"], ["flight"]],
    ),
    EvalCase(
        "c4", "cross_chunk",
        "Who booked the scooters, and how much did they cost?",
        must_include=[["Aditi"], ["500"]],
    ),

    EvalCase(
        "m1", "system_msg", "Who created the group?",
        must_include=[["Rohan"]],
        note="system messages are stripped from chunks - expected to fail until that's fixed",
    ),
    EvalCase(
        "m2", "system_msg", "Who added Sameer to the group?",
        must_include=[["Rohan"]],
        note="system messages are stripped from chunks - expected to fail until that's fixed",
    ),

    EvalCase(
        "g1", "grounding", "What was Sameer's budget for the trip?",
        must_not_include=["15k", "20k", "15,000", "20,000", "₹15", "₹20"],
        note="Sameer never states a budget; must not misattribute someone else's number",
    ),
    EvalCase(
        "g2", "grounding", "Did anyone cancel or bail on the trip?",
        must_not_include=["bailed", "cancelled", "canceled", "dropped out",
                           "backed out", "didn't come", "couldn't make it"],
        note="nobody bailed (all 5 booked); must not fabricate a dropout",
    ),
]


_NEGATIONS = (
    "no ", "not ", "n't", "nobody", "no one", "none", "never",
    "isn't", "doesn't", "didn't", "wasn't", "aren't",
)


def _claimed_positively(text: str, phrase: str, window: int = 60) -> bool:
    """True if `phrase` appears without a negation cue shortly before it.

    Plain substring matching can't tell "Dev bailed" from "nobody bailed" -
    both contain "bailed". This looks at a window of text right before each
    occurrence for a negation cue before counting it as an actual claim.
    """
    low = text.lower()
    phrase = phrase.lower()
    start = 0
    while True:
        idx = low.find(phrase, start)
        if idx == -1:
            return False
        ctx = low[max(0, idx - window):idx]
        if not any(neg in ctx for neg in _NEGATIONS):
            return True
        start = idx + len(phrase)


def _grade(answer: str, case: EvalCase) -> tuple[bool, str]:
    low = answer.lower()
    for group in case.must_include:
        if not any(kw.lower() in low for kw in group):
            return False, f"missing required: {' / '.join(group)}"
    for bad in case.must_not_include:
        if _claimed_positively(answer, bad):
            return False, f"contains forbidden: {bad!r}"
    return True, "ok"


def run_eval(engine: Engine, cases: list[EvalCase] = EVAL_CASES) -> list[EvalResult]:
    results = []
    for case in cases:
        engine.history = []  # keep cases independent of each other
        ans = engine.answer(case.question)
        passed, reason = _grade(ans.text, case)
        results.append(
            EvalResult(case=case, answer=ans.text, intent=ans.route.intent,
                       passed=passed, reason=reason)
        )
    return results
