"""Route a question to the right engine: retrieval, analytics, or both.

Embeddings faceplant on aggregations ("who's most active") and SQL
can't answer "what did Rohan say about the villa" - so the router
classifies first. LLM does the classification when available, with a
keyword heuristic as fallback so the tool degrades gracefully.
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

from .llm import BaseLLM

_ANALYTICS_HINTS = (
    "how many", "count", "number of", "most active", "least active",
    "average", "avg", "mean", "median", "per month", "per day", "per week",
    "stats", "statistics", "total", "percentage", "trend", "busiest",
    "reply time", "response time", "how often", "frequency", "most messages",
    "who sends", "who sent the most", "distribution",
)


@dataclass
class Route:
    intent: str  # "search" | "analytics" | "both"
    search_query: str | None = None
    sender: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    analytics_question: str | None = None


ROUTER_SYSTEM = """You route questions about a WhatsApp chat archive.
Return ONLY a JSON object, no prose, with keys:
  intent: "search" | "analytics" | "both"
  search_query: short keyword query for finding relevant messages (or null)
  sender: exact participant name to filter by (or null)
  date_from: "YYYY-MM-DD" lower bound implied by the question (or null)
  date_to: "YYYY-MM-DD" upper bound implied by the question (or null)
  analytics_question: restated stats question (or null)

"search" = answered by reading messages (what/who said something, plans, decisions).
"analytics" = needs counting/aggregation over STRUCTURED data already in the schema
(message counts, active senders, reply times, dates) - never a number that only exists
as free text someone typed (a budget, a price, a quote). "What's the total budget" or
"how much did X cost" are "search" even though they sound numeric, because that figure
lives inside a message's text, not a column - SQL can't extract it, only reading can.
"both" = needs stats AND message content.
Only set sender when the question filters BY author, not when a person is merely the
topic - e.g. "what did X say/talk about", "summarize X's messages", "X's activity"
-> set sender to X. But "who talked about X", "what happened to X", "news about X"
-> X is the topic, leave sender null.
Resolve relative dates ("last month", "before the trip") using the dates provided."""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in router output")
    return json.loads(raw[start : end + 1])


def _heuristic(question: str) -> Route:
    ql = question.lower()
    if any(h in ql for h in _ANALYTICS_HINTS):
        return Route(intent="analytics", analytics_question=question)
    return Route(intent="search", search_query=question)


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


# Relative-date phrases resolved from Python's real clock rather than LLM
# arithmetic. Even given "today" as text context, small local models get
# this wrong surprisingly often (llama3.2:3b resolved "this week" and "last
# 7 days" to the entire chat's date range in testing); a regex match here is
# unambiguous and always correct, so it overrides whatever the LLM guessed.
_REL_N_PATTERNS: tuple[tuple[re.Pattern, object], ...] = (
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+days?\b", re.I),
     lambda t, n: (t - timedelta(days=n), t)),
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+weeks?\b", re.I),
     lambda t, n: (t - timedelta(weeks=n), t)),
    (re.compile(r"\b(?:last|past)\s+(\d+)\s+months?\b", re.I),
     lambda t, n: (_add_months(t, -n), t)),
)

_REL_PATTERNS: tuple[tuple[re.Pattern, object], ...] = (
    (re.compile(r"\btoday\b", re.I), lambda t: (t, t)),
    (re.compile(r"\byesterday\b", re.I), lambda t: (t - timedelta(days=1),) * 2),
    (re.compile(r"\bthis week\b", re.I), lambda t: (t - timedelta(days=t.weekday()), t)),
    (re.compile(r"\blast week\b", re.I), lambda t: (
        t - timedelta(days=t.weekday() + 7), t - timedelta(days=t.weekday() + 1)
    )),
    (re.compile(r"\bthis month\b", re.I), lambda t: (t.replace(day=1), t)),
    (re.compile(r"\blast month\b", re.I), lambda t: (
        _add_months(t.replace(day=1), -1), t.replace(day=1) - timedelta(days=1)
    )),
)


def _resolve_relative_dates(question: str, today: date) -> tuple[str | None, str | None]:
    for pattern, fn in _REL_N_PATTERNS:
        m = pattern.search(question)
        if m:
            d_from, d_to = fn(today, int(m.group(1)))
            return d_from.isoformat(), d_to.isoformat()
    for pattern, fn in _REL_PATTERNS:
        if pattern.search(question):
            d_from, d_to = fn(today)
            return d_from.isoformat(), d_to.isoformat()
    return None, None


def _route_via_llm(
    question: str,
    llm: BaseLLM,
    participants: tuple[str, ...],
    date_range: tuple[str, str] | None,
    today: str | None,
) -> Route:
    ctx = []
    if participants:
        ctx.append("Chat participants: " + ", ".join(participants))
    if date_range:
        ctx.append(f"Chat spans: {date_range[0]} to {date_range[1]}")
    if today:
        ctx.append(f"Today: {today}")
    ctx.append(f"Question: {question}")

    try:
        raw = llm.complete(ROUTER_SYSTEM, [{"role": "user", "content": "\n".join(ctx)}], 300)
        data = _extract_json(raw)
        intent = data.get("intent")
        if intent not in ("search", "analytics", "both"):
            return _heuristic(question)
        sender = data.get("sender")
        if sender and participants:
            # Guard against hallucinated names: keep only real participants.
            match = next(
                (p for p in participants if p.lower() == str(sender).lower()
                 or str(sender).lower() in p.lower()),
                None,
            )
            sender = match
        return Route(
            intent=intent,
            search_query=data.get("search_query") or question,
            sender=sender,
            date_from=data.get("date_from"),
            date_to=data.get("date_to"),
            analytics_question=data.get("analytics_question") or question,
        )
    except Exception:
        return _heuristic(question)


def route_query(
    question: str,
    llm: BaseLLM | None = None,
    participants: tuple[str, ...] = (),
    date_range: tuple[str, str] | None = None,
    today: str | None = None,
) -> Route:
    route = (
        _route_via_llm(question, llm, participants, date_range, today)
        if llm is not None
        else _heuristic(question)
    )

    today_date = date.fromisoformat(today) if today else date.today()
    det_from, det_to = _resolve_relative_dates(question, today_date)
    if det_from:
        route.date_from = det_from
    if det_to:
        route.date_to = det_to

    return route
