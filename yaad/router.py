"""Route a question to the right engine: retrieval, analytics, or both.

Embeddings faceplant on aggregations ("who's most active") and SQL
can't answer "what did Rohan say about the villa" - so the router
classifies first. LLM does the classification when available, with a
keyword heuristic as fallback so the tool degrades gracefully.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

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
"analytics" = needs counting/aggregation (how many, most active, averages, trends).
"both" = needs stats AND message content.
Only set sender when the question filters BY author, not when a person is merely the topic.
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


def route_query(
    question: str,
    llm: BaseLLM | None = None,
    participants: tuple[str, ...] = (),
    date_range: tuple[str, str] | None = None,
    today: str | None = None,
) -> Route:
    if llm is None:
        return _heuristic(question)

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
