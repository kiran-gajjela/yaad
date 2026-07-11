from datetime import date

import pytest

from yaad.router import Route, _extract_json, _heuristic, _resolve_relative_dates, route_query


def test_heuristic_analytics():
    r = _heuristic("how many messages did priya send last month")
    assert r.intent == "analytics"


def test_heuristic_search():
    r = _heuristic("what did rohan say about the villa")
    assert r.intent == "search"
    assert r.search_query


def test_extract_json_plain():
    assert _extract_json('{"intent": "search"}') == {"intent": "search"}


def test_extract_json_fenced():
    raw = '```json\n{"intent": "analytics", "sender": null}\n```'
    assert _extract_json(raw)["intent"] == "analytics"


def test_extract_json_with_prose():
    raw = 'Sure! Here you go: {"intent": "both", "search_query": "villa"} hope that helps'
    assert _extract_json(raw)["intent"] == "both"


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, messages, max_tokens=1000):
        return self.reply


def test_route_validates_sender_against_participants():
    llm = FakeLLM('{"intent": "search", "search_query": "villa", "sender": "priya sharma"}')
    r = route_query("what did priya say", llm=llm, participants=("Priya", "Rohan"))
    assert r.sender is None or r.sender == "Priya"


def test_route_falls_back_on_garbage():
    llm = FakeLLM("total nonsense, no json here")
    r = route_query("how many messages total", llm=llm)
    assert r.intent == "analytics"  # heuristic fallback


# Saturday, so weekday() == 5 - exercises non-trivial week-boundary math.
TODAY = date(2026, 7, 11)


def test_resolve_this_week():
    d_from, d_to = _resolve_relative_dates("summarize this week", TODAY)
    assert d_from == "2026-07-06"  # Monday
    assert d_to == "2026-07-11"


def test_resolve_last_n_days():
    d_from, d_to = _resolve_relative_dates("what happened in the last 7 days", TODAY)
    assert d_from == "2026-07-04"
    assert d_to == "2026-07-11"


def test_resolve_last_month():
    d_from, d_to = _resolve_relative_dates("summarize last month", TODAY)
    assert d_from == "2026-06-01"
    assert d_to == "2026-06-30"


def test_resolve_last_n_months_crosses_year():
    d_from, d_to = _resolve_relative_dates("last 8 months", TODAY)
    assert d_from == "2025-11-11"
    assert d_to == "2026-07-11"


def test_resolve_no_match_returns_none():
    assert _resolve_relative_dates("what did priya say about the villa", TODAY) == (None, None)


def test_route_query_overrides_llm_date_with_deterministic_one():
    # LLM guesses a wrong/lazy date range; the deterministic parser must win
    # because "this week" is an unambiguous, regex-matchable phrase.
    llm = FakeLLM('{"intent": "search", "search_query": "x", "date_from": "2020-01-01", "date_to": "2020-01-01"}')
    r = route_query("summarize this week", llm=llm, today="2026-07-11")
    assert r.date_from == "2026-07-06"
    assert r.date_to == "2026-07-11"


def test_route_query_leaves_unmatched_dates_to_llm():
    llm = FakeLLM('{"intent": "search", "search_query": "x", "date_from": "2025-11-02", "date_to": "2025-11-02"}')
    r = route_query("what did we decide before the trip", llm=llm, today="2026-07-11")
    assert r.date_from == "2025-11-02"
    assert r.date_to == "2025-11-02"
