import pytest

from yaad.router import Route, _extract_json, _heuristic, route_query


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
