"""Eval-set smoke test: needs a real, running Ollama - skipped otherwise.

Unlike the rest of the suite this hits a live LLM, so it's slow and
non-deterministic. Use `yaad eval --db <db>` for the full report with a
per-category breakdown; this test just asserts the harness runs end to end
and prints a summary for visibility in CI logs when Ollama is available.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from yaad.db import ingest
from yaad.engine import Engine
from yaad.eval import run_eval
from yaad.llm import OllamaLLM
from yaad.parser import parse_chat

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_chat.txt"


def _ollama_up() -> bool:
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running on localhost:11434")
def test_eval_set_smoke(tmp_path):
    db_path = tmp_path / "eval.db"
    messages = parse_chat(SAMPLE)
    ingest(db_path, messages)
    engine = Engine(str(db_path), llm=OllamaLLM(), dense=False)

    results = run_eval(engine)

    passed = sum(r.passed for r in results)
    print(f"\neval: {passed}/{len(results)} passed")
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.case.id} ({r.case.category}): {r.reason}")

    assert len(results) == len(set(r.case.id for r in results))
