from pathlib import Path

import pytest

from yaad.analytics import overview, run_readonly_sql, sender_stats, top_emojis
from yaad.db import connect, ingest
from yaad.parser import parse_chat
from yaad.retrieve import fts_search, hybrid_search

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_chat.txt"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("data") / "chat.db"
    messages = parse_chat(SAMPLE)
    stats = ingest(db_path, messages)
    return db_path, stats


def test_ingest_stats(db):
    _, stats = db
    assert stats["messages"] > 60
    assert set(stats["participants"]) == {"Rohan", "Priya", "Sameer", "Aditi", "Dev"}
    assert stats["sessions"] > 5
    assert stats["chunks"] >= stats["sessions"]
    assert stats["date_from"] == "2025-10-04"
    assert stats["date_to"] == "2026-01-04"


def test_fts_finds_villa(db):
    db_path, _ = db
    con = connect(db_path, readonly=True)
    hits = fts_search(con, "villa")
    assert hits
    texts = [
        con.execute("SELECT text FROM messages WHERE id=?", (h,)).fetchone()["text"]
        for h in hits
    ]
    assert any("villa" in t.lower() for t in texts)


def test_fts_sender_filter(db):
    db_path, _ = db
    con = connect(db_path, readonly=True)
    hits = fts_search(con, "villa", sender="Priya")
    senders = {
        con.execute("SELECT sender FROM messages WHERE id=?", (h,)).fetchone()["sender"]
        for h in hits
    }
    assert senders == {"Priya"}


def test_sender_stats_sum(db):
    db_path, _ = db
    con = connect(db_path, readonly=True)
    total = sum(r["messages"] for r in sender_stats(con))
    n = con.execute("SELECT COUNT(*) AS c FROM messages WHERE is_system=0").fetchone()["c"]
    assert total == n
    assert overview(con)["participants"] == 5


def test_readonly_sql_rejects_writes(db):
    db_path, _ = db
    with pytest.raises(ValueError):
        run_readonly_sql(db_path, "INSERT INTO messages (ts) VALUES ('x')")
    with pytest.raises(Exception):
        run_readonly_sql(db_path, "WITH x AS (SELECT 1) DELETE FROM messages")


def test_readonly_sql_select_works(db):
    db_path, _ = db
    cols, rows = run_readonly_sql(db_path, "SELECT sender, messages FROM v_sender_stats")
    assert cols == ["sender", "messages"]
    assert len(rows) == 5


def test_hybrid_search_fts_only(db):
    db_path, _ = db
    con = connect(db_path, readonly=True)
    blocks = hybrid_search(con, "villa anjuna pool", top_k=3, dense_searcher=None)
    assert blocks
    assert any("villa" in b["text"].lower() for b in blocks)
    # blocks carry provenance
    assert all(b["start_ts"] and b["senders"] for b in blocks)


def test_emoji_stats(db):
    db_path, _ = db
    con = connect(db_path, readonly=True)
    emojis = dict(top_emojis(con))
    assert emojis  # sample chat definitely has emojis
