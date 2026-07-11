"""SQLite storage: schema, ingest pipeline, and connection helpers.

One database per chat export (v0.1). Re-running ingest on the same
db rebuilds it from scratch.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .parser import RawMessage
from .sessionize import DEFAULT_GAP_MINUTES, assign_sessions, build_chunks

TS_FMT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
DROP VIEW IF EXISTS v_sender_stats;
DROP VIEW IF EXISTS v_monthly;
DROP VIEW IF EXISTS v_hourly;
DROP VIEW IF EXISTS v_weekday;
DROP VIEW IF EXISTS v_reply_times;
DROP TABLE IF EXISTS messages_fts;
DROP TABLE IF EXISTS chunk_vectors;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS meta;

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,            -- 'YYYY-MM-DD HH:MM:SS'
    date TEXT NOT NULL,          -- 'YYYY-MM-DD' (easy filtering)
    time TEXT NOT NULL,          -- 'HH:MM'
    sender TEXT,                 -- NULL for system/service messages
    text TEXT NOT NULL,
    is_media INTEGER NOT NULL DEFAULT 0,
    is_system INTEGER NOT NULL DEFAULT 0,
    session_id INTEGER NOT NULL
);
CREATE INDEX idx_msg_sender ON messages(sender);
CREATE INDEX idx_msg_date ON messages(date);
CREATE INDEX idx_msg_session ON messages(session_id);

CREATE VIRTUAL TABLE messages_fts USING fts5(
    text,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    start_ts TEXT,
    end_ts TEXT,
    n_messages INTEGER,
    participants TEXT
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    start_msg INTEGER NOT NULL,  -- messages.id (inclusive)
    end_msg INTEGER NOT NULL,    -- messages.id (inclusive)
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    senders TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE chunk_vectors (
    chunk_id INTEGER PRIMARY KEY,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

-- ---------------------------------------------------------------- views ---

CREATE VIEW v_sender_stats AS
SELECT sender,
       COUNT(*)                    AS messages,
       SUM(is_media)               AS media,
       ROUND(AVG(LENGTH(text)), 1) AS avg_chars,
       MIN(date)                   AS first_day,
       MAX(date)                   AS last_day
FROM messages
WHERE is_system = 0
GROUP BY sender
ORDER BY messages DESC;

CREATE VIEW v_monthly AS
SELECT substr(date, 1, 7) AS month, COUNT(*) AS messages
FROM messages WHERE is_system = 0
GROUP BY month ORDER BY month;

CREATE VIEW v_hourly AS
SELECT CAST(substr(time, 1, 2) AS INTEGER) AS hour, COUNT(*) AS messages
FROM messages WHERE is_system = 0
GROUP BY hour ORDER BY hour;

CREATE VIEW v_weekday AS
SELECT CAST(strftime('%w', ts) AS INTEGER) AS weekday,  -- 0=Sunday
       COUNT(*) AS messages
FROM messages WHERE is_system = 0
GROUP BY weekday ORDER BY weekday;

-- Reply time: gap to the previous message when the author changes,
-- capped at 4h so fresh-conversation starts don't pollute the stats.
CREATE VIEW v_reply_times AS
WITH ordered AS (
    SELECT sender, ts,
           LAG(sender) OVER (ORDER BY id) AS prev_sender,
           LAG(ts)     OVER (ORDER BY id) AS prev_ts
    FROM messages WHERE is_system = 0
)
SELECT sender,
       (julianday(ts) - julianday(prev_ts)) * 1440.0 AS reply_minutes
FROM ordered
WHERE prev_sender IS NOT NULL
  AND sender <> prev_sender
  AND (julianday(ts) - julianday(prev_ts)) * 1440.0 BETWEEN 0 AND 240;
"""


def connect(db_path: str | Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def ingest(
    db_path: str | Path,
    messages: list[RawMessage],
    gap_minutes: int = DEFAULT_GAP_MINUTES,
    source: str = "",
) -> dict:
    """Write parsed messages into a fresh database. Returns summary stats."""
    if not messages:
        raise ValueError("no messages parsed - is this a WhatsApp export .txt?")

    session_ids = assign_sessions(messages, gap_minutes)
    chunks = build_chunks(messages, session_ids)

    con = connect(db_path)
    try:
        con.executescript(SCHEMA)

        con.executemany(
            "INSERT INTO messages (ts, date, time, sender, text, is_media, is_system, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    m.ts.strftime(TS_FMT),
                    m.ts.strftime("%Y-%m-%d"),
                    m.ts.strftime("%H:%M"),
                    m.sender,
                    m.text,
                    int(m.is_media),
                    int(m.is_system),
                    sid,
                )
                for m, sid in zip(messages, session_ids)
            ],
        )

        con.execute(
            """
            INSERT INTO sessions (id, start_ts, end_ts, n_messages, participants)
            SELECT session_id, MIN(ts), MAX(ts), COUNT(*), GROUP_CONCAT(DISTINCT sender)
            FROM messages WHERE is_system = 0
            GROUP BY session_id
            """
        )

        # Message list index -> db id: fresh table, ids are 1..n in order.
        con.executemany(
            "INSERT INTO chunks (session_id, start_msg, end_msg, start_ts, end_ts, senders, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.session_id,
                    c.start_idx + 1,
                    c.end_idx + 1,
                    c.start_ts.strftime(TS_FMT),
                    c.end_ts.strftime(TS_FMT),
                    ", ".join(c.senders),
                    c.text,
                )
                for c in chunks
            ],
        )

        con.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")

        meta = {
            "source": source,
            "ingested_at": datetime.now().strftime(TS_FMT),
            "gap_minutes": str(gap_minutes),
        }
        con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta.items())
        con.commit()
    finally:
        con.close()

    n_active = sum(1 for m in messages if not m.is_system)
    participants = sorted({m.sender for m in messages if m.sender})
    return {
        "messages": len(messages),
        "active_messages": n_active,
        "participants": participants,
        "sessions": max(session_ids) + 1,
        "chunks": len(chunks),
        "date_from": messages[0].ts.strftime("%Y-%m-%d"),
        "date_to": messages[-1].ts.strftime("%Y-%m-%d"),
    }
