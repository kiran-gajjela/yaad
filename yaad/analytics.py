"""Analytics over the chat db: prebuilt stats + guarded read-only SQL.

The LLM never touches a writable connection. Arbitrary analytics
questions become SQL executed against a `mode=ro` connection, one
statement at a time, row-capped.
"""
from __future__ import annotations

import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

from .db import connect

MAX_ROWS = 200

# Handed to the LLM when it writes SQL for an analytics question.
SCHEMA_DOC = """
TABLE messages(id, ts 'YYYY-MM-DD HH:MM:SS', date 'YYYY-MM-DD', time 'HH:MM',
               sender TEXT NULL for system msgs, text, is_media 0/1,
               is_system 0/1, session_id)
TABLE sessions(id, start_ts, end_ts, n_messages, participants)

Convenience views (prefer these when they fit):
  v_sender_stats(sender, messages, media, avg_chars, first_day, last_day)
  v_monthly(month 'YYYY-MM', messages)
  v_hourly(hour 0-23, messages)
  v_weekday(weekday 0=Sunday..6, messages)
  v_reply_times(sender, reply_minutes)   -- one row per reply, capped at 240 min

Notes: always exclude is_system=1 from people-stats; use LIKE for fuzzy
text matching; strftime works on ts. Each view above has ONLY the columns
listed for it - e.g. v_sender_stats has no `id`/`text`/`date`/`is_media`
column, only `sender, messages, media, avg_chars, first_day, last_day`.
Never reference a messages/sessions column against a view unless that
exact column is listed for that view. If you need raw per-message columns
(id, text, is_media, date, ts), query the messages table directly instead.
v_sender_stats.messages is ALREADY the per-sender message count - to get
one sender's count, SELECT messages FROM v_sender_stats WHERE sender=X;
never wrap it in COUNT(*), which just counts matching rows in the view
(always 1 for a single sender) and silently gives the wrong number.
""".strip()


def overview(con: sqlite3.Connection) -> dict:
    row = con.execute(
        "SELECT COUNT(*) AS n, MIN(date) AS d0, MAX(date) AS d1, "
        "SUM(is_media) AS media, COUNT(DISTINCT session_id) AS sessions "
        "FROM messages WHERE is_system = 0"
    ).fetchone()
    senders = con.execute(
        "SELECT COUNT(DISTINCT sender) AS c FROM messages WHERE is_system = 0"
    ).fetchone()
    return {
        "messages": row["n"],
        "date_from": row["d0"],
        "date_to": row["d1"],
        "media": row["media"],
        "sessions": row["sessions"],
        "participants": senders["c"],
    }


def sender_stats(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM v_sender_stats").fetchall()


def monthly(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM v_monthly").fetchall()


def hourly(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM v_hourly").fetchall()


def reply_time_stats(con: sqlite3.Connection) -> list[dict]:
    """Median + mean reply time per sender (minutes)."""
    per: dict[str, list[float]] = {}
    for row in con.execute("SELECT sender, reply_minutes FROM v_reply_times"):
        per.setdefault(row["sender"], []).append(row["reply_minutes"])
    out = []
    for sender, vals in per.items():
        out.append(
            {
                "sender": sender,
                "replies": len(vals),
                "median_min": round(statistics.median(vals), 1),
                "mean_min": round(statistics.fmean(vals), 1),
            }
        )
    out.sort(key=lambda r: r["median_min"])
    return out


_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"  # flags
    "\U0001f300-\U0001faff"  # symbols, pictographs, supplemental
    "\u2600-\u27bf"          # misc symbols, dingbats
    "]"
)


def top_emojis(con: sqlite3.Connection, n: int = 12) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for (text,) in con.execute(
        "SELECT text FROM messages WHERE is_system = 0 AND is_media = 0"
    ):
        counter.update(_EMOJI_RE.findall(text))
    return counter.most_common(n)


def run_readonly_sql(db_path: str | Path, sql: str, max_rows: int = MAX_ROWS):
    """Execute one read-only statement; returns (columns, rows)."""
    sql = sql.strip().rstrip(";").strip()
    if not sql.lower().startswith(("select", "with")):
        raise ValueError("only SELECT / WITH queries are allowed")
    con = connect(db_path, readonly=True)
    try:
        con.execute("PRAGMA query_only = ON")
        cur = con.execute(sql)  # sqlite3 rejects multiple statements here
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, [tuple(r) for r in rows]
    finally:
        con.close()


def rows_to_text(cols: list[str], rows: list[tuple], cap: int = 50) -> str:
    if not rows:
        return "(no rows)"
    lines = [" | ".join(cols)]
    for r in rows[:cap]:
        lines.append(" | ".join("" if v is None else str(v) for v in r))
    if len(rows) > cap:
        lines.append(f"... ({len(rows) - cap} more rows)")
    return "\n".join(lines)
