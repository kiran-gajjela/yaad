"""Hybrid retrieval: FTS5/BM25 over raw messages + dense over chunks.

Names and places are strong lexical anchors in chat, so BM25 pulls
its weight; dense catches paraphrases ("the beach house thing" ->
villa discussion). Both rank lists are fused with Reciprocal Rank
Fusion at the chunk level, then near-duplicate chunks (overlapping
message ranges from the sliding window) are deduped.
"""
from __future__ import annotations

import re
import sqlite3

FTS_TOP_K = 30
DENSE_TOP_K = 15
RRF_K = 60


def _fts_match_expr(query: str) -> str:
    tokens = re.findall(r"\w+", query, re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens)


def fts_search(
    con: sqlite3.Connection,
    query: str,
    top_k: int = FTS_TOP_K,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[int]:
    """Return message ids ranked by BM25 (best first)."""
    match = _fts_match_expr(query)
    if not match:
        return []
    sql = [
        "SELECT m.id AS id FROM messages_fts",
        "JOIN messages m ON m.id = messages_fts.rowid",
        "WHERE messages_fts MATCH ?",
    ]
    params: list = [match]
    if sender:
        sql.append("AND m.sender = ?")
        params.append(sender)
    if date_from:
        sql.append("AND m.date >= ?")
        params.append(date_from)
    if date_to:
        sql.append("AND m.date <= ?")
        params.append(date_to)
    sql.append("ORDER BY bm25(messages_fts) LIMIT ?")
    params.append(top_k)
    return [r["id"] for r in con.execute(" ".join(sql), params)]


def chunk_for_message(con: sqlite3.Connection, msg_id: int) -> int | None:
    row = con.execute(
        "SELECT id FROM chunks WHERE start_msg <= ? AND end_msg >= ? ORDER BY id LIMIT 1",
        (msg_id, msg_id),
    ).fetchone()
    return row["id"] if row else None


def _allowed_chunk_ids(
    con: sqlite3.Connection,
    sender: str | None,
    date_from: str | None,
    date_to: str | None,
) -> set[int] | None:
    if not (sender or date_from or date_to):
        return None
    sql = ["SELECT id FROM chunks WHERE 1=1"]
    params: list = []
    if sender:
        sql.append("AND senders LIKE ?")
        params.append(f"%{sender}%")
    if date_from:
        sql.append("AND end_ts >= ?")
        params.append(date_from)
    if date_to:
        sql.append("AND start_ts <= ?")
        params.append(date_to + " 23:59:59")
    return {r["id"] for r in con.execute(" ".join(sql), params)}


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, item in enumerate(ranks):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=lambda i: -scores[i])


def _overlap_frac(a: tuple[int, int], b: tuple[int, int]) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    return inter / max(1, (a[1] - a[0] + 1))


def hybrid_search(
    con: sqlite3.Connection,
    query: str,
    top_k: int = 6,
    sender: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    dense_searcher=None,
) -> list[dict]:
    """Return the top context blocks (chunk rows as dicts), fused + deduped."""
    # Lexical leg: message hits -> containing chunks (order-preserving dedupe).
    msg_hits = fts_search(con, query, sender=sender, date_from=date_from, date_to=date_to)
    fts_chunks: list[int] = []
    seen: set[int] = set()
    for mid in msg_hits:
        cid = chunk_for_message(con, mid)
        if cid is not None and cid not in seen:
            seen.add(cid)
            fts_chunks.append(cid)

    # Dense leg.
    dense_chunks: list[int] = []
    if dense_searcher is not None:
        allowed = _allowed_chunk_ids(con, sender, date_from, date_to)
        dense_chunks = [cid for cid, _ in dense_searcher.search(query, DENSE_TOP_K, allowed)]

    fused = rrf_fuse([fts_chunks, dense_chunks])

    # Fetch rows, dedupe heavily-overlapping windows.
    results: list[dict] = []
    taken_ranges: list[tuple[int, int]] = []
    for cid in fused:
        row = con.execute("SELECT * FROM chunks WHERE id = ?", (cid,)).fetchone()
        if row is None:
            continue
        rng = (row["start_msg"], row["end_msg"])
        if any(_overlap_frac(rng, t) > 0.5 for t in taken_ranges):
            continue
        taken_ranges.append(rng)
        results.append(dict(row))
        if len(results) >= top_k:
            break
    return results
