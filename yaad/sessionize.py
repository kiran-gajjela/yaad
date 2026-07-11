"""Group messages into conversation sessions and build retrieval chunks.

Individual WhatsApp messages ("haan", "ok", an emoji) are terrible
retrieval units on their own. We sessionize on time gaps, then cut each
session into overlapping windows of messages ("chunks") that carry
enough context to be embedded / retrieved meaningfully.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .parser import RawMessage

DEFAULT_GAP_MINUTES = 45
CHUNK_WINDOW = 25   # messages per chunk
CHUNK_OVERLAP = 5   # messages shared between consecutive chunks


def assign_sessions(
    messages: list[RawMessage], gap_minutes: int = DEFAULT_GAP_MINUTES
) -> list[int]:
    """Return a session id for every message; a new session starts after a gap."""
    session_ids: list[int] = []
    sid = 0
    prev_ts: datetime | None = None
    for m in messages:
        if prev_ts is not None and (m.ts - prev_ts).total_seconds() > gap_minutes * 60:
            sid += 1
        session_ids.append(sid)
        prev_ts = m.ts
    return session_ids


@dataclass
class Chunk:
    session_id: int
    start_idx: int  # index into the messages list (inclusive)
    end_idx: int    # inclusive
    text: str
    senders: list[str]
    start_ts: datetime
    end_ts: datetime


def format_line(m: RawMessage) -> str:
    who = m.sender or "system"
    body = "<media>" if m.is_media else m.text
    return f"[{m.ts:%Y-%m-%d %H:%M}] {who}: {body}"


def build_chunks(
    messages: list[RawMessage],
    session_ids: list[int],
    window: int = CHUNK_WINDOW,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = len(messages)
    step = max(1, window - overlap)

    start = 0
    while start < n:
        sid = session_ids[start]
        end = start
        while end + 1 < n and session_ids[end + 1] == sid:
            end += 1

        s = start
        while True:
            e = min(s + window - 1, end)
            idxs = [j for j in range(s, e + 1) if not messages[j].is_system]
            if idxs:
                text = "\n".join(format_line(messages[j]) for j in idxs)
                senders = sorted({messages[j].sender for j in idxs if messages[j].sender})
                chunks.append(
                    Chunk(
                        session_id=sid,
                        start_idx=s,
                        end_idx=e,
                        text=text,
                        senders=senders,
                        start_ts=messages[s].ts,
                        end_ts=messages[e].ts,
                    )
                )
            if e >= end:
                break
            s += step
        start = end + 1
    return chunks
