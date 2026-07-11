""""Surprise me" - a proactive, personalized recap with zero required input.

Combines deterministic stats (always correct, no LLM guessing) with a
couple of grounded excerpts (first message ever + recent activity), then
asks the LLM to pick out the genuinely interesting angles. Validated during
testing that small local models (~3B) reliably misattribute facts between
people even with a hardened citation prompt - this needs the model class
yaad already defaults to (gemma4:e4b or larger) to be trustworthy.
"""
from __future__ import annotations

import sqlite3

from .analytics import overview, reply_time_stats, sender_stats, top_emojis
from .llm import BaseLLM

SURPRISE_SYSTEM = """You are yaad, giving the user a delightful, personalized "surprise me" \
recap of their WhatsApp group chat.
You will be given real stats and real excerpts - use ONLY what is given, never invent names,
numbers, or quotes, and never attribute a quote or fact to a different person than the one it
is listed under in the input.
Cite every claim by referencing the fact label it came from (e.g. MOST ACTIVE, TOP EMOJI) so
it's traceable - pull names/dates/numbers exactly from the input, never guess or swap them.
If a comparison is already computed in the input (e.g. a lead margin), use that number as-is -
do not recompute or restate a raw total as if it were a margin.
Pick 3-4 of the most genuinely interesting angles, not all of them. Write short, warm, specific
callouts - like a friend noticing things about the group, not a report. No generic filler like
"this group is great". Format as a short bulleted list."""

_WEEKDAY_NAMES = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


def gather_facts(con: sqlite3.Connection, excerpt_len: int = 220) -> str:
    ov = overview(con)
    senders = sorted(sender_stats(con), key=lambda r: -r["messages"])
    replies = reply_time_stats(con)
    emojis = top_emojis(con, 3)
    weekday = con.execute("SELECT * FROM v_weekday ORDER BY messages DESC LIMIT 1").fetchone()
    hour = con.execute("SELECT * FROM v_hourly ORDER BY messages DESC LIMIT 1").fetchone()
    first_msg = con.execute(
        "SELECT ts, sender, text FROM messages WHERE is_system=0 ORDER BY ts LIMIT 1"
    ).fetchone()
    recent = con.execute(
        "SELECT ts, sender, text FROM messages WHERE is_system=0 ORDER BY ts DESC LIMIT 4"
    ).fetchall()

    lines = [
        f"GROUP VITALS: {ov['messages']} messages, {ov['participants']} people, "
        f"running since {ov['date_from']}, {ov['media']} media shared.",
    ]

    if ov["messages"] >= 10:
        next_milestone = ((ov["messages"] // 100) + 1) * 100
        lines.append(
            f"MILESTONE: {next_milestone - ov['messages']} messages away "
            f"from message #{next_milestone}."
        )

    if senders:
        top = senders[0]
        if len(senders) > 1:
            margin = top["messages"] - senders[1]["messages"]
            lines.append(
                f"MOST ACTIVE: {top['sender']} with {top['messages']} total messages - "
                f"leads 2nd place ({senders[1]['sender']}, {senders[1]['messages']}) "
                f"by {margin} messages."
            )
        else:
            lines.append(f"MOST ACTIVE: {top['sender']} with {top['messages']} messages.")
        quiet = senders[-1]
        if quiet["sender"] != top["sender"]:
            lines.append(
                f"QUIETEST: {quiet['sender']} with {quiet['messages']} "
                f"messages, last active {quiet['last_day']}."
            )

    if replies:
        fastest = replies[0]
        lines.append(f"FASTEST REPLIER: {fastest['sender']}, median {fastest['median_min']} min.")

    if emojis:
        lines.append(f"TOP EMOJI: {emojis[0][0]} used {emojis[0][1]} times.")

    if weekday and hour:
        lines.append(
            f"PEAK TIME: {_WEEKDAY_NAMES[weekday['weekday']]}s around {hour['hour']}:00."
        )

    if first_msg:
        lines.append(
            f"FIRST MESSAGE EVER ({first_msg['ts']}, {first_msg['sender']}): "
            f"{first_msg['text'][:excerpt_len]}"
        )

    if recent:
        lines.append("RECENT MESSAGES:")
        lines.extend(
            f"- ({m['ts']}, {m['sender']}): {m['text'][:150]}" for m in recent
        )

    return "\n".join(lines)


def generate_surprise(con: sqlite3.Connection, llm: BaseLLM) -> str:
    facts = gather_facts(con)
    return llm.complete(SURPRISE_SYSTEM, [{"role": "user", "content": facts}], 500).strip()
