"""Parse WhatsApp chat exports (.txt) into structured messages.

Handles both Android and iOS export formats, multiline messages,
system/service messages, media placeholders, the DD/MM vs MM/DD
ambiguity, and the invisible unicode marks WhatsApp sprinkles into
exports (LRM, narrow no-break space, etc).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

# Directional / invisible marks that appear in real exports.
_INVISIBLES = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\ufeff"), None)

# Android:  04/10/25, 22:41 - Rohan: text
#           12/25/23, 9:15 PM - John: text
_ANDROID_RE = re.compile(
    r"^(?P<date>\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\s+-\s+"
    r"(?P<rest>.*)$"
)
# iOS:      [04/10/25, 10:41:07 PM] Rohan: text
_IOS_RE = re.compile(
    r"^\[(?P<date>\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:[AaPp]\.?[Mm]\.?)?)\]\s+"
    r"(?P<rest>.*)$"
)

_MEDIA_EXACT = {
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "null",
}

# Some system/service messages contain ": " as part of ordinary English
# phrasing rather than as a sender delimiter (e.g. WhatsApp Communities:
# "You joined a group via invite in the community: <name>"). A real
# contact name never contains these, so treat a match as a signal that
# the whole line is system narration, not "sender: text".
_SYSTEM_PHRASES = (
    "joined", "left the group", "left this group", " added ", " removed ",
    "changed the subject", "changed this group's icon",
    "changed the group description", "created group", "created a community",
    "the community", "end-to-end encrypted", "security code", "turned on",
    "turned off", "is now an admin", "deleted this group",
)


@dataclass
class RawMessage:
    ts: datetime
    sender: str | None  # None => system/service message (group created, added, etc)
    text: str
    is_media: bool = False

    @property
    def is_system(self) -> bool:
        return self.sender is None


def _clean(line: str) -> str:
    return (
        line.translate(_INVISIBLES)
        .replace("\u202f", " ")  # narrow no-break space before AM/PM on iOS
        .replace("\u00a0", " ")
    )


def _split_date(date_s: str) -> tuple[int, int, int]:
    a, b, c = re.split(r"[/.\-]", date_s)
    return int(a), int(b), int(c)


def _detect_day_first(date_strs: Iterable[str]) -> bool:
    """Infer DD/MM vs MM/DD by looking for components > 12 across the file."""
    day_first_evidence = mm_dd_evidence = 0
    for d in date_strs:
        a, b, _ = _split_date(d)
        if a > 12:
            day_first_evidence += 1
        if b > 12:
            mm_dd_evidence += 1
    if day_first_evidence and not mm_dd_evidence:
        return True
    if mm_dd_evidence and not day_first_evidence:
        return False
    return True  # ambiguous -> default to DD/MM (most of the world)


def _parse_dt(date_s: str, time_s: str, day_first: bool) -> datetime:
    a, b, c = _split_date(date_s)
    day, month = (a, b) if day_first else (b, a)
    year = c + 2000 if c < 100 else c

    t = time_s.strip().upper().replace(".", "")
    m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]M)?$", t)
    if not m:
        raise ValueError(f"unparseable time: {time_s!r}")
    hh, mm = int(m.group(1)), int(m.group(2))
    ss = int(m.group(3) or 0)
    ampm = m.group(4)
    if ampm == "PM" and hh != 12:
        hh += 12
    elif ampm == "AM" and hh == 12:
        hh = 0
    return datetime(year, month, day, hh, mm, ss)


def _is_media(text: str) -> bool:
    tl = text.strip().lower()
    return tl in _MEDIA_EXACT or tl.endswith("(file attached)")


def parse_lines(lines: Sequence[str], day_first: bool | None = None) -> list[RawMessage]:
    # Pass 1: structural match, collect dates for DD/MM inference.
    parsed: list[tuple] = []
    dates: list[str] = []
    for raw in lines:
        line = _clean(raw.rstrip("\r\n"))
        m = _ANDROID_RE.match(line) or _IOS_RE.match(line)
        if m:
            dates.append(m.group("date"))
            parsed.append(("msg", m.group("date"), m.group("time"), m.group("rest")))
        else:
            parsed.append(("cont", line))

    if day_first is None:
        day_first = _detect_day_first(dates)

    # Pass 2: build messages, folding continuation lines into the previous one.
    messages: list[RawMessage] = []
    for item in parsed:
        if item[0] == "msg":
            _, d, t, rest = item
            ts = _parse_dt(d, t, day_first)
            sender: str | None = None
            text = rest
            if ": " in rest:
                cand_sender, cand_text = rest.split(": ", 1)
                # Real sender names are short and don't read like system
                # narration; this guards against exotic system messages
                # that happen to contain ": " (e.g. WhatsApp Communities
                # join notices).
                looks_like_system = any(p in cand_sender.lower() for p in _SYSTEM_PHRASES)
                if 0 < len(cand_sender) <= 100 and not looks_like_system:
                    sender, text = cand_sender, cand_text
            messages.append(
                RawMessage(ts=ts, sender=sender, text=text, is_media=_is_media(text))
            )
        else:
            if messages:  # multiline continuation
                messages[-1].text += "\n" + item[1]
            # else: stray preamble line before the first message -> drop

    for msg in messages:
        msg.text = msg.text.strip()
    return messages


def parse_chat(path: str | Path, day_first: bool | None = None) -> list[RawMessage]:
    """Parse a WhatsApp export file into a chronological list of messages."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_lines(text.splitlines(), day_first=day_first)
