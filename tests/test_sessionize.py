from datetime import datetime, timedelta

from yaad.parser import RawMessage
from yaad.sessionize import assign_sessions, build_chunks


def _msg(ts, sender="A", text="hello"):
    return RawMessage(ts=ts, sender=sender, text=text)


def test_gap_splits_sessions():
    t0 = datetime(2025, 10, 4, 22, 0)
    msgs = [
        _msg(t0),
        _msg(t0 + timedelta(minutes=10)),
        _msg(t0 + timedelta(hours=2)),  # > 45 min gap -> new session
    ]
    assert assign_sessions(msgs, gap_minutes=45) == [0, 0, 1]


def test_chunk_windows_cover_all_messages():
    t0 = datetime(2025, 10, 4, 22, 0)
    msgs = [_msg(t0 + timedelta(minutes=i), sender=f"P{i % 3}", text=f"msg {i}") for i in range(60)]
    sids = assign_sessions(msgs, gap_minutes=45)
    assert sids == [0] * 60  # one session

    chunks = build_chunks(msgs, sids, window=25, overlap=5)
    # step = 20 -> starts at 0, 20, 40 -> 3 chunks
    assert len(chunks) == 3

    covered = set()
    for c in chunks:
        covered.update(range(c.start_idx, c.end_idx + 1))
    assert covered == set(range(60))


def test_chunks_skip_system_messages_in_text():
    t0 = datetime(2025, 10, 4, 22, 0)
    msgs = [
        RawMessage(ts=t0, sender=None, text="X created group"),
        _msg(t0 + timedelta(minutes=1), sender="Rohan", text="hi all"),
    ]
    chunks = build_chunks(msgs, [0, 0])
    assert len(chunks) == 1
    assert "created group" not in chunks[0].text
    assert "Rohan" in chunks[0].text
