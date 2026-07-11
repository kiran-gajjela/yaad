from datetime import datetime

from yaad.parser import parse_lines

ANDROID = """04/10/25, 22:41 - Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.
04/10/25, 22:41 - Rohan created group "Goa Plan 🌴"
04/10/25, 22:42 - Rohan: yaar october long weekend
goa chalein?
04/10/25, 22:44 - Priya: <Media omitted>
04/10/25, 22:45 - Priya: villa in anjuna, 4k/night
25/12/25, 10:05 - Sameer: merry christmas 🎄""".splitlines()


def test_android_basic():
    msgs = parse_lines(ANDROID)
    assert len(msgs) == 6

    # system messages have no sender
    assert msgs[0].sender is None and msgs[0].is_system
    assert msgs[1].sender is None

    # multiline continuation folded in
    assert msgs[2].sender == "Rohan"
    assert msgs[2].text == "yaar october long weekend\ngoa chalein?"

    # media flag
    assert msgs[3].is_media
    assert not msgs[4].is_media

    # day-first inferred from the 25/12 date
    assert msgs[5].ts == datetime(2025, 12, 25, 10, 5)


def test_ios_format_with_narrow_space():
    line = "[25/12/25, 10:30:45\u202fPM] John Doe: hello there"
    msgs = parse_lines([line])
    assert len(msgs) == 1
    assert msgs[0].sender == "John Doe"
    assert msgs[0].text == "hello there"
    assert msgs[0].ts == datetime(2025, 12, 25, 22, 30, 45)


def test_mm_dd_inference():
    lines = ["12/25/23, 9:15 AM - John: merry christmas"]
    msgs = parse_lines(lines)
    assert msgs[0].ts == datetime(2023, 12, 25, 9, 15)


def test_invisible_marks_stripped():
    line = "\u200e04/10/25, 22:44 - Priya: \u200e<Media omitted>"
    msgs = parse_lines([line])
    assert msgs[0].is_media


def test_colon_in_message_body():
    msgs = parse_lines(["04/10/25, 22:42 - Rohan: check this: https://x.com"])
    assert msgs[0].sender == "Rohan"
    assert msgs[0].text == "check this: https://x.com"


def test_community_join_notice_is_system_not_sender():
    # WhatsApp Communities system messages contain ": " as ordinary
    # phrasing ("...in the community: <name>"), which can be misread as
    # a sender delimiter. Must stay a system message, not a fake sender.
    line = "12/04/26, 5:29 pm - You joined a group via invite in the community: Some Community"
    msgs = parse_lines([line])
    assert msgs[0].sender is None
    assert msgs[0].is_system
    assert "joined a group via invite" in msgs[0].text

    # a real "You: ..." message must still parse as a normal sender
    msgs = parse_lines(["04/10/25, 22:42 - You: on my way"])
    assert msgs[0].sender == "You"
    assert msgs[0].text == "on my way"
