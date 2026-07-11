"""yaad CLI: ingest, chat, stats, search."""
from __future__ import annotations

import argparse
import random
import sys
import threading

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .analytics import hourly, monthly, overview, reply_time_stats, sender_stats, top_emojis
from .db import connect, ingest
from .llm import LLMError, get_llm
from .parser import parse_chat
from .retrieve import hybrid_search

# Windows' legacy console defaults to cp1252, which can't encode the unicode
# glyphs (✓, ›, █, ...) rich writes. Force UTF-8 so output doesn't crash.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

console = Console()

_QUOTES = (
    # thinking / searching energy
    "42. — Deep Thought, The Hitchhiker's Guide to the Galaxy",
    "Don't Panic. — The Hitchhiker's Guide to the Galaxy",
    "Elementary, my dear Watson. — Sherlock Holmes",
    "I drink and I know things. — Tyrion Lannister, Game of Thrones",
    "Legen — wait for it — dary! — Barney Stinson, HIMYM",
    "Just keep swimming. — Dory, Finding Nemo",
    "Follow the white rabbit. — The Matrix",
    "I know kung fu. — Neo, The Matrix",
    "Wax on, wax off. — Mr. Miyagi, The Karate Kid",
    "Never tell me the odds. — Han Solo, Star Wars",
    "Do or do not. There is no try. — Yoda, Star Wars",
    "Make it so. — Captain Picard, Star Trek",
    "Great Scott! — Doc Brown, Back to the Future",
    "I solemnly swear I am up to no good. — Harry Potter",
    "It's super effective! — Pokémon",

    # certified classics
    "Love me. — Homelander, The Boys",
    "I am the one who knocks. — Walter White, Breaking Bad",
    "Say my name. — Walter White, Breaking Bad",
    "Yeah, science! — Jesse Pinkman, Breaking Bad",
    "Better call Saul! — Saul Goodman",
    "Winter is coming. — House Stark, Game of Thrones",
    "Not today. — Arya Stark, Game of Thrones",
    "Valar Morghulis. — Braavos, Game of Thrones",
    "Valar Dohaeris. — A man with no face",
    "Dracarys. — Daenerys Targaryen, Game of Thrones",
    "Chaos is a ladder. — Littlefinger, Game of Thrones",
    "This is the way. — The Mandalorian",
    "I'll be back. — The Terminator",
    "Why so serious? — The Joker, The Dark Knight",
    "I'm Batman. — Batman",
    "I am inevitable. — Thanos, Avengers: Endgame",
    "I am Iron Man. — Tony Stark, Avengers: Endgame",
    "I can do this all day. — Captain America",
    "I am Groot. — Groot, Guardians of the Galaxy",
    "May the Force be with you. — Star Wars",
    "Hello there. — Obi-Wan Kenobi",
    "To infinity and beyond! — Buzz Lightyear, Toy Story",
    "Hakuna Matata. — The Lion King",
    "You shall not pass! — Gandalf, The Lord of the Rings",
    "My precious. — Gollum, The Lord of the Rings",
    "One does not simply walk into Mordor. — Boromir, LOTR",
    "Fear is the mind-killer. — Dune",
    "The spice must flow. — Dune",

    # sitcom corner
    "How you doin'? — Joey, Friends",
    "We were on a break! — Ross, Friends",
    "PIVOT! — Ross, Friends",
    "That's what she said. — Michael Scott, The Office",
    "Bears. Beets. Battlestar Galactica. — Jim Halpert, The Office",
    "Identity theft is not a joke, Jim! — Dwight Schrute, The Office",
    "Bazinga! — Sheldon Cooper, The Big Bang Theory",
    "Wubba lubba dub dub! — Rick Sanchez, Rick and Morty",
    "D'oh! — Homer Simpson, The Simpsons",

    # anime + games
    "It's over 9000! — Vegeta, Dragon Ball Z",
    "Believe it! — Naruto",
    "Plus Ultra! — My Hero Academia",
    "Omae wa mou shindeiru. — Fist of the North Star",
    "The cake is a lie. — Portal",
    "It's dangerous to go alone! Take this. — The Legend of Zelda",
    "FUS RO DAH! — Skyrim",
    "Gotta catch 'em all! — Pokémon",

    # Bollywood
    "Kitne aadmi the? — Gabbar Singh, Sholay",
    "Mogambo khush hua. — Mr. India",
    "Picture abhi baaki hai, mere dost. — Om Shanti Om",
    "Don ko pakadna mushkil hi nahi, namumkin hai. — Don",
    "How's the josh? — Uri",
    "Jhukega nahi! — Pushpa",
)

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_DOTS = (".", "..", "...")


def _run_with_live(fn, *fn_args, render, tick: float = 0.4, **fn_kwargs):
    """Run a blocking call under a two-line rich Live display, redrawing via
    `render(i)` every `tick` seconds. `fn` runs on the current thread as
    normal - sqlite3 connections are bound to their creating thread, so the
    real work can't move to a worker thread. Only the cosmetic redraw runs
    on a background thread, ticking on a timer independent of the call."""
    stop = threading.Event()
    with Live(render(0), console=console, refresh_per_second=12, transient=True) as live:
        def tick_loop():
            i = 1
            while not stop.wait(timeout=tick):
                live.update(render(i))
                i += 1

        updater = threading.Thread(target=tick_loop, daemon=True)
        updater.start()
        try:
            return fn(*fn_args, **fn_kwargs)
        finally:
            stop.set()
            updater.join(timeout=1)


def _run_with_loading_spinner(fn, *fn_args, **fn_kwargs):
    """First-time model load: animated dots on line one, a rotating quote
    with a bulb on line two."""
    quotes = list(_QUOTES)
    random.shuffle(quotes)

    def render(i: int):
        spin = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
        dots = _DOTS[i % len(_DOTS)]
        line1 = Text.from_markup(f"[bold cyan]{spin} loading the model{dots}[/]")
        line2 = Text.from_markup(f"💡 [dim]{quotes[(i // 6) % len(quotes)]}[/]")
        return Group(line1, line2)

    return _run_with_live(fn, *fn_args, render=render, **fn_kwargs)


def _run_with_thinking_spinner(fn, *fn_args, **fn_kwargs):
    """Answer generation: a fixed joke on line one, a rotating quote below."""
    quotes = list(_QUOTES)
    random.shuffle(quotes)

    def render(i: int):
        spin = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
        line1 = Text.from_markup(
            f"[bold cyan]{spin} generating the response... until then, go touch grass 🌱[/]"
        )
        line2 = Text.from_markup(f"[dim]{quotes[(i // 6) % len(quotes)]}[/]")
        return Group(line1, line2)

    return _run_with_live(fn, *fn_args, render=render, **fn_kwargs)


def _bar(value: float, max_value: float, width: int = 28) -> str:
    if max_value <= 0:
        return ""
    return "█" * max(1, round(value / max_value * width))


# ------------------------------------------------------------------ ingest

def cmd_ingest(args) -> int:
    day_first = None
    if args.dd_mm:
        day_first = True
    elif args.mm_dd:
        day_first = False

    console.print(f"[bold]parsing[/] {args.export}")
    messages = parse_chat(args.export, day_first=day_first)
    stats = ingest(args.db, messages, gap_minutes=args.gap, source=str(args.export))

    console.print(
        f"[green]✓[/] {stats['messages']} messages "
        f"({stats['active_messages']} from people) · "
        f"{stats['sessions']} sessions · {stats['chunks']} chunks\n"
        f"  {stats['date_from']} → {stats['date_to']} · "
        f"participants: {', '.join(stats['participants'])}"
    )

    if not args.no_dense:
        try:
            from .embed import build_dense_index

            console.print("[dim]building dense index (first run downloads the model)...[/]")
            n = build_dense_index(args.db, model_name=args.embed_model)
            console.print(f"[green]✓[/] embedded {n} chunks")
        except (RuntimeError, ImportError) as e:
            console.print(f"[yellow]![/] skipping dense index: {e}")
            console.print("[dim]  FTS search still works. pip install 'yaad[dense]' to enable.[/]")

    console.print(f"\n[bold]db ready:[/] {args.db} — try: yaad chat --db {args.db}")
    return 0


# -------------------------------------------------------------------- chat

def cmd_chat(args) -> int:
    from .engine import Engine

    try:
        llm = get_llm(args.provider, args.model)
    except LLMError as e:
        console.print(f"[red]{e}[/]")
        return 1

    engine = _run_with_loading_spinner(Engine, args.db, llm=llm, dense=not args.no_dense)
    mode = "hybrid (fts + dense)" if engine.dense else "fts-only"
    console.print(
        Panel.fit(
            f"chatting with [bold]{args.db}[/]\n"
            f"{engine.date_range[0]} → {engine.date_range[1]} · "
            f"{', '.join(engine.participants)}\n"
            f"[dim]llm: {llm.name}/{llm.model} · retrieval: {mode} · /quit to exit[/]",
            title="yaad",
        )
    )

    while True:
        try:
            q = console.input("[bold cyan]you[/] › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not q:
            continue
        if q.lower() in ("/quit", "/exit", "quit", "exit"):
            break
        try:
            ans = _run_with_thinking_spinner(engine.answer, q)
        except LLMError as e:
            console.print(f"[red]{e}[/]")
            continue
        badge = ans.route.intent
        if ans.sources:
            badge += f" · {len(ans.sources)} excerpts"
        console.print(f"[dim]yaad · {badge}[/]")
        if ans.sql:
            console.print(f"[dim]  sql: {ans.sql}[/]")
        console.print(ans.text + "\n")
    return 0


# ------------------------------------------------------------------- embed

def cmd_embed(args) -> int:
    try:
        from .embed import build_dense_index
    except ImportError as e:
        console.print(f"[red]{e}[/]")
        console.print("[dim]pip install 'yaad[dense]' to enable.[/]")
        return 1

    console.print("[dim]building dense index on existing chunks (no reparse)...[/]")
    try:
        n = build_dense_index(args.db, model_name=args.embed_model)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        return 1
    console.print(f"[green]✓[/] embedded {n} chunks")
    return 0


# ------------------------------------------------------------------- stats

def cmd_stats(args) -> int:
    con = connect(args.db, readonly=True)
    ov = overview(con)
    console.print(
        Panel.fit(
            f"{ov['messages']} messages · {ov['participants']} people · "
            f"{ov['sessions']} sessions · {ov['media']} media\n"
            f"{ov['date_from']} → {ov['date_to']}",
            title="overview",
        )
    )

    t = Table(title="who talks")
    for col in ("sender", "messages", "media", "avg chars", "median reply"):
        t.add_column(col)
    replies = {r["sender"]: r for r in reply_time_stats(con)}
    for row in sender_stats(con):
        rep = replies.get(row["sender"])
        t.add_row(
            row["sender"],
            str(row["messages"]),
            str(row["media"]),
            str(row["avg_chars"]),
            f"{rep['median_min']} min" if rep else "—",
        )
    console.print(t)

    rows = monthly(con)
    if rows:
        mx = max(r["messages"] for r in rows)
        console.print("\n[bold]by month[/]")
        for r in rows:
            console.print(f"  {r['month']}  {_bar(r['messages'], mx)} {r['messages']}")

    rows = hourly(con)
    if rows:
        mx = max(r["messages"] for r in rows)
        console.print("\n[bold]by hour[/]")
        for r in rows:
            console.print(f"  {r['hour']:02d}:00  {_bar(r['messages'], mx)} {r['messages']}")

    emojis = top_emojis(con)
    if emojis:
        console.print("\n[bold]top emojis[/]  " + "  ".join(f"{e}×{c}" for e, c in emojis))
    con.close()
    return 0


# ------------------------------------------------------------------ search

def cmd_search(args) -> int:
    con = connect(args.db, readonly=True)
    dense = None
    if not args.no_dense:
        try:
            from .embed import DenseSearcher

            dense = DenseSearcher(con)
        except Exception:
            dense = None

    blocks = hybrid_search(
        con,
        args.query,
        top_k=args.top_k,
        sender=args.sender,
        date_from=args.after,
        date_to=args.before,
        dense_searcher=dense,
    )
    if not blocks:
        console.print("[yellow]no matches[/]")
        return 1
    for b in blocks:
        console.print(
            Panel(
                b["text"],
                title=f"{b['start_ts'][:16]} → {b['end_ts'][:16]}",
                subtitle=b["senders"],
                title_align="left",
            )
        )
    con.close()
    return 0


# ---------------------------------------------------------------- surprise

def cmd_surprise(args) -> int:
    from .surprise import generate_surprise

    try:
        llm = get_llm(args.provider, args.model)
    except LLMError as e:
        console.print(f"[red]{e}[/]")
        return 1

    con = connect(args.db, readonly=True)
    console.print(f"[dim]asking {llm.name}/{llm.model}...[/]")
    try:
        text = _run_with_thinking_spinner(generate_surprise, con, llm)
    except LLMError as e:
        console.print(f"[red]{e}[/]")
        return 1
    finally:
        con.close()

    console.print(Panel(text, title="surprise me", title_align="left"))
    return 0


# -------------------------------------------------------------------- eval

def cmd_eval(args) -> int:
    from .engine import Engine
    from .eval import EVAL_CASES, run_eval

    try:
        llm = get_llm(args.provider, args.model)
    except LLMError as e:
        console.print(f"[red]{e}[/]")
        return 1

    cases = EVAL_CASES
    if args.category:
        cases = [c for c in cases if c.category == args.category]
        if not cases:
            cats = sorted({c.category for c in EVAL_CASES})
            console.print(f"[red]no cases in category {args.category!r} (have: {', '.join(cats)})[/]")
            return 1

    engine = Engine(args.db, llm=llm, dense=not args.no_dense)
    console.print(f"[dim]running {len(cases)} eval cases against {llm.name}/{llm.model}...[/]\n")
    results = run_eval(engine, cases)

    t = Table(title="results")
    for col in ("id", "category", "question", "pass", "reason", "answer"):
        t.add_column(col, overflow="fold", max_width=None if col in ("id", "category", "pass") else 40)
    for r in results:
        mark = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        t.add_row(r.case.id, r.case.category, r.case.question, mark, r.reason, r.answer)
    console.print(t)

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r.case.category, []).append(r)

    s = Table(title="summary by category")
    for col in ("category", "passed", "total", "rate"):
        s.add_column(col)
    for cat, rs in sorted(by_cat.items()):
        passed = sum(r.passed for r in rs)
        s.add_row(cat, str(passed), str(len(rs)), f"{passed / len(rs) * 100:.0f}%")
    total_passed = sum(r.passed for r in results)
    s.add_row(
        "[bold]overall[/]", f"[bold]{total_passed}[/]", f"[bold]{len(results)}[/]",
        f"[bold]{total_passed / len(results) * 100:.0f}%[/]",
    )
    console.print()
    console.print(s)

    return 0 if total_passed == len(results) else 1


# -------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yaad", description="chat with your WhatsApp memories")
    p.add_argument("--version", action="version", version=f"yaad {__version__}")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("ingest", help="parse an export and build the database")
    pi.add_argument("export", help="path to WhatsApp export .txt")
    pi.add_argument("--db", default="chat.db", help="output database path")
    pi.add_argument("--gap", type=int, default=45, help="session gap in minutes")
    pi.add_argument("--dd-mm", action="store_true", help="force DD/MM date parsing")
    pi.add_argument("--mm-dd", action="store_true", help="force MM/DD date parsing")
    pi.add_argument("--no-dense", action="store_true", help="skip embedding index")
    pi.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model for dense retrieval",
    )
    pi.set_defaults(func=cmd_ingest)

    pe_embed = sub.add_parser("embed", help="build/refresh the dense index on an already-ingested db")
    pe_embed.add_argument("--db", default="chat.db")
    pe_embed.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model for dense retrieval",
    )
    pe_embed.set_defaults(func=cmd_embed)

    pc = sub.add_parser("chat", help="ask questions about the chat")
    pc.add_argument("--db", default="chat.db")
    pc.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    pc.add_argument("--model", default=None, help="override the default model")
    pc.add_argument("--no-dense", action="store_true", help="disable dense retrieval")
    pc.set_defaults(func=cmd_chat)

    ps = sub.add_parser("stats", help="pretty-printed chat statistics (no LLM needed)")
    ps.add_argument("--db", default="chat.db")
    ps.set_defaults(func=cmd_stats)

    pq = sub.add_parser("search", help="hybrid search without an LLM")
    pq.add_argument("query")
    pq.add_argument("--db", default="chat.db")
    pq.add_argument("--sender", default=None)
    pq.add_argument("--after", default=None, help="YYYY-MM-DD")
    pq.add_argument("--before", default=None, help="YYYY-MM-DD")
    pq.add_argument("--top-k", type=int, default=5)
    pq.add_argument("--no-dense", action="store_true")
    pq.set_defaults(func=cmd_search)

    psu = sub.add_parser("surprise", help="a proactive, personalized recap - no question needed")
    psu.add_argument("--db", default="chat.db")
    psu.add_argument(
        "--provider", default="ollama", choices=["ollama", "anthropic"],
        help="small local models (~3B) misattribute facts on this task even with a hardened "
             "prompt - stick to the default model class (gemma4:e4b) or bigger",
    )
    psu.add_argument("--model", default=None)
    psu.set_defaults(func=cmd_surprise)

    pe = sub.add_parser("eval", help="run the built-in retrieval/synthesis eval set against a live LLM")
    pe.add_argument("--db", default="chat.db")
    pe.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    pe.add_argument("--model", default=None)
    pe.add_argument("--no-dense", action="store_true")
    pe.add_argument(
        "--category", default=None,
        help="only run one category: single_hop, cross_chunk, system_msg, grounding",
    )
    pe.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
