"""yaad CLI: ingest, chat, stats, search."""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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

    engine = Engine(args.db, llm=llm, dense=not args.no_dense)
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
            ans = engine.answer(q)
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
