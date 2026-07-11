"""X-ray one question through yaad's real pipeline, step by step.

Mirrors Engine.answer() exactly (same router call, same retrieval
fallback logic, same synthesis prompt) but prints every intermediate
step instead of only the final answer - useful for demos/presentations
where you want to show *why* an answer is right, not just that it is.

Usage:
    python scripts/dry_run_demo.py --db ladakh.db \\
        --question "what happened with that permit problem back in april?"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from yaad.db import connect
from yaad.engine import ANSWER_SYSTEM
from yaad.llm import get_llm
from yaad.retrieve import FULL_RANGE_CHUNK_CAP, chunks_in_range, hybrid_search
from yaad.router import route_query


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--no-dense", action="store_true")
    args = p.parse_args()

    llm = get_llm(args.provider, args.model)
    con = connect(args.db, readonly=True)

    dense = None
    if not args.no_dense:
        try:
            from yaad.embed import DenseSearcher

            dense = DenseSearcher(con)
        except Exception:
            dense = None

    participants = tuple(r["sender"] for r in con.execute("SELECT sender FROM v_sender_stats"))
    row = con.execute("SELECT MIN(date) AS a, MAX(date) AS b FROM messages").fetchone()

    def header(title: str) -> None:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    header("STEP 1: ROUTER CLASSIFICATION")
    route = route_query(
        args.question, llm=llm, participants=participants,
        date_range=(row["a"], row["b"]), today=date.today().isoformat(),
    )
    print(json.dumps(
        {
            "intent": route.intent,
            "search_query": route.search_query,
            "sender": route.sender,
            "date_from": route.date_from,
            "date_to": route.date_to,
        },
        indent=2,
    ))

    header("STEP 2: RETRIEVAL")
    blocks = []
    if route.intent in ("search", "both"):
        if route.date_from or route.date_to:
            ranged = chunks_in_range(con, sender=route.sender, date_from=route.date_from, date_to=route.date_to)
            print(f"exhaustive range check: {len(ranged)} chunks match filters (cap={FULL_RANGE_CHUNK_CAP})")
            if 0 < len(ranged) <= FULL_RANGE_CHUNK_CAP:
                blocks = ranged
                print("-> using exhaustive chronological set (bypassing ranked search)\n")
        if not blocks:
            print("-> ranked hybrid_search (BM25 + dense, RRF fusion)\n")
            blocks = hybrid_search(
                con, route.search_query or args.question, top_k=6,
                sender=route.sender, date_from=route.date_from, date_to=route.date_to,
                dense_searcher=dense,
            )
        for i, b in enumerate(blocks, 1):
            print(f"--- chunk {i} ({b['start_ts']} to {b['end_ts']}; senders: {b['senders']}) ---")
            print(b["text"])
            print()

    sql_used = None
    result_text = None
    if route.intent in ("analytics", "both"):
        from yaad.engine import Engine

        engine = Engine(args.db, llm=llm, dense=dense is not None)
        sql_used, result_text = engine._run_analytics(route.analytics_question or args.question, sender=route.sender)
        print(f"SQL:\n{sql_used}\n\nResult:\n{result_text}")

    header("STEP 3: FINAL PROMPT")
    context_parts = []
    for b in blocks:
        span = f"{b['start_ts'][:10]} to {b['end_ts'][:10]}"
        context_parts.append(f"--- excerpt ({span}; {b['senders']}) ---\n{b['text']}")
    if sql_used:
        context_parts.append(f"--- stats SQL ---\n{sql_used}\n--- result ---\n{result_text}")
    prompt = "\n\n".join(context_parts) + f"\n\nQuestion: {args.question}\nAnswer using only the context above."
    print(prompt)

    header("STEP 4: FINAL ANSWER")
    answer = llm.complete(ANSWER_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=700)
    print(answer.strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())
