"""The answer engine: route -> gather evidence -> synthesize.

search    -> hybrid retrieval -> excerpts with dates/senders
analytics -> LLM writes SQL   -> read-only execution (retry once on error)
both      -> both legs feed the final synthesis
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .analytics import SCHEMA_DOC, rows_to_text, run_readonly_sql
from .db import connect
from .llm import BaseLLM
from .retrieve import hybrid_search
from .router import Route, route_query

SQLGEN_SYSTEM = f"""You write exactly one read-only SQLite query for a WhatsApp chat database.

{SCHEMA_DOC}

Rules:
- SELECT or WITH only. One statement. No comments, no fences, no prose.
- Prefer the convenience views when they fit the question.
- Add LIMIT 50 unless the question needs fewer rows.
Return ONLY the SQL."""

ANSWER_SYSTEM = """You are yaad, answering questions about the user's own WhatsApp chat export.
Rules:
- Ground every claim in the provided excerpts / stats. Cite like (Rohan, 2025-10-12).
- If the answer isn't in the context, say so plainly - never invent chat content.
- Be concise and match the user's tone. Numbers from stats should be exact."""

MAX_HISTORY = 3


@dataclass
class Answer:
    text: str
    route: Route
    sources: list[str] = field(default_factory=list)
    sql: str | None = None


def _strip_fences(s: str) -> str:
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", s.strip())


class Engine:
    def __init__(self, db_path: str, llm: BaseLLM, dense: bool = True):
        self.db_path = str(db_path)
        self.con = connect(db_path)
        self.llm = llm
        self.dense = None
        if dense:
            try:
                from .embed import DenseSearcher

                self.dense = DenseSearcher(self.con)
            except Exception:
                self.dense = None  # FTS-only mode

        self.participants = tuple(
            r["sender"] for r in self.con.execute("SELECT sender FROM v_sender_stats")
        )
        row = self.con.execute("SELECT MIN(date) AS a, MAX(date) AS b FROM messages").fetchone()
        self.date_range = (row["a"], row["b"])
        self.history: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ api

    def answer(self, question: str) -> Answer:
        route = route_query(
            question,
            llm=self.llm,
            participants=self.participants,
            date_range=self.date_range,
            today=date.today().isoformat(),
        )

        context_parts: list[str] = []
        sources: list[str] = []
        sql_used: str | None = None

        if route.intent in ("search", "both"):
            blocks = hybrid_search(
                self.con,
                route.search_query or question,
                top_k=6,
                sender=route.sender,
                date_from=route.date_from,
                date_to=route.date_to,
                dense_searcher=self.dense,
            )
            for b in blocks:
                span = f"{b['start_ts'][:10]} to {b['end_ts'][:10]}"
                context_parts.append(f"--- excerpt ({span}; {b['senders']}) ---\n{b['text']}")
                sources.append(f"{span} · {b['senders']}")
            if not blocks:
                context_parts.append("--- no matching messages found ---")

        if route.intent in ("analytics", "both"):
            sql_used, result_text = self._run_analytics(route.analytics_question or question)
            context_parts.append(f"--- stats SQL ---\n{sql_used}\n--- result ---\n{result_text}")

        prompt = (
            "\n\n".join(context_parts)
            + f"\n\nQuestion: {question}\nAnswer using only the context above."
        )
        messages: list[dict] = []
        for q, a in self.history[-MAX_HISTORY:]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": prompt})

        text = self.llm.complete(ANSWER_SYSTEM, messages, max_tokens=700).strip()
        self.history.append((question, text))
        return Answer(text=text, route=route, sources=sources, sql=sql_used)

    # ------------------------------------------------------------- internal

    def _run_analytics(self, question: str) -> tuple[str, str]:
        prompt = question
        sql = ""
        err = ""
        for _ in range(2):  # one retry with the error fed back
            sql = _strip_fences(
                self.llm.complete(SQLGEN_SYSTEM, [{"role": "user", "content": prompt}], 400)
            ).rstrip(";")
            try:
                cols, rows = run_readonly_sql(self.db_path, sql)
                return sql, rows_to_text(cols, rows)
            except Exception as e:
                err = str(e)
                prompt = (
                    f"{question}\n\nYour previous SQL failed.\n"
                    f"SQL: {sql}\nError: {err}\nReturn corrected SQL only."
                )
        return sql, f"(query failed: {err})"
