# yaad

**Chat with your WhatsApp memories.** Local-first RAG + analytics over your chat exports.

Your group chats remember everything — the villa someone found, the dates you finally agreed on, who bailed on the trek. `yaad` (याद, *memory*) turns a WhatsApp export into something you can actually ask questions of.

```
you › who suggested the villa?
yaad · search · 3 excerpts
Priya found it — a villa in Anjuna, 4k per night, sleeps 6, with a pool (Priya, 2025-10-12).

you › who's the most active in the group?
yaad · analytics
Rohan, with 21 messages. Priya is second with 14...

you › what did we finalize for dates?
yaad · search · 2 excerpts
29 Dec to 1 Jan — decided after Sameer's exams ruled out 19-21 Dec (Rohan, 2025-11-02).
```

## Why this isn't just "throw the txt at an LLM"

Chat data breaks naive RAG in two ways:

1. **Individual messages are terrible retrieval units.** "haan", "ok", "😂" embed to noise. `yaad` sessionizes on time gaps and embeds overlapping windows of conversation instead.
2. **Half the interesting questions are aggregations.** "Who's most active", "average reply time" — no amount of vector search answers these. They're SQL questions.

So a router classifies every question first:

```
question ──→ router (LLM) ──┬── search ────→ hybrid retrieval ─────┐
                            │                FTS5/BM25 ⊕ dense      │
                            │                fused with RRF         ├──→ synthesis ──→ answer
                            │                                       │     (grounded,
                            └── analytics ─→ LLM writes SQL ────────┘      cited)
                                             read-only execution
```

- **Hybrid retrieval**: names and places are strong lexical anchors in chat, so BM25 (SQLite FTS5) pulls its weight; dense embeddings catch paraphrases ("the beach house thing" → villa discussion). Reciprocal Rank Fusion merges both.
- **Guarded text-to-SQL**: the LLM writes SQLite against a documented schema, executed on a **read-only** connection, single statement, row-capped, with one auto-retry on error.
- **Grounded answers**: the synthesizer cites senders and dates, and says so when something isn't in the chat.

## Quickstart

**1. Export a chat** (without media): WhatsApp → open the chat → ⋮ → More → Export chat → *Without media*. You get a `.txt`.

**2. Install:**

```bash
pip install -e .              # core: FTS search + analytics
pip install -e ".[dense]"     # + semantic search (sentence-transformers)
```

**3. Ingest:**

```bash
yaad ingest "WhatsApp Chat with Goa Plan.txt" --db goa.db
```

**4. Ask:**

```bash
yaad chat --db goa.db                          # needs Ollama running locally
yaad chat --db goa.db --provider anthropic     # or the Claude API
```

No LLM handy? These work standalone:

```bash
yaad stats --db goa.db          # who talks, monthly/hourly activity, top emojis
yaad search "villa" --db goa.db --sender Priya --after 2025-10-01
```

Try it on the bundled sample first:

```bash
yaad ingest examples/sample_chat.txt --db demo.db
yaad chat --db demo.db
```

## LLM setup

| Provider | Default model | Config |
|---|---|---|
| `ollama` (default) | `llama3.2:3b` | `ollama serve` + `ollama pull llama3.2:3b`; host via `OLLAMA_HOST` |
| `anthropic` | `claude-sonnet-4-6` | `export ANTHROPIC_API_KEY=...` |

Override with `--model`. Everything else (parsing, indexing, FTS, stats) runs fully offline.

## What the parser survives

Android and iOS export formats, 12h/24h times, multiline messages, media placeholders, system messages (group created / added / left), the invisible unicode marks WhatsApp sprinkles everywhere, and the DD/MM vs MM/DD ambiguity (auto-inferred from the file; force with `--dd-mm` / `--mm-dd`).

## Privacy

- Everything stays on your machine: parsing, SQLite storage, FTS, embeddings, and (with Ollama) the LLM too.
- `*.db` and `exports/` are gitignored by default — chat exports contain *other people's* messages; don't commit them, don't ship them to third parties without thinking.
- One database per chat export. Delete the `.db`, and it's gone.

## Project layout

```
yaad/
├── parser.py       # WhatsApp export → structured messages
├── sessionize.py   # time-gap sessions → overlapping chunks
├── db.py           # SQLite schema, FTS5, analytics views, ingest
├── retrieve.py     # BM25 + dense, RRF fusion, filters, dedupe
├── embed.py        # optional sentence-transformers index
├── analytics.py    # prebuilt stats + guarded read-only SQL
├── router.py       # search vs analytics intent (LLM + heuristic fallback)
├── llm.py          # Ollama / Anthropic via plain requests
├── engine.py       # orchestration + grounded synthesis
└── cli.py          # ingest · chat · stats · search
```

## Roadmap

- [ ] LLM session summaries as an additional retrieval layer
- [ ] Multi-chat databases with cross-chat questions
- [ ] Temporal knowledge graph over entities/events (who-what-when across months)
- [ ] Small eval set for retrieval quality

## Dev

```bash
pip install -e ".[dev]"
pytest
```

MIT.
