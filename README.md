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

## Getting started, from scratch

Try the bundled sample chat instead of exporting your own:
```bash
git clone https://github.com/kiran-gajjela/yaad.git && cd yaad
pip install -e .
yaad ingest examples/sample_chat.txt --db demo.db
yaad surprise --db demo.db
```

### 1. Clone the repo

```bash
git clone https://github.com/kiran-gajjela/yaad.git
cd yaad
```

### 2. Install yaad

```bash
pip install -e .              # core: FTS search + analytics, no LLM needed yet
pip install -e ".[dense]"     # + semantic search (recommended — sentence-transformers)
```

### 3. Set up a local LLM

yaad defaults to [Ollama](https://ollama.com) running fully on your machine — install it, then:

```bash
ollama serve                  # start it, if it isn't already running
ollama pull llama3.2:3b       # or any model — see "LLM setup" below for other options
```

(Skip this if you'd rather use the Claude API instead — see "LLM setup" below.)

### 4. Export your WhatsApp chat

On your **phone**, open the specific chat or group you want to ask questions about:

- **Android**: tap ⋮ (top right) → **More** → **Export chat**
- **iPhone**: tap the chat/group name at the top → scroll down → **Export Chat**

Choose **Without Media** — smaller file, faster, and yaad only reads the text anyway.

> ⚠️ **If "Advanced Chat Privacy" is turned on for that chat, you can't export it at all.** WhatsApp disables the Export Chat option entirely while it's active — this is a real, documented WhatsApp behavior, not a yaad limitation. Any admin can turn it off first: open the chat → tap the chat name → **Advanced Chat Privacy** → toggle off. Then export.

Get the exported `.txt` off your phone and onto the computer running yaad — email it to yourself, save it via Drive/iCloud, AirDrop, cable, whatever's easiest.

### 5. Put the export where yaad can see it

Create an `exports/` folder at the root of this repo and drop the file in:

```bash
mkdir -p exports
```

```
yaad/
└── exports/
    └── WhatsApp Chat with Friends.txt   ← put it here
```

### 6. Ingest it

```bash
yaad ingest "exports/WhatsApp Chat with Friends.txt" --db mychat.db
```

This parses the export, groups messages into sessions, builds the search index, and creates one `mychat.db` SQLite file — everything for this chat lives in that single file.

### 7. Sanity-check the parse (no LLM needed)

```bash
yaad stats --db mychat.db
```

Check the participant list, message count, and date range actually look right before trusting anything downstream.

### 8. Ask it something — start with "surprise me"

```bash
yaad surprise --db mychat.db
```

No question to think of — it pulls a few genuinely interesting things out of the chat on its own: who's most active, a callback to the first message ever sent, that kind of thing.

### 9. Then ask whatever you're actually curious about

```bash
yaad chat --db mychat.db
```

```
you › who suggested the trip?
you › what did we decide about the venue?
you › summarize last week
you › /quit
```

## LLM setup

yaad is built for on-device LLMs first — that's the whole point: your chat never leaves your machine. Cloud is there if you specifically want the reasoning quality of a larger model and don't mind the trade-off.

| Provider | Model | Config |
|---|---|---|
| `ollama` (default) | `llama3.2:3b` (default) | `ollama serve` + `ollama pull llama3.2:3b`; host via `OLLAMA_HOST` |
| `ollama` | `gemma4:e4b` | `ollama pull gemma4:e4b` — noticeably better reasoning than the 3B default, at the cost of speed |
| `anthropic` | `claude-sonnet-4-6` | `export ANTHROPIC_API_KEY=...` — cloud, for when you want stronger reasoning and accept the trade-off |

Override with `--model` (and `--provider anthropic` for Claude). Everything else (parsing, indexing, FTS, stats) runs fully offline regardless of which LLM you point at it.

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
- [ ] Multimodal input (images) via vision-capable local models (e.g. Gemma 4) - v2

## Dev

```bash
pip install -e ".[dev]"
pytest
```

MIT.
