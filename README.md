# Arbitrage Fund Analyser (v0)

An internal tool that compares Indian arbitrage mutual funds on returns, costs and size, against the
NIFTY 50 Arbitrage index. Built to the agreed v0 spec: three screens (Fund comparison, Rolling
returns, Data health), SQLite storage, and data-quality checks on every load.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m arbitrage_analyser check-config
```

## Load data

TER and Average AUM are **switched off** for now (`[settings.features]` in `config/funds.toml`).
While off, their columns, uploads, import commands and freshness rows are hidden. Set `ter = true` or
`aaum = true` to turn them back on; the code and tests for both stay in place.

| Data | How | When |
| --- | --- | --- |
| NAV | `python -m arbitrage_analyser refresh` (or the **Refresh NAV now** button) | Daily |
| Benchmark | Download CSV, then upload on Data health or `import-benchmark FILE` | Monthly |
| TER | Download from AMFI, then upload or `import-ter FILE` | Monthly |
| Average AUM | Download per fund house from AMFI, then upload or `import-aaum FILE` | Quarterly |

Where to download:

- **Benchmark:** <https://www.niftyindices.com/reports/historical-data> → Equity → Strategy
  Indices → NIFTY 50 ARBITRAGE → pick dates → CSV. The first load can cover 01-Jan-2019 to today;
  after that, download the last two months each time. Dates already stored must match, or the file is
  rejected.
- **TER:** <https://www.amfiindia.com/ter-of-mf-schemes> → choose the month → export.
- **Average AUM:** <https://www.amfiindia.com/aum-data/average-aum> → Schemewise, Categorywise,
  one fund house, financial year, quarter → Go → Excel. Repeat for each fund house.

NAV comes from [mfapi.in](https://www.mfapi.in), a free unofficial API that republishes AMFI data.
AMFI's own NAV history download allows only 90 days per request. Every refresh checks that the
scheme code, ISIN and "Direct Plan - Growth" in the source match `config/funds.toml`.

## Run the app

```bash
streamlit run src/arbitrage_analyser/app.py
```

## Ask the analyst (AI agent, Phase 1)

The **Ask the analyst** tab is a chat that answers questions about the funds in the database.
An AI model uses read-only tools over the same data and calculations as the other screens, so
every figure comes from the database, with its source and as-of date. It declines questions
outside mutual funds, says when it is unsure or data is missing, and asks a clarifying question
when one is needed.

Setup (once):

```bash
pip install -e ".[agent]"
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Put a Google Gemini API key (free from <https://aistudio.google.com/apikey>) in `.env` after
`GEMINI_API_KEY=`, then restart the app. `.env` is git-ignored; never share it.

- Model: Gemini `gemini-3.8-flash` by default. Change it with `ARBITRAGE_AGENT_MODEL`. To use
  Claude, set `ARBITRAGE_AGENT_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`; for Groq, OpenRouter
  or Ollama use `openai_compatible` (see `.env.example`). The sidebar shows the model in use.
- Free tiers allow only a few requests per minute and per day, and one question takes 2 to 4
  requests. On the free tier Google may use prompts to improve its products, so do not type
  confidential information into the chat.
- Conversations are saved in `data/chat.db` (override with `ARBITRAGE_CHAT_DB`), under the name
  in the sidebar, so you can reopen them later. Each answer is stored with the tool calls
  behind it, model, tokens, time taken, Langfuse trace id and any thumbs up/down.
- Langfuse tracing: fill in `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and
  `LANGFUSE_BASE_URL` if not on Langfuse's EU cloud). Each question is one trace
  (session = conversation) showing every model call and tool call; thumbs up/down are sent as a
  `user_feedback` score.
- The instructions are in `src/arbitrage_analyser/agent/prompt.py`; bump `PROMPT_VERSION` when
  you change them.
- Limits for now: no web search or file uploads, the last 30 messages of a conversation are sent
  with each question, and answers are text and tables only (no charts yet).

## Daily NAV refresh

macOS/Linux (cron, 8:30 pm daily):

```
30 20 * * * cd /path/to/arbitrage-analyser && .venv/bin/python -m arbitrage_analyser refresh
```

Windows: Task Scheduler → daily task running
`C:\path\to\arbitrage-analyser\.venv\Scripts\python.exe -m arbitrage_analyser refresh`
with "Start in" set to the project folder. A missed day is caught up on the next run.

## Configuration

`config/funds.toml` holds the fund list, category, AMFI code, ISIN, exit load, fund managers, and
the TER name match. `[settings.features]` switches the TER and AAUM datasets on or off. It also holds the validation settings (outlier multiplier, gap days, overlap
tolerance) and freshness thresholds. Run `check-config` after editing.

Override paths with `ARBITRAGE_CONFIG` and `ARBITRAGE_DB` (default database: `data/arbitrage.db`).
Back up by copying the `.db` file.

## Data checks

| Data | Check | Result |
| --- | --- | --- |
| NAV | Scheme code, ISIN, Direct Growth plan match config | Fund skipped, error shown |
| NAV | Duplicate dates, zero/negative values, gaps over 5 days | Flag |
| NAV, benchmark | Daily move above 10x the median daily move of the last 250 days | Flag |
| Benchmark | Zero/negative, duplicate or weekend dates | File rejected |
| Benchmark | Stored dates must match within 0.01 | File rejected |
| TER | Value outside 0–5%, total below base | Flag |

Daily moves are divided by the calendar days since the previous value, so weekend accrual does not
trigger flags. A flag is reviewed on the Data health screen; a reviewed flag stays reviewed.

## Project layout

```
config/funds.toml          fund list and settings
src/arbitrage_analyser/
  config.py                load and validate config
  db.py                    SQLite schema, reads, writes
  sources/                 mfapi.in fetch; benchmark, TER, AAUM file parsers
  validation.py            data checks
  metrics.py               point-to-point, rolling returns, tracking difference
  ingest.py                fetch/parse -> validate -> store
  services.py              screen read models (reusable by a future API or chatbot)
  cli.py                   command line
  app.py                   Streamlit UI
  chat_ui.py               "Ask the analyst" chat tab
  agent/                   AI agent: tools, prompt, Claude loop, chat storage, Langfuse tracing
tests/                     unit and app tests
```

## Development

```bash
pytest
ruff check src tests && ruff format --check src tests
mypy
```

## Known limits (v0)

- The TER and AAUM parsers follow the layouts seen on 07-Oct-2026. A changed AMFI layout gives a
  clear error naming the missing column; the parser then needs updating.
- NAV relies on mfapi.in. If it goes away, add another fetcher in `sources/`; the rest is unchanged.
