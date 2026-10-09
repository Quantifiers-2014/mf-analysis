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

## First run, in order

```bash
cd arbitrage-analyser                       # the unzipped folder (it contains pyproject.toml)
python3 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                     # "." = this folder; [dev] adds test tools
python -m arbitrage_analyser refresh        # load NAVs
python -m arbitrage_analyser import-benchmark ~/Downloads/benchmark.csv
python -m arbitrage_analyser app            # opens the app in your browser
```

The benchmark can also be uploaded later on the app's Data health tab instead of the
`import-benchmark` command.

## Load data

TER is **on**; Average AUM is **switched off** (`[settings.features]` in `config/funds.toml`).
While a dataset is off, its columns, uploads, import commands and freshness rows are hidden. Set
`aaum = true` to turn AAUM back on; its code and tests stay in place.

| Data | How | When |
| --- | --- | --- |
| NAV | `python -m arbitrage_analyser refresh` (or the **Refresh NAV now** button) | Daily |
| Benchmark | Download CSV, then upload on Data health or `import-benchmark FILE` | Monthly |
| TER | **Fetch TER from AMFI** button (Data health) or `fetch-ter [--month MM-YYYY]`; fallback: download the Excel, then upload or `import-ter FILE` | Monthly |
| Average AUM | Download per fund house from AMFI, then upload or `import-aaum FILE` | Quarterly |

Where to download:

- **Benchmark:** <https://www.niftyindices.com/reports/historical-data> → Equity → Strategy
  Indices → NIFTY 50 ARBITRAGE → pick dates → CSV. The first load can cover 01-Jan-2019 to today;
  after that, download the last two months each time. Dates already stored must match, or the file is
  rejected.
- **TER (automatic):** `python -m arbitrage_analyser fetch-ter` downloads last month's TER for
  every configured category (`--month 09-2026` for another month). It calls the same request the
  AMFI page makes when you click GO. That request is not a published API: if AMFI changes the page,
  the fetch fails with an error and the Excel download below still works. AMFI returns one fund
  house per request, so each fund needs its fund house id (`amfi_mf_id`, e.g. Kotak = 17) in
  `config/funds.toml`; funds without one are listed as not fetched. The AMFI sub category id per
  fund category is in `[settings.amfi_ter_category_ids]` (Arbitrage = 46).
- **TER (manual):** <https://www.amfiindia.com/ter-of-mf-schemes> → Financial Year, Month → Category
  "Hybrid Scheme", Sub Category "Arbitrage Fund", Mutual Fund "All" → GO → Download Excel. The app
  reads the Direct Plan BER and Total TER (one row per day). Funds missing from the file, or days
  with blank values, are skipped and listed after the import; their columns stay blank.
- **Average AUM:** <https://www.amfiindia.com/aum-data/average-aum> → Schemewise, Categorywise,
  one fund house, financial year, quarter → Go → Excel. Repeat for each fund house.

NAV comes from [mfapi.in](https://www.mfapi.in), a free unofficial API that republishes AMFI data.
AMFI's own NAV history download allows only 90 days per request. Every refresh checks that the
scheme code, ISIN and "Direct Plan - Growth" in the source match `config/funds.toml`.

## Run the app

```bash
python -m arbitrage_analyser app
```

Run it with the virtual environment active. It starts Streamlit with the same Python, so the
app always finds this package.

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
- Langfuse tracing (optional): create a free account at <https://langfuse.com/cloud>, make a
  project, and put its API keys in `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
  `LANGFUSE_BASE_URL` for the US cloud or a self-hosted server). Traces follow Langfuse's best
  practices (`agent/tracing.py`):
  - one trace `answer-fund-question` per question (input: the question, output: the answer),
    one session per conversation, with the user name, `LANGFUSE_TRACING_ENVIRONMENT`
    (default `development`), prompt version and app version;
  - inside it, a `generate-response` generation per model call (messages, model, tokens; for
    Gemini via Langfuse's OpenAI integration) and a `tool` observation per tool call;
  - errors marked on the trace and failed tool calls as warnings;
  - thumbs up/down as the boolean score `response_rating`;
  - emails, phone, PAN and Aadhaar numbers masked before anything is sent.
  Names are used by Langfuse filters and evaluators, so rename them with care.
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
| Benchmark | Zero/negative values or duplicate dates | File rejected |
| Benchmark | Stored dates must match within 0.01 | File rejected |
| TER | Value outside 0–5%, Total TER below BER | Flag |

Daily moves are divided by the calendar days since the previous value, so weekend accrual does not
trigger flags. A flag is reviewed on the Data health screen; a reviewed flag stays reviewed.

## Project layout

```
config/funds.toml          fund list and settings
src/arbitrage_analyser/
  config.py                load and validate config
  db.py                    SQLite schema, reads, writes
  sources/                 mfapi.in and AMFI TER fetch; benchmark, TER, AAUM file parsers
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

- The TER parser follows the AMFI Excel for Sep-2026 (checked 08-Oct-2026; a copy is in
  `tests/data`). The AAUM parser follows the layout seen on 07-Oct-2026. A changed AMFI layout gives a
  clear error naming the missing column; the parser then needs updating.
- NAV relies on mfapi.in. If it goes away, add another fetcher in `sources/`; the rest is unchanged.
