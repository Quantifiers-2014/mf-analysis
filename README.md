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

TER and factsheet values are **on** (`[settings.features]` in `config/funds.toml`). While a
dataset is off, its columns, uploads, commands and freshness rows are hidden.

| Data | How | When |
| --- | --- | --- |
| NAV | `python -m arbitrage_analyser refresh` (or the **Refresh NAV now** button) | Daily |
| Benchmark | Download CSV, then upload on Data health or `import-benchmark FILE` | Monthly |
| TER | **Fetch TER from AMFI** button (Data health) or `fetch-ter [--month MM-YYYY]`; fallback: download the Excel, then upload or `import-ter FILE` | Monthly |
| AUM, AAUM, turnover, Direct BER | Automatic: monthly GitHub job (see below), or **Fetch factsheets now** (Data health) / `fetch-factsheets` | Monthly |

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
- **Factsheet values (automatic):** `python -m arbitrage_analyser fetch-factsheets` downloads
  each fund's latest factsheet PDF from the AMC site (`[funds.factsheet]` in `config/funds.toml`),
  finds the fund's page and reads AUM, AAUM, portfolio turnover and the Direct Plan BER. See
  "Factsheet values" below.

NAV comes from [mfapi.in](https://www.mfapi.in), a free unofficial API that republishes AMFI data.
AMFI's own NAV history download allows only 90 days per request. Every refresh checks that the
scheme code, ISIN and "Direct Plan - Growth" in the source match `config/funds.toml`.

## Factsheet values

AUM, AAUM, portfolio turnover and the Direct Plan BER have no free API, so they are read from the
AMC factsheet PDFs.

- **Where they are kept:** `published/factsheet_values.csv` in the repository, one row per fund
  per month, with the PDF link and page. The first rows (Sep-2026; Nippon Aug-2026) were read with these
  rules from factsheets downloaded by hand on 18-Oct-2026. The app loads it on start, so the values survive
  Streamlit Cloud restarts (its database starts empty). The Factsheet column opens the PDF at the
  fund's page.
- **Monthly job:** `.github/workflows/factsheets.yml` runs on the 12th, 18th and 25th, and on
  demand (GitHub > Actions > "Factsheet values" > Run workflow). It fetches the last two months of
  AMFI TER (for the BER check), reads the factsheets, and commits the values file to `main` only
  when a value changed, so `main` must accept pushes from `github-actions[bot]`. The run is marked
  failed when a fund's factsheet could not be read or a value was flagged; the step log names
  them, and the other values are still saved. If an AMC site blocks GitHub, run
  `python -m arbitrage_analyser fetch-factsheets` on a laptop and commit
  `published/factsheet_values.csv` through a pull request.
- **Values file checks:** every line is checked when loaded (dates, link, page, the value ranges
  below); a bad line is skipped and shown as a warning in the app. The app also re-checks the
  latest values against any AMFI TER fetched later, so a BER mismatch shows as a flag.
- **Finding the file:** a link with placeholders is tried for this month's edition, then the two
  before; Edelweiss PDF names carry an upload timestamp, so its listing page is read instead.
- **Accuracy rules** (`sources/factsheet_pdf.py`, tested on the Sep-2026 factsheets of all 8 AMCs
  in `tests/data/factsheets`):
  - The fund's page is the one whose heading (largest text) is `page_title`; pages that only
    mention the fund, such as fund-of-fund pages, are skipped. Two such pages showing AUM is an
    error.
  - Each value is taken next to its printed label, by position on the page. A value that is not
    found, or not clear (two % values on the Direct BER row), stays blank: nothing is estimated
    or carried over from another PDF. A re-read of the same month that lacks a stored value
    keeps the stored row.
  - The as-of date is the one printed on the page, never the file name: Nippon's "September"
    factsheet holds August data and ABSL's "October" one holds September data.
  - Turnover is kept as printed. AMCs use different units (a ratio, or %) and bases (equity, or
    gross exposure including derivatives), so it is not comparable across AMCs.
- **Checks:** AUM and AAUM above 0 and below Rs 10 lakh crore, BER between 0 and 2.5%, else the
  value is not stored and an error is shown. Flags: AAUM outside 0.5-2 times AUM; factsheet BER
  more than 0.02 points from the AMFI BER on the same date, with and without 18% GST (Kotak
  prints BER with GST). Older data never replaces newer.
- **When an AMC changes its factsheet:** the fetch reports the fund (no page titled ..., a value
  not found), the other funds still load. Update `[funds.factsheet]` (link, page title, turnover
  labels); a layout change may need a rule change and a new fixture page.

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
the TER name match, and where each fund's factsheet is. `[settings.features]` switches the TER
and factsheet datasets on or off. It also holds the validation settings (outlier multiplier, gap days, overlap
tolerance) and freshness thresholds. Run `check-config` after editing.

Override paths with `ARBITRAGE_CONFIG`, `ARBITRAGE_DB` (default database: `data/arbitrage.db`)
and `ARBITRAGE_FACTSHEET_CSV` (default: `published/factsheet_values.csv`).
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
| Factsheet | AUM/AAUM not above 0 or above Rs 10 lakh crore; BER outside 0–2.5% | Value not stored, error shown |
| Factsheet | AAUM outside 0.5–2x AUM; BER more than 0.02 from AMFI BER (with and without GST) | Flag |

Daily moves are divided by the calendar days since the previous value, so weekend accrual does not
trigger flags. A flag is reviewed on the Data health screen; a reviewed flag stays reviewed.

## Project layout

```
config/funds.toml          fund list and settings
src/arbitrage_analyser/
  config.py                load and validate config
  db.py                    SQLite schema, reads, writes
  sources/                 mfapi.in and AMFI TER fetch; benchmark and TER file parsers;
                           factsheet download (factsheet_links.py) and PDF rules (factsheet_pdf.py)
  validation.py            data checks
  metrics.py               point-to-point, rolling returns, tracking difference
  ingest.py                fetch/parse -> validate -> store
  factsheets.py            factsheet values: fetch, checks, values file
  services.py              screen read models (reusable by a future API or chatbot)
  cli.py                   command line
  app.py                   Streamlit UI
  chat_ui.py               "Ask the analyst" chat tab
  agent/                   AI agent: tools, prompt, Claude loop, chat storage, Langfuse tracing
published/                factsheet values kept in the repository (written by the monthly job)
.github/workflows/         monthly factsheet job
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
  `tests/data`). A changed AMFI layout gives a clear error naming the missing column; the parser
  then needs updating.
- Factsheet links follow the AMC sites as seen in Oct-2026, but have not yet been downloaded by the
  job (this was built without access to the AMC sites).
  The SBI and Invesco links are used without the `?sfvrsn=...` the sites add; if they fail, the
  first job run shows it.
- NAV relies on mfapi.in. If it goes away, add another fetcher in `sources/`; the rest is unchanged.
