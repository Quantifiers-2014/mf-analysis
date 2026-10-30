"""Search the text of the funds' factsheet pages, for the chatbot.

Chunks (see sources/factsheet_text.py) are stored per fund and factsheet month. A question is
answered from at most MAX_RESULTS chunks and MAX_RESULT_TOKENS tokens, so each model call stays
within free-tier token limits.

Modes (settings.search_mode):
- keyword: BM25 over the chunk text, with a small list of synonyms (derivative ~ futures ~
  hedge, ...). Chunks scoring below MIN_KEYWORD_SCORE are not returned: then the answer is "not
  stated in the factsheet".
- semantic: cosine similarity of Mistral embeddings (question vs chunk), above MIN_SIMILARITY.
- hybrid: both lists merged by reciprocal rank.
Semantic and hybrid fall back to keyword when Mistral cannot be reached or a chunk has no
embedding yet.

Filters: the funds named in the question (or all funds), and a factsheet month (the month of
the data, e.g. "August" = data as of 31-Aug); without a month, each fund's latest factsheet.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from arbitrage_analyser import embeddings
from arbitrage_analyser.config import AppConfig, Fund
from arbitrage_analyser.sources import SourceError
from arbitrage_analyser.sources.factsheet_text import Chunk, estimate_tokens, text_hash

MAX_RESULTS = 5
MAX_RESULT_TOKENS = 1500
MIN_KEYWORD_SCORE = 1.5  # BM25 score; set with the evals: answer chunks scored 1.8+ (Oct-2026)
MIN_SIMILARITY = 0.75  # cosine similarity for mistral-embed; to tune with the evals
BM25_K1 = 1.5
BM25_B = 0.75
SYNONYM_WEIGHT = 0.5  # a synonym counts half as much as the word asked for
RRF_K = 60  # reciprocal rank fusion constant (hybrid)

# Words that carry no meaning for search.
_STOP = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "per", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with", "fund", "funds",
    "scheme", "schemes", "factsheet", "say", "said", "tell", "me", "about", "any", "there",
    "their", "they",
))  # fmt: skip
# Groups of words that mean the same in a factsheet; a question word also finds the others.
SYNONYMS: tuple[frozenset[str], ...] = tuple(
    frozenset(g.split())
    for g in (
        "derivative futures future option f&o hedge hedged hedging",
        "expense ter ber cost charge",
        "aum asset size corpus",
        "aaum average",
        "manager managed managing",
        "exit load redemption redeem",
        "benchmark index",
        "inception launch allotment",
        "riskometer risk",
        "holding portfolio position stock",
        "yield ytm",
        "return performance",
        "turnover churn",
        "minimum lumpsum sip",
        "objective aim",
        "duration maturity",
    )
)
_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS = {m: i + 1 for i, m in enumerate(_MONTH_NAMES)}
_FULL_MONTHS = [date(2026, m, 1).strftime("%B").lower() for m in range(1, 13)]
_MONTH = re.compile(  # "August", "Aug", "Sept 2026", "Sep-26"; not "market" or "decide"
    r"\b(" + "|".join([*_FULL_MONTHS, "sept", *_MONTH_NAMES]) + r")\b\.?"
    r"(?:\s*[-,']?\s*(\d{4}|\d{2})\b)?",
    re.IGNORECASE,
)
# Words in AMC names that do not identify one AMC.
_COMMON_AMC_WORDS = frozenset({"india", "mutual", "fund", "life", "sun", "asset", "management"})


@dataclass(frozen=True)
class Hit:
    amfi_code: int
    fund: str
    as_of: str
    chunk_no: int
    page: int
    section: str
    text: str
    source_url: str
    score: float

    @property
    def link(self) -> str:
        return f"{self.source_url}#page={self.page}"

    def record(self) -> dict[str, object]:
        """What the model sees (and the chat shows under "Data used")."""
        return {
            "fund": self.fund,
            "as_of": self.as_of,
            "page": self.page,
            "section": self.section,
            "link": self.link,
            "score": round(self.score, 3),
            "text": self.text,
        }


@dataclass
class SearchResult:
    hits: list[Hit] = field(default_factory=list)
    mode: str = "keyword"  # the mode used (keyword when semantic fell back)
    funds: list[str] = field(default_factory=list)  # funds searched
    month: str | None = None  # YYYY-MM asked for, or None for the latest
    note: str | None = None  # e.g. why semantic search fell back to keyword


# ---------- storing chunks ----------


def embed_text(fund: Fund, as_of: str, chunk: Chunk) -> str:
    """The text embedded for a chunk: the chunk with its fund and month, so a vector carries
    them too. Its hash is the chunk's text_hash."""
    month = date.fromisoformat(as_of).strftime("%d-%b-%Y")
    return f"{fund.name} factsheet, data as of {month}. {chunk.section}\n{chunk.text}"


def store_chunks(
    conn: sqlite3.Connection,
    fund: Fund,
    as_of: date,
    page: int,
    source_url: str,
    chunks: Sequence[Chunk],
) -> int:
    """Replace the fund's chunks for this factsheet month. Returns the number of rows changed
    (0 when the text is the same as stored, or when there are no chunks: the stored ones stay)."""
    if not chunks:
        return 0
    day = as_of.isoformat()
    rows = [
        (fund.amfi_code, day, n, fund.amc, page, c.section, c.text,
         text_hash(embed_text(fund, day, c)), source_url)
        for n, c in enumerate(chunks, start=1)
    ]  # fmt: skip
    stored = conn.execute(
        "SELECT amfi_code, as_of, chunk_no, amc, page, section, text, text_hash, source_url "
        "FROM factsheet_chunks WHERE amfi_code=? AND as_of=? ORDER BY chunk_no",
        (fund.amfi_code, day),
    ).fetchall()
    if stored == rows:
        return 0
    conn.execute(
        "DELETE FROM factsheet_chunks WHERE amfi_code=? AND as_of=?", (fund.amfi_code, day)
    )
    conn.executemany(
        "INSERT INTO factsheet_chunks (amfi_code, as_of, chunk_no, amc, page, section, text, "
        "text_hash, source_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def chunks_without_embedding(
    conn: sqlite3.Connection, config: AppConfig, model: str
) -> list[tuple[str, str]]:
    """(text_hash, text to embed) for chunks with no embedding for `model`, one per hash."""
    names = {f.amfi_code: f for f in config.funds}
    found: dict[str, str] = {}
    for code, as_of, section, text, hashed in conn.execute(
        "SELECT c.amfi_code, c.as_of, c.section, c.text, c.text_hash FROM factsheet_chunks c "
        "WHERE NOT EXISTS (SELECT 1 FROM factsheet_embeddings e "
        "WHERE e.text_hash = c.text_hash AND e.model = ?) ORDER BY c.amfi_code, c.as_of, "
        "c.chunk_no",
        (model,),
    ):
        fund = names.get(code)
        if fund is None or hashed in found:
            continue  # a fund no longer in config: not searched, not embedded
        text = embed_text(fund, as_of, Chunk(section, text))
        if text_hash(text) == hashed:  # else the fund was renamed: re-chunked on next fetch
            found[hashed] = text
    return list(found.items())


# ---------- question parsing ----------


_SUFFIXES = ("ings", "ing", "ers", "er", "ed", "es", "e", "s")
MIN_STEM = 4  # letters a word keeps after its ending is cut


def _stem(word: str) -> str:
    """Cut a common English ending, so 'manages', 'managers', 'managing' and 'managed' all
    become 'manag', 'derivatives' and 'derivative' 'derivativ'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ss") or not word.isalpha():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    """Search words: lower case, numbers kept whole ('0.33'), simple plural stemming."""
    words = re.findall(r"f&o|[a-z]+|\d+(?:[.,]\d+)*", text.lower())
    return [_stem(w) for w in words if w not in _STOP]


def fund_aliases(fund: Fund) -> set[str]:
    """Words that identify the fund in a question: its AMC's name words (e.g. 'nippon',
    'aditya', 'birla', 'icici', 'prudential')."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", fund.amc)}
    return {w for w in words - _COMMON_AMC_WORDS if len(w) > 2}


def funds_in(question: str, config: AppConfig) -> list[Fund]:
    """Funds the question names (by AMC). Empty when it names none."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    return [f for f in config.funds if fund_aliases(f) & words]


def month_in(question: str, stored_months: Iterable[str], today: date) -> str | None:
    """The factsheet month (YYYY-MM) a question asks about, e.g. 'in August' -> the latest
    stored August not after today; None when no month is named."""
    for match in _MONTH.finditer(question):
        word = match.group(1).lower()
        if word == "may" and not match.group(2):
            continue  # "may" is more often the verb
        month = _MONTHS[word[:3]]
        if match.group(2):
            year = int(match.group(2)) + (2000 if len(match.group(2)) == 2 else 0)
            return f"{year:04d}-{month:02d}"
        candidates = sorted(
            (m for m in stored_months if int(m[5:7]) == month and m <= today.strftime("%Y-%m")),
            reverse=True,
        )
        year = int(candidates[0][:4]) if candidates else (
            today.year if month <= today.month else today.year - 1
        )  # fmt: skip
        return f"{year:04d}-{month:02d}"
    return None


# ---------- search ----------


@dataclass(frozen=True)
class _Row:
    amfi_code: int
    as_of: str
    chunk_no: int
    page: int
    section: str
    text: str
    text_hash: str
    source_url: str


def _rows(conn: sqlite3.Connection, codes: Sequence[int], month: str | None) -> list[_Row]:
    """Chunks of the funds, for the month asked (YYYY-MM) or each fund's latest month."""
    marks = ", ".join("?" * len(codes))
    columns = "amfi_code, as_of, chunk_no, page, section, text, text_hash, source_url"
    if month is not None:
        query = (
            f"SELECT {columns} FROM factsheet_chunks WHERE amfi_code IN ({marks}) "
            "AND substr(as_of, 1, 7) = ?"
        )
        params: list[object] = [*codes, month]
    else:
        query = (
            f"SELECT {columns} FROM factsheet_chunks c WHERE amfi_code IN ({marks}) AND as_of = "
            "(SELECT MAX(as_of) FROM factsheet_chunks d WHERE d.amfi_code = c.amfi_code)"
        )
        params = list(codes)
    return [_Row(*r) for r in conn.execute(query + " ORDER BY amfi_code, chunk_no", params)]


def name_words(config: AppConfig) -> set[str]:
    """Search words of the fund and AMC names. Dropped from questions: the fund filter already
    uses them, and they would favour chunks that print the fund's name."""
    return {t for f in config.funds for t in tokens(f"{f.name} {f.amc}")}


def keyword_scores(
    question: str,
    docs: Sequence[str],
    ignore: Iterable[str] = (),
    corpus: Sequence[str] | None = None,
) -> list[float]:
    """BM25 score of each doc for the question; synonyms count SYNONYM_WEIGHT. Question words
    in `ignore` are not searched for. Word rarity and average length come from `corpus` (all
    stored chunks; default: the docs), so a score means the same however few docs are
    searched, and one threshold fits every fund and month."""
    skip = set(ignore)
    terms: dict[str, float] = {}
    for word in tokens(question):
        if word in skip:
            continue
        terms[word] = 1.0
        for group in SYNONYMS:
            if word in {_stem(g) for g in group}:
                for other in group:
                    terms.setdefault(_stem(other), SYNONYM_WEIGHT)
    bags = [Counter(tokens(d)) for d in docs]
    if not bags or not terms:
        return [0.0] * len(docs)
    stats = bags if corpus is None else [Counter(tokens(d)) for d in corpus] or bags
    average = sum(sum(b.values()) for b in stats) / len(stats) or 1.0
    having = {t: sum(1 for b in stats if t in b) for t in terms}
    scores = []
    for bag in bags:
        length = sum(bag.values())
        score = 0.0
        for term, weight in terms.items():
            count = bag.get(term, 0)
            if not count:
                continue
            idf = math.log(1 + (len(stats) - having[term] + 0.5) / (having[term] + 0.5))
            norm = count + BM25_K1 * (1 - BM25_B + BM25_B * length / average)
            score += weight * idf * count * (BM25_K1 + 1) / norm
        scores.append(score)
    return scores


# embed_query(text, model) -> vector; tests pass a fake.
QueryEmbedder = Callable[[str, str], Sequence[float]]


def _mistral_query(text: str, model: str) -> Sequence[float]:
    return embeddings.mistral_embed([text], model)[0]


def _semantic_scores(
    conn: sqlite3.Connection, rows: Sequence[_Row], question: str, model: str,
    embed_query: QueryEmbedder,
) -> list[float] | None:  # fmt: skip
    """Cosine similarity of each row to the question; None when a row has no embedding."""
    vectors = dict(
        conn.execute(
            "SELECT text_hash, vector FROM factsheet_embeddings WHERE model = ?", (model,)
        ).fetchall()
    )
    if any(r.text_hash not in vectors for r in rows):
        return None
    query = np.asarray(embed_query(question, model), dtype="float32")
    scores = []
    for row in rows:
        vector = embeddings.decode(vectors[row.text_hash])
        denominator = float(np.linalg.norm(query) * np.linalg.norm(vector)) or 1.0
        scores.append(float(np.dot(query, vector)) / denominator)
    return scores


def _ranked(scores: Sequence[float], minimum: float) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [i for i in order if scores[i] >= minimum]


def search(
    conn: sqlite3.Connection,
    config: AppConfig,
    question: str,
    *,
    funds: Sequence[Fund] | None = None,
    month: str | None = None,
    mode: str | None = None,
    today: date | None = None,
    max_results: int = MAX_RESULTS,
    max_tokens: int = MAX_RESULT_TOKENS,
    embed_query: QueryEmbedder | None = None,
) -> SearchResult:
    """Chunks that answer the question, best first. Funds and month default to those named in
    the question (all funds, latest month when none)."""
    today = today or date.today()
    chosen = list(funds) if funds else (funds_in(question, config) or list(config.funds))
    if month is None:
        stored = [
            r[0] for r in conn.execute("SELECT DISTINCT substr(as_of, 1, 7) FROM factsheet_chunks")
        ]
        month = month_in(question, stored, today)
    mode = mode or config.settings.search_mode
    result = SearchResult(mode=mode, funds=[f.name for f in chosen], month=month)
    rows = _rows(conn, [f.amfi_code for f in chosen], month)
    if not rows:
        return result
    docs = [f"{r.section}\n{r.text}" for r in rows]

    corpus = [
        f"{section}\n{text}"
        for section, text in conn.execute(
            f"SELECT section, text FROM factsheet_chunks WHERE amfi_code IN "
            f"({', '.join('?' * len(config.funds))})",
            [f.amfi_code for f in config.funds],
        )
    ]
    scores = keyword_scores(question, docs, ignore=name_words(config), corpus=corpus)
    keyword = _ranked(scores, MIN_KEYWORD_SCORE)
    order = keyword
    if mode in ("semantic", "hybrid"):
        try:
            similarity = _semantic_scores(
                conn, rows, question, config.settings.embedding_model,
                embed_query or _mistral_query,
            )  # fmt: skip
        except SourceError as exc:
            similarity = None
            result.note = f"search by meaning unavailable ({exc}); used keyword search"
        if similarity is None:
            result.note = result.note or "some chunks have no embedding yet; used keyword search"
            result.mode = "keyword"
        else:
            semantic = _ranked(similarity, MIN_SIMILARITY)
            if mode == "semantic":
                order, scores = semantic, similarity
            else:
                fused: dict[int, float] = {}
                for ranking in (keyword, semantic):
                    for rank, i in enumerate(ranking):
                        fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
                order = sorted(fused, key=lambda i: -fused[i])
                scores = [fused.get(i, 0.0) for i in range(len(rows))]

    names = {f.amfi_code: f.name for f in config.funds}
    used = 0
    for i in order:
        if len(result.hits) >= max_results:
            break
        row = rows[i]
        size = estimate_tokens(row.text)
        if used + size > max_tokens:
            continue
        used += size
        result.hits.append(
            Hit(row.amfi_code, names.get(row.amfi_code, str(row.amfi_code)), row.as_of,
                row.chunk_no, row.page, row.section, row.text, row.source_url, scores[i])
        )  # fmt: skip
    return result


def extracts_message(result: SearchResult) -> str:
    """The found chunks, as text added to the question for the model."""
    parts = ["[Factsheet extracts found for this question. Use them only if relevant; quote "
             "the words you rely on and link the page.]"]  # fmt: skip
    for n, hit in enumerate(result.hits, start=1):
        month = date.fromisoformat(hit.as_of).strftime("%d-%b-%Y")
        parts.append(
            f"<extract {n}> {hit.fund}, factsheet data as of {month}, page {hit.page}, "
            f"link {hit.link}\n{hit.text}\n</extract {n}>"
        )
    return "\n\n".join(parts)
