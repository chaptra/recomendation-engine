#!/usr/bin/env python3
"""
scripts/export_to_d1.py
=======================
Local training script: reads books from **either a local CSV/JSON file or
PostgreSQL**, computes TF-IDF cosine-similarity scores, and writes two SQL
files ready for import into Cloudflare D1:

  d1_books.sql        – INSERT statements for the books table
  d1_similarities.sql – INSERT statements for the similarities table

The Cloudflare Worker then serves recommendations by querying these
pre-computed similarity scores, so the heavyweight ML work only ever runs
on your local machine.

Quick-start (no database needed)
---------------------------------
1.  Install requirements:
        pip install -r requirements.txt

2.  Train from the included sample CSV:
        python scripts/export_to_d1.py --csv data/sample_books.csv

    Or use your own CSV (columns: id, title, authors, categories, description,
    and optionally slug, rating, rating_count, cover_path, year, isbn):
        python scripts/export_to_d1.py --csv /path/to/books.csv

    Or use a JSON file (list of book objects with the same fields):
        python scripts/export_to_d1.py --json /path/to/books.json

3.  Apply schema to D1 (one-time):
        npx wrangler d1 execute recommendation-engine --file=schema.sql

4.  Import books:
        npx wrangler d1 execute recommendation-engine --file=d1_books.sql

5.  Import pre-computed similarities:
        npx wrangler d1 execute recommendation-engine --file=d1_similarities.sql

6.  Deploy the Worker:
        npx wrangler deploy

PostgreSQL source (existing database)
--------------------------------------
1.  Copy .env.example to .env and fill in RDS_DB_URL.
2.  Run without --csv/--json:
        python scripts/export_to_d1.py [--top-n 50] [--threshold 0.1]

Notes
-----
- --top-n   controls how many similar books are stored per book (default 50).
  Higher values give better recall but increase D1 storage and import time.
- Rows are inserted in batches to stay within D1's per-statement limits.
- Re-running the script is safe: INSERT OR REPLACE keeps data up to date.
- TF-IDF min_df is scaled automatically for small datasets (< 500 books).
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── SQL batch size ─────────────────────────────────────────────────────────
# D1 has a 1 MB / statement limit; keep batches small.
BOOKS_BATCH = 200
SIM_BATCH = 500


# ─── Database helpers (PostgreSQL source) ────────────────────────────────────

def _require_psycopg2():
    try:
        import psycopg2
        return psycopg2
    except ImportError:
        logger.error("psycopg2 is required for PostgreSQL support. Install it with: pip install psycopg2-binary")
        sys.exit(1)


def connect_to_db() -> Any:
    psycopg2 = _require_psycopg2()
    db_url = os.getenv("RDS_DB_URL")
    if not db_url:
        logger.error("RDS_DB_URL environment variable is not set.")
        sys.exit(1)
    parsed = urlparse(db_url)
    return psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port,
        database=parsed.path[1:],
        user=parsed.username,
        password=unquote(parsed.password or ""),
    )


# ─── Local file sources (CSV / JSON) ────────────────────────────────────────

# Canonical column names used internally; input files may use any of the
# aliases listed in COLUMN_ALIASES.
_REQUIRED_COLS = {"id", "title", "authors", "categories", "description"}
_OPTIONAL_COLS = {"slug", "rating", "rating_count", "cover_path", "year", "isbn"}

COLUMN_ALIASES: dict[str, list[str]] = {
    "id":           ["id", "book_id"],
    "title":        ["title", "book_title", "name"],
    "authors":      ["authors", "author", "author_name", "writer"],
    "categories":   ["categories", "category", "genre", "genres", "tags"],
    "description":  ["description", "desc", "summary", "synopsis", "blurb"],
    "slug":         ["slug", "url_slug", "book_slug"],
    "rating":       ["rating", "average_rating", "avg_rating", "score"],
    "rating_count": ["rating_count", "ratings_count", "num_ratings", "ratings"],
    "cover_path":   ["cover_path", "cover", "cover_image", "cover_url",
                     "cover_local_path", "image_url"],
    "year":         ["year", "published_year", "publication_year", "pub_year"],
    "isbn":         ["isbn", "isbn13", "isbn10"],
}

# Reverse map: alias → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias: canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from any supported alias to the canonical name."""
    rename = {}
    for col in df.columns:
        canonical = _ALIAS_TO_CANONICAL.get(col.lower().strip())
        if canonical and canonical not in rename.values():
            rename[col] = canonical
    df = df.rename(columns=rename)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}.\n"
            f"Required: {sorted(_REQUIRED_COLS)}\n"
            f"Found:    {sorted(df.columns.tolist())}"
        )
    return df


def load_books_from_csv(path: str) -> pd.DataFrame:
    """Load books from a CSV file."""
    logger.info("Loading books from CSV: %s", path)
    try:
        df = pd.read_csv(path, dtype=str)
        df = _normalise_columns(df)
    except ValueError as exc:
        logger.error("Invalid CSV structure: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to read CSV file: %s", exc)
        sys.exit(1)
    logger.info("Loaded %d books from CSV.", len(df))
    return df


def load_books_from_json(path: str) -> pd.DataFrame:
    """Load books from a JSON file (list of objects or records-orient dict)."""
    logger.info("Loading books from JSON: %s", path)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            # pandas records / split / index orient
            df = pd.DataFrame.from_dict(raw)
        elif isinstance(raw, list):
            df = pd.DataFrame(raw)
        else:
            raise ValueError("JSON root must be a list or dict.")
        df = df.astype(str)
        df = _normalise_columns(df)
    except ValueError as exc:
        logger.error("Invalid JSON structure: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to read JSON file: %s", exc)
        sys.exit(1)
    logger.info("Loaded %d books from JSON.", len(df))
    return df


def load_books(conn: Any) -> pd.DataFrame:
    query = """
        SELECT
            b.id,
            b.title,
            b.slug,
            b.cover_local_path,
            b.description,
            b.average_rating,
            b.rating_count,
            b.isbn,
            EXTRACT(YEAR FROM b.created_at)::INTEGER AS year,
            string_agg(DISTINCT a.name, '|')          AS authors,
            string_agg(DISTINCT c.name, '|')          AS categories
        FROM books_with_details b
        LEFT JOIN book_authors    ba ON b.id = ba.canonical_book_id
        LEFT JOIN authors          a ON ba.author_id = a.id
                                     AND a.deleted_at IS NULL
        LEFT JOIN book_categories  bc ON b.id = bc.canonical_book_id
        LEFT JOIN categories        c ON bc.category_id = c.id
                                     AND c.deleted_at IS NULL
        WHERE b.deleted_at   IS NULL
          AND b.description  IS NOT NULL
          AND b.description  != ''
          AND b.title        IS NOT NULL
          AND b.language     = 'en'
        GROUP BY b.id, b.title, b.slug, b.cover_local_path, b.description,
                 b.average_rating, b.rating_count, b.isbn, b.created_at
        ORDER BY b.id
    """
    logger.info("Fetching books from PostgreSQL …")
    df = pd.read_sql_query(query, conn)
    logger.info("Fetched %d books.", len(df))
    return df


# ─── TF-IDF helpers ──────────────────────────────────────────────────────────

def get_db_hash(conn: Any) -> str:
    result = pd.read_sql_query(
        "SELECT COUNT(*) AS n, MAX(updated_at) AS ts FROM books_with_details WHERE deleted_at IS NULL",
        conn,
    )
    raw = f"{result.iloc[0, 0]}_{result.iloc[0, 1]}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def build_tfidf(df: pd.DataFrame, cache_dir: Path, cache_key: str):
    tfidf_path = cache_dir / f"tfidf_{cache_key}.npz"
    vec_path = cache_dir / f"vectorizer_{cache_key}.pkl"

    if tfidf_path.exists() and vec_path.exists():
        logger.info("Loading TF-IDF matrix from local cache …")
        matrix = load_npz(str(tfidf_path))
        with open(vec_path, "rb") as fh:
            vectorizer = pickle.load(fh)
        return matrix, vectorizer

    logger.info("Building TF-IDF matrix …")
    df["clean_text"] = (
        df["title"].fillna("") + " "
        + df["description"].fillna("").str[:500] + " "
        + df["authors"].fillna("") + " "
        + df["categories"].fillna("")
    ).str.lower().str.replace(r"[^a-zA-Z\s]", " ", regex=True)

    # Scale min_df so it never exceeds the number of documents (important for
    # small local datasets where min_df=3 would drop almost every term).
    n_docs = len(df)
    min_df = min(3, max(1, n_docs // 10))

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.7,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df["clean_text"])
    logger.info("TF-IDF matrix shape: %s (min_df=%d)", matrix.shape, min_df)

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(str(tfidf_path), matrix)
    with open(vec_path, "wb") as fh:
        pickle.dump(vectorizer, fh)
    logger.info("Cached TF-IDF matrix to %s", cache_dir)
    return matrix, vectorizer


# ─── SQL generation ───────────────────────────────────────────────────────────

def escape(value: str) -> str:
    """Escape a string for SQLite single-quote literals."""
    return value.replace("'", "''")


def write_books_sql(df: pd.DataFrame, out_path: Path) -> dict[int, int]:
    """
    Write INSERT OR REPLACE statements for all books.
    Returns a mapping from source book_id → D1 id so similarity rows can
    reference the right id.

    We include an explicit id column mirroring the source id to keep
    cross-table references stable across re-imports.
    """
    logger.info("Writing %s …", out_path)
    id_map: dict[int, int] = {}

    # Helper: fetch a column by canonical name, fall back to empty string.
    def col(row: pd.Series, name: str, default="") -> str:
        return str(row.get(name) or default)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("-- Auto-generated by scripts/export_to_d1.py\n")
        fh.write("-- Import with: npx wrangler d1 execute recommendation-engine --file=d1_books.sql\n\n")

        rows_buf: list[str] = []

        def flush():
            if not rows_buf:
                return
            fh.write(
                "INSERT OR REPLACE INTO books "
                "(id, title, slug, author, categories, description, cover_path, rating, rating_count, year, isbn) VALUES\n"
            )
            fh.write(",\n".join(rows_buf))
            fh.write(";\n\n")
            rows_buf.clear()

        for _, row in df.iterrows():
            book_id = int(float(col(row, "id", "0") or "0"))
            id_map[book_id] = book_id

            title       = escape(col(row, "title"))
            slug        = escape(col(row, "slug", re.sub(r"[^a-z0-9]+", "-", col(row, "title").lower()).strip("-")))
            author      = escape(col(row, "authors", "Unknown"))
            categories  = escape(col(row, "categories", "General"))
            desc        = escape(col(row, "description")[:500])
            cover       = escape(col(row, "cover_path"))
            # rating / rating_count: may be NaN or "nan" from CSV read
            raw_rating  = col(row, "rating", "0")
            rating      = float(raw_rating) if raw_rating not in ("", "nan", "None") else 0.0
            raw_count   = col(row, "rating_count", "0")
            rating_count = int(float(raw_count)) if raw_count not in ("", "nan", "None") else 0
            raw_year    = col(row, "year", "")
            year: int | None = int(float(raw_year)) if raw_year not in ("", "nan", "None") else None
            isbn        = escape(col(row, "isbn"))

            rows_buf.append(
                f"  ({book_id}, '{title}', '{slug}', '{author}', '{categories}', "
                f"'{desc}', '{cover}', {rating}, {rating_count}, "
                f"{'NULL' if year is None else year}, '{isbn}')"
            )

            if len(rows_buf) >= BOOKS_BATCH:
                flush()

        flush()

    logger.info("Wrote %d book rows to %s", len(id_map), out_path)
    return id_map


def write_similarities_sql(
    df: pd.DataFrame,
    tfidf_matrix,
    id_map: dict[int, int],
    top_n: int,
    threshold: float,
    out_path: Path,
) -> None:
    logger.info("Computing and writing similarities (top_n=%d, threshold=%.2f) …", top_n, threshold)
    n_books = len(df)
    total_pairs = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("-- Auto-generated by scripts/export_to_d1.py\n")
        fh.write("-- Import with: npx wrangler d1 execute recommendation-engine --file=d1_similarities.sql\n\n")

        rows_buf: list[str] = []

        def flush():
            nonlocal total_pairs
            if not rows_buf:
                return
            fh.write(
                "INSERT OR REPLACE INTO similarities (book_id, similar_book_id, score) VALUES\n"
            )
            fh.write(",\n".join(rows_buf))
            fh.write(";\n\n")
            total_pairs += len(rows_buf)
            rows_buf.clear()

        for i in range(n_books):
            src_id = int(df.iloc[i]["id"])
            d1_src_id = id_map.get(src_id)
            if d1_src_id is None:
                continue

            # Compute cosine similarity for this single book vs all others
            src_vec = tfidf_matrix[i]
            sims = cosine_similarity(src_vec, tfidf_matrix).flatten()

            # Collect (index, score) pairs above threshold, excluding self
            pairs = [
                (j, float(sims[j]))
                for j in range(n_books)
                if j != i and sims[j] >= threshold
            ]
            pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = pairs[:top_n]

            for j, score in pairs:
                tgt_id = int(df.iloc[j]["id"])
                d1_tgt_id = id_map.get(tgt_id)
                if d1_tgt_id is None:
                    continue
                rows_buf.append(f"  ({d1_src_id}, {d1_tgt_id}, {score:.6f})")
                if len(rows_buf) >= SIM_BATCH:
                    flush()

            if (i + 1) % 500 == 0:
                logger.info("  Processed %d / %d books …", i + 1, n_books)

        flush()

    logger.info("Wrote %d similarity pairs to %s", total_pairs, out_path)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train locally, export to Cloudflare D1 SQL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples
            --------
            # Train from the bundled sample CSV (no database needed):
              python scripts/export_to_d1.py --csv data/sample_books.csv

            # Train from your own CSV:
              python scripts/export_to_d1.py --csv /path/to/books.csv

            # Train from a JSON file:
              python scripts/export_to_d1.py --json /path/to/books.json

            # Train from PostgreSQL (set RDS_DB_URL in .env first):
              python scripts/export_to_d1.py

            After running, import into D1 and deploy:
              npx wrangler d1 execute recommendation-engine --file=schema.sql
              npx wrangler d1 execute recommendation-engine --file=d1_books.sql
              npx wrangler d1 execute recommendation-engine --file=d1_similarities.sql
              npx wrangler deploy
        """),
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv",  metavar="FILE",
                        help="Path to a CSV file containing book data (no database required).")
    source.add_argument("--json", metavar="FILE",
                        help="Path to a JSON file containing book data (no database required).")

    parser.add_argument("--top-n", type=int, default=50,
                        help="Similar books to store per book (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum cosine similarity score to store (default: 0.1)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="Directory for local TF-IDF cache (default: ./cache)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output directory for SQL files (default: current directory)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Choose data source ────────────────────────────────────────────────────
    if args.csv:
        df = load_books_from_csv(args.csv)
        # Stable cache key from file content hash
        with open(args.csv, "rb") as fh:
            cache_key = hashlib.md5(fh.read()).hexdigest()[:8]
    elif args.json:
        df = load_books_from_json(args.json)
        with open(args.json, "rb") as fh:
            cache_key = hashlib.md5(fh.read()).hexdigest()[:8]
    else:
        conn = connect_to_db()
        cache_key = get_db_hash(conn)
        df = load_books(conn)
        conn.close()

    if df.empty:
        logger.error("No books found in the data source. Check your file or database connection.")
        sys.exit(1)

    tfidf_matrix, _ = build_tfidf(df, cache_dir, cache_key)

    books_sql = out_dir / "d1_books.sql"
    sims_sql  = out_dir / "d1_similarities.sql"

    id_map = write_books_sql(df, books_sql)
    write_similarities_sql(df, tfidf_matrix, id_map, args.top_n, args.threshold, sims_sql)

    elapsed = time.time() - t0
    logger.info("Done in %.1f s.", elapsed)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Apply the schema (once):  npx wrangler d1 execute recommendation-engine --file=schema.sql")
    logger.info("  2. Import books:             npx wrangler d1 execute recommendation-engine --file=%s", books_sql)
    logger.info("  3. Import similarities:      npx wrangler d1 execute recommendation-engine --file=%s", sims_sql)
    logger.info("  4. Deploy the worker:        npx wrangler deploy")


if __name__ == "__main__":
    main()
