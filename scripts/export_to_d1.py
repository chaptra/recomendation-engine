#!/usr/bin/env python3
"""
scripts/export_to_d1.py
=======================
One-time migration script: reads books from PostgreSQL, computes TF-IDF
cosine-similarity scores using the existing recommendation logic, and writes
two SQL files that can be imported into Cloudflare D1:

  d1_books.sql       – INSERT statements for the books table
  d1_similarities.sql – INSERT statements for the similarities table

Usage
-----
1. Copy .env.example to .env and fill in RDS_DB_URL.
2. Install requirements:
       pip install -r requirements.txt
3. Run:
       python scripts/export_to_d1.py [--top-n 50] [--threshold 0.1]
4. Apply schema to D1 (one-time):
       npx wrangler d1 execute recommendation-engine --file=schema.sql
5. Import books (may take a while for large catalogues):
       npx wrangler d1 execute recommendation-engine --file=d1_books.sql
6. Import pre-computed similarities:
       npx wrangler d1 execute recommendation-engine --file=d1_similarities.sql

Notes
-----
- The --top-n flag controls how many similar books are stored per book (default 50).
  Higher values give better recall but increase D1 storage and import time.
- Rows are inserted in batches to stay within D1's per-statement limits.
- Re-running the script is safe: it uses INSERT OR REPLACE so existing rows
  are updated rather than duplicated.
"""

import argparse
import hashlib
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
import psycopg2
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


# ─── Database helpers ────────────────────────────────────────────────────────

def connect_to_db() -> Any:
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

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(df["clean_text"])
    logger.info("TF-IDF matrix shape: %s", matrix.shape)

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
    Returns a mapping from PostgreSQL book_id → D1 row position (1-based)
    so similarity rows can reference the right id.

    Because D1 uses AUTOINCREMENT we include an explicit id column that
    mirrors the PostgreSQL id, ensuring cross-table references are stable.
    """
    logger.info("Writing %s …", out_path)
    id_map: dict[int, int] = {}

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("-- Auto-generated by scripts/export_to_d1.py\n")
        fh.write("-- Import with: npx wrangler d1 execute recommendation-engine --file=d1_books.sql\n\n")

        rows_buf: list[str] = []

        def flush():
            if not rows_buf:
                return
            fh.write(
                "INSERT OR REPLACE INTO books (id, title, slug, author, categories, description, cover_path, rating, rating_count, year, isbn) VALUES\n"
            )
            fh.write(",\n".join(rows_buf))
            fh.write(";\n\n")
            rows_buf.clear()

        for _, row in df.iterrows():
            book_id = int(row["id"])
            id_map[book_id] = book_id  # keep PostgreSQL id as D1 id

            title = escape(str(row.get("title") or ""))
            slug = escape(str(row.get("slug") or ""))
            author = escape(str(row.get("authors") or "Unknown"))
            categories = escape(str(row.get("categories") or "General"))
            desc = escape((str(row.get("description") or ""))[:500])
            cover = escape(str(row.get("cover_local_path") or ""))
            rating = float(row.get("average_rating") or 0.0)
            rating_count = int(row.get("rating_count") or 0)
            year_val = row.get("year")
            year = int(year_val) if pd.notna(year_val) and year_val else "NULL"
            isbn = escape(str(row.get("isbn") or ""))

            rows_buf.append(
                f"  ({book_id}, '{title}', '{slug}', '{author}', '{categories}', '{desc}', '{cover}', {rating}, {rating_count}, {year}, '{isbn}')"
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
    parser = argparse.ArgumentParser(description="Export PostgreSQL books to D1 SQL files.")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of similar books to store per book (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Minimum cosine similarity score to store (default: 0.1)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="Directory for local TF-IDF cache (default: ./cache)")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Output directory for SQL files (default: current directory)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    conn = connect_to_db()
    cache_key = get_db_hash(conn)
    df = load_books(conn)
    conn.close()

    if df.empty:
        logger.error("No books found in the database. Check your query and connection.")
        sys.exit(1)

    tfidf_matrix, _ = build_tfidf(df, cache_dir, cache_key)

    books_sql = out_dir / "d1_books.sql"
    sims_sql = out_dir / "d1_similarities.sql"

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
