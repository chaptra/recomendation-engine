"""Tests for the CSV/JSON local-training path in scripts/export_to_d1.py."""

import csv
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

# Make the scripts package importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from export_to_d1 import (  # noqa: E402
    load_books_from_csv,
    load_books_from_json,
    _normalise_columns,
    write_books_sql,
    write_similarities_sql,
    build_tfidf,
    escape,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_ROWS = [
    {
        "id": "1",
        "title": "The Great Gatsby",
        "authors": "F. Scott Fitzgerald",
        "categories": "Fiction|Classic",
        "description": "A story of wealth, love, and the American Dream set in the Jazz Age.",
        "slug": "great-gatsby",
        "rating": "4.2",
        "rating_count": "4823456",
        "cover_path": "covers/gatsby.jpg",
        "year": "1925",
        "isbn": "9780743273565",
    },
    {
        "id": "2",
        "title": "1984",
        "authors": "George Orwell",
        "categories": "Science Fiction|Dystopian",
        "description": "A chilling dystopian novel about a totalitarian society.",
        "slug": "1984",
        "rating": "4.6",
        "rating_count": "4987234",
        "cover_path": "covers/1984.jpg",
        "year": "1949",
        "isbn": "9780451524935",
    },
    {
        "id": "3",
        "title": "Brave New World",
        "authors": "Aldous Huxley",
        "categories": "Science Fiction|Dystopian",
        "description": "A dystopian novel set in a futuristic World State controlled by pleasure.",
        "slug": "brave-new-world",
        "rating": "4.1",
        "rating_count": "2345678",
        "cover_path": "covers/brave.jpg",
        "year": "1932",
        "isbn": "9780060850524",
    },
]


def _make_csv(rows: list[dict], path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def _make_json(rows: list[dict], path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return path


# ─── escape() ────────────────────────────────────────────────────────────────

class TestEscape:
    def test_no_special_chars(self):
        assert escape("Hello World") == "Hello World"

    def test_single_quote_doubled(self):
        assert escape("It's a test") == "It''s a test"

    def test_multiple_quotes(self):
        assert escape("I'd say it's fine") == "I''d say it''s fine"

    def test_empty_string(self):
        assert escape("") == ""


# ─── Column normalisation ─────────────────────────────────────────────────────

class TestNormaliseColumns:
    def test_canonical_columns_unchanged(self):
        df = pd.DataFrame([{"id": "1", "title": "T", "authors": "A", "categories": "C", "description": "D"}])
        out = _normalise_columns(df)
        assert set(["id", "title", "authors", "categories", "description"]).issubset(out.columns)

    def test_author_alias_mapped(self):
        df = pd.DataFrame([{"id": "1", "title": "T", "author": "A", "categories": "C", "description": "D"}])
        out = _normalise_columns(df)
        assert "authors" in out.columns
        assert "author" not in out.columns

    def test_average_rating_alias(self):
        df = pd.DataFrame([{
            "id": "1", "title": "T", "author": "A", "genre": "G",
            "description": "D", "average_rating": "4.5",
        }])
        out = _normalise_columns(df)
        assert "rating" in out.columns
        assert "categories" in out.columns   # genre → categories

    def test_missing_required_column_raises(self):
        df = pd.DataFrame([{"id": "1", "title": "T", "authors": "A"}])  # no categories/description
        with pytest.raises(ValueError, match="missing required columns"):
            _normalise_columns(df)


# ─── CSV loader ───────────────────────────────────────────────────────────────

class TestLoadBooksFromCsv:
    def test_loads_sample_rows(self, tmp_path):
        path = _make_csv(SAMPLE_ROWS, tmp_path / "books.csv")
        df = load_books_from_csv(str(path))
        assert len(df) == 3
        assert "title" in df.columns
        assert "authors" in df.columns

    def test_accepts_alias_columns(self, tmp_path):
        rows = [
            {"id": "1", "title": "Book A", "author": "Auth A", "genre": "G", "summary": "Desc A"},
        ]
        path = tmp_path / "alias.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        df = load_books_from_csv(str(path))
        assert "authors" in df.columns
        assert "categories" in df.columns
        assert "description" in df.columns

    def test_missing_required_column_exits(self, tmp_path):
        rows = [{"id": "1", "title": "Book A"}]
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(SystemExit):
            load_books_from_csv(str(path))

    def test_sample_csv_file_ships_with_project(self):
        sample = Path(__file__).parent.parent / "data" / "sample_books.csv"
        assert sample.exists(), "data/sample_books.csv must exist"
        df = load_books_from_csv(str(sample))
        assert len(df) >= 10
        assert "description" in df.columns


# ─── JSON loader ──────────────────────────────────────────────────────────────

class TestLoadBooksFromJson:
    def test_loads_list_of_objects(self, tmp_path):
        path = _make_json(SAMPLE_ROWS, tmp_path / "books.json")
        df = load_books_from_json(str(path))
        assert len(df) == 3
        assert "description" in df.columns

    def test_missing_required_column_exits(self, tmp_path):
        rows = [{"id": "1", "title": "Book A"}]
        path = tmp_path / "bad.json"
        with open(path, "w") as fh:
            json.dump(rows, fh)
        with pytest.raises(SystemExit):
            load_books_from_json(str(path))

    def test_invalid_json_exits(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json")
        with pytest.raises(SystemExit):
            load_books_from_json(str(path))


# ─── TF-IDF builder ───────────────────────────────────────────────────────────

class TestBuildTfidf:
    def _df(self):
        return pd.DataFrame(SAMPLE_ROWS)

    def test_matrix_rows_match_book_count(self, tmp_path):
        df = self._df()
        matrix, vec = build_tfidf(df, tmp_path / "cache", "test")
        assert matrix.shape[0] == len(df)

    def test_small_dataset_min_df_scaled(self, tmp_path):
        """min_df should be ≤ 1 for a 3-book dataset so terms are not dropped."""
        df = self._df()
        matrix, vec = build_tfidf(df, tmp_path / "cache", "small")
        # With 3 docs, min_df becomes 1; the matrix must have features
        assert matrix.shape[1] > 0

    def test_cache_is_reused(self, tmp_path):
        df = self._df()
        cache_dir = tmp_path / "cache"
        matrix1, _ = build_tfidf(df, cache_dir, "reuse")
        # Second call must hit cache (no error, same shape)
        matrix2, _ = build_tfidf(df, cache_dir, "reuse")
        assert matrix1.shape == matrix2.shape


# ─── SQL writers ─────────────────────────────────────────────────────────────

class TestWriteBooksSql:
    def _df(self):
        return pd.DataFrame(SAMPLE_ROWS)

    def test_creates_file(self, tmp_path):
        df = self._df()
        out = tmp_path / "books.sql"
        write_books_sql(df, out)
        assert out.exists()

    def test_contains_insert_statements(self, tmp_path):
        df = self._df()
        out = tmp_path / "books.sql"
        write_books_sql(df, out)
        content = out.read_text()
        assert "INSERT OR REPLACE INTO books" in content
        assert "The Great Gatsby" in content

    def test_single_quotes_escaped(self, tmp_path):
        rows = SAMPLE_ROWS + [{
            "id": "99",
            "title": "Harry Potter and the Sorcerer's Stone",
            "authors": "J.K. Rowling",
            "categories": "Fantasy",
            "description": "A boy who didn't know he was a wizard.",
            "slug": "hp1",
            "rating": "4.5",
            "rating_count": "1000",
            "cover_path": "",
            "year": "1997",
            "isbn": "",
        }]
        df = pd.DataFrame(rows)
        out = tmp_path / "books.sql"
        write_books_sql(df, out)
        content = out.read_text()
        # Apostrophe in title must be doubled for SQLite
        assert "Sorcerer''s Stone" in content

    def test_returns_id_map(self, tmp_path):
        df = self._df()
        id_map = write_books_sql(df, tmp_path / "books.sql")
        assert 1 in id_map
        assert 2 in id_map
        assert id_map[1] == 1

    def test_missing_optional_columns_ok(self, tmp_path):
        """Books with only required columns should export without errors."""
        df = pd.DataFrame([{
            "id": "1",
            "title": "Minimal Book",
            "authors": "Author",
            "categories": "Fiction",
            "description": "A minimal test book.",
        }])
        out = tmp_path / "books.sql"
        write_books_sql(df, out)
        assert "Minimal Book" in out.read_text()


class TestWriteSimilaritiesSql:
    def test_creates_file_with_pairs(self, tmp_path):
        df = pd.DataFrame(SAMPLE_ROWS)
        matrix, _ = build_tfidf(df, tmp_path / "cache", "sim_test")
        id_map = {1: 1, 2: 2, 3: 3}
        out = tmp_path / "sims.sql"
        write_similarities_sql(df, matrix, id_map, top_n=2, threshold=0.0, out_path=out)
        content = out.read_text()
        assert "INSERT OR REPLACE INTO similarities" in content
        # VALUES rows contain numeric similarity scores (e.g.  0.123456)
        assert "0." in content

    def test_threshold_filters_low_scores(self, tmp_path):
        df = pd.DataFrame(SAMPLE_ROWS)
        matrix, _ = build_tfidf(df, tmp_path / "cache", "sim_thresh")
        id_map = {1: 1, 2: 2, 3: 3}

        out_low = tmp_path / "sims_low.sql"
        write_similarities_sql(df, matrix, id_map, top_n=10, threshold=0.0, out_path=out_low)

        out_high = tmp_path / "sims_high.sql"
        write_similarities_sql(df, matrix, id_map, top_n=10, threshold=0.99, out_path=out_high)

        low_lines = out_low.read_text().count("(")
        high_lines = out_high.read_text().count("(")
        assert low_lines >= high_lines
