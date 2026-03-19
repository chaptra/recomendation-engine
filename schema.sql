-- D1 (SQLite) schema for the Cloudflare Workers recommendation engine.
-- Run via: npx wrangler d1 execute recommendation-engine --file=schema.sql

-- Core books table
CREATE TABLE IF NOT EXISTS books (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    slug        TEXT    NOT NULL DEFAULT '',
    author      TEXT    NOT NULL DEFAULT '',
    categories  TEXT    NOT NULL DEFAULT '',
    description TEXT    NOT NULL DEFAULT '',
    cover_path  TEXT    NOT NULL DEFAULT '',
    rating      REAL    NOT NULL DEFAULT 0.0,
    rating_count INTEGER NOT NULL DEFAULT 0,
    year        INTEGER,
    isbn        TEXT    NOT NULL DEFAULT ''
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_books_title_author ON books (title, author);
CREATE INDEX IF NOT EXISTS idx_books_title             ON books (title);

-- Full-text search index over title, author, and categories
CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
    title,
    author,
    categories,
    content = books,
    content_rowid = id
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS books_ai AFTER INSERT ON books BEGIN
    INSERT INTO books_fts(rowid, title, author, categories)
    VALUES (new.id, new.title, new.author, new.categories);
END;

CREATE TRIGGER IF NOT EXISTS books_ad AFTER DELETE ON books BEGIN
    INSERT INTO books_fts(books_fts, rowid, title, author, categories)
    VALUES ('delete', old.id, old.title, old.author, old.categories);
END;

CREATE TRIGGER IF NOT EXISTS books_au AFTER UPDATE ON books BEGIN
    INSERT INTO books_fts(books_fts, rowid, title, author, categories)
    VALUES ('delete', old.id, old.title, old.author, old.categories);
    INSERT INTO books_fts(rowid, title, author, categories)
    VALUES (new.id, new.title, new.author, new.categories);
END;

-- Pre-computed pairwise cosine-similarity scores.
-- Populated offline by scripts/export_to_d1.py; Workers only read from this table.
CREATE TABLE IF NOT EXISTS similarities (
    book_id        INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    similar_book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    score          REAL    NOT NULL,
    PRIMARY KEY (book_id, similar_book_id)
);

CREATE INDEX IF NOT EXISTS idx_similarities_lookup
    ON similarities (book_id, score DESC);
