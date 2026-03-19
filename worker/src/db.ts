// D1 database query helpers.

import { BookRow, SimilarityRow } from './types.js';

// ─── Books ─────────────────────────────────────────────────────────────────

/** Return total number of books in the catalogue. */
export async function countBooks(db: D1Database): Promise<number> {
  const result = await db
    .prepare('SELECT COUNT(*) AS n FROM books')
    .first<{ n: number }>();
  return result?.n ?? 0;
}

/**
 * Find a book by title.
 * Tries exact match first (case-insensitive); falls back to LIKE prefix match.
 */
export async function findBookByTitle(
  db: D1Database,
  title: string,
): Promise<BookRow | null> {
  // Exact match
  const exact = await db
    .prepare('SELECT * FROM books WHERE title = ? COLLATE NOCASE LIMIT 1')
    .bind(title)
    .first<BookRow>();
  if (exact) return exact;

  // Prefix / contains match
  const pattern = `%${title.replace(/[%_]/g, '\\$&')}%`;
  return db
    .prepare(
      "SELECT * FROM books WHERE title LIKE ? ESCAPE '\\' LIMIT 1",
    )
    .bind(pattern)
    .first<BookRow>();
}

/** Fetch a book by its primary key. */
export async function getBookById(
  db: D1Database,
  id: number,
): Promise<BookRow | null> {
  return db.prepare('SELECT * FROM books WHERE id = ?').bind(id).first<BookRow>();
}

/**
 * Full-text search over title, author, and categories.
 * Uses the FTS5 virtual table for fast partial-word matching.
 */
export async function searchBooks(
  db: D1Database,
  query: string,
  limit: number,
): Promise<BookRow[]> {
  // Escape FTS special characters and append a wildcard so partial words match
  const ftsQuery = query
    .replace(/["]/g, '""')
    .split(/\s+/)
    .filter(Boolean)
    .map((t) => `"${t}"*`)
    .join(' ');

  const { results } = await db
    .prepare(
      `SELECT books.*
       FROM books_fts
       JOIN books ON books.id = books_fts.rowid
       WHERE books_fts MATCH ?
       ORDER BY rank
       LIMIT ?`,
    )
    .bind(ftsQuery, limit)
    .all<BookRow>();

  return results ?? [];
}

// ─── Similarities ──────────────────────────────────────────────────────────

/**
 * Retrieve the top-N pre-computed similar books for a given source book.
 * Results are already sorted by score descending.
 */
export async function getSimilarBooks(
  db: D1Database,
  bookId: number,
  limit: number,
  threshold: number,
): Promise<SimilarityRow[]> {
  const { results } = await db
    .prepare(
      `SELECT similar_book_id, score
       FROM similarities
       WHERE book_id = ? AND score >= ?
       ORDER BY score DESC
       LIMIT ?`,
    )
    .bind(bookId, threshold, limit)
    .all<SimilarityRow>();

  return results ?? [];
}

/**
 * Bulk-fetch book rows by a list of IDs, preserving the order of `ids`.
 * Cloudflare D1 does not support arrays natively so we build the placeholder
 * list manually.
 */
export async function getBooksByIds(
  db: D1Database,
  ids: number[],
): Promise<Map<number, BookRow>> {
  if (ids.length === 0) return new Map();

  const placeholders = ids.map(() => '?').join(', ');
  const { results } = await db
    .prepare(`SELECT * FROM books WHERE id IN (${placeholders})`)
    .bind(...ids)
    .all<BookRow>();

  const map = new Map<number, BookRow>();
  for (const row of results ?? []) {
    map.set(row.id, row);
  }
  return map;
}
