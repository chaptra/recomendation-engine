// TypeScript types shared across Worker modules.

// ─── Cloudflare bindings ───────────────────────────────────────────────────

export interface Env {
  DB: D1Database;
  CACHE: KVNamespace;
  // [vars] from wrangler.toml
  ENVIRONMENT: string;
  CACHE_TTL_SECONDS: string;
  MAX_RECOMMENDATIONS: string;
  DEFAULT_RECOMMENDATIONS: string;
  SIMILARITY_THRESHOLD: string;
}

// ─── Database row shapes ───────────────────────────────────────────────────

export interface BookRow {
  id: number;
  title: string;
  slug: string;
  author: string;
  categories: string;
  description: string;
  cover_path: string;
  rating: number;
  rating_count: number;
  year: number | null;
  isbn: string;
}

export interface SimilarityRow {
  similar_book_id: number;
  score: number;
}

// ─── API response shapes ───────────────────────────────────────────────────

export interface BookDetails {
  id: number;
  title: string;
  slug: string;
  cover_local_path: string;
  authors: string;
  categories: string;
  description: string;
  rating: number;
  rating_count: number;
  similarity_score?: number;
  reason?: string;
}

export interface RecommendationResponse {
  query: string;
  found: boolean;
  message?: string;
  source_book?: BookDetails;
  suggestions: BookDetails[];
  total: number;
}

export interface SearchResponse {
  query: string;
  results: BookDetails[];
  count: number;
}

export interface HealthResponse {
  status: string;
  books_count: number;
  timestamp: string;
}

export interface InfoResponse {
  name: string;
  version: string;
  status: string;
  docs: string;
  endpoints: string[];
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/** Convert a BookRow from D1 into the public BookDetails shape. */
export function rowToDetails(row: BookRow, similarityScore?: number): BookDetails {
  return {
    id: row.id,
    title: row.title,
    slug: row.slug,
    cover_local_path: row.cover_path,
    authors: row.author,
    categories: row.categories,
    description: row.description,
    rating: row.rating,
    rating_count: row.rating_count,
    similarity_score: similarityScore,
    reason: similarityScore !== undefined
      ? `${Math.round(similarityScore * 100)}% content similarity`
      : undefined,
  };
}
