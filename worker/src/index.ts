/**
 * Cloudflare Worker – Book Recommendation Engine
 *
 * Replaces the EC2-hosted FastAPI server.  All heavy ML computation
 * (TF-IDF + cosine similarity) is performed **offline** by
 * scripts/export_to_d1.py which writes the results into D1 (SQLite).
 * This Worker simply queries those pre-computed results and returns them.
 *
 * Bindings (wrangler.toml):
 *   DB    – Cloudflare D1 (books + pre-computed similarities)
 *   CACHE – Cloudflare KV  (response-level cache, 24 h TTL)
 */

import {
  Env,
  RecommendationResponse,
  SearchResponse,
  HealthResponse,
  InfoResponse,
  ErrorResponse,
  rowToDetails,
} from './types.js';

import {
  countBooks,
  findBookByTitle,
  searchBooks,
  getSimilarBooks,
  getBooksByIds,
} from './db.js';

import {
  recCacheKey,
  searchCacheKey,
  healthCacheKey,
  cacheGet,
  cachePut,
} from './cache.js';

// ─── Constants ──────────────────────────────────────────────────────────────

const VERSION = '2.0.0';
const CORS_HEADERS: HeadersInit = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

// ─── Helpers ────────────────────────────────────────────────────────────────

function json<T>(data: T, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
  });
}

function err(message: string, status = 400, detail?: string): Response {
  const body: ErrorResponse = { error: message, detail };
  return json(body, status);
}

/** Parse ?limit= / &count= query param with bounds checking. */
function parseLimit(url: URL, defaultVal: number, max: number): number {
  const raw = url.searchParams.get('limit') ?? url.searchParams.get('count');
  if (!raw) return defaultVal;
  const n = parseInt(raw, 10);
  if (isNaN(n) || n < 1) return defaultVal;
  return Math.min(n, max);
}

// ─── Route handlers ─────────────────────────────────────────────────────────

async function handleRoot(_req: Request, _env: Env): Promise<Response> {
  const body: InfoResponse = {
    name: 'Book Recommendation Engine',
    version: VERSION,
    status: 'running',
    docs: '/docs',
    endpoints: [
      'GET  /',
      'GET  /health',
      'GET  /suggest?title=<title>&limit=<n>',
      'POST /recommendations',
      'GET  /search?q=<query>&count=<n>',
      'GET  /search/<title>',
    ],
  };
  return json(body);
}

async function handleHealth(_req: Request, env: Env): Promise<Response> {
  const ttl = parseInt(env.CACHE_TTL_SECONDS, 10);

  const cached = await cacheGet<HealthResponse>(env.CACHE, healthCacheKey());
  if (cached) return json(cached);

  const booksCount = await countBooks(env.DB);
  const body: HealthResponse = {
    status: 'ok',
    books_count: booksCount,
    timestamp: new Date().toISOString(),
  };
  // Cache health for 60 s to avoid repeated COUNT(*) queries
  await cachePut(env.CACHE, healthCacheKey(), body, Math.min(ttl, 60));
  return json(body);
}

/** GET /suggest?title=X&limit=N  (mirrors clean_api.py /suggest) */
async function handleSuggest(req: Request, env: Env): Promise<Response> {
  const url = new URL(req.url);
  const title = (url.searchParams.get('title') ?? '').trim();
  if (!title) return err('title query parameter is required');

  const maxRecs = parseInt(env.MAX_RECOMMENDATIONS, 10);
  const defaultRecs = parseInt(env.DEFAULT_RECOMMENDATIONS, 10);
  const threshold = parseFloat(env.SIMILARITY_THRESHOLD);
  const limit = parseLimit(url, defaultRecs, maxRecs);
  const ttl = parseInt(env.CACHE_TTL_SECONDS, 10);

  const cacheKey = recCacheKey(title, limit, threshold);
  const cached = await cacheGet<RecommendationResponse>(env.CACHE, cacheKey);
  if (cached) return json(cached);

  const book = await findBookByTitle(env.DB, title);
  if (!book) {
    const body: RecommendationResponse = {
      query: title,
      found: false,
      message: `Book not found: "${title}". Try /search?q=${encodeURIComponent(title)} to find it.`,
      suggestions: [],
      total: 0,
    };
    return json(body, 404);
  }

  const similarities = await getSimilarBooks(env.DB, book.id, limit, threshold);
  const ids = similarities.map((s) => s.similar_book_id);
  const bookMap = await getBooksByIds(env.DB, ids);

  const suggestions = similarities
    .map((s) => {
      const related = bookMap.get(s.similar_book_id);
      return related ? rowToDetails(related, s.score) : null;
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);

  const body: RecommendationResponse = {
    query: title,
    found: true,
    source_book: rowToDetails(book),
    suggestions,
    total: suggestions.length,
  };

  await cachePut(env.CACHE, cacheKey, body, ttl);
  return json(body);
}

/** POST /recommendations  (mirrors api.py /recommendations) */
async function handleRecommendationsPost(req: Request, env: Env): Promise<Response> {
  let payload: Record<string, unknown>;
  try {
    payload = await req.json() as Record<string, unknown>;
  } catch {
    return err('Request body must be valid JSON');
  }

  const title = typeof payload['title'] === 'string' ? payload['title'].trim() : '';
  if (!title) return err('title is required in the request body');

  const maxRecs = parseInt(env.MAX_RECOMMENDATIONS, 10);
  const defaultRecs = parseInt(env.DEFAULT_RECOMMENDATIONS, 10);
  const defaultThreshold = parseFloat(env.SIMILARITY_THRESHOLD);

  const rawLimit = payload['limit'];
  const limit = (typeof rawLimit === 'number' && rawLimit >= 1)
    ? Math.min(Math.floor(rawLimit), maxRecs)
    : defaultRecs;

  const rawThreshold = payload['similarity_threshold'];
  const threshold = (typeof rawThreshold === 'number')
    ? Math.max(0, Math.min(1, rawThreshold))
    : defaultThreshold;

  const ttl = parseInt(env.CACHE_TTL_SECONDS, 10);
  const cacheKey = recCacheKey(title, limit, threshold);
  const cached = await cacheGet<RecommendationResponse>(env.CACHE, cacheKey);
  if (cached) return json(cached);

  const book = await findBookByTitle(env.DB, title);
  if (!book) {
    const body: RecommendationResponse = {
      query: title,
      found: false,
      message: `Book not found: "${title}"`,
      suggestions: [],
      total: 0,
    };
    return json(body, 404);
  }

  const similarities = await getSimilarBooks(env.DB, book.id, limit, threshold);
  const ids = similarities.map((s) => s.similar_book_id);
  const bookMap = await getBooksByIds(env.DB, ids);

  const suggestions = similarities
    .map((s) => {
      const related = bookMap.get(s.similar_book_id);
      return related ? rowToDetails(related, s.score) : null;
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);

  const body: RecommendationResponse = {
    query: title,
    found: true,
    source_book: rowToDetails(book),
    suggestions,
    total: suggestions.length,
  };

  await cachePut(env.CACHE, cacheKey, body, ttl);
  return json(body);
}

/** GET /search?q=X&count=N  (full-text search) */
async function handleSearch(req: Request, env: Env): Promise<Response> {
  const url = new URL(req.url);
  const query = (url.searchParams.get('q') ?? '').trim();
  if (!query) return err('q query parameter is required');

  const maxRecs = parseInt(env.MAX_RECOMMENDATIONS, 10);
  const limit = parseLimit(url, 10, maxRecs);
  const ttl = parseInt(env.CACHE_TTL_SECONDS, 10);

  const cacheKey = searchCacheKey(query, limit);
  const cached = await cacheGet<SearchResponse>(env.CACHE, cacheKey);
  if (cached) return json(cached);

  const books = await searchBooks(env.DB, query, limit);
  const body: SearchResponse = {
    query,
    results: books.map((b) => rowToDetails(b)),
    count: books.length,
  };

  await cachePut(env.CACHE, cacheKey, body, ttl);
  return json(body);
}

/** GET /search/:title  (exact-title lookup, mirrors api.py /search/{title}) */
async function handleSearchByTitle(title: string, _env: Env, db: D1Database): Promise<Response> {
  const book = await findBookByTitle(db, decodeURIComponent(title));
  if (!book) {
    return json({ found: false, message: `Book "${title}" not found` }, 404);
  }
  return json({ found: true, book: rowToDetails(book) });
}

// ─── Router ─────────────────────────────────────────────────────────────────

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    // CORS pre-flight
    if (req.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(req.url);
    const path = url.pathname.replace(/\/$/, '') || '/';

    try {
      // Exact-path routes
      if (path === '/' && req.method === 'GET') return handleRoot(req, env);
      if (path === '/health' && req.method === 'GET') return handleHealth(req, env);
      if (path === '/suggest' && req.method === 'GET') return handleSuggest(req, env);
      if (path === '/recommendations' && req.method === 'POST') return handleRecommendationsPost(req, env);
      if (path === '/search' && req.method === 'GET') return handleSearch(req, env);

      // /search/:title  (GET)
      const searchMatch = path.match(/^\/search\/(.+)$/);
      if (searchMatch && req.method === 'GET') {
        return handleSearchByTitle(searchMatch[1], env, env.DB);
      }

      // Docs redirect (optional convenience)
      if (path === '/docs') {
        return Response.redirect('https://developers.cloudflare.com/', 301);
      }

      return err('Not Found', 404);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Internal server error';
      return err('Internal server error', 500, message);
    }
  },
} satisfies ExportedHandler<Env>;
