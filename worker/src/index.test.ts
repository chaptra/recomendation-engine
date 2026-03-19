/**
 * Unit tests for the Cloudflare Worker.
 *
 * We mock the D1 and KV bindings so that tests run without a real Cloudflare
 * account – the same philosophy as the Python tests which mock the database
 * and cache layers.
 */

import { describe, it, expect, vi } from 'vitest';

// ─── Minimal binding mocks ───────────────────────────────────────────────────

function makeD1Mock(overrides: Record<string, unknown> = {}): D1Database {
  const stmt = {
    bind: vi.fn().mockReturnThis(),
    first: vi.fn().mockResolvedValue(null),
    all: vi.fn().mockResolvedValue({ results: [] }),
    run: vi.fn().mockResolvedValue({}),
  };
  return {
    prepare: vi.fn().mockReturnValue(stmt),
    batch: vi.fn().mockResolvedValue([]),
    dump: vi.fn().mockResolvedValue(new ArrayBuffer(0)),
    exec: vi.fn().mockResolvedValue({ count: 0, duration: 0 }),
    ...overrides,
  } as unknown as D1Database;
}

function makeKvMock(store: Record<string, string> = {}): KVNamespace {
  return {
    get: vi.fn(async (key: string) => store[key] ?? null),
    put: vi.fn(async (key: string, value: string) => { store[key] = value; }),
    delete: vi.fn(async (key: string) => { delete store[key]; }),
    list: vi.fn().mockResolvedValue({ keys: [], list_complete: true, cursor: '' }),
    getWithMetadata: vi.fn().mockResolvedValue({ value: null, metadata: null }),
  } as unknown as KVNamespace;
}

function makeEnv(db: D1Database, cache: KVNamespace): import('../src/types.js').Env {
  return {
    DB: db,
    CACHE: cache,
    ENVIRONMENT: 'test',
    CACHE_TTL_SECONDS: '86400',
    MAX_RECOMMENDATIONS: '50',
    DEFAULT_RECOMMENDATIONS: '10',
    SIMILARITY_THRESHOLD: '0.1',
  };
}

// ─── Import worker after mocks are defined ──────────────────────────────────

// We import the handler under test
import worker from '../src/index.js';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function get(path: string, env: import('../src/types.js').Env): Promise<Response> {
  return worker.fetch(new Request(`http://localhost${path}`), env);
}

function post(path: string, body: unknown, env: import('../src/types.js').Env): Promise<Response> {
  return worker.fetch(
    new Request(`http://localhost${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
    env,
  );
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('GET /', () => {
  it('returns API info', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await get('/', env);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data['name']).toBe('Book Recommendation Engine');
    expect(Array.isArray(data['endpoints'])).toBe(true);
  });
});

describe('GET /health', () => {
  it('returns ok and books_count from D1', async () => {
    const stmt = {
      bind: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue({ n: 42 }),
      all: vi.fn().mockResolvedValue({ results: [] }),
      run: vi.fn(),
    };
    const db = { prepare: vi.fn().mockReturnValue(stmt) } as unknown as D1Database;
    const env = makeEnv(db, makeKvMock());

    const res = await get('/health', env);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data['status']).toBe('ok');
    expect(data['books_count']).toBe(42);
  });

  it('returns cached health without hitting D1', async () => {
    const db = makeD1Mock();
    const cached = JSON.stringify({ status: 'ok', books_count: 99, timestamp: new Date().toISOString() });
    const kv = makeKvMock({ health: cached });
    const env = makeEnv(db, kv);

    const res = await get('/health', env);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data['books_count']).toBe(99);
    // D1 should not have been queried
    expect((db.prepare as ReturnType<typeof vi.fn>).mock.calls.length).toBe(0);
  });
});

describe('GET /suggest', () => {
  it('returns 400 when title is missing', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await get('/suggest', env);
    expect(res.status).toBe(400);
  });

  it('returns 404 when book is not found', async () => {
    const stmt = {
      bind: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue({ results: [] }),
      run: vi.fn(),
    };
    const db = { prepare: vi.fn().mockReturnValue(stmt) } as unknown as D1Database;
    const env = makeEnv(db, makeKvMock());

    const res = await get('/suggest?title=UnknownBook', env);
    expect(res.status).toBe(404);
    const data = await res.json() as Record<string, unknown>;
    expect(data['found']).toBe(false);
  });

  it('returns suggestions when book is found', async () => {
    const book = {
      id: 1, title: 'The Great Gatsby', slug: 'gatsby', author: 'F. Scott Fitzgerald',
      categories: 'Fiction', description: 'A story', cover_path: '', rating: 4.2,
      rating_count: 1000, year: 1925, isbn: '',
    };
    const similar = {
      id: 2, title: 'Tender Is the Night', slug: 'tender', author: 'F. Scott Fitzgerald',
      categories: 'Fiction', description: 'Another story', cover_path: '', rating: 4.0,
      rating_count: 800, year: 1934, isbn: '',
    };

    let callCount = 0;
    const db = {
      prepare: vi.fn().mockImplementation(() => ({
        bind: vi.fn().mockReturnThis(),
        first: vi.fn().mockImplementation(async () => {
          callCount++;
          return callCount === 1 ? book : null; // first call = findBookByTitle
        }),
        all: vi.fn().mockImplementation(async () => {
          // similarities query
          if (callCount <= 1) return { results: [{ similar_book_id: 2, score: 0.85 }] };
          // getBooksByIds query
          return { results: [similar] };
        }),
        run: vi.fn(),
      })),
    } as unknown as D1Database;

    const env = makeEnv(db, makeKvMock());
    const res = await get('/suggest?title=The+Great+Gatsby&limit=5', env);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data['found']).toBe(true);
    expect(data['query']).toBe('The Great Gatsby');
  });

  it('returns cached response without hitting D1', async () => {
    const db = makeD1Mock();
    const cached = JSON.stringify({
      query: 'Cached Book', found: true, suggestions: [], total: 0,
    });
    // The cache key is rec:cached book:5:0.1
    const kv = makeKvMock({ 'rec:cached book:5:0.1': cached });
    const env = makeEnv(db, kv);

    const res = await get('/suggest?title=Cached+Book&limit=5', env);
    expect(res.status).toBe(200);
    expect((db.prepare as ReturnType<typeof vi.fn>).mock.calls.length).toBe(0);
  });
});

describe('POST /recommendations', () => {
  it('returns 400 when body is not JSON', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await worker.fetch(
      new Request('http://localhost/recommendations', {
        method: 'POST',
        body: 'not json',
      }),
      env,
    );
    expect(res.status).toBe(400);
  });

  it('returns 400 when title is missing', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await post('/recommendations', { limit: 5 }, env);
    expect(res.status).toBe(400);
  });

  it('returns 404 when book is not found', async () => {
    const stmt = {
      bind: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue({ results: [] }),
      run: vi.fn(),
    };
    const db = { prepare: vi.fn().mockReturnValue(stmt) } as unknown as D1Database;
    const env = makeEnv(db, makeKvMock());

    const res = await post('/recommendations', { title: 'Unknown Book' }, env);
    expect(res.status).toBe(404);
  });
});

describe('GET /search', () => {
  it('returns 400 when q param is missing', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await get('/search', env);
    expect(res.status).toBe(400);
  });

  it('returns search results from FTS', async () => {
    const books = [
      { id: 1, title: 'Gatsby', slug: 'gatsby', author: 'Fitzgerald', categories: '', description: '', cover_path: '', rating: 4.0, rating_count: 100, year: 1925, isbn: '' },
    ];
    const db = {
      prepare: vi.fn().mockReturnValue({
        bind: vi.fn().mockReturnThis(),
        first: vi.fn().mockResolvedValue(null),
        all: vi.fn().mockResolvedValue({ results: books }),
        run: vi.fn(),
      }),
    } as unknown as D1Database;
    const env = makeEnv(db, makeKvMock());

    const res = await get('/search?q=gatsby', env);
    expect(res.status).toBe(200);
    const data = await res.json() as Record<string, unknown>;
    expect(data['query']).toBe('gatsby');
    expect((data['results'] as unknown[]).length).toBe(1);
    expect(data['count']).toBe(1);
  });
});

describe('GET /search/:title', () => {
  it('returns 404 for unknown title', async () => {
    const stmt = {
      bind: vi.fn().mockReturnThis(),
      first: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue({ results: [] }),
      run: vi.fn(),
    };
    const db = { prepare: vi.fn().mockReturnValue(stmt) } as unknown as D1Database;
    const env = makeEnv(db, makeKvMock());

    const res = await get('/search/SomeRandomBook', env);
    expect(res.status).toBe(404);
  });
});

describe('OPTIONS (CORS pre-flight)', () => {
  it('returns 204 with CORS headers', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await worker.fetch(
      new Request('http://localhost/suggest', { method: 'OPTIONS' }),
      env,
    );
    expect(res.status).toBe(204);
    expect(res.headers.get('Access-Control-Allow-Origin')).toBe('*');
  });
});

describe('Unknown routes', () => {
  it('returns 404 for unknown path', async () => {
    const env = makeEnv(makeD1Mock(), makeKvMock());
    const res = await get('/unknown-path', env);
    expect(res.status).toBe(404);
  });
});
