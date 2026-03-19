// KV-based response cache helpers.

// ─── Key generation ────────────────────────────────────────────────────────

function normalizeTitle(title: string): string {
  return title.toLowerCase().trim().replace(/\s+/g, ' ');
}

export function recCacheKey(title: string, limit: number, threshold: number): string {
  return `rec:${normalizeTitle(title)}:${limit}:${threshold}`;
}

export function searchCacheKey(query: string, limit: number): string {
  return `search:${normalizeTitle(query)}:${limit}`;
}

export function healthCacheKey(): string {
  return 'health';
}

// ─── Generic read / write ──────────────────────────────────────────────────

export async function cacheGet<T>(kv: KVNamespace, key: string): Promise<T | null> {
  const raw = await kv.get(key, 'text');
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function cachePut<T>(
  kv: KVNamespace,
  key: string,
  value: T,
  ttlSeconds: number,
): Promise<void> {
  await kv.put(key, JSON.stringify(value), { expirationTtl: ttlSeconds });
}
