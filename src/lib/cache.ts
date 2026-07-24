const store = new Map<string, { data: unknown; expiry: number }>();

export async function withCache<T>(
  key: string,
  ttlMs: number,
  fetcher: () => Promise<T>,
): Promise<T> {
  const cached = store.get(key);
  if (cached && cached.expiry > Date.now()) {
    return cached.data as T;
  }
  const data = await fetcher();
  store.set(key, { data, expiry: Date.now() + ttlMs });
  return data;
}
