import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from langcache import LangCache
from langcache.models import SearchStrategy

# === Load environment variables ===
load_dotenv()

# === LangCache Configuration ===
# Prefer LANGCACHE_API_KEY; fall back to legacy LANGCACHE_SERVICE_KEY for compatibility
LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY") or os.getenv("LANGCACHE_SERVICE_KEY")
LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
LANGCACHE_BASE_URL = os.getenv("LANGCACHE_BASE_URL", "https://gcp-us-east4.langcache.redis.io")

# === OpenAI Configuration ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def call_openai_llm(prompt: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] OpenAI request failed: {e}"


def main():
    if not LANGCACHE_API_KEY or not LANGCACHE_CACHE_ID:
        print("[WARN] Missing LangCache config (LANGCACHE_API_KEY and/or LANGCACHE_CACHE_ID). Caching disabled.")
    if not OPENAI_API_KEY:
        raise SystemExit("Missing OPENAI_API_KEY in env.")

    print("LangCache Semantic Cache Chat - Type 'exit' to quit.\n")

    # Use a no-op context manager when LangCache isn't configured
    class _Noop:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    cache_ctx = (
        LangCache(server_url=LANGCACHE_BASE_URL, cache_id=LANGCACHE_CACHE_ID, api_key=LANGCACHE_API_KEY)
        if (LANGCACHE_API_KEY and LANGCACHE_CACHE_ID)
        else _Noop()
    )

    with cache_ctx as lang_cache:
        while True:
            query = input("Ask something: ").strip()
            if query.lower() in {"exit", "quit"}:
                break

            cached_resp = None
            start_time = time.perf_counter()

            # Try cache only if available
            if lang_cache:
                try:
                    # First: semantic search with threshold
                    results = lang_cache.search(prompt=query, similarity_threshold=0.7)
                    # Fallback: exact + semantic if nothing found
                    if not results or not getattr(results, "data", None):
                        results = lang_cache.search(
                            prompt=query,
                            search_strategies=[SearchStrategy.EXACT, SearchStrategy.SEMANTIC],
                        )
                    if results and getattr(results, "data", None):
                        cached_resp = results.data[0].response
                except Exception as e:
                    print(f"[LangCache search error] {e}")

            elapsed_time = time.perf_counter() - start_time

            if cached_resp:
                print("[CACHE HIT]")
                print(f"[Latency] Cache hit in {elapsed_time:.3f} seconds")
                print("Response:", cached_resp)
            else:
                print("[CACHE MISS]" if lang_cache else "[CACHE DISABLED]")
                if lang_cache:
                    print(f"[Latency] Cache search took {elapsed_time:.3f} seconds")

                start_llm = time.perf_counter()
                response = call_openai_llm(query)
                elapsed_llm = time.perf_counter() - start_llm

                # Best-effort: store in cache if available
                if lang_cache:
                    try:
                        lang_cache.set(prompt=query, response=response)
                    except Exception as e:
                        print(f"[LangCache set error] {e}")

                print(f"[Latency] OpenAI response took {elapsed_llm:.3f} seconds")
                print("Response:", response)

            print("-" * 60)


if __name__ == "__main__":
    main()