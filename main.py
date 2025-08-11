import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from langcache import LangCache

# === Load environment variables ===
load_dotenv()

# === LangCache Configuration ===
LANGCACHE_SERVICE_KEY = os.getenv("LANGCACHE_SERVICE_KEY")
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
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] OpenAI request failed: {e}"


def main():
    print("LangCache Semantic Cache Chat - Type 'exit' to quit.\n")

    with LangCache(
        server_url=LANGCACHE_BASE_URL,
        cache_id=LANGCACHE_CACHE_ID,
        service_key=LANGCACHE_SERVICE_KEY
    ) as lang_cache:

        while True:
            query = input("Ask something: ").strip()
            if query.lower() in {"exit", "quit"}:
                break

            start_time = time.perf_counter()
            results = lang_cache.search(prompt=query, similarity_threshold=0.7)
            elapsed_time = time.perf_counter() - start_time

            if results and results.data:
                print("[CACHE HIT]")
                print(f"[Latency] Cache hit in {elapsed_time:.3f} seconds")
                print("Response:", results.data[0].response)
            else:
                print("[CACHE MISS]")
                print(f"[Latency] Cache miss search took {elapsed_time:.3f} seconds")

                start_llm = time.perf_counter()
                response = call_openai_llm(query)
                lang_cache.set(prompt=query, response=response)
                elapsed_llm = time.perf_counter() - start_llm

                print(f"[Latency] OpenAI response took {elapsed_llm:.3f} seconds")
                print("Response:", response)

            print("-" * 60)


if __name__ == "__main__":
    main()