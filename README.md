# Redis LangCache Demo

A minimal Python demo showing how to use [Redis LangCache](https://redis.io/docs/latest/solutions/semantic-caching/langcache/) with OpenAI to implement semantic caching for LLM queries.  
This example caches responses based on semantic similarity, reducing latency and API usage costs.

---

## 📂 Project Structure

```
.
├── main.py              # Main script for running the demo
├── requirements.txt     # Python dependencies
├── .env.EXAMPLE         # Example environment variable configuration
└── .env                 # Your actual environment variables (not committed)
```

---

## 🚀 Prerequisites

- Python **3.10+**
- A Redis LangCache instance (Redis Cloud)
- An OpenAI API key

---

## ⚙️ Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-repo>/gabs-redis-langcache.git
   cd gabs-redis-langcache
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   - Copy `.env.EXAMPLE` to `.env`
   - Fill in your credentials:
     ```env
     OPENAI_API_KEY=sk-proj-<your-openai-key>
     OPENAI_MODEL=gpt-4o

     LANGCACHE_SERVICE_KEY=<your-langcache-service-key>
     LANGCACHE_CACHE_ID=<your-langcache-cache-id>
     LANGCACHE_BASE_URL=https://gcp-us-east4.langcache.redis.io
     ```

---

## ▶️ Usage

Run the demo:

```bash
python main.py
```

Example interaction:

```
LangCache Semantic Cache Chat - Type 'exit' to quit.

Ask something: What is Redis LangCache?
[CACHE MISS]
[Latency] Cache miss search took 0.023 seconds
[Latency] OpenAI response took 0.882 seconds
Response: Redis LangCache is a semantic caching solution...
------------------------------------------------------------
Ask something: Tell me about LangCache
[CACHE HIT]
[Latency] Cache hit in 0.002 seconds
Response: Redis LangCache is a semantic caching solution...
------------------------------------------------------------
```

---

## 🧠 How It Works

1. **Search** in Redis LangCache for a semantically similar question.
2. If a **cache hit** is found (above the similarity threshold), return it instantly.
3. If a **cache miss** occurs:
   - Query OpenAI.
   - Store the response in Redis LangCache for future reuse.

---

## 📄 License

MIT
