# Redis LangCache — English Demo (Gradio UI)

A fully functional demo showing **Redis LangCache** + **OpenAI** in action, implementing **semantic caching** with **scoped isolation** by Company / Business Unit / Person — all in a **Gradio web interface**.

> Main demo file: [`main_demo_released.py`](https://github.com/Redislabs-Solution-Architects/redis-langcache-python-example/blob/main/main_demo_released.py)

---

## ✨ What This Demo Does

- Demonstrates **semantic caching** for LLM responses to reduce **latency** and **API cost**.  
- **Scoped reuse** of answers by **Company / Business Unit / Person** — adjustable isolation levels.  
- **Domain disambiguation**: ambiguous questions (“cell”, “network”, “bank”) are automatically interpreted in the correct domain.  
- **Identity handling**:
  - **Name** → not cached (display only when asked).  
  - **Role/Function** → stored under exact key (`[IDENTITY:ROLE]`) and supports “set” (e.g., “My role is …”).  
- **Cache management UI**: clear cached entries by scope (A, B, or both) — *the index is never deleted.*  
- **Real-time KPIs**: cache hits, misses, hit rate, estimated tokens saved, and $ savings.

---

## 📁 Project Structure

```
.
├── main_demo_released.py   # Main Gradio app (this demo)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build
├── docker-compose.yml      # Example local orchestration
└── .env                    # Environment variables (not committed)
```

> The repository also includes additional examples (RAG, attribute-based caching, etc.).  
> This demo uses **`main_demo_released.py`** as its entry point.

---

## 🔐 Environment Variables

Create a `.env` file in the project root with:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-<your-openai-key>
OPENAI_MODEL=gpt-4o-mini

# LangCache (Redis Cloud)
LANGCACHE_SERVICE_KEY=<your-service-key>  # or LANGCACHE_API_KEY
LANGCACHE_CACHE_ID=<your-cache-id>
LANGCACHE_BASE_URL=https://gcp-us-east4.langcache.redis.io

# (Optional) Redis local or other configs
REDIS_URL=redis://localhost:6379/0

# Embedding model (for RAG examples)
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
```

> `LANGCACHE_API_KEY` and `LANGCACHE_SERVICE_KEY` are interchangeable for this app — use one of them.

---

## 🚀 Running the Demo

### 1) Locally (Python)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt

# Ensure your .env is configured
python main_demo_released.py
```

The UI will start at: **http://localhost:7860**

---

### 2) With Docker (prebuilt image)

```bash
docker run -d \
  --name langcache-demo \
  --env-file .env \
  -p 7860:7860 \
  gacerioni/gabs-redis-langcache:1.1.0
```

> Apple Silicon (arm64): if needed, add `--platform linux/amd64` when running the image.

---

### 3) Docker Compose (optional)

```yaml
# docker-compose.yml
version: "3.9"
services:
  langcache-demo:
    image: gacerioni/gabs-redis-langcache:1.1.0
    # platform: linux/amd64  # uncomment on Apple Silicon if needed
    env_file:
      - .env
    ports:
      - "7860:7860"
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

```bash
docker compose up -d
```

---

## 🧑‍💻 Using the UI

1. Set **Company**, **Business Unit**, and **Person** for both **Scenario A and B**.  
2. Ask questions in both panels to observe **cache hits/misses** and **domain-aware disambiguation**.  
3. Use the **🧹 Clear Cache** buttons to delete entries by scope (A, B, or both).  
   > ⚠️ This clears cached **entries only** — the index is **never deleted**.

Recommended questions for demonstration:

- “**My role is Doctor.**” / “**My role is Software Engineer.**”  
- “**What is my role in the company?**”  
- “**What is a cell?**” (see difference between healthcare vs software)  
- “**Explain what machine learning is.**” / “**What is machine learning?**”  
- “**What is my name?**”  

---

## 🧠 How It Works

1. **Search** Redis LangCache for semantically similar prompts.  
2. If a **cache hit** (above threshold) is found, return the cached response.  
3. If a **miss** occurs:  
   - Query OpenAI.  
   - Store a **neutral** response (no user identity) in the cache.  
4. Isolation is managed via attributes: `company`, `business_unit`, and `person`.  
5. Ambiguous prompts are internally **rewritten** with explicit domain context (e.g., “(in the context of healthcare)”).

---

## ⚙️ CI/CD Pipeline (optional)

You can automate Docker build & release with GitHub Actions.  
The existing workflow builds a **multi-arch** image and publishes it on new tags (`vX.Y.Z`).

Required repository secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub PAT)
- `GITHUB_TOKEN` (provided automatically)

---

## 🔗 Useful Links

- **Redis LangCache Documentation:** https://redis.io/docs/latest/solutions/semantic-caching/langcache/  
- **Redis Website:** https://redis.io/  
- **LinkedIn (Gabriel Cerioni):** https://www.linkedin.com/in/gabrielcerioni/

---

## 📜 License

MIT — feel free to use, adapt, and share.
