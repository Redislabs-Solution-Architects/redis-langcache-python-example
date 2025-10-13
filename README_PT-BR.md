# Redis LangCache — PT‑BR Demo (Gradio UI)

Uma demo 100% prática mostrando **Redis LangCache** + **OpenAI** com **cache semântico** e **escopo por Company / BU / Person**, usando uma **interface Gradio**.

> Código principal desta demo: [`main_demo_released.py`](https://github.com/Redislabs-Solution-Architects/redis-langcache-python-example/blob/main/main_demo_released.py)

---

## ✨ O que esta demo faz

- **Cache semântico** para respostas de LLMs, reduzindo **latência** e **custo**.
- **Escopo de reutilização** por **Company / BU / Person** — você controla o isolamento.
- **Desambiguação por domínio**: perguntas ambíguas (ex.: “célula”, “rede”, “banco”) são respondidas no **contexto certo**.
- **Identidade**:
  - **Nome** → sem cache (apenas exibição quando perguntado).
  - **Cargo/Função** → armazenado com **chave exata** (`[IDENTITY:ROLE]`) e suporta **“set”** via texto (“Minha função é …”).
- **UI de manutenção**: botões para **limpar entradas do cache** por escopo (A, B ou A+B). *Nunca apaga o índice.*
- **KPIs na tela**: hits, misses, taxa de acerto, tokens estimados e economia.

---

## 📦 Estrutura do projeto (mínima)

```
.
├── main_demo_released.py   # App Gradio (esta demo)
├── requirements.txt        # Dependências Python
├── Dockerfile              # Build da imagem
├── docker-compose.yml      # (exemplo) Orquestração local
└── .env                    # Variáveis de ambiente (não versionar)
```

> O repositório também contém exemplos adicionais (RAG, atributos etc.). Esta demo usa **`main_demo_released.py`**.

---

## 🔐 Variáveis de ambiente

Crie um arquivo `.env` na raiz (ou exporte no ambiente) com:

```env
# OpenAI
OPENAI_API_KEY=sk-proj-<sua-chave>
OPENAI_MODEL=gpt-4o-mini

# LangCache (Redis Cloud)
LANGCACHE_SERVICE_KEY=<sua-service-key>   # ou use LANGCACHE_API_KEY
LANGCACHE_CACHE_ID=<seu-cache-id>
LANGCACHE_BASE_URL=https://gcp-us-east4.langcache.redis.io

# (opcional) Redis local / outros
REDIS_URL=redis://localhost:6379/0

# Embeddings (se usados em exemplos de RAG)
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
```

> **Dica:** `LANGCACHE_API_KEY` e `LANGCACHE_SERVICE_KEY` são equivalentes para este app. Use **um** deles.

---

## 🚀 Como rodar

### 1) Local (Python)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt

# certifique-se de ter o .env configurado
python main_demo_released.py
```

A UI subirá em: **http://localhost:7860**

---

### 2) Docker (imagem pronta)

```bash
docker run -d \
  --name langcache-demo \
  --env-file .env \
  -p 7860:7860 \
  gacerioni/gabs-redis-langcache:1.1.0
```

> Apple Silicon (arm64): se necessário, force `--platform linux/amd64` ao rodar a imagem.

---

### 3) Docker Compose (opcional)

```yaml
# docker-compose.yml
version: "3.9"
services:
  langcache-demo:
    image: gacerioni/gabs-redis-langcache:1.1.0
    # platform: linux/amd64  # descomente em Apple Silicon se precisar
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

## 🧑‍💻 Como usar a UI

1. Preencha **Company**, **Business Unit** e **Person** para os **Cenários A e B**.
2. Faça perguntas nos dois cenários para ver **cache hit/miss** e **desambiguação por domínio**.
3. Use os botões **🧹 Limpar Cache** para apagar **entradas** por escopo (A, B ou A+B).  
   > **Importante:** a limpeza remove **entradas** do cache **sem apagar o índice**.

Exemplos úteis para testar:

- “**Minha função é Médico**.” / “**Minha função é Engenheiro de Software**.”  
- “**Qual é a minha função na empresa?**”  
- “**O que é uma célula?**” (saúde vs. software)  
- “**Explique o que é aprendizado de máquina**.” / “**O que é machine learning?**”  
- “**Qual é o meu nome?**”

---

## 🏗️ Como funciona (resumo técnico)

- **Busca** no LangCache por semelhança semântica (ou exata, quando apropriado).  
- **Hit** → responde imediato do cache.  
- **Miss** → consulta OpenAI, **salva resposta neutra** no LangCache e exibe.  
- **Isolamento** por atributos (**company**, **business_unit**, **person**).  
- **Perguntas ambíguas** → reescritas internamente para incluir “(no contexto de …)” com base no domínio inferido.

---

## 🧪 CI/CD (opcional)

Se quiser automatizar build e push para Docker Hub + release por tag:
- GitHub Actions com workflow de **build multi‑arch** e **release ao criar `vX.Y.Z`**.
- Secrets necessários no repositório:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN` (PAT)
  - (`GITHUB_TOKEN` já é fornecido pelo Actions)

> O pipeline compõe tags automaticamente (`latest`, `sha`, `branch`, `semver`).

---

## 🔗 Links úteis

- **Redis LangCache (docs):** https://redis.io/docs/latest/solutions/semantic-caching/langcache/  
- **Redis (site):** https://redis.io/  
- **LinkedIn (Gabriel Cerioni):** https://www.linkedin.com/in/gabrielcerioni/

---

## 📝 Licença

MIT — use à vontade.
