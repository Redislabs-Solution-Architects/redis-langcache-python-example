#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI Visual Refresh (apenas UI)
- Space Grotesk nos títulos/KPIs, Inter no corpo
- Header vermelho Redis com logo oficial + links (LinkedIn do Gabs / Redis)
- Cenário A e B lado a lado (layout preservado)
- Cards leves, contraste melhorado
- Sem CSV / healthcheck
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

try:
    from langcache import LangCache
    from langcache.models import SearchStrategy
except Exception:
    LangCache = None
    SearchStrategy = None

# ============== Env & Clients ==============
load_dotenv()

LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY") or os.getenv("LANGCACHE_SERVICE_KEY")
LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
LANGCACHE_BASE_URL = os.getenv("LANGCACHE_BASE_URL", "https://gcp-us-east4.langcache.redis.io")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise SystemExit("Faltou OPENAI_API_KEY no ambiente.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

lang_cache: Optional[LangCache] = None
if LANGCACHE_API_KEY and LANGCACHE_CACHE_ID and LangCache is not None:
    lang_cache = LangCache(
        server_url=LANGCACHE_BASE_URL,
        cache_id=LANGCACHE_CACHE_ID,
        api_key=LANGCACHE_API_KEY,
    )
else:
    print("[AVISO] LangCache não configurado; a UI roda, mas sem cache real.")

# ============== Intenção / Identidade ==============

def is_name_prompt(p: str) -> bool:
    p = (p or "").strip().lower()
    gatilhos = [
        "qual é meu nome", "qual eh meu nome", "qual o meu nome", "qual meu nome",
        "como me chamo", "diga meu nome", "fale meu nome", "quem sou eu",
        "qual é o meu nome", "qual eh o meu nome",
    ]
    return any(g in p for g in gatilhos)

def is_role_prompt(p: str) -> bool:
    p = (p or "").strip().lower()
    gatilhos = [
        "qual é a minha função", "qual eh a minha funcao", "qual a minha função", "qual a minha funcao",
        "qual é meu cargo", "qual eh meu cargo", "qual meu cargo",
        "qual é a minha posição", "qual a minha posição", "minha posição", "minha posicao",
        "qual é o meu papel", "qual o meu papel", "meu papel",
        "o que eu faço na empresa", "o que eu faco na empresa",
    ]
    return any(g in p for g in gatilhos)

ROLE_SET_PATTERNS = [
    r"\bminha função é\s+(?P<role>.+)$",
    r"\bmeu cargo é\s+(?P<role>.+)$",
    r"\beu sou\s+(?P<role>.+)$",
    r"\btrabalho como\s+(?P<role>.+)$",
]
def try_extract_role_set(p: str) -> Optional[str]:
    txt = (p or "").strip()
    txt = re.sub(r"[.!?]\s*$", "", txt)
    for pat in ROLE_SET_PATTERNS:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            role = m.group("role").strip()
            role = re.sub(r"[.!?]\s*$", "", role)
            return role
    return None

def is_identity_prompt(p: str) -> bool:
    return is_name_prompt(p) or is_role_prompt(p)

def should_personalize_name(p: str) -> bool:
    return is_name_prompt(p)

KEY_NAME = "[IDENTITY:NAME]"
KEY_ROLE = "[IDENTITY:ROLE]"

def normalize_prompt_for_cache(prompt: str) -> Tuple[str, str]:
    if is_name_prompt(prompt):
        return KEY_NAME, "identity:name"
    if is_role_prompt(prompt):
        return KEY_ROLE, "identity:role"
    return f"[FACT]\n{prompt.strip()}", "fact"

def depersonalize_safe(text: str, person: Optional[str]) -> str:
    if not text or not person:
        return text
    original = text.strip()
    t = original
    patterns = [
        rf"^ol[áa],\s*{re.escape(person)}\s*!\s*",
        rf"^seu\s+nome\s+é\s*{re.escape(person)}[.!]?\s*",
        rf"^seu\s+nome\s+eh\s*{re.escape(person)}[.!]?\s*",
        rf"^voc[êe]\s+se\s+chama\s*{re.escape(person)}[.!]?\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = t.strip()
    return t if t else original

# ============== Contexto / Desambiguação ==============

AMBIGUOUS_TERMS = [
    r"\bc[eé]lula\b",
    r"\bbanco\b",
    r"\brede\b",
    r"\bmodelo\b",
    r"\bpipeline\b",
]

def infer_domain(company: str, bu: str, role: Optional[str]) -> str:
    text = f"{company} {bu} {role or ''}".lower()
    if any(k in text for k in ["saude", "clínica", "clinica", "medic", "hospital"]):
        return "saúde"
    if any(k in text for k in ["engenharia", "software", "dev", "produto", "ti", "tecnologia", "tech"]):
        return "engenharia de software"
    if any(k in text for k in ["dados", "data", "bi", "analytics"]):
        return "dados"
    if any(k in text for k in ["finan", "banco", "invest", "asset", "seguro"]):
        return "finanças"
    if any(k in text for k in ["turismo", "eco", "aventura", "hotel", "viagem"]):
        return "turismo"
    return "geral da área do usuário"

def looks_ambiguous(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(re.search(pat, p, flags=re.IGNORECASE) for pat in AMBIGUOUS_TERMS)

def rewrite_with_domain(prompt: str, domain_label: str) -> str:
    clean = prompt.strip()
    if not clean.endswith("?"):
        clean += "?"
    return f"{clean} (no contexto de {domain_label})"

# ============== LLM ==============

def call_openai(
    prompt: str,
    person: Optional[str] = None,
    company: Optional[str] = None,
    bu: Optional[str] = None,
    role: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    domain = infer_domain(company or "", bu or "", role)
    system_ctx = (
        "Responda de forma breve e direta. "
        "Não mencione o nome do usuário a menos que a pergunta seja sobre o nome/identidade. "
        f"Contexto principal: {domain}. "
        "Se a pergunta for ambígua (ex.: 'deploy', 'pipeline', 'modelo'), "
        f"RESPONDA APENAS no sentido de {domain} e NÃO mencione outros significados."
    )

    examples = [
        # Engenharia de Software
        {"role": "user", "content": "O que é um deploy? (no contexto de engenharia de software)"},
        {"role": "assistant",
         "content": "Deploy é o processo de disponibilizar uma nova versão de software em produção."},
        {"role": "user", "content": "O que é um pipeline? (no contexto de engenharia de software)"},
        {"role": "assistant",
         "content": "Um pipeline é uma sequência automatizada de etapas para construir, testar e implantar código."},

        # Financeiro
        {"role": "user", "content": "O que é um deploy? (no contexto de finanças corporativas)"},
        {"role": "assistant",
         "content": "No contexto financeiro, deploy pode se referir à liberação de um novo processo, sistema ou investimento para uso interno."},
        {"role": "user", "content": "O que é um pipeline? (no contexto de vendas e finanças)"},
        {"role": "assistant",
         "content": "Pipeline é a lista de oportunidades ou previsões de receita que ainda estão em andamento."},

        # Bilíngue — cache semântico cross-language
        {"role": "user", "content": "Explique o que é aprendizado de máquina."},
        {"role": "assistant",
         "content": "Aprendizado de máquina é uma área da IA que permite que sistemas aprendam padrões a partir de dados sem programação explícita."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant",
         "content": "Machine learning is a branch of AI that enables systems to learn patterns from data and make predictions or decisions without being explicitly programmed."},
    ]
    msgs = [{"role": "system", "content": system_ctx}, *examples, {"role": "user", "content": prompt}]
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=msgs,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def estimate_tokens(*texts: str) -> int:
    total_chars = sum(len(t or "") for t in texts)
    return max(1, total_chars // 4)

# ============== Atributos do Cache ==============

def build_attributes(company: str, bu: str, person: str, isolation: str) -> Dict[str, str]:
    if isolation == "company":
        return {"company": company}
    if isolation == "company+bu":
        return {"company": company, "business_unit": bu}
    if isolation == "company+bu+person":
        return {"company": company, "business_unit": bu, "person": person}
    return {}

# ============== Núcleo Busca/Resposta ==============

def search_and_answer(
    company: str,
    bu: str,
    person: str,
    prompt_original: str,
    isolation: str,
    similarity_threshold: Optional[float],
    use_exact_then_semantic: bool,
    ttl_ms: Optional[int],
) -> Tuple[str, str, str, str, int]:
    if not prompt_original.strip():
        return "", "—", "{}", "Ocioso", 0

    role_set = try_extract_role_set(prompt_original)
    attrs = build_attributes(company, bu, person, isolation)

    if role_set and lang_cache:
        try:
            lang_cache.set(
                prompt=KEY_ROLE,
                response=f"Sua função é {role_set}.",
                attributes=attrs,
                ttl_millis=ttl_ms,
            )
            display = f"Anotado: sua função é **{role_set}** (escopo {attrs})."
            debug_set = {
                "attributes_usados": attrs,
                "intent": "identity:role:set",
                "stored_under": KEY_ROLE,
                "value": role_set,
            }
            return display, "llm", json.dumps(debug_set, indent=2, ensure_ascii=False), "[Perfil] gravado no cache", estimate_tokens(prompt_original, display)
        except Exception as e:
            err = f"Falha ao salvar sua função: {e}"
            return err, "llm", json.dumps({"erro": str(e)}, indent=2, ensure_ascii=False), "Erro ao salvar", estimate_tokens(prompt_original, err)

    key_for_cache, intent = normalize_prompt_for_cache(prompt_original)

    debug = {
        "attributes_usados": attrs,
        "intent": intent,
        "similarity_threshold": similarity_threshold,
        "estrategia": "EXACT→SEMANTIC" if use_exact_then_semantic else "SEMANTIC only",
        "ttl_ms_on_set": ttl_ms,
    }

    if intent == "identity:name":
        if person:
            ans = f"Olá, {person}! Seu nome é {person}."
        else:
            ans = "Não tenho seu nome salvo. Diga: “Meu nome é <seu nome>” ou preencha o campo Person."
        tokens_est = estimate_tokens(prompt_original, ans)
        debug["prompt_normalizado"] = KEY_NAME
        return ans, "llm", json.dumps(debug, indent=2, ensure_ascii=False), "[Identidade:Nome] direto (sem cache)", tokens_est

    strategies = None
    sim_thr = similarity_threshold
    if intent == "identity:role":
        strategies = [SearchStrategy.EXACT] if (SearchStrategy is not None) else None
        sim_thr = None  # desliga semântico

    rewritten_prompt = prompt_original
    domain_label = infer_domain(company, bu, None)
    if looks_ambiguous(prompt_original):
        rewritten_prompt = rewrite_with_domain(prompt_original, domain_label)
        if intent == "fact":
            key_for_cache = f"[FACT]\n{rewritten_prompt}"
    debug["prompt_normalizado"] = key_for_cache

    t0 = time.perf_counter()
    cached_answer = None
    if lang_cache:
        try:
            if strategies is not None:
                results = lang_cache.search(
                    prompt=key_for_cache,
                    attributes=attrs,
                    search_strategies=strategies,
                )
            else:
                results = lang_cache.search(
                    prompt=key_for_cache,
                    similarity_threshold=sim_thr,
                    attributes=attrs,
                )

            if (not results) or (not getattr(results, "data", None)):
                if use_exact_then_semantic and SearchStrategy is not None and strategies is None:
                    results = lang_cache.search(
                        prompt=key_for_cache,
                        attributes=attrs,
                        search_strategies=[SearchStrategy.EXACT, SearchStrategy.SEMANTIC],
                    )

            if results and getattr(results, "data", None):
                cached_answer = results.data[0].response
        except Exception as e:
            debug["cache_search_error"] = str(e)
    cache_latency = time.perf_counter() - t0

    if cached_answer:
        display_answer = depersonalize_safe(cached_answer, person if not should_personalize_name(prompt_original) else None)
        tokens_est = estimate_tokens(prompt_original, display_answer)
        latency_txt = f"[Cache Hit] busca: {cache_latency:.3f}s"
        return display_answer, "cache", json.dumps(debug, indent=2, ensure_ascii=False), latency_txt, tokens_est

    t1 = time.perf_counter()
    if intent == "identity:role":
        llm_answer_neutral = "Não tenho sua função ainda. Diga: “Minha função é <cargo>” para eu guardar."
    else:
        llm_answer_neutral = call_openai(
            rewritten_prompt if intent == "fact" else key_for_cache,
            person=None,
            company=company, bu=bu, role=None
        )
    llm_latency = time.perf_counter() - t1

    if lang_cache and intent != "identity:name":
        try:
            lang_cache.set(
                prompt=key_for_cache,
                response=llm_answer_neutral,
                attributes=attrs,
                ttl_millis=ttl_ms,
            )
        except Exception as e:
            debug["cache_set_error"] = str(e)

    display_answer = depersonalize_safe(llm_answer_neutral, person if not should_personalize_name(prompt_original) else None)
    tokens_est = estimate_tokens(prompt_original, display_answer)
    latency_txt = f"[Cache Miss] busca: {cache_latency:.3f}s, llm: {llm_latency:.3f}s"
    return display_answer, "llm", json.dumps(debug, indent=2, ensure_ascii=False), latency_txt, tokens_est

# ============== FLUSH helpers ==============

def parse_deleted_count(res: Any) -> Optional[int]:
    if hasattr(res, "deleted_entries_count"):
        return getattr(res, "deleted_entries_count", None)
    if isinstance(res, dict):
        return res.get("deleted_entries_count") or res.get("deleted") or res.get("deleted_count")
    return None

def flush_entries_with_attrs(attrs: Dict[str, str]) -> Tuple[str, str]:
    if not lang_cache:
        return "⚠️ LangCache não configurado; nenhum flush executado.", json.dumps({"attributes": attrs, "ok": False}, ensure_ascii=False, indent=2)
    try:
        res = lang_cache.delete_query(attributes=attrs)
        deleted = parse_deleted_count(res)
        msg = f"✅ Flush executado. Escopo={attrs}. Removidos={deleted if deleted is not None else '—'}"
        debug = {"attributes": attrs, "response": getattr(res, '__dict__', res)}
        return msg, json.dumps(debug, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"❌ Erro no flush: {e}", json.dumps({"attributes": attrs, "error": str(e)}, ensure_ascii=False, indent=2)

def handle_flush_scope(company: str, bu: str, person: str, isolation: str):
    attrs = build_attributes(company or "", bu or "", person or "", isolation)
    if not attrs:
        return ("⚠️ Selecione um nível de isolamento diferente de 'none' para poder limpar por escopo.",
                json.dumps({"attributes": attrs, "error": "attributes cannot be blank"}, ensure_ascii=False, indent=2))
    return flush_entries_with_attrs(attrs)

def handle_flush_both(
    a_company: str, a_bu: str, a_person: str,
    b_company: str, b_bu: str, b_person: str,
    isolation: str,
):
    attrs_a = build_attributes(a_company or "", a_bu or "", a_person or "", isolation)
    attrs_b = build_attributes(b_company or "", b_bu or "", b_person or "", isolation)

    msgs = []
    debugs = []

    if attrs_a:
        msg_a, dbg_a = flush_entries_with_attrs(attrs_a)
        msgs.append(msg_a)
        debugs.append(json.loads(dbg_a))
    else:
        msgs.append("⚠️ Escopo A: isolamento 'none' não pode ser limpo.")
        debugs.append({"attributes": attrs_a, "error": "attributes cannot be blank"})

    if attrs_b:
        msg_b, dbg_b = flush_entries_with_attrs(attrs_b)
        msgs.append(msg_b)
        debugs.append(json.loads(dbg_b))
    else:
        msgs.append("⚠️ Escopo B: isolamento 'none' não pode ser limpo.")
        debugs.append({"attributes": attrs_b, "error": "attributes cannot be blank"})

    final_msg = "<br/>".join(msgs)
    return final_msg, json.dumps({"A": debugs[0], "B": debugs[1]}, ensure_ascii=False, indent=2)

# ============== UI / KPIs ==============

DESCRICAO_LONGA = """
- O LangCache guarda respostas neutras do LLM e as reutiliza por escopo (Company/BU/Person).
- A UI abaixo permite comparar dois cenários em paralelo (A x B).
- Pedidos ambíguos são reescritos automaticamente para o domínio do usuário.
- É possível limpar entradas do cache por escopo (A, B ou ambos) — o índice NUNCA é apagado.
"""

def format_currency(v: float, currency: str = "USD") -> str:
    return f"{currency} ${v:,.4f}"

def atualizar_kpis(state: dict) -> Dict[str, str]:
    hits = state.get("hits", 0)
    misses = state.get("misses", 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total else 0.0
    saved_tokens = state.get("saved_tokens", 0)
    saved_usd = state.get("saved_usd", 0.0)
    return {
        "hits": f"{hits}",
        "misses": f"{misses}",
        "rate": f"{hit_rate:.1f}%",
        "tokens": f"{saved_tokens}",
        "usd": format_currency(saved_usd),
    }

def calc_savings(tokens_est: int, price_in: float, price_out: float, frac_in: float = 0.5) -> float:
    tokens_in = int(tokens_est * frac_in)
    tokens_out = max(0, tokens_est - tokens_in)
    return (tokens_in / 1000.0) * price_in + (tokens_out / 1000.0) * price_out

# ===================== CSS (VISUAL APENAS) =====================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
  --redis-red:#D82C20; --ink:#0b1220; --soft:#475569; --muted:#64748b;
  --line:#e5e7eb; --bg:#f6f7f9; --white:#ffffff; --radius:14px;
  --success:#10b981; --warning:#f59e0b;
}

* { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }
body, #app-root { background: var(--bg); }

/* HEADER */
.app-header {
  position: sticky; top: 0; z-index: 50;
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:12px 16px; background: var(--redis-red); color:#fff;
  box-shadow: 0 2px 8px rgba(0,0,0,.18);
}
.app-header .brand { display:flex; align-items:center; gap:12px; }
.app-header .brand img { height:22px; display:block; }
.app-header .title { font-family: 'Space Grotesk', Inter, sans-serif; font-size:18px; font-weight:700; letter-spacing:.2px; }
.app-header .links { display:flex; gap:8px; }
.app-header .links a {
  display:inline-flex; align-items:center; gap:8px; color:#fff; text-decoration:none;
  border:1px solid rgba(255,255,255,.35); padding:7px 12px; border-radius:999px; font-weight:600; font-size:12px;
  transition: background .15s ease, transform .15s ease;
}
.app-header .links a:hover { background: rgba(255,255,255,.14); transform: translateY(-1px); }

/* HEADINGS */
.h1 {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:26px; font-weight:700; color:var(--ink); margin:16px 16px 6px;
}
.h2 {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:16px; font-weight:600; color:var(--soft); margin:0 16px 14px;
}

/* Config box (clean) */
.config-card {
  margin: 10px 16px 14px; padding:12px;
  background: var(--white);
  border:1px solid var(--line); border-radius: var(--radius);
}

/* KPIs */
.kpi-row { display:flex; gap:12px; margin: 0 16px 16px; flex-wrap: wrap; }
.kpi {
  flex:1; min-width: 140px; background: var(--white); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; transition: transform .2s ease, box-shadow .2s ease;
}
.kpi:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.08); }
.kpi .kpi-num {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:24px; font-weight:700; color:var(--ink); line-height:1.1;
}
.kpi .kpi-label {
  font-size:11px; color:var(--muted); margin-top:6px;
  text-transform:uppercase; letter-spacing:.8px; font-weight:600;
}
.kpi-accent { border-color: var(--redis-red); border-width: 2px; }
.kpi-accent .kpi-num { color: var(--redis-red); }

/* Cenários lado a lado */
.scenarios { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 10px 16px; }
@media (max-width: 1024px) { .scenarios { grid-template-columns: 1fr; } }

.card {
  background: var(--white); border:2px solid var(--line); border-radius: var(--radius);
  padding:16px; transition: border-color .2s ease;
}
.card:hover { border-color: var(--redis-red); }
.card .card-title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size:18px; font-weight:700; color:var(--ink); margin-bottom:12px;
  display: flex; align-items: center; gap: 8px;
}

/* Source badges */
.source-badge {
  display: inline-block; padding: 4px 10px; border-radius: 6px;
  font-size: 11px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .5px;
}
.source-cache { background: #d1fae5; color: #065f46; }
.source-llm { background: #fef3c7; color: #92400e; }

/* History */
.dataframe { background: var(--white); border:1px solid var(--line); border-radius: var(--radius); }
.dataframe thead tr th { font-size:12px; font-weight:600; }
.dataframe tbody tr td { font-size:12px; }

/* Buttons */
button.primary, .gr-button-primary {
  background: var(--redis-red) !important; border-color: var(--redis-red) !important; color:#fff !important;
  font-weight: 600 !important; transition: all .2s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
  background: #c02518 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(216,44,32,.3) !important;
}

/* Secondary buttons */
.secondary-btn {
  background: var(--white) !important; border: 1px solid var(--line) !important;
  color: var(--soft) !important; font-weight: 600 !important;
}

/* --- HERO (título + subtítulo) --- */
.hero {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  margin: 16px;
  padding: 16px 18px;
}

.hero-title {
  font-family: 'Space Grotesk', Inter, sans-serif;
  font-size: 26px;
  font-weight: 700;
  color: var(--ink);      /* força contraste alto */
  letter-spacing: .2px;
  margin: 0 0 8px 0;
}

.hero-sub {
  font-size: 14px;
  color: var(--soft);
  line-height: 1.6;
  margin: 0;
}

/* Se em algum tema o título estiver “preto no preto”, garante contraste: */
.h1 { color: var(--ink) !important; background: transparent !important; }
"""

# ============== APP (layout preservado A/B) ==============
with gr.Blocks(title="Redis LangCache — Demo PT-BR", css=CUSTOM_CSS, elem_id="app-root") as demo:
    st = gr.State({"hits": 0, "misses": 0, "saved_tokens": 0, "saved_usd": 0.0, "history": []})

    # Header com logo + links
    gr.HTML("""
      <div class="app-header">
        <div class="brand">
          <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Redis_logo.svg/2560px-Redis_logo.svg.png" alt="Redis">
          <div class="title">Redis LangCache — Demo PT-BR - SemVer: v2.0.4 - PR GitHub: gacerioni</div>
        </div>
        <div class="links">
          <a href="https://www.linkedin.com/in/gabrielcerioni/" target="_blank" rel="noopener">💼 LinkedIn do Gabs</a>
          <a href="https://redis.io/" target="_blank" rel="noopener">🔗 Redis</a>
        </div>
      </div>
    """)

    # Título + Subtítulo (claros e legíveis)
    gr.HTML("""
      <div class="hero">
        <div class="hero-title">Cache semântico por Empresa / Business Unit / Pessoa</div>
        <p class="hero-sub">
          Esta demo mostra como o LangCache guarda respostas neutras dos LLMs e
          as reutiliza por escopo (empresa/BU/pessoa).<br/>
          Você pode limpar o cache por escopo diretamente na interface.<br/>
          Lembrando que o cache pode ser para TODOS, pra uma BU, ou pra uma pessoa específica!
          
        </p>
      </div>
    """)

    # Configurações
    with gr.Group(elem_classes=["config-card"]):
        with gr.Row():
            isolation_global = gr.Radio(
                choices=["company+bu+person", "company+bu", "company", "none"],
                value="company+bu+person",
                label="Isolamento",
            )
            threshold_global = gr.Slider(label="Similaridade (−1 = off)", minimum=-1.0, maximum=1.0, value=0.85, step=0.05)
            exact_sem_global = gr.Checkbox(label="Fallback EXACT→SEMANTIC", value=True)
            ttl_global = gr.Number(label="TTL (s)", value=0, precision=0)
        with gr.Row():
            price_in = gr.Number(label="Preço 1k tokens (Entrada)", value=0.15)
            price_out = gr.Number(label="Preço 1k tokens (Saída)", value=0.60)
            frac_in = gr.Slider(label="% Entrada", minimum=0.1, maximum=0.9, value=0.5, step=0.05)
            currency = gr.Dropdown(label="Moeda", choices=["USD"], value="USD")
        with gr.Accordion("Como funciona (detalhes)", open=False):
            gr.Markdown(DESCRICAO_LONGA)

    # KPIs
    with gr.Row(elem_classes=["kpi-row"]):
        kpi_hits = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Hits</div></div>")
        kpi_misses = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Misses</div></div>")
        kpi_rate = gr.HTML("<div class='kpi'><div class='kpi-num'>0.0%</div><div class='kpi-label'>Hit Rate</div></div>")
        kpi_tokens = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Tokens</div></div>")
        kpi_savings = gr.HTML("<div class='kpi kpi-accent'><div class='kpi-num'>USD $0.0000</div><div class='kpi-label'>Economia</div></div>")

    # Cenários A e B (lado a lado)
    with gr.Row(elem_classes=["scenarios"]):
        # --- Cenário A ---
        with gr.Column(elem_classes=["card"]):
            gr.Markdown("<div class='card-title'>Chat 🅰️</div>")
            with gr.Row():
                a_company = gr.Textbox(label="Company", value="RedisLabs")
                a_bu = gr.Textbox(label="Business Unit", value="Engenharia-de-Software")
                a_person = gr.Textbox(label="Person", value="Gabriel")
            a_prompt = gr.Textbox(label="Pergunta", placeholder="Pergunte algo…", lines=3)
            a_btn = gr.Button("Perguntar (A)", variant="primary")
            a_answer = gr.Textbox(label="Resposta", lines=6, interactive=False)
            with gr.Row():
                a_source = gr.HTML(label="Origem")
                a_latency = gr.Label(label="Latência")
            with gr.Accordion("🔍 Debug Info", open=False):
                a_debug = gr.Code(label="Debug JSON", language="json")
            with gr.Accordion("🧹 Gerenciar Cache", open=False):
                a_flush_btn = gr.Button("Limpar Cache (Escopo A)", variant="secondary")
                a_flush_status = gr.HTML()
                with gr.Accordion("Debug do Flush", open=False):
                    a_flush_debug = gr.Code(language="json")

        # --- Cenário B ---
        with gr.Column(elem_classes=["card"]):
            gr.Markdown("<div class='card-title'>Chat 🅱️</div>")
            with gr.Row():
                b_company = gr.Textbox(label="Company", value="RedisLabs")
                b_bu = gr.Textbox(label="Business Unit", value="Financeiro")
                b_person = gr.Textbox(label="Person", value="Diego")
            b_prompt = gr.Textbox(label="Pergunta", placeholder="Pergunte algo…", lines=3)
            b_btn = gr.Button("Perguntar (B)", variant="primary")
            b_answer = gr.Textbox(label="Resposta", lines=6, interactive=False)
            with gr.Row():
                b_source = gr.HTML(label="Origem")
                b_latency = gr.Label(label="Latência")
            with gr.Accordion("🔍 Debug Info", open=False):
                b_debug = gr.Code(label="Debug JSON", language="json")
            with gr.Accordion("🧹 Gerenciar Cache", open=False):
                b_flush_btn = gr.Button("Limpar Cache (Escopo B)", variant="secondary")
                b_flush_status = gr.HTML()
                with gr.Accordion("Debug do Flush", open=False):
                    b_flush_debug = gr.Code(language="json")

    # Copy A → B button
    with gr.Row():
        gr.HTML("<div style='flex:1'></div>")
        copy_btn = gr.Button("📋 Copiar A → B", variant="secondary", size="sm")
        gr.HTML("<div style='flex:1'></div>")

    # Flush ambos
    with gr.Group():
        gr.Markdown("### 🧹 Limpeza Combinada (A + B)")
        with gr.Accordion("Opções Avançadas", open=False):
            flush_both_btn = gr.Button("🧹 Limpar Ambos (A+B)", variant="secondary")
            flush_both_status = gr.HTML()
            with gr.Accordion("Debug do Flush", open=False):
                flush_both_debug = gr.Code(language="json")

    # Histórico
    gr.Markdown("### 📊 Histórico de Consultas (últimos 50)")
    history_table = gr.Dataframe(
        headers=["Hora", "Cenário", "Company", "BU", "Person", "Fonte", "Latência", "Tokens (est.)", "Economia", "Prompt"],
        datatype=["str", "str", "str", "str", "str", "str", "str", "number", "str", "str"],
        interactive=False,
        wrap=True,
        row_count=(0, "dynamic"),
        col_count=(10, "fixed"),
    )

    # ==== Eventos (mesmos do seu código) ====
    def handle_submit(
        side: str,
        company: str,
        bu: str,
        person: str,
        prompt: str,
        isolation: str,
        threshold: float,
        use_exact_sem: bool,
        ttl_seconds: int,
        price_in: float,
        price_out: float,
        frac_in: float,
        currency: str,
        state: dict,
    ):
        sim = None if threshold < 0 else threshold
        ttl_ms = None if ttl_seconds <= 0 else ttl_seconds * 1000

        company = (company or "Acme").strip()
        bu = (bu or "BU-1").strip()
        person = (person or "user-1").strip()

        answer, source, debug_json, latency, tokens_est = search_and_answer(
            company=company,
            bu=bu,
            person=person,
            prompt_original=prompt,
            isolation=isolation,
            similarity_threshold=sim,
            use_exact_then_semantic=use_exact_sem,
            ttl_ms=ttl_ms,
        )

        if source == "cache":
            state["hits"] = state.get("hits", 0) + 1
            saved = calc_savings(tokens_est, price_in, price_out, frac_in)
            state["saved_usd"] = state.get("saved_usd", 0.0) + saved
            state["saved_tokens"] = state.get("saved_tokens", 0) + tokens_est
            saved_str = f"Economia: {format_currency(saved, currency)}"
        else:
            state["misses"] = state.get("misses", 0) + 1
            saved_str = "Sem economia neste turno"

        history = state.setdefault("history", [])
        history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lado": side,
            "company": company,
            "business_unit": bu,
            "person": person,
            "fonte": source,
            "latencia": latency,
            "tokens_est": tokens_est,
            "economia_turno": (calc_savings(tokens_est, price_in, price_out, frac_in) if source == "cache" else 0.0),
            "prompt": prompt[:60] + ("…" if len(prompt) > 60 else ""),
        })
        if len(history) > 200:
            del history[: len(history) - 200]

        k = atualizar_kpis(state)
        table_rows = [
            [h["timestamp"], h["lado"], h["company"], h["business_unit"], h["person"], h["fonte"], h["latencia"], h["tokens_est"], f"{format_currency(h['economia_turno'])}", h["prompt"]]
            for h in reversed(history[-50:])
        ]

        kpi_hits_html = f"<div class='kpi'><div class='kpi-num'>{k['hits']}</div><div class='kpi-label'>Hits</div></div>"
        kpi_miss_html = f"<div class='kpi'><div class='kpi-num'>{k['misses']}</div><div class='kpi-label'>Misses</div></div>"
        kpi_rate_html = f"<div class='kpi'><div class='kpi-num'>{k['rate']}</div><div class='kpi-label'>Hit Rate</div></div>"
        kpi_tok_html = f"<div class='kpi'><div class='kpi-num'>{k['tokens']}</div><div class='kpi-label'>Tokens</div></div>"
        kpi_usd_html = f"<div class='kpi kpi-accent'><div class='kpi-num'>{k['usd']}</div><div class='kpi-label'>Economia</div></div>"

        # Create visual badge for source
        source_badge = (
            f"<span class='source-badge source-{source}'>"
            f"{'✓ CACHE HIT' if source == 'cache' else '⚡ LLM CALL'}"
            f"</span>"
        )

        return (
            answer,
            source_badge,
            debug_json,
            f"{latency} · {saved_str}",
            gr.update(value=kpi_hits_html),
            gr.update(value=kpi_miss_html),
            gr.update(value=kpi_rate_html),
            gr.update(value=kpi_tok_html),
            gr.update(value=kpi_usd_html),
            table_rows,
            state,
        )

    # Scenario A submit handlers
    a_btn.click(
        fn=handle_submit,
        inputs=[
            gr.State("A"),
            a_company, a_bu, a_person, a_prompt,
            isolation_global, threshold_global, exact_sem_global, ttl_global,
            price_in, price_out, frac_in, currency,
            st,
        ],
        outputs=[
            a_answer, a_source, a_debug, a_latency,
            kpi_hits, kpi_misses, kpi_rate, kpi_tokens, kpi_savings,
            history_table,
            st,
        ],
    )

    # Enable Enter key submission for A
    a_prompt.submit(
        fn=handle_submit,
        inputs=[
            gr.State("A"),
            a_company, a_bu, a_person, a_prompt,
            isolation_global, threshold_global, exact_sem_global, ttl_global,
            price_in, price_out, frac_in, currency,
            st,
        ],
        outputs=[
            a_answer, a_source, a_debug, a_latency,
            kpi_hits, kpi_misses, kpi_rate, kpi_tokens, kpi_savings,
            history_table,
            st,
        ],
    )

    # Scenario B submit handlers
    b_btn.click(
        fn=handle_submit,
        inputs=[
            gr.State("B"),
            b_company, b_bu, b_person, b_prompt,
            isolation_global, threshold_global, exact_sem_global, ttl_global,
            price_in, price_out, frac_in, currency,
            st,
        ],
        outputs=[
            b_answer, b_source, b_debug, b_latency,
            kpi_hits, kpi_misses, kpi_rate, kpi_tokens, kpi_savings,
            history_table,
            st,
        ],
    )

    # Enable Enter key submission for B
    b_prompt.submit(
        fn=handle_submit,
        inputs=[
            gr.State("B"),
            b_company, b_bu, b_person, b_prompt,
            isolation_global, threshold_global, exact_sem_global, ttl_global,
            price_in, price_out, frac_in, currency,
            st,
        ],
        outputs=[
            b_answer, b_source, b_debug, b_latency,
            kpi_hits, kpi_misses, kpi_rate, kpi_tokens, kpi_savings,
            history_table,
            st,
        ],
    )

    # Copy A → B functionality
    def copy_a_to_b(company, bu, person, prompt):
        return company, bu, person, prompt

    copy_btn.click(
        fn=copy_a_to_b,
        inputs=[a_company, a_bu, a_person, a_prompt],
        outputs=[b_company, b_bu, b_person, b_prompt]
    )

    a_flush_btn.click(
        fn=handle_flush_scope,
        inputs=[a_company, a_bu, a_person, isolation_global],
        outputs=[a_flush_status, a_flush_debug],
    )

    b_flush_btn.click(
        fn=handle_flush_scope,
        inputs=[b_company, b_bu, b_person, isolation_global],
        outputs=[b_flush_status, b_flush_debug],
    )

    flush_both_btn.click(
        fn=handle_flush_both,
        inputs=[a_company, a_bu, a_person, b_company, b_bu, b_person, isolation_global],
        outputs=[flush_both_status, flush_both_debug],
    )

if __name__ == "__main__":
    if lang_cache:
        with lang_cache:
            demo.launch()
    else:
        demo.launch()