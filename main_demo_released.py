#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis LangCache — Demo PT-BR (Contexto forte por BU/Empresa/Cargo)
- Nome (IDENTITY:NAME): sem cache (usa atributo person)
- Cargo (IDENTITY:ROLE): EXACT ONLY (chave constante) + SET suportado
- Fatos: Semântico (threshold) + fallback EXACT→SEMANTIC
- Resposta no cache é NEUTRA (sem nome/cargo); personalização só na exibição
- Contexto forte: reescreve prompts AMBÍGUOS com "(no contexto de ...)" + system rígido "não cite outros sentidos"
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

# termos ambíguos -> reforçar contexto
AMBIGUOUS_TERMS = [
    r"\bc[eé]lula\b",      # célula (biologia vs computação/planilhas)
    r"\bbanco\b",          # banco (financeiro vs margem de rio)
    r"\brede\b",           # rede (computadores vs hospital/saúde pública)
    r"\bmodelo\b",         # modelo (ML vs negócio)
    r"\bpipeline\b",       # pipeline (dados/software vs oleoduto)
]

def infer_domain(company: str, bu: str, role: Optional[str]) -> str:
    """
    Retorna rótulo curto de domínio para instrução: 'saúde', 'software', 'dados', 'finanças', 'turismo', etc.
    """
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
    # fallback neutro
    return "geral da área do usuário"

def looks_ambiguous(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(re.search(pat, p, flags=re.IGNORECASE) for pat in AMBIGUOUS_TERMS)

def rewrite_with_domain(prompt: str, domain_label: str) -> str:
    """
    Se a pergunta for ambígua, reforça explicitamente o contexto.
    Ex.: "O que é uma célula? (no contexto de saúde)"
    """
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
        "Se a pergunta for ambígua (ex.: 'célula', 'rede', 'banco'), "
        f"RESPONDA APENAS no sentido de {domain} e NÃO mencione outros significados."
    )

    # few-shot mínimo para ancorar comportamento
    examples = [
        {"role": "user", "content": "O que é uma célula? (no contexto de saúde)"},
        {"role": "assistant", "content": "Uma célula é a menor unidade estrutural e funcional dos seres vivos."},
        {"role": "user", "content": "O que é uma célula? (no contexto de engenharia de software)"},
        {"role": "assistant", "content": "Em computação, célula costuma se referir a uma unidade em uma tabela/planilha ou a um componente isolado de execução."},
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

    # Nome: não usa cache
    if intent == "identity:name":
        if person:
            ans = f"Olá, {person}! Seu nome é {person}."
        else:
            ans = "Não tenho seu nome salvo. Diga: “Meu nome é <seu nome>” ou preencha o campo Person."
        tokens_est = estimate_tokens(prompt_original, ans)
        debug["prompt_normalizado"] = KEY_NAME
        return ans, "llm", json.dumps(debug, indent=2, ensure_ascii=False), "[Identidade:Nome] direto (sem cache)", tokens_est

    # Estratégia por intenção
    strategies = None
    sim_thr = similarity_threshold
    if intent == "identity:role":
        strategies = [SearchStrategy.EXACT] if (SearchStrategy is not None) else None
        sim_thr = None  # desliga semântico

    # REESCRITA de prompt se for ambíguo (para o LLM e para o cache!)
    # - Mantém isolamento adicional por atributos; a chave de cache leva a versão reescrita.
    rewritten_prompt = prompt_original
    domain_label = infer_domain(company, bu, None)
    if looks_ambiguous(prompt_original):
        rewritten_prompt = rewrite_with_domain(prompt_original, domain_label)
        if intent == "fact":
            key_for_cache = f"[FACT]\n{rewritten_prompt}"
    debug["prompt_normalizado"] = key_for_cache

    # ====== Cache
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

    # ====== LLM
    t1 = time.perf_counter()
    if intent == "identity:role":
        llm_answer_neutral = "Não tenho sua função ainda. Diga: “Minha função é <cargo>” para eu guardar."
    else:
        # usar prompt reescrito quando ambíguo
        llm_answer_neutral = call_openai(
            rewritten_prompt if intent == "fact" else key_for_cache,
            person=None,  # não usar nome em fatos
            company=company, bu=bu, role=None
        )
    llm_latency = time.perf_counter() - t1

    # ====== Grava no cache (exceto nome)
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

# ============== UI / KPIs ==============

DESCRICAO_LONGA = """
- Isolamento por atributos: company, business_unit, person.
- Nome: sem cache; Cargo: EXACT ONLY.
- Fatos: SEMANTIC + fallback EXACT→SEMANTIC.
- Desambiguação forte: prompts ambíguos são reescritos com “(no contexto de …)”.
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

    return (
        answer,
        json.dumps({"fonte": source}, ensure_ascii=False),
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

# ============== Custom CSS ==============
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
:root {
  --redis-red: #D82C20; --text-strong: #111; --text-soft: #444;
  --bg-card: #fff; --bg-soft: #f7f7f7; --border-soft: #e6e6e6; --radius: 14px;
}
* { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; }
.kpi-row { display:flex; gap:12px; }
.kpi { flex:1; background:var(--bg-card); border:1px solid var(--border-soft); border-radius:var(--radius);
       padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.kpi .kpi-num { font-size:22px; font-weight:700; color:var(--text-strong); line-height:1.1; }
.kpi .kpi-label { font-size:12px; color:var(--text-soft); margin-top:4px; text-transform:uppercase; letter-spacing:.6px; }
.kpi-accent { border-color:var(--redis-red); box-shadow:0 2px 8px rgba(216,44,32,.12); }
button.primary, .gr-button-primary { background:var(--redis-red) !important; border-color:var(--redis-red) !important; color:#fff !important; }
"""

# ============== App ==============
with gr.Blocks(title="Redis LangCache — Demo PT-BR", css=CUSTOM_CSS) as demo:
    st = gr.State({"hits": 0, "misses": 0, "saved_tokens": 0, "saved_usd": 0.0, "history": []})

    gr.Markdown("# Redis LangCache — Demo PT-BR")
    gr.Markdown("### Cache semântico isolado por Company / BU / Person · Reaproveitamento + Economia")

    with gr.Accordion("Como funciona (detalhes)", open=False):
        gr.Markdown(DESCRICAO_LONGA)

    with gr.Accordion("Configurações", open=True):
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

    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Cenário A")
            a_company = gr.Textbox(label="Company", value="ClinicaMedicaABC")
            a_bu = gr.Textbox(label="Business Unit", value="Saude-Medicos")
            a_person = gr.Textbox(label="Person", value="Gabriel")
            a_prompt = gr.Textbox(label="Pergunta", placeholder="Pergunte algo…", lines=3)
            a_btn = gr.Button("Perguntar (A)", variant="primary")
            a_answer = gr.Textbox(label="Resposta", lines=6)
            with gr.Row():
                a_source = gr.Label(label="Origem")
                a_latency = gr.Label(label="Latência")
            a_debug = gr.Code(label="Debug")

        with gr.Column():
            gr.Markdown("#### Cenário B")
            b_company = gr.Textbox(label="Company", value="TechNova")
            b_bu = gr.Textbox(label="Business Unit", value="Engenharia-de-Software")
            b_person = gr.Textbox(label="Person", value="Diego")
            b_prompt = gr.Textbox(label="Pergunta", placeholder="Pergunte algo…", lines=3)
            b_btn = gr.Button("Perguntar (B)", variant="primary")
            b_answer = gr.Textbox(label="Resposta", lines=6)
            with gr.Row():
                b_source = gr.Label(label="Origem")
                b_latency = gr.Label(label="Latência")
            b_debug = gr.Code(label="Debug")

    gr.Markdown("### Indicadores")
    with gr.Row(elem_classes=["kpi-row"]):
        kpi_hits = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Hits</div></div>")
        kpi_misses = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Misses</div></div>")
        kpi_rate = gr.HTML("<div class='kpi'><div class='kpi-num'>0.0%</div><div class='kpi-label'>Hit Rate</div></div>")
        kpi_tokens = gr.HTML("<div class='kpi'><div class='kpi-num'>0</div><div class='kpi-label'>Tokens</div></div>")
        kpi_savings = gr.HTML("<div class='kpi kpi-accent'><div class='kpi-num'>USD $0.0000</div><div class='kpi-label'>Economia</div></div>")

    gr.Markdown("### Histórico (últimos 50)")
    history_table = gr.Dataframe(
        headers=["Hora", "Cenário", "Company", "BU", "Person", "Fonte", "Latência", "Tokens (est.)", "Economia", "Prompt"],
        datatype=["str", "str", "str", "str", "str", "str", "str", "number", "str", "str"],
        interactive=False,
        wrap=True,
        row_count=(0, "dynamic"),
        col_count=(10, "fixed"),
    )

    # Eventos
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

if __name__ == "__main__":
    if lang_cache:
        with lang_cache:
            demo.launch()
    else:
        demo.launch()