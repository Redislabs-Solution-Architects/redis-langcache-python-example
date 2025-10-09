#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from langcache import LangCache
from langcache.models import SearchStrategy

# ============== Env & Clients ==============
load_dotenv()

# LangCache
LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY") or os.getenv("LANGCACHE_SERVICE_KEY")
LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
LANGCACHE_BASE_URL = os.getenv("LANGCACHE_BASE_URL", "https://gcp-us-east4.langcache.redis.io")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# LangCache (optional fail-soft if not configured)
lang_cache: Optional[LangCache] = None
if LANGCACHE_API_KEY and LANGCACHE_CACHE_ID:
    lang_cache = LangCache(
        server_url=LANGCACHE_BASE_URL,
        cache_id=LANGCACHE_CACHE_ID,
        api_key=LANGCACHE_API_KEY,
    )
else:
    print("[WARN] LangCache not fully configured; UI will still run, but caching will be disabled.")

# ============== Core Helpers ==============
def call_openai(prompt: str, temperature: float = 0.2) -> str:
    """Simple LLM call."""
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Be concise."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def build_attributes(tenant_id: str, session_id: str, isolate_mode: str) -> Dict[str, str]:
    """
    isolate_mode:
      - "tenant": cache scoped by tenant only
      - "tenant+session": cache scoped by tenant AND session
      - "none": no attributes (shared across everyone; for contrast)
    """
    if isolate_mode == "tenant":
        return {"tenant": tenant_id}
    if isolate_mode == "tenant+session":
        return {"tenant": tenant_id, "session": session_id}
    return {}

def search_and_answer(
    tenant_id: str,
    session_id: str,
    prompt: str,
    isolate_mode: str,
    similarity_threshold: Optional[float],
    use_exact_then_semantic: bool,
    ttl_ms: Optional[int],
) -> Tuple[str, str, str, str]:
    """
    Returns (answer, source, debug_json, latency_text)
    """
    if not prompt.strip():
        return "", "—", "{}", "Idle"

    attrs = build_attributes(tenant_id, session_id, isolate_mode)
    debug = {
        "attributes_used": attrs,
        "similarity_threshold": similarity_threshold,
        "strategy": "EXACT→SEMANTIC" if use_exact_then_semantic else "SEMANTIC only",
        "ttl_ms_on_set": ttl_ms,
    }

    # 1) Try cache
    t0 = time.perf_counter()
    cached_answer = None
    cache_latency = None

    if lang_cache:
        try:
            # First pass: semantic with threshold (optional)
            results = lang_cache.search(
                prompt=prompt,
                similarity_threshold=similarity_threshold,
                attributes=attrs,
            )

            # Optional fallback: exact → semantic combo
            if (not results) or (not getattr(results, "data", None)):
                if use_exact_then_semantic:
                    results = lang_cache.search(
                        prompt=prompt,
                        attributes=attrs,
                        search_strategies=[SearchStrategy.EXACT, SearchStrategy.SEMANTIC],
                    )

            if results and getattr(results, "data", None):
                cached_answer = results.data[0].response
        except Exception as e:
            cached_answer = None
            debug["cache_search_error"] = str(e)

    cache_latency = time.perf_counter() - t0

    # 2) If cache hit, return
    if cached_answer:
        latency_txt = f"[Cache Hit] search: {cache_latency:.3f}s"
        return cached_answer, "cache", json.dumps(debug, indent=2), latency_txt

    # 3) Cache miss → LLM
    t1 = time.perf_counter()
    llm_answer = call_openai(prompt)
    llm_latency = time.perf_counter() - t1

    # 4) Store answer with same attributes so it won’t leak
    if lang_cache:
        try:
            lang_cache.set(
                prompt=prompt,
                response=llm_answer,
                attributes=attrs,
                ttl_millis=ttl_ms,
            )
        except Exception as e:
            debug["cache_set_error"] = str(e)

    latency_txt = f"[Cache Miss] search: {cache_latency:.3f}s, llm: {llm_latency:.3f}s"
    return llm_answer, "llm", json.dumps(debug, indent=2), latency_txt

# ============== UI ==============
DESCRIPTION = """
# LangCache Isolation Demo

Use **tenant** and **session** attributes to prevent cache from leaking between customers.

- **Isolation Mode**:
  - **tenant+session**: strictest isolation (same tenant, different sessions don’t share)
  - **tenant**: shared within tenant but isolated across tenants
  - **none**: shared across everyone (for contrast / anti-pattern)

Try entering **the same prompt** on both sides with different tenants or sessions and observe whether results come from the cache.
"""

def handle_submit(
    tenant_id: str,
    session_id: str,
    prompt: str,
    isolate_mode: str,
    threshold: float,
    use_exact_sem: bool,
    ttl_seconds: int,
):
    sim = None if threshold < 0 else threshold
    ttl_ms = None if ttl_seconds <= 0 else ttl_seconds * 1000
    answer, source, debug_json, lat = search_and_answer(
        tenant_id=tenant_id.strip() or "A",
        session_id=session_id.strip() or "s1",
        prompt=prompt,
        isolate_mode=isolate_mode,
        similarity_threshold=sim,
        use_exact_then_semantic=use_exact_sem,
        ttl_ms=ttl_ms,
    )
    return answer, source, debug_json, lat

with gr.Blocks(title="LangCache Isolation Demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer A")
            a_tenant = gr.Textbox(label="Tenant", value="tenant-A")
            a_session = gr.Textbox(label="Session", value="sess-A1")
            a_prompt = gr.Textbox(label="Prompt", placeholder="Ask something...", lines=3)

            a_isolation = gr.Radio(
                choices=["tenant+session", "tenant", "none"],
                value="tenant+session",
                label="Isolation Mode",
            )
            a_threshold = gr.Slider(
                label="Similarity Threshold (use -1 to disable)",
                minimum=-1.0, maximum=1.0, value=0.7, step=0.05,
            )
            a_exact_sem = gr.Checkbox(
                label="Fallback to EXACT → SEMANTIC if no hit",
                value=True,
            )
            a_ttl = gr.Number(label="TTL on set (seconds, 0 = default cache TTL)", value=0, precision=0)

            a_btn = gr.Button("Ask (Customer A)")
            a_answer = gr.Textbox(label="Answer", lines=6)
            a_source = gr.Label(label="Answer Source (cache or llm)")
            a_debug = gr.Code(label="Debug (attributes & strategy)")
            a_latency = gr.Label(label="Latency")

        with gr.Column():
            gr.Markdown("### Customer B")
            b_tenant = gr.Textbox(label="Tenant", value="tenant-B")
            b_session = gr.Textbox(label="Session", value="sess-B1")
            b_prompt = gr.Textbox(label="Prompt", placeholder="Ask something...", lines=3)

            b_isolation = gr.Radio(
                choices=["tenant+session", "tenant", "none"],
                value="tenant+session",
                label="Isolation Mode",
            )
            b_threshold = gr.Slider(
                label="Similarity Threshold (use -1 to disable)",
                minimum=-1.0, maximum=1.0, value=0.7, step=0.05,
            )
            b_exact_sem = gr.Checkbox(
                label="Fallback to EXACT → SEMANTIC if no hit",
                value=True,
            )
            b_ttl = gr.Number(label="TTL on set (seconds, 0 = default cache TTL)", value=0, precision=0)

            b_btn = gr.Button("Ask (Customer B)")
            b_answer = gr.Textbox(label="Answer", lines=6)
            b_source = gr.Label(label="Answer Source (cache or llm)")
            b_debug = gr.Code(label="Debug (attributes & strategy)")
            b_latency = gr.Label(label="Latency")

    # Wire events
    a_btn.click(
        handle_submit,
        inputs=[a_tenant, a_session, a_prompt, a_isolation, a_threshold, a_exact_sem, a_ttl],
        outputs=[a_answer, a_source, a_debug, a_latency],
    )

    b_btn.click(
        handle_submit,
        inputs=[b_tenant, b_session, b_prompt, b_isolation, b_threshold, b_exact_sem, b_ttl],
        outputs=[b_answer, b_source, b_debug, b_latency],
    )

if __name__ == "__main__":
    # Use a context manager only if LangCache is configured to avoid errors on exit
    if lang_cache:
        with lang_cache:
            demo.launch()
    else:
        demo.launch()