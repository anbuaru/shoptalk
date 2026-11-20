"""HTML / Markdown rendering for the Phase 8 UI. Pure functions (no Gradio), so they are unit-tested.

Everything that comes from the catalog or the LLM is HTML-escaped before it is shown.
"""
from __future__ import annotations

import html
import re
from typing import Optional
from urllib.parse import quote

ID_RE = re.compile(r"\[ID:\s*([A-Za-z0-9_\-]+)\]")

CSS = """
.st-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px;}
.st-card {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px;
          background:var(--background-fill-secondary); display:flex; flex-direction:column; gap:6px;}
.st-img {width:100%; aspect-ratio:1/1; object-fit:contain; border-radius:6px; background:#fff;}
.st-noimg {width:100%; aspect-ratio:1/1; border-radius:6px; display:flex; align-items:center;
           justify-content:center; font-size:42px; background:var(--background-fill-primary);}
.st-title {font-weight:600; font-size:0.92em; line-height:1.25; display:-webkit-box;
           -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;}
.st-meta {font-size:0.8em; opacity:0.8;}
.st-price {font-weight:700;}
.st-id {font-family:monospace; font-size:0.78em; opacity:0.75;}
.st-chip {display:inline-block; font-size:0.72em; padding:1px 6px; border-radius:8px; margin-right:4px;
          background:var(--color-accent-soft);}
.st-link {margin-top:auto; font-size:0.85em; font-weight:600;}
.st-note {font-size:0.85em; padding:8px 10px; border-radius:8px; background:var(--background-fill-secondary);
          margin-bottom:10px;}
"""

EMPTY_PRODUCTS = "<div class='st-note'>Product recommendations will appear here.</div>"


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def click_href(link_base: str, request_id: Optional[str], product_id: str, rank: int,
               session_id: Optional[str] = None, fallback: Optional[str] = None) -> str:
    """
    Link used on a product card. It points at the UI's /go route, which logs the click through the
    API's /click endpoint and then redirects to the product page. Without a request_id there is
    nothing to log against, so the direct product link is used.
    """
    if not request_id:
        return fallback or "#"
    href = f"{link_base.rstrip('/')}/go/{quote(request_id)}/{quote(product_id)}?rank={rank}"
    if session_id:
        href += f"&session_id={quote(session_id)}"
    return href


def format_price(p: dict) -> str:
    if p.get("price") is None or p.get("price_unknown"):
        return "price n/a"
    return f"${p['price']:,.2f}"


def product_cards_html(products: list, request_id: Optional[str] = None, link_base: str = "",
                       session_id: Optional[str] = None, unmet: Optional[list] = None,
                       no_match: bool = False) -> str:
    """Renders the product grid shown next to the chat."""
    if no_match or not products:
        return ("<div class='st-note'>No matching products in the catalog. Try a broader description "
                "or relax a filter such as price or color.</div>")
    parts = []
    if unmet:
        parts.append(f"<div class='st-note'>⚠️ None of these has everything you asked for "
                     f"(not matched: {esc(', '.join(unmet))}). These are the closest options.</div>")
    parts.append("<div class='st-grid'>")
    for rank, p in enumerate(products, 1):
        pid = str(p.get("id", ""))
        img = p.get("image_url")
        img_html = (f"<img class='st-img' src='{esc(img)}' alt='{esc(p.get('title'))}' loading='lazy'>"
                    if isinstance(img, str) and img.startswith(("http://", "https://"))
                    else "<div class='st-noimg'>🛍️</div>")
        meta = " · ".join(esc(x) for x in (p.get("brand"), p.get("color"), p.get("product_type")) if x)
        chips = "".join(f"<span class='st-chip'>✓ {esc(k)}</span>"
                        for k, v in (p.get("matched") or {}).items() if v)
        href = click_href(link_base, request_id, pid, rank, session_id, fallback=p.get("link"))
        parts.append(
            f"<div class='st-card'>{img_html}"
            f"<div class='st-title'>{rank}. {esc(p.get('title'))}</div>"
            f"<div class='st-meta'>{meta}</div>"
            f"<div><span class='st-price'>{esc(format_price(p))}</span></div>"
            f"<div>{chips}</div>"
            f"<div class='st-id'>ID: {esc(pid)}</div>"
            f"<a class='st-link' href='{esc(href)}' target='_blank' rel='noopener'>View product ↗</a>"
            f"</div>")
    parts.append("</div>")
    return "".join(parts)


def format_answer(text: str) -> str:
    """Chat answer as Markdown: product citations like [ID: B082SP22KL] are shown in bold code style.
    Angle brackets are escaped so catalog titles or LLM output cannot inject HTML into the chat."""
    text = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return ID_RE.sub(lambda m: f"**`{m.group(1)}`**", text).strip()


def latency_line(latency_ms: Optional[dict], client_ms: Optional[dict] = None) -> str:
    """Small status line under the chat: server-side stage times and what the user actually waited."""
    if not latency_ms:
        return ""
    keys = [k for k in ("parse", "retrieve", "generate", "queue", "total") if latency_ms.get(k) is not None]
    s = " · ".join(f"{k} {latency_ms[k]:,.0f} ms" for k in keys)
    if client_ms:
        s += " | UI: " + " · ".join(f"{k.replace('_ms', '').replace('_', ' ')} {v:,.0f} ms"
                                    for k, v in client_ms.items() if v is not None)
    return f"⏱️ {s}"


def health_banner(h: dict, api_url: str) -> str:
    status = h.get("status")
    if status == "ok":
        voice = ""
        if h.get("asr"):
            voice = f" · 🎤 `{h.get('asr')}`"
        elif h.get("asr_status") and h.get("asr_status") != "ok":
            voice = f" · 🎤 voice input unavailable ({esc(h.get('asr_status'))})"
        if h.get("tts"):
            voice += " · 🔊 read-aloud available"
        return (f"🟢 **API ready** · LLM `{h.get('llm')}` · index {h.get('index_size') or 0:,} products"
                f"{voice} · `{api_url}`")
    if status == "loading":
        return f"🟡 **API is loading models** (1–3 min). Refresh the status in a moment · `{api_url}`"
    return f"🔴 **API not available** ({esc(h.get('error') or status)}) · `{api_url}`"
