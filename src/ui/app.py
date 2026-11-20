"""ShopTalk chat UI (Phase 8): a Gradio app on top of the Phase 7 inference API.

What the user gets:
  * a chat box: the answer streams in token by token (POST /recommend/stream)
  * product cards next to the chat (image, title, brand, price, ID, "View product" link)
  * follow-ups ("something similar but cheaper"): each response's `history_entry` is kept and sent back
  * 👍 / 👎 (+ optional comment) on the latest answer -> POST /feedback
  * clicks on "View product" go through /go/... on this UI server, which logs them via the API's
    /click endpoint and then redirects to the product page
  * Phase 10: 🎤 microphone input -> POST /transcribe -> the transcript is put in the text box, so the user
    sees what was heard and can fix it before pressing Send; optional 🔊 "read answers aloud" (POST /speak),
    off by default

The UI holds no model: it only talks HTTP to the API, so the API can stay private (127.0.0.1) and
only the UI needs a public URL. The same file runs in Colab and in Docker / AWS (Phase 11).

Run:
    SHOPTALK_API_URL=http://127.0.0.1:8000 python -m src.ui.app        # serves on port 7860
"""
from __future__ import annotations

import os
import re
import tempfile
import time
import uuid
from typing import Optional

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .client import ApiError, ShopTalkClient
from .render import (CSS, EMPTY_PRODUCTS, format_answer, health_banner, latency_line, product_cards_html)

MAX_HISTORY = 3                      # the API uses the last 3 turns
PRODUCT_URL = "https://www.amazon.com/dp/{id}"
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

EXAMPLES = ["Show me a red shirt for men under $50", "Waterproof hiking boots, size 10",
            "A gift for someone who loves coffee", "Something cozy for a small living room",
            "Find me a black leather wallet"]


# ---------------- per-browser-session state ----------------
def new_state(session_id: Optional[str] = None) -> dict:
    """State kept per browser tab: session id, last turns for follow-ups, and the latest request."""
    return {"session_id": session_id or f"ui-{uuid.uuid4().hex[:12]}", "history": [],
            "last_request_id": None, "voted": {}}


def _assistant(chat: list, text: str) -> list:
    """Sets the text of the last assistant message (adds one if the last message is the user's)."""
    if chat and chat[-1].get("role") == "assistant":
        chat[-1] = {"role": "assistant", "content": text}
    else:
        chat.append({"role": "assistant", "content": text})
    return chat


# ---------------- handlers (plain functions, unit-tested without a browser) ----------------
def add_user_message(message: str, chat: Optional[list]):
    """Step 1 of a turn: show the user's message at once and clear the textbox."""
    chat = list(chat or [])
    message = (message or "").strip()
    if message:
        chat.append({"role": "user", "content": message})
    return "", chat, message


def chat_turn(client: ShopTalkClient, message: str, chat: Optional[list], state: Optional[dict],
              link_base: str = "", use_stream: bool = True):
    """
    Step 2 of a turn: call the API and stream the result into the UI.

    Yields (chat, products_html, latency_markdown, state) every time something new arrives:
    first the product cards (meta event), then the growing answer (token events), then the final
    answer (done event; the API cleans citations, so it can differ slightly from the streamed text).
    """
    chat = list(chat or [])
    state = dict(state or new_state())
    if not message:
        yield chat, gr.update(), gr.update(), state
        return
    if len(message) > 500:
        yield _assistant(chat, "That message is too long (500 characters max). Please shorten it."), \
            gr.update(), "", state
        return

    sid, history = state["session_id"], state["history"][-MAX_HISTORY:]
    t0 = time.perf_counter()
    products_html, text = gr.update(), ""
    client_ms = {"products_ms": None, "first_token_ms": None, "total_ms": None}
    done = None
    try:
        if use_stream:
            yield _assistant(chat, "🔎 Searching the catalog…"), gr.update(), "", state
            for event, data in client.stream(message, history=history, session_id=sid):
                if event == "meta":
                    client_ms["products_ms"] = (time.perf_counter() - t0) * 1000
                    state["last_request_id"] = data.get("request_id")
                    products_html = product_cards_html(
                        data.get("products", []), data.get("request_id"), link_base, sid,
                        data.get("unmet_constraints"), data.get("no_match", False))
                    yield _assistant(chat, "✍️ Writing an answer…"), products_html, "", state
                elif event == "token":
                    if client_ms["first_token_ms"] is None:
                        client_ms["first_token_ms"] = (time.perf_counter() - t0) * 1000
                    text += data.get("text", "")
                    yield _assistant(chat, format_answer(text) + " ▌"), products_html, "", state
                elif event == "done":
                    done = data
                elif event == "error":
                    raise ApiError(f"{data.get('detail', 'error')} (request_id={data.get('request_id')})")
        else:
            yield _assistant(chat, "🔎 Searching and writing an answer…"), gr.update(), "", state
            done = client.recommend(message, history=history, session_id=sid)
    except ApiError as e:
        msg = str(e)
        if "503" in msg:
            msg = "The models are still loading on the server. Please try again in a minute."
        yield _assistant(chat, f"⚠️ {msg}"), products_html, "", state
        return

    if done is None:
        yield _assistant(chat, "⚠️ The answer was cut off. Please try again."), products_html, "", state
        return

    client_ms["total_ms"] = (time.perf_counter() - t0) * 1000
    rid = done.get("request_id")
    state["last_request_id"] = rid
    state["history"] = (state["history"] + [done["history_entry"]])[-MAX_HISTORY:]
    products_html = product_cards_html(done.get("products", []), rid, link_base, sid,
                                       done.get("unmet_constraints"), done.get("no_match", False))
    state["last_response"] = {"request_id": rid, "products": done.get("products", []),
                              "no_match": done.get("no_match"), "latency_ms": done.get("latency_ms"),
                              "client_ms": {k: round(v) for k, v in client_ms.items() if v is not None},
                              "answer": done.get("answer", "")}
    yield (_assistant(chat, format_answer(done.get("answer", ""))), products_html,
           latency_line(done.get("latency_ms"), {k: v for k, v in client_ms.items() if v is not None}), state)


def send_feedback(client: ShopTalkClient, rating: str, comment: str, state: Optional[dict]):
    """👍 / 👎 on the latest answer. Returns (status_markdown, comment_box_value, state)."""
    state = dict(state or new_state())
    rid = state.get("last_request_id")
    if not rid:
        return "Ask something first, then rate the answer.", comment, state
    voted = dict(state.get("voted") or {})
    if voted.get(rid) == rating and not (comment or "").strip():
        return "Already recorded, thanks!", comment, state
    try:
        client.feedback(rid, rating, (comment or "").strip()[:1000] or None, state["session_id"])
    except ApiError as e:
        return f"⚠️ Could not send feedback: {e}", comment, state
    voted[rid] = rating
    state["voted"] = voted
    return ("Thanks! 👍 recorded." if rating == "up" else "Thanks, 👎 recorded. It will help us improve."), "", state


def transcribe_audio(client: ShopTalkClient, audio_path: Optional[str], state: Optional[dict]):
    """🎤 -> text box. Returns (textbox_update, heard_markdown). The query is NOT sent automatically:
    the user checks the transcript (brands, sizes and prices are easy to mishear) and presses Send."""
    if not audio_path:
        return gr.update(), ""
    sid = (state or {}).get("session_id")
    try:
        out = client.transcribe(audio_path, session_id=sid)
    except ApiError as e:
        msg = str(e)
        if "503" in msg:
            msg = "Speech-to-text is not available on the server right now; please type your request."
        elif "400" in msg:
            msg = "Could not understand that recording; please try again or type your request."
        return gr.update(), f"⚠️ {msg}"
    text = (out.get("text") or "").strip()
    if not text:
        return gr.update(), "🎤 I didn't catch any words; please try again."
    lat = out.get("latency_ms") or {}
    return text, (f"🎤 Heard: *{text}* — check it, edit if needed, then press **Send** "
                  f"(speech-to-text {lat.get('asr', 0):,.0f} ms)")


_ID_TAG = re.compile(r"\[\s*ID\s*:[^\]]*\]")


def speak_answer(client: ShopTalkClient, enabled: bool, state: Optional[dict]):
    """🔊 After a turn: read the latest answer aloud when the checkbox is on. Returns an Audio update."""
    answer = ((state or {}).get("last_response") or {}).get("answer", "")
    if not enabled or not answer:
        return gr.update(value=None, visible=False)
    try:
        wav = client.speak(_ID_TAG.sub("", answer))
    except ApiError:
        return gr.update(value=None, visible=False)
    f = tempfile.NamedTemporaryFile(prefix="shoptalk_tts_", suffix=".wav", delete=False)
    f.write(wav)
    f.close()
    return gr.update(value=f.name, visible=True)


def clear_chat(state: Optional[dict]):
    """New conversation: empties the chat and the follow-up history, keeps the session id."""
    s = new_state((state or {}).get("session_id"))
    return [], EMPTY_PRODUCTS, "", "", s


# ---------------- Gradio layout ----------------
def build_ui(client: ShopTalkClient, api_url: str, link_base: str = "", use_stream: bool = True) -> gr.Blocks:
    """Builds the Gradio Blocks app (not launched)."""
    with gr.Blocks(title="ShopTalk") as demo:
        gr.Markdown("# 🛍️ ShopTalk\nAsk for products in plain language. Follow-ups such as "
                    "*\"something similar but cheaper\"* use the previous results.")
        with gr.Row():
            status = gr.Markdown(health_banner(client.health(), api_url))
            refresh = gr.Button("Refresh status", size="sm", scale=0)
        state = gr.State(new_state)
        pending = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="Chat", height=520)
                with gr.Row():
                    msg = gr.Textbox(placeholder="e.g. Waterproof hiking boots, size 10", show_label=False,
                                     scale=5, max_lines=3, autofocus=True)
                    send = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    mic = gr.Audio(sources=["microphone"], type="filepath", format="wav",
                                   label="🎤 Or say it (the words appear in the box above)", scale=5)
                    speak_on = gr.Checkbox(value=False, label="🔊 Read answers aloud", scale=1)
                heard = gr.Markdown("")
                answer_audio = gr.Audio(label="Answer (audio)", autoplay=True, interactive=False, visible=False)
                gr.Examples(EXAMPLES, inputs=msg, label="Try")
                latency = gr.Markdown("")
                with gr.Row():
                    up = gr.Button("👍 Helpful", size="sm")
                    down = gr.Button("👎 Not helpful", size="sm")
                    clear = gr.Button("🧹 New chat", size="sm")
                comment = gr.Textbox(placeholder="Optional comment for your rating", show_label=False, max_lines=2)
                fb_status = gr.Markdown("")
            with gr.Column(scale=6):
                gr.Markdown("### Products")
                products = gr.HTML(EMPTY_PRODUCTS)

        def _turn(message, chat, st):
            yield from chat_turn(client, message, chat, st, link_base, use_stream)

        for trigger in (msg.submit, send.click):
            trigger(add_user_message, [msg, chatbot], [msg, chatbot, pending], queue=False) \
                .then(_turn, [pending, chatbot, state], [chatbot, products, latency, state]) \
                .then(lambda on, st: speak_answer(client, on, st), [speak_on, state], answer_audio)
        for rec_event in (mic.stop_recording, mic.upload):
            rec_event(lambda path, st: transcribe_audio(client, path, st), [mic, state], [msg, heard])
        up.click(lambda c, s: send_feedback(client, "up", c, s), [comment, state], [fb_status, comment, state])
        down.click(lambda c, s: send_feedback(client, "down", c, s), [comment, state], [fb_status, comment, state])
        clear.click(clear_chat, [state], [chatbot, products, latency, fb_status, state]) \
            .then(lambda: ("", gr.update(value=None, visible=False)), None, [heard, answer_audio])
        refresh.click(lambda: health_banner(client.health(), api_url), None, status)
        demo.load(lambda: health_banner(client.health(), api_url), None, status)
    demo.queue(default_concurrency_limit=8)
    return demo


def create_server(api_url: str, use_stream: bool = True) -> FastAPI:
    """
    FastAPI app that serves the Gradio UI at / and the click-tracking redirect at /go/....

    /go/{request_id}/{product_id}?rank=N logs the click through the API (server side, so the API does
    not need to be public) and redirects the browser to the product page.
    """
    client = ShopTalkClient(api_url)
    server = FastAPI(title="ShopTalk UI")

    @server.get("/go/{request_id}/{product_id}")
    def go(request_id: str, product_id: str, rank: Optional[int] = None, session_id: Optional[str] = None):
        if not (_ID_RE.match(request_id) and _ID_RE.match(product_id)):
            return RedirectResponse("/", status_code=307)
        target = client.click(request_id, product_id, rank=rank, session_id=session_id)
        return RedirectResponse(target or PRODUCT_URL.format(id=product_id), status_code=307)

    @server.get("/ui-health")
    def ui_health():
        return {"ui": "ok", "api": client.health().get("status"), "api_url": api_url}

    demo = build_ui(client, api_url, link_base="", use_stream=use_stream)
    return gr.mount_gradio_app(server, demo, path="/", css=CSS, theme=gr.themes.Soft())


def main():
    import uvicorn
    api_url = os.environ.get("SHOPTALK_API_URL", "http://127.0.0.1:8000")
    port = int(os.environ.get("SHOPTALK_UI_PORT", "7860"))
    use_stream = os.environ.get("SHOPTALK_UI_STREAM", "1") != "0"
    uvicorn.run(create_server(api_url, use_stream), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
