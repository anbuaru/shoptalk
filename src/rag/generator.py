"""Stage 3 of the ShopTalk RAG pipeline: grounded answer generation.

The LLM sees only the top 3-5 retrieved products, numbered [1]..[k], and is told to recommend only
from them, cite them by number, describe only features present in the product details, and say so when nothing
matches. Code then turns [n] into [ID: <real id>] (`resolve_citations`): an ID tag costs ~10 output tokens,
a number ~3, so answers are shorter/faster and the model cannot invent an ID.
After that, `check_grounding` verifies the IDs and `enforce_ids` appends any missing
ones, so the "answers list product IDs" target (100%) does not depend on the model following
instructions.

Note: ID grounding does not catch invented *features* (sizes, materials, brand claims). That is
covered by the manual review (Phase 9).
"""
from __future__ import annotations

import re
import threading
import time
from typing import Iterator, Optional

SYSTEM_PROMPT = """You are ShopTalk, a shopping assistant.
Rules:
1. Recommend ONLY products from the PRODUCTS list. Never invent products, prices, sizes, materials or features.
2. Only state facts that appear in that product's Title, Brand, Price, Color, Type or Details.
3. Refer to a product by its number in square brackets, e.g. [1] or [2]. Never write product IDs.
4. If a price is "unknown", do not guess it.
5. If none of the products fit the request, say so plainly and suggest how to rephrase.
6. If there is a NOTE, start by saying plainly what could not be found, then offer the products only as the
   closest alternatives. Never call a product something it is not (a hat is not a shirt).
7. Keep the answer under 50 words: one short intro sentence, then one short line per recommended product
   (recommend 2-3 when several fit) explaining why it fits the request."""

USER_TEMPLATE = """Customer request: {query}
{constraints}{note}
PRODUCTS:
{context}

Write your recommendation. Cite products by number, e.g. [1]."""

ID_TAG_RE = re.compile(r"\[\s*ID\s*:\s*([A-Za-z0-9_\-]+)\s*\]")
# loose fallback for IDs written without the tag, e.g. ABO/ASIN-style "B07H9GMYXS"
LOOSE_ID_RE = re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8,14}\b")


def _fmt_price(v) -> str:
    try:
        f = float(v)
        if f != f:  # NaN
            return "unknown"
        return f"${f:,.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _or_unknown(v) -> str:
    return str(v) if v not in (None, "") else "unknown"


def format_context(products: list, max_chars: int = 350, title_key="title", brand_key="brand",
                   price_key="price", color_key="color", type_key: Optional[str] = None) -> str:
    lines = []
    for i, p in enumerate(products, 1):
        m = p.get("metadata", {})
        desc = re.sub(r"\s+", " ", p.get("document", ""))[:max_chars]
        ptype = f" | Type: {_or_unknown(m.get(type_key))}" if type_key else ""
        lines.append(
            f"[{i}] {m.get(title_key, '') or '(no title)'}\n"
            f"   Brand: {_or_unknown(m.get(brand_key))} | Price: {_fmt_price(m.get(price_key))} | "
            f"Color: {_or_unknown(m.get(color_key))}{ptype}\n"
            f"   Details: {desc}"
        )
    return "\n".join(lines)


def format_constraints(pq) -> str:
    if pq is None:
        return ""
    parts = []
    if pq.max_price is not None:
        parts.append(f"max price ${pq.max_price:g}")
    if pq.min_price is not None:
        parts.append(f"min price ${pq.min_price:g}")
    if pq.colors:
        parts.append("color " + "/".join(pq.colors))
    if pq.gender:
        parts.append(f"for {pq.gender}")
    if pq.brand:
        parts.append(f"brand {pq.brand}")
    if pq.size:
        parts.append(f"size {pq.size}")
    if getattr(pq, "type_term", None):
        parts.append(f"product type {pq.type_term}")
    return ("Constraints: " + ", ".join(parts) + "\n") if parts else ""


ATTR_LABELS = {"type": "product type", "color": "color", "gender": "for", "brand": "brand", "size": "size"}


def describe_unmet(unmet: list, pq) -> str:
    """'a shirt, color red' for the NOTE line."""
    if not unmet or pq is None:
        return ""
    vals = {"type": getattr(pq, "type_term", None), "color": "/".join(pq.colors) if pq.colors else None,
            "gender": pq.gender, "brand": pq.brand, "size": pq.size}
    return ", ".join(f"{ATTR_LABELS[a]} {vals[a]}" for a in unmet if vals.get(a))


def format_note(unmet: list, pq) -> str:
    d = describe_unmet(unmet, pq)
    return f"NOTE: none of these products matches: {d}. Say so first.\n" if d else ""


def check_grounding(answer: str, products: list) -> dict:
    """ID-level groundedness: which IDs were cited, and were any cited that we did not retrieve."""
    allowed = {p["id"] for p in products}
    cited = set(ID_TAG_RE.findall(answer))
    loose = {t for t in LOOSE_ID_RE.findall(answer) if t not in cited}
    cited_all = (cited | loose) & allowed
    hallucinated = sorted(cited - allowed)  # only tagged IDs count; loose tokens may be model numbers
    return {
        "cited_ids": sorted(cited_all),
        "hallucinated_ids": hallucinated,
        "lists_ids": len(cited_all) > 0,
        "grounded_ids": len(hallucinated) == 0,
    }


CITE_RE = re.compile(r"\[\s*(\d{1,2}(?:\s*(?:,|and)\s*\d{1,2})*)\s*\]")


def resolve_citations(text: str, products: list) -> str:
    """[2] -> [ID: <id of product 2>]; [1, 3] -> [ID: a], [ID: c]. A number outside 1..k becomes
    [ID: REF<n>], which check_grounding counts as hallucinated and remove_unshown_ids deletes."""
    def _sub(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        return ", ".join(f"[ID: {products[n - 1]['id']}]" if 1 <= n <= len(products) else f"[ID: REF{n}]"
                         for n in nums)
    return CITE_RE.sub(_sub, text)


class StreamingCitationResolver:
    """Resolves [n] in a token stream: holds back text after an unclosed '[' until it can be resolved."""

    def __init__(self, products: list):
        self.products, self.buf = products, ""

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        cut = self.buf.rfind("[")
        if cut != -1 and "]" not in self.buf[cut:] and len(self.buf) - cut < 16:
            out, self.buf = self.buf[:cut], self.buf[cut:]
        else:
            out, self.buf = self.buf, ""
        return resolve_citations(out, self.products)

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return resolve_citations(out, self.products)


def remove_unshown_ids(answer: str, products: list) -> tuple:
    """Delete [ID: ...] tags whose ID was not in the products shown (copied placeholders, typos,
    invented IDs). Returns (clean_answer, removed_ids). Keeps the 0% hallucinated-ID target
    independent of the model; the raw rate is still reported from check_grounding."""
    allowed = {p["id"] for p in products}
    removed = []

    def _sub(m):
        if m.group(1) in allowed:
            return m.group(0)
        removed.append(m.group(1))
        return ""

    clean = ID_TAG_RE.sub(_sub, answer)
    clean = re.sub(r"[ \t]+([,.;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", clean))
    return clean.strip(), removed


def enforce_ids(answer: str, products: list, grounding: Optional[dict] = None) -> str:
    """Deterministic post-step: if the model cited no IDs, append the shown products' IDs."""
    g = grounding or check_grounding(answer, products)
    if g["lists_ids"] or not products:
        return answer
    ids = ", ".join(p["id"] for p in products)
    return answer.rstrip() + f"\n\nProduct IDs: {ids}"


NO_MATCH_ANSWER = ("I couldn't find products matching that request in the catalog. "
                   "Try a broader description or relax a filter such as price or color.")


class HFGenerator:
    """Hugging Face causal LM. fp16 by default: on a T4, bitsandbytes 4-bit decoding is much slower
    than fp16 for 1-4B models, and those fit in fp16 anyway. Use load_in_4bit=True only for 7B+."""

    def __init__(self, model_id: str, load_in_4bit: bool = False, max_new_tokens: int = 100,
                 context_max_chars: int = 350, device_map: str = "auto", trust_remote_code: bool = False,
                 context_keys: Optional[dict] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.context_max_chars = context_max_chars
        self.context_keys = context_keys or {}
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        kwargs = {"device_map": device_map, "trust_remote_code": trust_remote_code}
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        t0 = time.perf_counter()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **kwargs, **({} if "quantization_config" in kwargs else {"dtype": dtype}))
        except TypeError:  # older transformers only know torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, **kwargs, **({} if "quantization_config" in kwargs else {"torch_dtype": dtype}))
        self.model.eval()
        self.load_seconds = time.perf_counter() - t0
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_inputs(self, system: str, user: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        tmpl_kwargs = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
        try:
            # enable_thinking=False keeps Qwen3-style "thinking" models fast; ignored by others
            return self.tokenizer.apply_chat_template(msgs, enable_thinking=False, **tmpl_kwargs)
        except Exception:
            # some templates (e.g. Gemma) reject the system role: fold it into the user turn
            msgs = [{"role": "user", "content": system + "\n\n" + user}]
            return self.tokenizer.apply_chat_template(msgs, **tmpl_kwargs)

    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def complete(self, system: str, user: str, max_new_tokens: Optional[int] = None) -> tuple:
        inputs = self._build_inputs(system, user).to(self.model.device)
        n_in = inputs["input_ids"].shape[1]
        self._sync()
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        self._sync()
        dt = time.perf_counter() - t0
        new_tokens = out[0][n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, {"seconds": dt, "input_tokens": int(n_in), "output_tokens": int(len(new_tokens))}

    def build_user_prompt(self, query: str, products: list, parsed=None, unmet: Optional[list] = None) -> str:
        return USER_TEMPLATE.format(query=query, constraints=format_constraints(parsed),
                                    note=format_note(unmet or [], parsed),
                                    context=format_context(products, self.context_max_chars, **self.context_keys))

    def generate(self, query: str, products: list, parsed=None, unmet: Optional[list] = None) -> tuple:
        """(answer with [n] already turned into [ID: ...], info). info["raw"] keeps the model's text and
        info["hit_token_cap"] says whether the answer was cut off by max_new_tokens."""
        text, info = self.complete(SYSTEM_PROMPT, self.build_user_prompt(query, products, parsed, unmet))
        info["raw"] = text
        info["hit_token_cap"] = info["output_tokens"] >= self.max_new_tokens
        return resolve_citations(text, products), info

    def stream(self, query: str, products: list, parsed=None, unmet: Optional[list] = None) -> Iterator[str]:
        """Yield text chunks as they are generated (for the Phase 7 streaming endpoint / TTFT)."""
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(SYSTEM_PROMPT, self.build_user_prompt(query, products, parsed, unmet)).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                  pad_token_id=self.tokenizer.pad_token_id, streamer=streamer)
        th = threading.Thread(target=lambda: self.model.generate(**kw), daemon=True)
        th.start()
        resolver = StreamingCitationResolver(products)
        for chunk in streamer:
            out = resolver.feed(chunk) if chunk else ""
            if out:
                yield out
        tail = resolver.flush()
        if tail:
            yield tail
        th.join()

    def time_to_first_token(self, query: str, products: list, parsed=None, unmet: Optional[list] = None) -> tuple:
        """(seconds to first streamed chunk, total seconds)."""
        t0 = time.perf_counter()
        ttft = None
        for _ in self.stream(query, products, parsed, unmet):
            if ttft is None:
                ttft = time.perf_counter() - t0
        return ttft, time.perf_counter() - t0

    def as_text_fn(self, max_new_tokens: int = 120):
        """A `prompt -> str` callable, e.g. for LLMQueryParser."""
        return lambda prompt: self.complete("You output JSON only.", prompt, max_new_tokens)[0]

    def unload(self):
        import gc
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
