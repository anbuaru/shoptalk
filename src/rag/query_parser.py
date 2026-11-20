"""Stage 1 of the ShopTalk RAG pipeline: query understanding.

Turns a natural-language shopping query into
  * a semantic query (what gets embedded), and
  * structured filters (price, color, gender, brand, size) applied against metadata.

Two interchangeable parsers share the same interface (`parse(query) -> ParsedQuery`):
  * RuleBasedParser - regex/lookup based, ~1 ms, deterministic. Default.
  * LLMQueryParser  - asks an instruction LLM for JSON, falls back to rules on any error.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional

COLORS = [
    "black", "white", "red", "blue", "navy", "green", "yellow", "orange", "purple",
    "pink", "brown", "beige", "tan", "grey", "gray", "silver", "gold", "ivory",
    "cream", "maroon", "burgundy", "teal", "turquoise", "olive", "khaki", "charcoal",
    "multicolor", "multi-color", "clear", "transparent", "bronze", "copper", "lavender",
    "coral", "mint", "rose", "wine", "aqua",
]
# Synonyms that should match each other when filtering on the catalog's color field.
COLOR_SYNONYMS = {
    "gray": {"gray", "grey", "charcoal"},
    "grey": {"gray", "grey", "charcoal"},
    "navy": {"navy", "blue"},
    "burgundy": {"burgundy", "maroon", "wine"},
    "maroon": {"burgundy", "maroon", "wine"},
    "beige": {"beige", "tan", "cream", "khaki"},
    "multicolor": {"multicolor", "multi-color", "multicoloured", "multi"},
    "multi-color": {"multicolor", "multi-color", "multicoloured", "multi"},
}

GENDER_PATTERNS = {
    "women": r"\b(women|woman|womens|ladies|lady|female)\b",
    "men": r"\b(men|man|mens|male|gentlemen)\b",
    "kids": r"\b(boys?|girls?|kids?|child|children|toddlers?|baby|babies|infants?|youth)\b",
    "unisex": r"\bunisex\b",
}

_NUM = r"\$?\s*(\d+(?:[.,]\d{1,2})?)\s*(?:dollars|usd|bucks|\$)?"
PRICE_PATTERNS = [
    # between $20 and $50 / $20-$50 / 20 to 50 dollars
    ("range", rf"\b(?:between\s+)?{_NUM}\s*(?:-|to|and)\s*{_NUM}"),
    ("max", rf"\b(?:under|below|less\s+than|cheaper\s+than|up\s+to|at\s+most|max(?:imum)?|no\s+more\s+than|within)\s+{_NUM}"),
    ("max", rf"<\s*{_NUM}"),
    ("min", rf"\b(?:over|above|more\s+than|at\s+least|min(?:imum)?)\s+{_NUM}"),
    ("min", rf">\s*{_NUM}"),
]

SIZE_PATTERN = (
    r"\bsize\s+(\d{1,2}(?:\.5)?|xxs|xs|s|m|l|xl|xxl|xxxl|small|medium|large|x-large|xx-large)\b"
)

FILLER_PATTERN = (
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:show me|find me|find|get me|i want|i need|i'?m looking for|looking for|"
    r"search for|recommend|suggest)\s+"
)

FOLLOWUP_PATTERN = (
    r"\b(similar|like (?:that|this|these|those|the (?:first|second|last) one)|"
    r"more like|another one|other options|something else|cheaper|less expensive|"
    r"more expensive|pricier|in (?:a )?different colou?r|same but)\b"
)
CHEAPER_PATTERN = r"\b(cheaper|less expensive|lower price|more affordable|budget)\b"
PRICIER_PATTERN = r"\b(more expensive|pricier|higher end|premium|fancier)\b"

# Brand names in ABO that are also everyday words; matching them would add false brand filters.
BRAND_STOPLIST = {
    "amazon", "amazon brand", "amazonbasics", "generic", "basics", "essentials", "home",
    "find", "show", "small", "large", "classic", "simple", "modern", "cozy", "gift",
    "the", "for", "and", "with", "men", "women", "kids", "size", "black", "white",
}


@dataclass
class ParsedQuery:
    raw: str
    semantic_query: str
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    colors: list = field(default_factory=list)
    gender: Optional[str] = None
    brand: Optional[str] = None
    size: Optional[str] = None
    product_types: list = field(default_factory=list)   # catalog ptype codes, e.g. ["SHIRT"]
    type_term: Optional[str] = None                      # the query word they came from, e.g. "shirt"
    type_unavailable: bool = False                       # the query names a product the catalog doesn't sell
    is_followup: bool = False
    wants_cheaper: bool = False
    wants_pricier: bool = False
    parser: str = "rules"

    def has_filters(self) -> bool:
        return any([self.min_price is not None, self.max_price is not None,
                    self.colors, self.gender, self.brand, self.size, self.product_types])

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(s: str) -> float:
    return float(s.replace(",", "."))


def _clean_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    return s.strip(" ,.-")


# ---------------- product type detection ----------------
TYPE_STOPWORDS = {
    "for", "with", "and", "the", "that", "this", "from", "your", "you", "set", "pack", "piece", "pieces",
    "size", "sizes", "inch", "inches", "new", "amazon", "brand", "basics", "essentials", "premium", "classic",
    "style", "design", "quality", "small", "medium", "large", "extra", "mini", "big", "one", "two", "three",
    "men", "man", "mens", "women", "woman", "womens", "ladies", "lady", "boys", "boy", "girls", "girl",
    "kids", "kid", "unisex", "adult", "adults", "baby", "under", "over", "between", "cheap", "best",
    "something", "someone", "who", "loves", "love", "gift", "gifts", "show", "find", "want", "need",
}


# Common product nouns (singular, after norm_token). If the query names one of these and the catalog
# titles (almost) never mention it, the catalog doesn't sell it: "shirt" in a catalog with no shirts.
GENERIC_PRODUCT_NOUNS = {
    "shirt", "tee", "blouse", "jean", "pant", "trouser", "legging", "dress", "skirt", "jacket", "coat", "sweater",
    "hoodie", "sweatshirt", "cardigan", "suit", "blazer", "vest", "sock", "underwear", "bra", "pajama", "swimsuit",
    "scarf", "glove", "hat", "cap", "beanie", "shoe", "sneaker", "boot", "sandal", "slipper", "heel", "loafer",
    "bag", "backpack", "handbag", "purse", "wallet", "belt", "watch", "necklace", "earring", "bracelet", "jewelry",
    "sunglass", "eyeglass", "laptop", "computer", "tablet", "phone", "smartphone", "television", "monitor",
    "camera", "headphone", "earbud", "speaker", "printer", "keyboard", "charger", "console", "toy", "doll",
    "book", "sofa", "couch", "chair", "table", "desk", "bed", "mattress", "pillow", "blanket", "lamp", "rug",
    "curtain", "shelf", "bookcase", "dresser", "cabinet", "mirror", "mug", "plate", "bowl", "pan", "knife",
    "kettle", "blender", "toaster", "microwave", "vacuum", "heater", "espresso", "grinder", "tire", "bicycle",
    "bike", "helmet", "tent", "ticket", "insurance", "salmon", "chainsaw", "drill", "perfume", "shampoo",
    "lipstick", "vitamin", "guitar", "piano",
}


def norm_token(w: str) -> str:
    """Tiny plural normaliser: boots->boot, dresses->dress, watches->watch, batteries->battery."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _tokens(text: str) -> list:
    return [norm_token(t) for t in re.findall(r"[a-z]+", str(text).lower()) if len(t) >= 3]


class ProductTypeLexicon:
    """Maps a query word to catalog product types, learned from the catalog itself.

    For every title word, look at the product types of the items whose title contains it. If the word is
    concentrated in a few types (top type >= min_share of its items), it is a product noun: "shirt" -> SHIRT,
    "boot" -> BOOT, "wallet" -> WALLET. Words spread over many types ("cozy", "coffee", "room") are ignored.
    """

    def __init__(self, mapping: Optional[dict] = None, counts: Optional[dict] = None, min_mentions: int = 3,
                 type_words: Optional[Iterable[str]] = None):
        self.mapping = mapping or {}   # token -> [ptype, ...]
        self.counts = counts or {}     # token -> number of catalog titles containing it
        self.min_mentions = min_mentions
        self.type_words = set(type_words or [])   # words of the catalog's ptype names: SOFA -> sofa, DRINKING_CUP -> cup

    @classmethod
    def from_catalog(cls, titles, ptypes, stop: Optional[Iterable[str]] = None, min_count: int = 5,
                     min_share: float = 0.5, coverage: float = 0.8, max_types: int = 4):
        from collections import Counter, defaultdict

        stop_set = set(TYPE_STOPWORDS) | set(COLORS) | {norm_token(w) for w in (stop or [])}
        by_tok = defaultdict(Counter)
        for title, pt in zip(titles, ptypes):
            if not isinstance(pt, str) or not pt.strip():
                continue
            for tok in set(_tokens(title)):
                if tok not in stop_set and not tok.isdigit():
                    by_tok[tok][pt] += 1
        mapping = {}
        for tok, cnt in by_tok.items():
            n = sum(cnt.values())
            if n < min_count:
                continue
            top = cnt.most_common()
            if top[0][1] / n < min_share:
                continue
            chosen, acc = [], 0
            for pt, c in top[:max_types]:
                chosen.append(pt); acc += c
                if acc / n >= coverage:
                    break
            mapping[tok] = chosen
        counts = {tok: sum(cnt.values()) for tok, cnt in by_tok.items()}
        type_words = {norm_token(w) for pt in set(p for p in ptypes if isinstance(p, str))
                      for w in pt.lower().split("_") if len(w) >= 3}
        # generic product nouns are also counted even if they are in stop words, so "not sold" can be detected
        for title in titles:
            for tok in set(_tokens(title)):
                if tok in GENERIC_PRODUCT_NOUNS and tok in stop_set:
                    counts[tok] = counts.get(tok, 0) + 1
        return cls(mapping, counts, type_words=type_words)

    def unavailable(self, text: str) -> Optional[str]:
        """A common product noun in the text that catalog titles (almost) never mention, e.g. 'shirt'."""
        if not self.counts:
            return None
        for tok in _tokens(text):
            if (tok in GENERIC_PRODUCT_NOUNS and tok not in self.mapping and tok not in self.type_words
                    and self.counts.get(tok, 0) < self.min_mentions):
                return tok
        return None

    def detect(self, text: str) -> tuple:
        """(term, ptypes) for the LAST product noun in the text (the head noun in English:
        'black leather wallet' -> wallet, 'waterproof hiking boots' -> boot)."""
        hit = (None, [])
        for tok in _tokens(text):
            if tok in self.mapping:
                hit = (tok, list(self.mapping[tok]))
        return hit

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"mapping": self.mapping, "counts": self.counts, "min_mentions": self.min_mentions,
                       "type_words": sorted(self.type_words)}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        if "mapping" not in data:          # v3/v4 format: the mapping only
            return cls(data)
        return cls(data["mapping"], data.get("counts"), data.get("min_mentions", 3), data.get("type_words"))


class RuleBasedParser:
    """Fast, deterministic parser. Pass `known_brands` (from the catalog) to enable brand detection."""

    def __init__(self, known_brands: Optional[Iterable[str]] = None, min_brand_len: int = 3,
                 stoplist: Optional[set] = None, type_lexicon: Optional[ProductTypeLexicon] = None):
        self.type_lexicon = type_lexicon
        stop = BRAND_STOPLIST if stoplist is None else stoplist
        self.known_brands = {}
        for b in known_brands if known_brands is not None else []:
            if not isinstance(b, str):
                continue
            key = b.strip().lower()
            if len(key) >= min_brand_len and key not in stop and key not in COLORS:
                self.known_brands[key] = b.strip()
        self._max_brand_words = max((len(b.split()) for b in self.known_brands), default=1)

    def _find_brand(self, q: str) -> Optional[str]:
        if not self.known_brands:
            return None
        tokens = re.findall(r"[a-z0-9&'\-]+", q.lower())
        # longest n-gram first so "under armour" wins over "under"
        for n in range(min(self._max_brand_words, 4), 0, -1):
            for i in range(len(tokens) - n + 1):
                cand = " ".join(tokens[i:i + n])
                if cand in self.known_brands:
                    return self.known_brands[cand]
        return None

    def parse(self, query: str) -> ParsedQuery:
        q = query.strip()
        work = q
        pq = ParsedQuery(raw=q, semantic_query=q)

        # follow-up intent (resolved against history by the pipeline)
        if re.search(FOLLOWUP_PATTERN, q, re.I):
            pq.is_followup = True
            pq.wants_cheaper = bool(re.search(CHEAPER_PATTERN, q, re.I))
            pq.wants_pricier = bool(re.search(PRICIER_PATTERN, q, re.I))

        # size first, so "size 8-10" is never read as a price range
        m = re.search(SIZE_PATTERN, work, re.I)
        if m:
            pq.size = m.group(1).upper() if not m.group(1)[0].isdigit() else m.group(1)
            work = work[:m.start()] + " " + work[m.end():]

        # price: remove from the semantic query, embeddings handle numbers badly
        for kind, pat in PRICE_PATTERNS:
            m = re.search(pat, work, re.I)
            if not m:
                continue
            if kind == "range":
                # need a $ / currency word / "between" so "2-pack" or "3 to 4 people" is not a price
                span = m.group(0).lower()
                if not ("$" in span or "between" in span or re.search(r"dollars|usd|bucks", span)):
                    continue
                lo, hi = sorted([_to_float(m.group(1)), _to_float(m.group(2))])
                pq.min_price, pq.max_price = lo, hi
            elif kind == "max" and pq.max_price is None:
                pq.max_price = _to_float(m.group(1))
            elif kind == "min" and pq.min_price is None:
                pq.min_price = _to_float(m.group(1))
            else:
                continue
            work = work[:m.start()] + " " + work[m.end():]

        # colors and gender stay in the semantic query (they help the embedding too)
        low = work.lower()
        pq.colors = [c for c in COLORS if re.search(rf"\b{re.escape(c)}\b", low)]
        for g, pat in GENDER_PATTERNS.items():
            if re.search(pat, low):
                pq.gender = g
                break
        pq.brand = self._find_brand(work)
        if self.type_lexicon is not None:
            missing = self.type_lexicon.unavailable(work)
            if missing:   # "gaming laptop ..." in a catalog without laptops: say so instead of laptop bags
                pq.type_term, pq.product_types, pq.type_unavailable = missing, [], True
            else:
                pq.type_term, pq.product_types = self.type_lexicon.detect(work)

        work = re.sub(FILLER_PATTERN, "", work, flags=re.I)
        work = re.sub(r"\b(for|with|in)\s*$", "", _clean_spaces(work), flags=re.I)
        pq.semantic_query = _clean_spaces(work) or q
        return pq


LLM_PARSER_PROMPT = """Extract shopping filters from the query. Reply with one JSON object only, no prose, no code fences.
Keys: "semantic_query" (the product being searched for, without price words),
"min_price" (number or null), "max_price" (number or null), "colors" (list of lowercase colors),
"gender" ("men", "women", "kids", "unisex" or null), "brand" (string or null), "size" (string or null).

Example
Query: navy running shoes for men under $60, size 10
JSON: {{"semantic_query": "navy running shoes for men", "min_price": null, "max_price": 60, "colors": ["navy"], "gender": "men", "brand": null, "size": "10"}}

Query: {query}
JSON:"""


def _num(v) -> Optional[float]:
    if v in (None, "", "null"):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else None


class LLMQueryParser:
    """Uses an LLM (any `generate_fn(prompt) -> str`) for parsing; falls back to rules on failure.

    Rule-based detection still decides follow-up intent, since that is cheap and reliable.
    `last_raw` / `last_error` keep the most recent model output and failure reason for debugging.
    """

    def __init__(self, generate_fn: Callable[[str], str], fallback: Optional[RuleBasedParser] = None):
        self.generate_fn = generate_fn
        self.fallback = fallback or RuleBasedParser()
        self.last_raw: Optional[str] = None
        self.last_error: Optional[str] = None

    def parse(self, query: str) -> ParsedQuery:
        base = self.fallback.parse(query)
        self.last_raw, self.last_error = None, None
        try:
            out = self.generate_fn(LLM_PARSER_PROMPT.format(query=query))
            self.last_raw = out
            out = re.sub(r"```(?:json)?", "", out)
            m = re.search(r"\{.*?\}", out, re.S)
            if not m:
                raise ValueError("no JSON object in model output")
            data = json.loads(m.group(0))
            colors = data.get("colors") or []
            if isinstance(colors, str):
                colors = [colors]
            gender = (data.get("gender") or "")
            gender = gender.lower() if isinstance(gender, str) else ""
            return ParsedQuery(
                raw=query,
                semantic_query=(data.get("semantic_query") or base.semantic_query).strip(),
                min_price=_num(data.get("min_price")),
                max_price=_num(data.get("max_price")),
                colors=[str(c).lower() for c in colors],
                gender=gender if gender in ("men", "women", "kids", "unisex") else None,
                brand=data.get("brand") or None,
                size=str(data["size"]) if data.get("size") else None,
                product_types=base.product_types, type_term=base.type_term, type_unavailable=base.type_unavailable,
                is_followup=base.is_followup, wants_cheaper=base.wants_cheaper,
                wants_pricier=base.wants_pricier, parser="llm",
            )
        except Exception as e:  # noqa: BLE001 - any failure falls back to rules
            self.last_error = f"{type(e).__name__}: {e}"
            base.parser = "rules(fallback)"
            return base
