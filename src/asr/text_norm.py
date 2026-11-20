"""Text normalisation and speech-to-text metrics (Phase 10).

* `normalize` makes a reference and a transcript comparable: lower case, no punctuation, "$30" ->
  "30 dollars", number words -> digits ("thirty" -> 30, "nine point five" -> 9.5), "men's" -> "mens".
* `wer` / `corpus_wer`: word error rate on normalised text.
* `key_terms` / `key_term_recall`: the words that change search results (brands, numbers, colours,
  product type) and whether the transcript kept them.
* `filter_agreement`: run the ShopTalk query parser on the reference and on the transcript and compare
  the filters it extracts (price, brand, colour, size, gender, product type). This is the most direct
  "did mishearing change the search?" measure.
* `speakable`: the reverse direction for text-to-speech ("$11.99" -> "eleven dollars ninety nine").
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


# ----------------------------------------------------------------------------- numbers -> words (TTS)
def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_to_words(n % 100))
    if n < 1_000_000:
        return int_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + int_to_words(n % 1000))
    return " ".join(_ONES[int(d)] for d in str(n))


def _decimal_words(s: str) -> str:
    whole, _, frac = s.partition(".")
    out = int_to_words(int(whole or 0))
    return out + (" point " + " ".join(_ONES[int(d)] for d in frac) if frac else "")


def speakable(text: str) -> str:
    """Text for a TTS voice that cannot read digits: removes [ID: …] tags and markdown, spells numbers."""
    t = re.sub(r"\[\s*ID\s*:[^\]]*\]", "", text or "")
    t = re.sub(r"[*_`#>|]", " ", t)

    def money(m):
        whole = int(m.group(1).replace(",", ""))
        cents = int((m.group(2) or "0").ljust(2, "0")[:2])
        s = int_to_words(whole) + (" dollar" if whole == 1 else " dollars")
        return s + (f" {int_to_words(cents)}" if cents else "")

    t = re.sub(r"\$\s?(\d[\d,]*)(?:\.(\d{1,2}))?", money, t)
    t = re.sub(r"(\d+)\s*%", lambda m: int_to_words(int(m.group(1))) + " percent", t)
    t = re.sub(r"\d+(?:\.\d+)?", lambda m: _decimal_words(m.group()), t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------- words -> numbers (WER)
_UNITS = {w: i for i, w in enumerate(_ONES)}            # zero..nineteen
_TENS_W = {w: i * 10 for i, w in enumerate(_TENS) if w}  # twenty..ninety


def _is_num_word(t: str) -> bool:
    return t in _UNITS or t in _TENS_W


def _parse_run(tokens: list, i: int) -> tuple:
    """Parse one spoken number starting at tokens[i]. Returns (value_str, next_index)."""
    total, cur, last, j = 0, 0, None, i
    while j < len(tokens):
        t = tokens[j]
        if t == "a" and j + 1 < len(tokens) and tokens[j + 1] in ("hundred", "thousand") and last is None:
            cur, last = 1, "unit"
        elif t in _UNITS:
            v = _UNITS[t]
            if last in ("unit", "teen") or (last == "tens" and v >= 10):
                break
            cur += v
            last = "teen" if v >= 10 else "unit"
        elif t in _TENS_W:
            if last in ("unit", "teen", "tens"):
                break
            cur += _TENS_W[t]
            last = "tens"
        elif t == "hundred" and last in ("unit", "teen"):
            cur *= 100
            last = "hundred"
        elif t == "thousand" and (cur or total):
            total, cur, last = total + max(cur, 1) * 1000, 0, "thousand"
        elif t == "and" and last in ("hundred", "thousand") and j + 1 < len(tokens) and _is_num_word(tokens[j + 1]):
            pass
        else:
            break
        j += 1
    value = str(total + cur)
    # decimals: "nine point five" -> 9.5
    if j + 1 < len(tokens) and tokens[j] == "point" and tokens[j + 1] in _UNITS and _UNITS[tokens[j + 1]] < 10:
        digits, k = [], j + 1
        while k < len(tokens) and tokens[k] in _UNITS and _UNITS[tokens[k]] < 10:
            digits.append(str(_UNITS[tokens[k]]))
            k += 1
        value, j = value + "." + "".join(digits), k
    return value, j


def _words_to_numbers(tokens: list) -> list:
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        starts = _is_num_word(t) or (t == "a" and i + 1 < len(tokens) and tokens[i + 1] in ("hundred", "thousand"))
        if starts:
            value, j = _parse_run(tokens, i)
            if j > i:
                out.append(value)
                i = j
                continue
        out.append(t)
        i += 1
    return out


_TENS_RE = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_UNITS_RE = "one|two|three|four|five|six|seven|eight|nine"


def transcript_to_query(text: str) -> str:
    """Speech-to-text output -> the query the pipeline sees: spelled numbers become digits and
    "30 dollars" becomes "$30", so prices and sizes reach the parser the same way as typed queries
    (the parser reads "$30"/"under 30" but not "thirty", and the product-type lexicon even maps
    "thirty" to HAT). Everything else is kept as transcribed."""
    t = re.sub(rf"\b({_TENS_RE})-({_UNITS_RE})\b", r"\1 \2", text or "", flags=re.I)
    chunks = t.split()
    cores = [re.sub(r"^[^\w$]+|[^\w]+$", "", c).lower() for c in chunks]
    out, i = [], 0
    while i < len(chunks):
        c = cores[i]
        starts = _is_num_word(c) or (c == "a" and i + 1 < len(cores) and cores[i + 1] in ("hundred", "thousand"))
        if starts:
            # a run ends at a chunk carrying punctuation ("thirty," "dollars.")
            stop = i
            while stop < len(chunks) - 1 and chunks[stop].lower() == cores[stop]:
                stop += 1
            value, j = _parse_run(cores[: stop + 1], i)
            if j > i and not (value == "1" and c == "one" and j == i + 1):   # keep "one" alone ("one piece")
                tail = re.sub(r"^.*?([^\w]*)$", r"\1", chunks[j - 1])
                out.append(value + tail)
                i = j
                continue
        out.append(chunks[i])
        i += 1
    q = " ".join(out)
    q = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:us\s+)?(?:dollars?|bucks?)\b", r"$\1", q, flags=re.I)
    return q


def normalize(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"\$\s?(\d[\d,]*(?:\.\d+)?)", lambda m: m.group(1).replace(",", "") + " dollars", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = t.replace("'s ", "s ").replace("'", "")
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)          # keep decimal points only
    t = re.sub(r"[^a-z0-9.\s]", " ", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)            # "10in" -> "10 in"
    t = re.sub(r"([a-z])(\d)", r"\1 \2", t)
    tokens = _words_to_numbers(t.split())
    tokens = [{"dollar": "dollars", "bucks": "dollars", "buck": "dollars"}.get(x, x) for x in tokens]
    return " ".join(tokens)


# ----------------------------------------------------------------------------- WER
def edit_counts(ref: list, hyp: list) -> dict:
    """Levenshtein alignment counts: substitutions, deletions, insertions."""
    n, m = len(ref), len(hyp)
    d = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, n + 1):
        d[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        d[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
                continue
            s, de, ins = d[i - 1][j - 1], d[i - 1][j], d[i][j - 1]
            d[i][j] = min((s[0] + 1, s[1] + 1, s[2], s[3]), (de[0] + 1, de[1], de[2] + 1, de[3]),
                          (ins[0] + 1, ins[1], ins[2], ins[3] + 1))
    c = d[n][m]
    return {"S": c[1], "D": c[2], "I": c[3], "N": n}


def wer(reference: str, hypothesis: str) -> float:
    r, h = normalize(reference).split(), normalize(hypothesis).split()
    c = edit_counts(r, h)
    return (c["S"] + c["D"] + c["I"]) / max(c["N"], 1)


def corpus_wer(pairs: Iterable[tuple]) -> float:
    errs = n = 0
    for ref, hyp in pairs:
        c = edit_counts(normalize(ref).split(), normalize(hyp).split())
        errs += c["S"] + c["D"] + c["I"]
        n += c["N"]
    return errs / max(n, 1)


# ----------------------------------------------------------------------------- what matters for search
def key_terms(reference: str, parser=None) -> list:
    """Normalised words that change the search: numbers + brand / colour / product-type words."""
    norm = normalize(reference)
    terms = [t for t in norm.split() if re.fullmatch(r"\d+(\.\d+)?", t)]
    if parser is not None:
        pq = parser.parse(reference)
        for v in [pq.brand, getattr(pq, "type_term", None), *(pq.colors or [])]:
            if v:
                terms += normalize(str(v)).split()
    return list(dict.fromkeys(_stem(t) for t in terms))


def key_term_recall(reference: str, hypothesis: str, parser=None) -> Optional[float]:
    terms = key_terms(reference, parser)
    if not terms:
        return None
    hyp = {_stem(t) for t in normalize(hypothesis).split()}
    return sum(_stem(t) in hyp for t in terms) / len(terms)


def _stem(t: str) -> str:
    """Plural-insensitive match: the parser reports 'sandal' for 'sandals'."""
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


FILTER_FIELDS = ("brand", "min_price", "max_price", "colors", "gender", "size", "product_types")


def _canon(v):
    if isinstance(v, (list, tuple, set)):
        return tuple(sorted(str(x).lower() for x in v))
    if isinstance(v, str):
        return normalize(v)
    return v


def filter_agreement(reference: str, hypothesis: str, parser) -> dict:
    """Per-field equality of the parsed filters; 'all' is True when every field matches."""
    a, b = parser.parse(reference), parser.parse(hypothesis)
    out = {f: _canon(getattr(a, f, None)) == _canon(getattr(b, f, None)) for f in FILTER_FIELDS}
    out["all"] = all(out.values())
    out["mismatched"] = [f for f in FILTER_FIELDS if not out[f]]
    return out
