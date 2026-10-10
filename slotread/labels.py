"""Slot labels: the placeholder token, the digit labels, and the bin labels each arm uses.

An arm decides how every number is written:

  arm      coarse slot   bins   bin labels
  A        yes           256    ordinary words
  F        yes           256    bytes
  E        no            256    bytes            (the blog's main arm)
  Bplus    no            101    single characters (GPC-1's grid)
  C        no            256    ordinary words   (never learns without a coarse slot)
  Bplusw   no            101    ordinary words

"Bytes" are ASCII characters followed by two-letter tokens (`ab` ... `zz`); the first 101 are exactly
B+'s single-character labels. Every label is one token right after "x: ".
"""

from dataclasses import dataclass, field

PLACEHOLDER = " ___"
# Gemma 4's ids. configure_tokenizer() replaces them for another model's tokenizer.
PH = 52551  # " ___"
DIGITS = [236771, 236770, 236778, 236800, 236812, 236810, 236825, 236832, 236828, 236819]  # "0".."9"

ARMS = {"A": (True, 256, "words"), "F": (True, 256, "bytes"), "E": (False, 256, "bytes"),
        "Bplus": (False, 101, "bytes"), "C": (False, 256, "words"), "Bplusw": (False, 101, "words")}


@dataclass
class Arm:
    name: str
    coarse: bool
    bins: int
    tokens: str
    fine: list = field(default_factory=list)  # label ids of every fine (number) slot


def configure_tokenizer(tok):
    """Derive the placeholder and digit ids from a non-Gemma tokenizer (e.g. GUI-Owl / Qwen3-VL):
    the placeholder is the last token of "x: ___", each digit the last token of "x: <d>"."""
    global PH, DIGITS
    PH = tok("x: ___", add_special_tokens=False)["input_ids"][-1]
    assert tok.decode([PH]) == PLACEHOLDER, f"placeholder is not one token: {tok.decode([PH])!r}"
    DIGITS = [tok(f"x: {d}", add_special_tokens=False)["input_ids"][-1] for d in "0123456789"]
    assert len(set(DIGITS)) == 10 and [tok.decode([i]).strip() for i in DIGITS] == list("0123456789")


def byte_labels(tok, n=256):
    """ASCII characters, then two-letter tokens, each a single token after "x: " (GPC-1-style)."""
    ids, seen = [], set()
    ctx = tok("x: ___", add_special_tokens=False)["input_ids"][:-1]
    codes = [chr(c) for c in range(33, 127)] + [f"{chr(a)}{chr(b)}" for a in range(97, 123) for b in range(97, 123)]
    for code in codes:
        t = tok("x: " + code, add_special_tokens=False)["input_ids"]
        if len(t) == len(ctx) + 1 and t[-1] not in seen:
            seen.add(t[-1])
            ids.append(t[-1])
        if len(ids) >= n:
            break
    return ids


def word_labels(tok, n=256):
    """n distinct ordinary lowercase words (3-7 letters), each one token after "x: ", spread evenly
    over the vocabulary."""
    ctx = tok("x: ___", add_special_tokens=False)["input_ids"][:-1]
    cands = []
    for t, i in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
        w = t[1:]
        if 3000 < i < 60000 and t.startswith("▁") and w.isascii() and w.isalpha() and w.islower() and 3 <= len(w) <= 7:
            if tok("x: " + w, add_special_tokens=False)["input_ids"] == ctx + [i]:
                cands.append(i)
    step = max(1, len(cands) // n)
    return [cands[k * step] for k in range(n)]


def make_arm(name, tok):
    coarse, bins, tokens = ARMS[name]
    ids = word_labels(tok, bins) if tokens == "words" else byte_labels(tok, bins)[:bins]
    assert len(set(ids)) == bins, "bin labels are not distinct"
    if coarse:
        assert not set(ids) & set(DIGITS), "a bin label is also a digit label"
    return Arm(name, coarse, bins, tokens, list(ids))


def word_label_ids(tok, prefix, words):
    """The label id of each word as the next token after "<prefix>: " (categorical slots)."""
    return [tok(f"{prefix}: {w}", add_special_tokens=False)["input_ids"][-1] for w in words]
