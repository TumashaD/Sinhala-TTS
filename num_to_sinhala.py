# sinhala_number_expander.py
# Produce Sinhala word expansions for numeric tokens.
# Two modes:
#  - mode="spoken"        -> natural spoken expansion using Indian grouping (thousand/lakh/crore)
#  - mode="digit_by_digit"-> safe fallback that spells each digit
#
# Notes:
# - Uses common Sinhala word forms (based on Wikibooks/education resources).
# - Orthography choices (spacing/concatenation) can be tuned in the SMALL/TEENS/TENS/SCALES dicts.
# - Decimal point reads as "දශම" + digits (digit-by-digit) which is a standard TTS-friendly choice.
#
# Tested for integers, negative numbers, decimals, and comma-separated numbers like "1,234,567".
# Recommended integration: run replace_numbers_in_text() before your G2P pipeline.

from __future__ import annotations
import re
import unicodedata
from typing import List

# -------------------------
# Lexicon (editable)
# Sources & conventions: Wikibooks / common Sinhala number lists (1..100) and scale words.
# See: Wikibooks (Numbers in Sinhala), CLDR compact forms and TacoSi discussion for motivation.
# -------------------------
ZERO = "ශූන්‍ය"   # alternative: "බිංදුව" — pick one consistently

# Explicit small numbers (0..19) using common modern Sinhala forms (Wikibooks)
SMALL = {
    0: ZERO,
    1: "එක",
    2: "දෙ",
    3: "තුන",
    4: "හතර",
    5: "පහ",
    6: "හය",
    7: "හත",
    8: "අට",
    9: "නවය",
    10: "දහය",
    11: "එකොළහ",
    12: "දොළහ",
    13: "දහතුන",
    14: "දහහතර",
    15: "පහලොව",   # some variants: "පහලොව" / "පහළොව"
    16: "දහසය",
    17: "දහහත",
    18: "දහඅට",
    19: "දහනවය"
}

# Tens names (20..90) from common lists (Wikibooks etc.)
TENS = {
    20: "විසි",
    30: "තිහ",
    40: "හතළිහ",
    50: "පනහ",
    60: "හැට",
    70: "හැත්තෑව",
    80: "අසූව",
    90: "අනූව"
}

# Scales (Indian grouping): value -> spoken scale word
# You can tune strings (e.g., "දහස" vs "දහස්") depending on orthography you prefer.
SCALES = [
    (10_000_000, "කෝටි"),
    (100_000,    "ලක්ෂ"),
    (1_000,      "දහස"),
    (100,        "සිය")
]

# Decimal point word
DECIMAL_WORD = "දශම"

# Separator when composing multi-part words (TTS-friendly spacing is safe).
SEP = " "

# Regex for detecting numbers (integers, optionally comma grouped, optional decimal)
NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

# -------------------------
# Helpers
# -------------------------
def _norm_number_token(tok: str) -> str:
    """Normalize commas and Unicode for the numeric token; returns a canonical string."""
    tok = tok.replace(",", "")
    tok = unicodedata.normalize("NFC", tok.strip())
    return tok

def _small_to_words(n: int) -> str:
    """Return words for 0 <= n < 1000 using SMALL and TENS and 'සිය' style composition."""
    assert 0 <= n < 1000
    parts: List[str] = []

    # hundreds
    if n >= 100:
        h = n // 100
        # single hundred: 'සිය' (100)
        if h == 1:
            parts.append("සිය")
        else:
            # e.g., 200 -> 'දෙ' + 'සිය' => 'දෙ සිය' (we keep a space for clarity)
            parts.append(SMALL.get(h, str(h)) + SEP + "සිය")
        n = n % 100

    # tens and ones
    if n == 0:
        pass
    elif n < 20:
        parts.append(SMALL[n])
    elif n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        tens_word = TENS.get(tens)
        if tens_word:
            # e.g., 25 -> 'විසි' + 'පහ' -> 'විසි පහ'
            if ones:
                parts.append(tens_word + SEP + SMALL[ones])
            else:
                parts.append(tens_word)
        else:
            # fallback (shouldn't happen for standard tens)
            if ones:
                parts.append(str(tens) + SEP + SMALL[ones])
            else:
                parts.append(str(tens))
    return SEP.join(parts) if parts else SMALL[0]

def _compose_spoken_integer(n: int) -> str:
    """Compose a spoken Sinhala string for non-negative integer n using SCALES.
       Uses space-separated components for reliability in TTS. """
    if n == 0:
        return SMALL[0]

    parts: List[str] = []
    remaining = n

    for val, label in SCALES:
        if remaining >= val:
            count = remaining // val
            remaining = remaining % val

            # for scales other than 100 we show "<count> <label>" but collapse when count==1
            if count == 1:
                # prefer the bare label (e.g., 'ලක්ෂ' rather than 'එක් ලක්ෂ')
                parts.append(label)
            else:
                # use small-to-words for the count (e.g., 'දෙ ලක්ෂ' or 'දෙලක්ෂ' if you prefer no space)
                count_words = _small_to_words(count)
                parts.append(count_words + SEP + label)

    # leftover < 100 or <1000 after scales
    if remaining:
        parts.append(_small_to_words(remaining))

    return SEP.join(parts)

def _read_decimal_fraction(frac: str) -> str:
    """Read each digit of the fractional part individually (safe approach)."""
    out = []
    for ch in frac:
        if ch.isdigit():
            out.append(SMALL[int(ch)])
        else:
            out.append(ch)
    return SEP.join(out)

# -------------------------
# Public API
# -------------------------
def number_to_sinhala(number_token: str | int | float, mode: str = "spoken") -> str:
    """
    Convert a numeric token to Sinhala words.

    mode:
      - "spoken": natural reading: 2025 -> "දෙ දහස විසි පහ"
      - "digit_by_digit": safe: "2 0 2 5" -> "දෙ ශූන්‍ය දෙ පහ"
    """
    if mode not in {"spoken", "digit_by_digit"}:
        raise ValueError("mode must be 'spoken' or 'digit_by_digit'")

    if isinstance(number_token, (int, float)):
        s = str(number_token)
    else:
        s = str(number_token)

    s = _norm_number_token(s)

    # handle sign
    negative = False
    if s.startswith("-"):
        negative = True
        s = s[1:]

    if mode == "digit_by_digit":
        parts: List[str] = []
        for ch in s:
            if ch.isdigit():
                parts.append(SMALL[int(ch)])
            elif ch == ".":
                parts.append(DECIMAL_WORD)
            else:
                parts.append(ch)
        out = SEP.join(parts)
        return ("මයිනස් " + out) if negative else out

    # spoken mode
    if "." in s:
        intpart, fracpart = s.split(".", 1)
    else:
        intpart, fracpart = s, None

    intpart = intpart or "0"

    try:
        ival = int(intpart)
    except ValueError:
        # fallback to digit-by-digit if parsing fails
        return number_to_sinhala(s, mode="digit_by_digit")

    words = _compose_spoken_integer(ival)

    if fracpart:
        frac_words = _read_decimal_fraction(fracpart)
        words = SEP.join([words, DECIMAL_WORD, frac_words])

    if negative:
        words = "මයිනස් " + words

    return words

def replace_numbers_in_text(text: str, mode: str = "spoken") -> str:
    """Replace numeric tokens in `text` with their Sinhala word expansions."""
    text = unicodedata.normalize("NFC", text)

    def repl(m):
        tok = m.group(0)
        return number_to_sinhala(tok, mode=mode)

    return NUM_RE.sub(repl, text)

# -------------------------
# Quick manual tests (run this module directly)
# -------------------------
if __name__ == "__main__":
    examples = [
        "2025", "0", "15", "25", "100", "101", "110", "999",
        "1000", "2000", "12500", "100000", "2500000", "3.14", "-42",
        "1,234,567", "2025 දින"
    ]
    print("spoken mode:")
    for ex in examples:
        print(f"{ex} -> {number_to_sinhala(ex, mode='spoken')}")
    print("\ndigit_by_digit:")
    for ex in examples:
        print(f"{ex} -> {number_to_sinhala(ex, mode='digit_by_digit')}")
