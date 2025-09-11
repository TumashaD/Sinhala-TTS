import re
import unicodedata
from typing import List
import csv


ZWJ = "\u200D"      # Zero-width joiner
VIRAMA = "\u0DCA"   # Hal (virama) - kills inherent vowel

# Pre-base vowel signs (they appear BEFORE the consonant in Unicode order)
PRE_BASE_SIGNS = {"ෙ", "ේ", "ො", "ෝ", "ෞ"}

# Post/base vowel signs
VOWEL_SIGNS = {
    "ා": "aː",
    "ැ": "æ",
    "ෑ": "æː",
    "ි": "i",
    "ී": "iː",
    "ු": "u",
    "ූ": "uː",
    "ෘ": "ru",   # vocalic r (contextually 'ru' in medial; independent 'ඍ' -> 'ri')
    "ෲ": "ruː",
    "ෙ": "e",   # pre-base
    "ේ": "eː",  # pre-base
    "ො": "o",   # pre-base
    "ෝ": "oː",  # pre-base
    "ෞ": "au",  # pre-base
    "ෛ": "ai",  # Add missing vowel sign AI - Tumasha
}


INDEP_VOWELS = {
    "අ": "a",
    "ආ": "aː",
    "ඇ": "æ",
    "ඈ": "æː",
    "ඉ": "i",
    "ඊ": "iː",
    "උ": "u",
    "ඌ": "uː",
    "එ": "e",
    "ඒ": "eː",
    "ඔ": "o",
    "ඕ": "oː",
    "ඓ": "ai",
    "ඖ": "au",
    "ඍ": "ri",   
}


SPECIAL_SIGNS = {
    "ං": "ŋ",   # anusvara
    "ඃ": "h",   # visarga
}


CONS_MAP = {
    "ක": "k",  "ඛ": "k",  "ග": "g",  "ඝ": "g",  "ඞ": "ŋ", "ඟ": "ŋɡ",
    "ච": "c",  "ඡ": "c",  "ජ": "dʒ","ඣ": "dʒ","ඤ": "ɲ","ඦ": "dʒɲ",
    "ට": "ʈ",  "ඨ": "ʈ",  "ඩ": "ɖ",  "ඪ": "ɖ",  "ණ": "n", "ඬ": "ɖn",
    "ත": "t̪", "ථ": "t̪", "ද": "d̪", "ධ": "d̪", "න": "n", "ඳ": "nd̪",
    "ප": "p",  "ඵ": "p",  "බ": "b",  "භ": "b",  "ම": "m", "ඹ": "mb",
    "ය": "j",  "ර": "r",  "ල": "l",  "ව": "v",  "ෆ": "f",
    "ශ": "ʃ",  "ෂ": "ʂ",  "ස": "s",  "හ": "h",  "ළ": "ɭ",
    "ඥ": "ɲ",  # Add missing GNA character - Tumasha
}

# Helpers for regex classes (treat these as “consonant symbols” in the phoneme string)
CONSONANTS = sorted({
    "k","g","ŋ","c","dʒ","ɲ","ʈ","ɖ","n","t̪","d̪","p","b","m","mb","nd̪","ɖn","ŋɡ",
    "j","r","l","v","f","ʃ","ʂ","s","h","ɭ"
}, key=len, reverse=True)


VOWELS = ["a","aː","æ","æː","i","iː","u","uː","e","eː","o","oː","ə","ai","au","ri","ru","ruː"]


CONS_CLASS = r"[bcdfghjklmnprstʃʂvɖʈɭɲŋ]"

def is_vowel_token(tok: str) -> bool:
    return tok in VOWELS

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def word_to_initial_phonemes(word: str) -> str:

    out: List[str] = []
    pending_pre_vowel: str | None = None
    last_vowel_idx: int | None = None

    i = 0
    while i < len(word):
        ch = word[i]

        # Ignore ZWJ
        if ch == ZWJ:
            i += 1
            continue

        # Pre-base vowel sign buffer
        if ch in PRE_BASE_SIGNS:
            pending_pre_vowel = ch
            i += 1
            continue

        # Independent vowels
        if ch in INDEP_VOWELS:
            out.append(INDEP_VOWELS[ch])
            last_vowel_idx = None
            pending_pre_vowel = None
            i += 1
            continue

        # Nasal/visarga
        if ch in SPECIAL_SIGNS:
            out.append(SPECIAL_SIGNS[ch])
            last_vowel_idx = None
            i += 1
            continue

        # Consonants
        if ch in CONS_MAP:
            base = CONS_MAP[ch]
            out.append(base)
            # default: add schwa
            out.append("ə")
            last_vowel_idx = len(out) - 1

            # apply pending pre-base sign to this consonant
            if pending_pre_vowel:
                vow = VOWEL_SIGNS.get(pending_pre_vowel)
                if vow:
                    out[last_vowel_idx] = vow
                pending_pre_vowel = None


            j = i + 1
            while j < len(word) and word[j] in VOWEL_SIGNS and word[j] not in PRE_BASE_SIGNS:
                out[last_vowel_idx] = VOWEL_SIGNS[word[j]]
                j += 1

            # Virama right after consonant (kills inherent vowel)
            if j < len(word) and word[j] == VIRAMA:
                # remove the last vowel we just placed (schwa or sign)
                if last_vowel_idx is not None and last_vowel_idx == len(out) - 1:
                    out.pop()
                    last_vowel_idx = None
                j += 1  # consume virama

            i = j
            continue

        # Any other char (space, punctuation, digits): pass through
        out.append(ch)
        last_vowel_idx = None
        i += 1

    return "".join(out)


def apply_rule_1(word: str) -> str:
    """
    Rule #1:
    If the nucleus of the first syllable is schwa, replace by 'a'
    EXCEPT if:
      (a) word starts with 'sv' cluster
      (b) word starts with 'kər' (k + schwa + r)
      (c) the word is a single-syllable CV (e.g., 'də')
    """
    # quick exits (spaces/punct)
    if not word or word[0].isspace():
        return word

    if re.fullmatch(fr"{CONS_CLASS}ə", word):
        return word

    m = re.match(fr"^({CONS_CLASS}+?)ə", word)
    if not m:
        return word

    if word.startswith("sv"):
        return word

    if word.startswith("kər"):
        return word


    return re.sub(fr"^({CONS_CLASS}+?)ə", r"\1a", word, count=1)

def apply_rules_2_3_4_7_repeated(word: str) -> str:
    """
    Apply Rules #2, #3, #4, #7 repeatedly until no more changes.
    """
    changed = True
    s = word
    while changed:
        before = s


        # (a) C r ə h → C r a h
        s = re.sub(fr"({CONS_CLASS})rəh", r"\1rah", s)
        # (b) C r ə (C≠h) → C r a (C)
        s = re.sub(fr"({CONS_CLASS})rə({CONS_CLASS})(?!)", r"\1ra\2", s)
        # (c) C r a (C≠h) → C r ə (C)
        s = re.sub(fr"({CONS_CLASS})ra({CONS_CLASS})", r"\1rə\2", s)
        # (d) C r a h → keep (no-op)

        # Rule #3: V ə h with V ∈ {a,e,æ,o,ə} → V a h
        s = re.sub(r"([aeæoə])əh", r"\1ah", s)

        # Rule #4: ə followed by a consonant cluster (two consonants) → a
        s = re.sub(fr"ə({CONS_CLASS})({CONS_CLASS})", r"a\1\2", s)

        # Rule #7: k ə (r|l) u → k a (r|l) u
        s = re.sub(r"kə([rl])u", r"ka\1u", s)

        changed = (s != before)
    return s

def apply_rule_5(word: str) -> str:
    """
    Rule #5: Word-final ... ə C$ → ... a C$, except when C ∈ {r, b, ɖ, ʈ}
    """
    EXCEPT = {"r", "b", "ɖ", "ʈ"}
    def repl(m):
        c = m.group(1)
        return f"ə{c}" if c in EXCEPT else f"a{c}"
    return re.sub(fr"ə({CONS_CLASS})$", repl, word)

def apply_rule_6(word: str) -> str:
    """
    Rule #6: ə j i $ → a j i $
    """
    return re.sub(r"əji$", r"aji", word)

def apply_rule_8(word: str) -> str:
    """
    Rule #8: 'kal' contexts → change a→ə in specific patterns
      - kal(aː|eː|oː)j  → kəl(aː|eː|oː)j
      - kale(m|h)(u|i) → kəle...
      - kaləh(u|i)     → kəleh...
      - kalə           → kələ
    """
    s = word
    s = re.sub(r"kal(aː|eː|oː)j", r"kəl\1j", s)
    s = re.sub(r"kale([mh])([ui])", r"kəle\1\2", s)
    s = re.sub(r"kaləh([ui])", r"kəleh\1", s)
    s = re.sub(r"kalə", r"kələ", s)
    return s

def apply_all_rules(word: str) -> str:
    parts = word.split(" ")
    out_parts = []
    for p in parts:
        if not p:
            out_parts.append(p)
            continue
        w = p
        w = apply_rule_1(w)                # #1 (single pass)
        w = apply_rules_2_3_4_7_repeated(w)# #2,#3,#4,#7 (repeat until stable)
        w = apply_rule_5(w)                # #5
        w = apply_rule_6(w)                # #6
        w = apply_rule_8(w)                # #8 (single pass set)
        out_parts.append(w)
    return " ".join(out_parts)


def sinhala_to_ipa(word: str) -> str:
    raw = word_to_initial_phonemes(word)
    return apply_all_rules(raw)

def convert_text(text: str) -> str:
    text = normalize_text(text)
    # Convert token-by-token (preserve punctuation & whitespace)
    tokens = re.findall(r"\S+|\s+", text)
    out = []
    for tok in tokens:
        if tok.isspace():
            out.append(tok)
        else:
            out.append(sinhala_to_ipa(tok))
    return "".join(out)

def convert_file(input_path: str, output_path: str):
    new_rows = []
    count = 0
    
    with open(input_path, "r", encoding="utf-8", newline='') as f:
        reader = csv.reader(f, delimiter='|')
        
        for row in reader:
            if len(row) >= 4:
                file_id, _, sinhala, speaker = row[:4]
                if speaker.lower() == "mettananda":
                    count += 1
                    sinhala = sinhala.replace("\n", " ").replace("\r", " ").strip()
                    print(f"[{count}] Converting: {sinhala}")
                    ipa = convert_text(sinhala)
                    new_rows.append([file_id, sinhala, ipa])

    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(new_rows)

    print(f"[✓] Saved filtered metadata → {output_path}")

if __name__ == "__main__":
    # Example usage
    convert_file("original.csv", "output_ipa.csv")