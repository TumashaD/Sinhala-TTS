import re
import unicodedata
from typing import List


ZWJ = "\u200D"      # Zero-width joiner
VIRAMA = "්"   # Hal (virama) - kills inherent vowel

# Post/base vowel signs
VOWEL_SIGNS = {
    "ා": "aː", "ැ": "æ", "ෑ": "æː",
    "ි": "i", "ී": "iː", "ු": "u", "ූ": "uː",
    "ෘ": "ru", "ෲ": "ruː",
    "ෙ": "e", "ේ": "eː", "ො": "o", "ෝ": "oː",
    "ෞ": "au", "ෛ": "ai"
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
}

# Helpers for regex classes (treat these as “consonant symbols” in the phoneme string)
CONSONANTS = sorted({
    "k","g","ŋ","c","dʒ","ɲ","ʈ","ɖ","n","t̪","d̪","p","b","m","mb","nd̪","ɖn","ŋɡ",
    "j","r","l","v","f","ʃ","ʂ","s","h","ɭ"
}, key=len, reverse=True)


CONS_CLASS = r"[bcdfghjklmnprstʃʂvɖʈɭɲŋ]"
CONSONANT_TOKENS = set(CONS_MAP.values())
VOWEL_TOKENS = set(["a","aː","æ","æː","i","iː","u","uː","e","eː","o","oː","au","ai","ru","ri","ə"])


def is_consonant_token(tok: str) -> bool:
    return tok in CONSONANT_TOKENS

def is_vowel_token(tok: str) -> bool:
    return tok in VOWEL_TOKENS

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def word_to_initial_phonemes(word: str) -> str:
    out = []
    last_vowel_idx = None
    i = 0
    L = len(word)
    print("Processing word:", word, "length:", L)

    while i < L:
        ch = word[i]

        # Independent vowels
        if ch in INDEP_VOWELS:
            out.append(INDEP_VOWELS[ch])
            last_vowel_idx = None
            i += 1
            continue

        # Special signs
        if ch in SPECIAL_SIGNS:
            out.append(SPECIAL_SIGNS[ch])
            last_vowel_idx = None
            i += 1
            continue

        # --- Repaya check (ර් + consonant) ---
        if (
            i + 1 < L
            and ch == "ර"
            and word[i+1] == VIRAMA 
            and i + 2 < L
            and word[i+2] in CONS_MAP
        ):
            # Add leading "r"
            out.append("r")
            # Process the following consonant normally
            ch = word[i+2]
            out.append(CONS_MAP[ch])
            out.append("ə")  # default schwa
            last_vowel_idx = len(out) - 1
            j = i + 3

            # Check for dependent vowels after repaya cluster
            while j < L and word[j] in VOWEL_SIGNS:
                out[last_vowel_idx] = VOWEL_SIGNS[word[j]]
                j += 1

            # Virama cancels schwa
            if j < L and word[j] == VIRAMA:
                if last_vowel_idx is not None and last_vowel_idx == len(out) - 1:
                    out.pop()
                    last_vowel_idx = None
                j += 1

            i = j
            continue

        # --- Consonants (normal flow) ---
        if ch in CONS_MAP:
            base = CONS_MAP[ch]
            out.append(base)
            out.append("ə")  # schwa
            last_vowel_idx = len(out) - 1
            print(out)
            j = i + 1

            if j < L and word[j] == ZWJ: j+=1  # skip ZWJ if present

            # Handle rakaransaya / yansaya
            if (
                j + 2 < L
                and word[j] == VIRAMA
                and word[j+1] == ZWJ
                and word[j+2] in ("ර", "ය")
            ):
                if last_vowel_idx is not None and last_vowel_idx == len(out) - 1:
                    out.pop()
                    last_vowel_idx = None
                out.append("r" if word[j+2] == "ර" else "j")
                j += 3
                if j < L and word[j] in VOWEL_SIGNS:
                    out.append(VOWEL_SIGNS[word[j]])
                    j += 1
                i = j
                continue

            # Normal dependent vowels
            while j < L and word[j] in VOWEL_SIGNS:
                out[last_vowel_idx] = VOWEL_SIGNS[word[j]]
                j += 1

            # Virama cancels schwa
            if j < L and word[j] == VIRAMA:
                if last_vowel_idx is not None and last_vowel_idx == len(out) - 1:
                    out.pop()
                    last_vowel_idx = None
                j += 1

            i = j
            continue

        # Dependent vowel by itself
        if ch in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[ch])
            last_vowel_idx = None
            i += 1
            continue

        # Virama alone
        if ch == VIRAMA:
            if last_vowel_idx is not None and last_vowel_idx == len(out) - 1:
                out.pop()
                last_vowel_idx = None
            i += 1
            continue

        # Other characters (punctuation, whitespace, etc.)
        out.append(ch)
        last_vowel_idx = None
        i += 1

    return "".join(out)


def rule1_initial_schwa_to_a(tokens: List[str]) -> bool:
    """
    Rule #1: If the nucleus of the first syllable is schwa, replace with 'a'
    EXCEPT:
      - single-syllable CV words (tokens length == 2 and pattern C 'ə')
      - words starting with 's' 'v' cluster (sv...). (conservative check)
      - words starting with [k, 'ə', 'r'] (paper exception) - mapped approximately
    """
    # find index of first vowel token
    for idx, t in enumerate(tokens):
        if t in VOWEL_TOKENS:
            first_v_idx = idx
            break
    else:
        return False

    if tokens[first_v_idx] != "ə":
        return False

    # Exception: single CV (e.g., ['d', 'ə'])
    if len(tokens) == 2 and is_consonant_token(tokens[0]) and tokens[1] == "ə":
        return False

    # Exception: starts with sv cluster
    if len(tokens) >= 2 and tokens[0] == "s" and tokens[1] == "ʋ":
        return False

    # Exception approximation: k ə r (if this exact sequence appears at start)
    if len(tokens) >= 3 and tokens[0] == "k" and tokens[1] == "ə" and tokens[2] == "r":
        return False

    # otherwise change first schwa to 'a'
    tokens[first_v_idx] = "a"
    return True

def rule2_r_context(tokens: List[str]) -> bool:
    """
    Rule #2 family: r-context alternations. Implemented as several passes:
     - C r ə h  -> C r a h
     - C r ə C(not h) -> C r a C
     - C r a C -> C r ə C  (an alternating rule per the paper; we implement both and rely on iteration)
    """
    changed = False
    i = 0
    while i + 3 <= len(tokens) - 1:
        # pattern C r ə h
        if is_consonant_token(tokens[i]) and tokens[i+1] == "r" and tokens[i+2] == "ə" and tokens[i+3] == "h":
            tokens[i+2] = "a"
            changed = True
            i += 4
            continue
        # pattern C r ə C (C != 'h')
        if is_consonant_token(tokens[i]) and tokens[i+1] == "r" and tokens[i+2] == "ə" and is_consonant_token(tokens[i+3]) and tokens[i+3] != "h":
            tokens[i+2] = "a"
            changed = True
            i += 4
            continue
        # pattern C r a C -> C r ə C  (may toggle)
        if is_consonant_token(tokens[i]) and tokens[i+1] == "r" and tokens[i+2] == "a" and is_consonant_token(tokens[i+3]):
            tokens[i+2] = "ə"
            changed = True
            i += 4
            continue
        i += 1
    return changed

def rule3_v_ә_h(tokens: List[str]) -> bool:
    """
    Rule #3: V ə h  (V in {a,e,æ,o,ə}) -> V a h
    """
    changed = False
    i = 0
    while i + 2 < len(tokens):
        if tokens[i] in {"a", "e", "æ", "o", "ə"} and tokens[i+1] == "ə" and tokens[i+2] == "h":
            tokens[i+1] = "a"
            changed = True
            i += 3
            continue
        i += 1
    return changed

def rule4_schwa_before_cluster(tokens: List[str]) -> bool:
    """
    Rule #4: ə C1 C2 -> a C1 C2 (schwa -> a before consonant cluster)
    """
    changed = False
    i = 0
    while i + 2 < len(tokens):
        if tokens[i] == "ə" and is_consonant_token(tokens[i+1]) and is_consonant_token(tokens[i+2]):
            tokens[i] = "a"
            changed = True
            i += 3
            continue
        i += 1
    return changed

def rule7_k_r_l_u(tokens: List[str]) -> bool:
    """
    Rule #7: k ə (r|l) u -> k a (r|l) u
    """
    changed = False
    i = 0
    while i + 3 < len(tokens):
        if tokens[i] == "k" and tokens[i+1] == "ə" and tokens[i+2] in {"r", "l"} and tokens[i+3] == "u":
            tokens[i+1] = "a"
            changed = True
            i += 4
            continue
        i += 1
    return changed

def rule5_wordfinal(tokens: List[str]) -> bool:
    """
    Rule #5: Word-final ... ə C$ -> ... a C$ except when C in {r,b,ɖ,ʈ}
    """
    if len(tokens) >= 2 and tokens[-2] == "ə" and is_consonant_token(tokens[-1]):
        if tokens[-1] not in {"r", "b", "ɖ", "ʈ"}:
            tokens[-2] = "a"
            return True
    return False

def rule6_aji(tokens: List[str]) -> bool:
    """
    Rule #6: ... ə j i $ -> ... a j i $
    (end of token list pattern)
    """
    if len(tokens) >= 3 and tokens[-3] == "ə" and tokens[-2] == "j" and tokens[-1] == "i":
        tokens[-3] = "a"
        return True
    return False

def rule8_kal_contexts(tokens: List[str]) -> bool:
    """
    Rule #8: Several kal-specific alternations (conservative implementation)
    We implement:
      - k a l (aː|eː|oː) j  -> k ə l (aː|eː|oː) j   (turn 'a' into 'ə' at pos 1)
      - k a l e (m|h) (u|i) -> k ə l e ...
      - k a l ə -> k ə l ə  (ensure k a l ə -> k ə l ə)
    This captures the general intent: switch 'a' -> 'ə' in 'kal...' conditions.
    """
    changed = False
    i = 0
    while i + 3 < len(tokens):
        # pattern k a l X j  where X is a long vowel
        if tokens[i] == "k" and tokens[i+1] in {"a", "aː"} and tokens[i+2] == "l" and i+3 < len(tokens):
            # If next is long vowel and later 'j'
            if i+3 < len(tokens) and tokens[i+3] in {"aː","eː","oː"}:
                # require 'j' after it
                if i+4 < len(tokens) and tokens[i+4] == "j":
                    tokens[i+1] = "ə"
                    changed = True
            # other kal patterns
        i += 1
    # Additional simpler rule: if starts with ['k','a','l','ə'] change a->ə
    if len(tokens) >= 3 and tokens[0] == "k" and tokens[1] == "a" and tokens[2] == "l":
        # convert tokens[1] -> 'ə' if not already
        if tokens[1] != "ə":
            tokens[1] = "ə"
            changed = True
    return changed


def apply_all_rules(tokens: List[str]) -> List[str]:
    """
    Apply the rule set in order. Repeat the group of rules that must be
    applied until no change (these rules can trigger each other).
    """
    # Work on a copy
    toks = list(tokens)
    # Rule #1 once (initial schwa -> a) — paper applies just once
    _ = rule1_initial_schwa_to_a(toks)

    # Rules #2,#3,#4,#7 repeat until stable
    changed = True
    while changed:
        changed = False
        if rule2_r_context(toks):
            changed = True
        if rule3_v_ә_h(toks):
            changed = True
        if rule4_schwa_before_cluster(toks):
            changed = True
        if rule7_k_r_l_u(toks):
            changed = True

    # Then rule #5, #6, #8 (single final passes)
    _ = rule5_wordfinal(toks)
    _ = rule6_aji(toks)
    _ = rule8_kal_contexts(toks)

    return toks


def tokens_to_string(tokens: List[str]) -> str:
    """
    Convert the token list back to a readable IPA string.
    We join tokens without separator but keep punctuation and whitespace tokens unchanged.
    """
    # punctuation / whitespace tokens are single characters from original; keep them as is.
    out = []
    for t in tokens:
        out.append(t)
    # Simply concatenate tokens; tokens are IPA pieces (multi-char allowed).
    return "".join(out)


def sinhala_to_ipa(word: str) -> str:
    toks = word_to_initial_phonemes(word)
    toks2 = apply_all_rules(toks)
    return tokens_to_string(toks2)

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
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    ipa = convert_text(content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ipa)
    print(f"[✓] Converted {input_path} → {output_path}")

if __name__ == "__main__":
    # input_text = """මෛත‍්‍රී පාලනයක්' හදන්න ඇවිල්ලා අද මේ අය ගෙන යන්නේ තුච්ඡ, නින්දිත පාලනයක්
    # කාටවත් ලෙඩේ නම් හොඳ කරන්න බැරි වුණා. මං වතුපිටිවල ඉස්පිරිත‍ාලෙ ළඟ ආයතනයක වැඩ කරනවා පරිගණක නිලධාරිනියක් හැටියට.
    # අප දැක්කා නේ ජාතික වශයෙන් ව‍ූ මේ විපතේදී ඒකාබද්ධ විපක්ෂය බොරදියේ මාළු බාපු ආකාරය.
    # මම කම්පියුටර් භාවිතා කරනවා. අම්මා ගියා, තත්ත්‍රි ගුරුත්‍රාණය කියලා කියනවා. අද 2025 දින රත්මලානේ යුර්සිටි තුළ කාර්යය තියනවා.
    # "ශ්‍රී ලංකා" කියන රටේ නාමය ලොව පුරා ප්‍රසිද්ධයි. අපේ ක්‍යාලේජ් ළමයි නින්දිත රැකියාවක් ගැන කතා කළා.
    # කුමරු කාර්යං කරලා ගියේ නාගරික මණ්ඩපය."""
    # print("input:", input_text)
    # print("output:", convert_text(input_text))
    # Example usage
    convert_file("input.txt", "output_ipa.txt")
