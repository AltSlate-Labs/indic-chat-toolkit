"""Rule-based fallback romanizer for words missing from the human romanization dictionaries (aksharantar, Dakshina lexicons).
Brahmic scripts share the ISCII-derived Unicode layout, so one table of offsets (relative to the block start) serves Devanagari, Bengali,
Gurmukhi, Gujarati, Tamil, Telugu, Kannada and Malayalam. Chat-style lower-case output ("kya", "nahin"); not a standard scheme.
Urdu: consonant letters only (short vowels are not written), which is roughly how Urdu is often typed in Latin."""
BLOCK = {"hi": 0x900, "mr": 0x900, "bn": 0x980, "pa": 0xA00, "gu": 0xA80, "ta": 0xB80, "te": 0xC00, "kn": 0xC80, "ml": 0xD00}
V = {0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ee", 0x09: "u", 0x0A: "oo", 0x0B: "ri", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x12: "o", 0x13: "o",
     0x14: "au", 0x0D: "e", 0x11: "o"}
M = {0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri", 0x46: "e", 0x47: "e", 0x48: "ai", 0x4A: "o", 0x4B: "o",
     0x4C: "au", 0x45: "e", 0x49: "o", 0x57: "au", 0x55: "", 0x56: "ai"}
C = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t",
     0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p",
     0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v",
     0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "gh", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f",
     0x5F: "y", 0x4E: "t"}
NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}       # candrabindu, anusvara, visarga, Gurmukhi tippi
CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}  # Malayalam chillu letters
VIRAMA, NUKTA, ADDAK = 0x4D, 0x3C, 0x71
INDO_ARYAN = {"hi", "mr", "pa", "gu", "bn"}
UR = {"ا": "a", "آ": "aa", "ب": "b", "پ": "p", "ت": "t", "ٹ": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ڈ": "d",
      "ذ": "z", "ر": "r", "ڑ": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh",
      "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "ں": "n", "و": "o", "ؤ": "o", "ہ": "h", "ھ": "h",
      "ه": "h", "ۃ": "h", "ة": "h", "ی": "i", "ي": "i", "ئ": "i", "ے": "e", "ۓ": "e", "ء": ""}


def romanize_word(w, lang):
    if lang == "ur":
        return "".join(UR.get(c, c if c.isascii() else "") for c in w)
    base = BLOCK[lang]; out = []; pending = False; double = False
    cs = list(w)
    for i, ch in enumerate(cs):
        o = ord(ch) - base
        if not 0 <= o < 0x80:
            if pending: out.append("a"); pending = False
            out.append(ch if ch.isascii() else ""); continue
        if o in C:
            if pending: out.append("a")
            r = C[o]
            if double: r = r[0] + r; double = False
            out.append(r); pending = True
        elif o == NUKTA: continue
        elif o == VIRAMA: pending = False
        elif o in M: out.append(M[o]); pending = False
        elif o in V:
            if pending: out.append("a"); pending = False
            out.append(V[o])
        elif o in NASAL:
            if pending: out.append("a"); pending = False
            out.append("m" if lang == "ml" and o == 0x02 and i == len(cs) - 1 else NASAL[o])
        elif o == ADDAK: double = True
        elif o in CHILLU:
            if pending: out.append("a"); pending = False
            out.append(CHILLU[o])
        elif 0x66 <= o <= 0x6F:
            if pending: out.append("a"); pending = False
            out.append(str(o - 0x66))
        else:
            if pending: out.append("a"); pending = False
    if pending and lang not in INDO_ARYAN: out.append("a")   # Indo-Aryan: final inherent vowel is not pronounced
    return "".join(out)


if __name__ == "__main__":
    for l, w in [("hi", "नहीं"), ("hi", "क्षेत्र"), ("hi", "द्वारा"), ("ta", "தமிழ்"), ("ta", "வெற்றி"), ("te", "తెలుగు"), ("kn", "ಕನ್ನಡ"),
                 ("ml", "മലയാളം"), ("ml", "അവൻ"), ("pa", "ਪੱਕਾ"), ("bn", "বাংলা"), ("gu", "ગુજરાત"), ("mr", "महाराष्ट्र"), ("ur", "پاکستان")]:
        print(l, w, "->", romanize_word(w, l))
