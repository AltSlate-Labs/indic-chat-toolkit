"""TTS-normalization training data: fill Ornith carriers (gen_carriers.py) with written + spoken values from indic-numtowords + CLDR
(months, units) + the hand lexicon (lex_hand.py), so targets are exact. Plus Dakshina TRAIN-split wiki sentences with small integers (rules fill) and
identity rows (no digits). Carriers are split 90/10 BEFORE filling, so the programmatic test never shares a carrier with training.
python fill_data.py CARRIERS.jsonl OUT_DIR [PER_CARRIER=3]   (envs/tn)
Row: {"src": "<lang>-tn: <written>", "tgt": <spoken>, "lang", "kind"}.  Never reads the Dakshina valid/test splits or tn test B."""
import gzip, json, os, random, re, sys, unicodedata
import indic_numtowords as n
from babel import Locale
from lex_hand import LEX, ordinal, W
CLDR_UNITS = {"percent": "concentr-percent", "km": "length-kilometer", "m": "length-meter", "cm": "length-centimeter",
  "kg": "mass-kilogram", "g": "mass-gram", "l": "volume-liter", "ml": "volume-milliliter", "C": "temperature-celsius",
  "GB": "digital-gigabyte", "MB": "digital-megabyte", "hour": "duration-hour", "minute": "duration-minute"}
LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
DIG0 = {"hi": 0x966, "mr": 0x966, "bn": 0x9E6, "pa": 0xA66, "gu": 0xAE6, "ta": 0xBE6, "te": 0xC66, "kn": 0xCE6, "ml": 0xD66, "ur": 0x6F0}
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0/")
UNIT_ABBR = {"km": ["km", "KM"], "m": ["m"], "cm": ["cm"], "kg": ["kg", "KG"], "g": ["g", "gm"], "l": ["L", "l", "ltr"], "ml": ["ml", "mL"],
             "C": ["°C", "° C"], "GB": ["GB"], "MB": ["MB"], "hour": ["hr", "hrs", "h"], "minute": ["min", "mins"]}


SCRIPT = {"hi": "DEVANAGARI", "mr": "DEVANAGARI", "bn": "BENGALI", "pa": "GURMUKHI", "gu": "GUJARATI", "ta": "TAMIL", "te": "TELUGU",
          "kn": "KANNADA", "ml": "MALAYALAM", "ur": "ARABIC"}
CRICKET = {"hi": "क्रिकेट|मैच|रन|विकेट|पारी", "mr": "क्रिकेट|सामन|धाव|बाद|डाव", "bn": "ক্রিকেট|ম্যাচ|রান|উইকেট|ইনিংস", "gu": "ક્રિકેટ|મેચ|રન|વિકેટ",
           "pa": "ਕ੍ਰਿਕਟ|ਮੈਚ|ਦੌੜ|ਵਿਕਟ", "ur": "کرکٹ|میچ|رن|وکٹ", "ta": "கிரிக்கெட்|போட்டி|ரன்|விக்கெட்|இன்னிங்ஸ்", "te": "క్రికెట్|మ్యాచ్|పరుగ|వికెట్",
           "kn": "ಕ್ರಿಕೆಟ್|ಪಂದ್ಯ|ರನ್|ವಿಕೆಟ್", "ml": "ക്രിക്കറ്റ്|മത്സര|റൺ|വിക്കറ്റ്"}
# words that must NOT directly follow the placeholder (the filled value already says them -> "rupees rupees")
AFTER = {"AMOUNT": "रुप|रूप|টাকা|ரூபா|రూపా|రుపా|ರೂಪಾ|രൂപ|ਰੁਪ|روپ|رپے|રૂપ|पैसे|पैसा",
         "PERCENT": "प्रतिशत|टक्के|শতাংশ|சதவீத|శాతం|ಶೇಕಡ|ശതമാന|ਪ੍ਰਤੀਸ਼ਤ|فیصد|ટકા|%",
         "TIME": "बजे|वाज|টা|மணி|గంట|ಗಂಟೆ|മണി|ਵਜੇ|بجے|વાગ",
         "QUANTITY": "किलो|लीटर|लिटर|मीटर|डिग्री|घंट|मिनट|मिनिट|গ্রাম|কিলো|লিটার|মিটার|ডিগ্রি|ঘণ্টা|মিনিট|கிலோ|லிட்டர|மீட்டர|டிகிரி|மணி|நிமிட|"
                     "కిలో|లీటర|మీటర|డిగ్రీ|గంట|నిమిష|ಕಿಲೋ|ಲೀಟರ|ಮೀಟರ|ಡಿಗ್ರಿ|ಗಂಟೆ|ನಿಮಿಷ|കിലോ|ലിറ്റർ|മീറ്റർ|ഡിഗ്രി|മണിക്കൂർ|മിനിറ്റ്|"
                     "ਕਿਲੋ|ਲੀਟਰ|ਮੀਟਰ|ਡਿਗਰੀ|ਘੰਟ|ਮਿੰਟ|کلو|لیٹر|میٹر|ڈگری|گھنٹ|منٹ|કિલો|લિટર|મીટર|ડિગ્રી|કલાક|મિનિટ|GB|MB|km|kg",
         "SCORE": "रन|धाव|রান|ரன்|పరుగ|ರನ್|റൺ|ਦੌੜ|رن|રન"}


def good_carrier(c):
    """Reject Ornith carriers with mixed scripts (e.g. 'प्रashासन'), a unit word repeated right after the value, or a cricket score outside cricket."""
    ph = f"⟦{c['kind']}⟧"; t = c["carrier"]; rest = t.replace(ph, " ")
    scripts = {unicodedata.name(ch, "?").split()[0] for ch in rest if ch.isalpha()}
    if scripts - {SCRIPT[c["lang"]]}: return False
    if re.search(r"(\S+)(?: \1){4}", rest): return False  # v4b: a token repeated 5+ times in a row (ur "ټیل ټیل ټیل ...")
    if re.search(r"[₹$€£%°]", rest): return False  # v4: "₹⟦AMOUNT⟧" kept ₹ in the gold (tn-v3 said "₹एक हज़ार ... रुपये"); 202 carriers
    if re.search(r"\d", rest): return False  # native digits in the carrier (gen_carriers only rejected ASCII 0-9)
    after = t.split(ph, 1)[1].strip()[:12]
    if c["kind"] in AFTER and re.match(f"(?:{AFTER[c['kind']]})", after): return False
    if c["kind"] == "SCORE" and not re.search(CRICKET[c["lang"]], rest): return False
    return True


class Lang:
    def __init__(s, lang):
        loc = Locale.parse(lang); up = loc._data["unit_patterns"]
        def unit(k):
            try: d = up[k]["long"]; return d.get("other") or d.get("one")
            except Exception: return None
        s.l, s.r = lang, random.Random()
        s.x = {**LEX[lang], "zero": W(0, lang), "months": [loc.months["format"]["wide"][i] for i in range(1, 13)],
               "units": {k: unit(v) for k, v in CLDR_UNITS.items()}}

    def digits(s, t):  # 20% native digits where the language uses them in print
        if s.l in ("ta", "ml", "te", "kn") or s.r.random() > 0.2: return t
        return "".join(chr(DIG0[s.l] + int(c)) if c.isdigit() else c for c in t)

    def group(s, v):  # Indian grouping 1,23,456 half the time
        t = str(v)
        if v < 1000 or s.r.random() < 0.5: return t
        return s.igroup(v)

    @staticmethod
    def igroup(v):  # always Indian grouping (ITN canonical form for amounts)
        t = str(v)
        if v < 1000: return t
        head, tail = t[:-3], t[-3:]; parts = []
        while len(head) > 2: parts.insert(0, head[-2:]); head = head[:-2]
        return ",".join([head] + parts + [tail]) if head else ",".join(parts + [tail])

    def dec(s, a, b):  # 12.5 -> twelve point five (digits read one by one)
        return f"{W(a, s.l)} {s.x['point']} " + " ".join(s.x["zero"] if c == "0" else W(c, s.l) for c in b)

    HUNDRED = {"hi": " सौ ", "ur": " سو ", "pa": " ਸੌ ", "mr": "शे ", "gu": "સો ", "bn": "শো ", "te": " వందల "}  # year style "nineteen hundred forty-two"

    def year(s, y):  # ta/kn/ml read years as plain cardinals
        h = s.HUNDRED.get(s.l)
        if h and 1100 <= y <= 1999:
            if y % 100 == 0 and s.l == "te": h = " వందలు "
            return (W(y // 100, s.l) + h + (W(y % 100, s.l) if y % 100 else "")).strip()
        return W(y, s.l)

    def unit(s, k, words):
        p = s.x["units"].get(k) or s.x.get(k)
        return p.replace("{0}", words) if p and "{0}" in p else None

    def value(s, kind):  # -> (written, spoken) or None if the lexicon lacks a needed part
        r, x, l = s.r, s.x, s.l
        if kind == "NUMBER":
            v = r.choice([r.randint(2, 99), r.randint(100, 999), r.randint(1000, 99999), r.randint(100000, 99999999)])
            s.canon = str(v); return s.digits(s.group(v)), W(v, l)
        if kind == "AMOUNT":
            v = r.choice([r.randint(5, 999), r.randint(1000, 99999), r.randint(100000, 9999999)]); p = r.choice([0, 0, 0, r.randint(1, 99)])
            num = s.group(v) + (f".{p:02d}" if p else ""); w = r.choice(["₹{}", "₹ {}", "Rs. {}", "Rs {}", "INR {}"]).format(num)
            sp = f"{W(v, l)} {x['rupees']}" + (" ".join(["", x["and_"], W(p, l), x["paise"]]).replace("  ", " ") if p else "")
            s.canon = "₹" + s.igroup(v) + (f".{p:02d}" if p else ""); return s.digits(w), sp
        if kind == "TIME":
            h, m = r.randint(1, 12), r.choice([0, 0, 15, 30, 45, r.randint(1, 59)]); ap = r.choice(["", "", "AM", "PM", "am", "pm", "a.m.", "p.m."])
            w = f"{h}:{m:02d}" + (" " + ap if ap else "")
            sp = x["oclock"].format(h=W(h, l)) if m == 0 else x["time"].format(h=W(h, l), m=W(m, l))
            if ap:
                pm = ap.lower().startswith("p")
                word = (x["night"] if h == 12 or h <= 3 else x["am"]) if not pm else x["noon"] if h == 12 or h <= 4 else x["eve"] if h <= 7 else x["night"]
                sp = f"{word} {sp}"
            s.canon = (f"{word} " if ap else "") + (x["oclock"].format(h=h) if m == 0 else f"{h}:{m:02d}")
            return s.digits(w), sp
        if kind in ("DATE", "YEAR"):
            y = r.choice([r.randint(1900, 1999), r.randint(2000, 2030), r.randint(1100, 1899)]) if kind == "YEAR" else r.randint(1950, 2030)
            if kind == "YEAR": s.canon = str(y); return s.digits(str(y)), s.year(y)
            d, mo = r.randint(1, 28), r.randint(1, 12); f = r.randint(0, 3)
            w = [f"{d:02d}/{mo:02d}/{y}", f"{d}-{mo}-{y}", f"{d} {x['months'][mo - 1]} {y}", f"{d}.{mo:02d}.{y}"][f]
            s.canon = f"{d} {x['months'][mo - 1]} {y}"; return s.digits(w), f"{W(d, l)} {x['months'][mo - 1]} {s.year(y)}"
        if kind == "PERCENT":
            a = r.randint(1, 99); b = str(r.randint(1, 9)) if r.random() < 0.3 else ""
            words = s.dec(a, b) if b else W(a, l); sp = s.unit("percent", words); s.canon = f"{a}.{b}%" if b else f"{a}%"
            return (s.digits(f"{a}.{b}%" if b else f"{a}%"), sp) if sp else None
        if kind == "DECIMAL":
            a, b = r.randint(0, 999), str(r.randint(1, 99))
            s.canon = f"{a}.{b}"; return s.digits(f"{a}.{b}"), s.dec(a, b)
        if kind == "PHONE":
            ph = str(r.randint(6, 9)) + "".join(str(r.randint(0, 9)) for _ in range(9))
            w = r.choice([ph, ph[:5] + " " + ph[5:], ph[:5] + "-" + ph[5:]])
            s.canon = ph; return s.digits(w), " ".join(x["zero"] if c == "0" else W(c, l) for c in ph)
        if kind == "QUANTITY":
            k = r.choice(list(UNIT_ABBR)); v = r.choice([r.randint(1, 99), r.randint(100, 999)])
            if k == "C": v = r.randint(5, 48)
            sp = s.unit(k, W(v, l)); s.canon = s.unit(k, str(v))
            return (s.digits(f"{v}{r.choice(['', ' '])}{r.choice(UNIT_ABBR[k])}"), sp) if sp else None
        if kind == "ORDINAL":
            v = r.choice([r.randint(1, 10), r.randint(11, 50)]); o = ordinal(l, v, r)
            s.canon = o[1] if o else None; return (s.digits(o[0]), o[1]) if o else None  # ITN keeps ordinal words
        if kind == "RANGE":
            a = r.randint(1, 90); b = a + r.randint(1, 30)
            s.canon = x["range"].format(a=a, b=b)
            return s.digits(f"{a}{r.choice(['-', '–', ' - ', ' to '])}{b}"), x["range"].format(a=W(a, l), b=W(b, l))
        if kind == "SCORE":
            ru, wk = r.randint(40, 420), r.randint(1, 9)
            s.canon = x["score"].format(r=ru, w=wk)
            return s.digits(r.choice([f"{ru}/{wk}", f"{ru}-{wk}"])), x["score"].format(r=W(ru, l), w=W(wk, l))
        return None


PAIRK = ("AMOUNT", "QUANTITY", "NUMBER", "PERCENT", "DECIMAL")


def fill(langs, rows, per, seed, pair=0.0):
    """pair (v4): with this probability a slot of a PAIRK kind gets a second value after ", " (tn-v3 fused "₹15,999, 128 GB" into one
    number: a comma + space between two numbers looked like digit grouping); such rows get kind PAIR."""
    out = []
    for i, c in enumerate(rows):
        L = langs.get(c["lang"])
        if not L: continue
        for j in range(per):
            L.r.seed(f"{seed}-{i}-{j}"); L.canon = None
            try: v = L.value(c["kind"])
            except Exception: v = None
            canon = L.canon
            if not v or re.search(r"[0-9]", v[1]): continue
            kind = c["kind"]
            if pair and kind in PAIRK and L.r.random() < pair:
                try: v2 = L.value(L.r.choice(PAIRK))
                except Exception: v2 = None
                if not v2 or re.search(r"[0-9]", v2[1]): continue
                v, kind = (f"{v[0]}, {v2[0]}", f"{v[1]}, {v2[1]}"), "PAIR"; canon = f"{canon}, {L.canon}" if canon and L.canon else None
            ph = f"⟦{c['kind']}⟧"
            out.append({"src": f"{c['lang']}-tn: " + c["carrier"].replace(ph, v[0]), "tgt": c["carrier"].replace(ph, v[1]),
                        "lang": c["lang"], "kind": kind, "itn": c["carrier"].replace(ph, canon) if canon else None})  # itn: canonical written form
    return out


WIKI_BAD = re.compile(r"[₹$%#]|\d[.,:/]\d|[A-Za-z]-?\d|\d-?[A-Za-z]|\d{8,}")  # decimals, times, dates, grouped numbers, Latin codes (J-8)


def combine(rows, n, k, seed):
    """v3: n multi-number rows, each joining k (2 or 3) filled carrier rows of the same language (tn-v2 misread or invented numbers
    when a sentence had more than one, since every carrier has exactly one slot)."""
    r = random.Random(seed); by = {}
    for x in rows: by.setdefault(x["lang"], []).append(x)
    out = []
    for i in range(n):
        lang = r.choice(sorted(by)); parts = r.sample(by[lang], k if k else r.choice([2, 2, 3]))
        out.append({"src": parts[0]["src"] + " " + " ".join(x["src"].split(": ", 1)[1] for x in parts[1:]),
                    "tgt": " ".join(x["tgt"] for x in parts), "lang": lang, "kind": "MULTI",
                    "itn": " ".join(x["itn"] for x in parts) if all(x.get("itn") for x in parts) else None})
    return out


def wiki(L, k_num, k_id):
    """Dakshina TRAIN split only. v2: any integer up to 7 digits, incl. years (read in the language's year style when 4 digits,
    1100-1999), numbers glued to suffixes (500க்கும், ১টি) and number-number ranges (1859-1941); the suffix/hyphen is kept as written."""
    lang = L.l; num, ident = [], []
    def rep(m):
        d = m.group(0); v = int(d)
        return L.year(v) if len(d) == 4 and 1100 <= v <= 2099 else W(v, lang)
    with gzip.open(f"{D}{lang}/native_script_wikipedia/{lang}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as g:
        for t in g:
            t = " ".join(t.split())
            if not 30 <= len(t) <= 400: continue
            if not re.search(r"\d", t):
                if len(ident) < k_id: ident.append({"src": f"{lang}-tn: {t}", "tgt": t, "lang": lang, "kind": "IDENTITY", "itn": t})
            elif len(num) < k_num and not WIKI_BAD.search(t):
                num.append({"src": f"{lang}-tn: {t}", "tgt": re.sub(r"\d+", rep, t), "lang": lang, "kind": "WIKI",
                            "itn": re.sub(r"\d", lambda m: str(int(m.group(0))), t)})  # ITN target: native digits -> ASCII
            if len(num) >= k_num and len(ident) >= k_id: break
    return num + ident


if __name__ == "__main__":
    cf, out = sys.argv[1:3]; per = int(sys.argv[3]) if len(sys.argv) > 3 else 3; os.makedirs(out, exist_ok=True)
    langs = {l: Lang(l) for l in LANGS}
    car = [json.loads(x) for x in open(cf, encoding="utf-8")]; n0 = len(car)
    car = [c for c in car if good_carrier(c)]
    print("carriers kept", len(car), "of", n0)
    r = random.Random(0); r.shuffle(car)
    test_c = [c for i, c in enumerate(car) if i % 10 == 0]; train_c = [c for i, c in enumerate(car) if i % 10]
    train = fill(langs, train_c, per, "train", pair=0.15) + fill(langs, [c for c in train_c if c["kind"] == "SCORE"], 7, "score")  # SCORE: 10 fills
    test = fill(langs, test_c, 1, "test")
    train += combine(train, 30000, 0, "multi-train")
    test_multi = combine(test, 1000, 2, "multi-test")  # built only from held-out carriers
    test_pair = [x for x in fill(langs, [c for c in test_c if c["kind"] in PAIRK], 1, "test-pair", pair=1.0) if x["kind"] == "PAIR"]
    for l in langs: train += wiki(langs[l], 6000, 1500)
    n1 = len(train); train = [x for x in train if len(x["src"].encode()) <= 760 and len(x["tgt"].encode()) <= 1500]
    print("dropped over-length rows", n1 - len(train))
    r.shuffle(train); dev, train = train[:2000], train[2000:]
    for name, rows in (("train", train), ("dev", dev), ("test-prog", test), ("test-multi", test_multi), ("test-pair", test_pair)):
        with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from collections import Counter
    print("langs", sorted(langs), "train", len(train), "dev", len(dev), "test-prog", len(test))
    print("train kinds", dict(Counter(x["kind"] for x in train)))
    print("test kinds", dict(Counter(x["kind"] for x in test)))
    for k in ("AMOUNT", "TIME", "DATE", "ORDINAL", "SCORE", "PERCENT"):
        ex = [x for x in train if x["kind"] == k][:3]
        for x in ex: print(k, "|", x["src"], "=>", x["tgt"])
