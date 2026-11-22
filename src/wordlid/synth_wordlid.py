"""Word-level LID data: one label per word. python synth_wordlid.py OUT_DIR [N_TRAIN=200000]   (envs/pii: needs pandas)
Labels: the 12 Dakshina languages (script is not part of the label), en, other (digits, punctuation, other languages).
Rows mix segments, so a row can hold 2-3 languages; this is what indic-chat-lid-small (one label per message) cannot express.
Sources (same as src/lid/synth_lid.py): Dakshina v1.0 (CC BY-SA 4.0) native wiki TRAIN sentences (minus romanized dev/test
sentences), TRAIN word lexicons for synthetic romanization, romanized DEV sentences (human romanizations) except a held-out part;
jobs/background-v1 'en' sentences (Apache-2.0); wikimedia/wikipedia shards of 27 Latin-script languages (CC BY-SA) as 'other'.
Test (test-mix.jsonl): built ONLY from Dakshina TEST split (native + human romanizations), held-out English sentences and held-out
'other' sentences, so no test sentence is in training."""
import collections, glob, gzip, json, os, random, re, sys
import pandas as pd

D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
LABELS = LANGS + ["en", "other"]
BG1 = os.path.expanduser("~/agent/rnd/jobs/background-v1/out/background.jsonl")
EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
OTHER_TOK = re.compile(r"^[\W\d_]+$")  # digits, punctuation, symbols, emoji


def lab(w, l):  # label for one token of a language-l segment
    return "other" if OTHER_TOK.match(w) else l


EV = None  # v3: English-word convention (envocab.py); set via ENVOCAB env var
LAT = re.compile(r"^[A-Za-z'.,!?;:()\"-]+$")


def is_en(w):
    k = re.sub(r"[^a-z]", "", w.lower())
    if not (EV and k) or k in EV["hi"]: return False
    return (k in EV["top"] and (len(k) >= 4 or k in EV["en"])) or k in EV["comi_en"]


def conv(ws):  # Latin English words inside romanized Indic segments -> en (COMI-LINGUA convention)
    return [(w, "en" if y in LANGS and LAT.match(w) and is_en(w) else y) for w, y in ws]


def lexicon(l):
    lex = {}
    for line in open(f"{D}/{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3: lex.setdefault(p[0], []).append((p[1], int(p[2])))
    return lex


def chat(r, w):  # chat noise on one romanized word
    if r.random() < 0.12 and len(w) > 3: w = w[0] + re.sub(r"[aeiou]", lambda m: "" if r.random() < 0.6 else m.group(), w[1:-1]) + w[-1]
    if r.random() < 0.03: w = w + w[-1] * r.randint(1, 3)
    return w


def win(r, words, lo=2, hi=14):
    k = r.randint(lo, hi)
    if len(words) <= k: return words
    s = r.randrange(len(words) - k + 1); return words[s:s + k]


def native_lines(l, split, n, excl=()):
    if split == "train":
        f = gzip.open(f"{D}/{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8")
    else:
        f = open(f"{D}/{l}/romanized/{l}.romanized.rejoined.test.native.txt", encoding="utf-8")
    out = []
    for line in f:
        t = line.strip()
        if t and t not in excl and len(t.split()) >= 3: out.append(t.split())
        if len(out) >= n: break
    return out


class Src:
    """Segment sources for one split ('train' or 'test')."""
    def __init__(s, split, r):
        s.r, s.split = r, split; s.nat, s.rom, s.lex = {}, {}, {}
        for l in LANGS:
            rp = f"{D}/{l}/romanized/{l}.romanized.rejoined."
            if split == "train":
                excl = {x.strip() for sp in ("dev", "test") for x in open(rp + sp + ".native.txt", encoding="utf-8")}
                s.nat[l] = native_lines(l, "train", 40000, excl); s.lex[l] = lexicon(l)
                devr = open(rp + "dev.roman.txt", encoding="utf-8").read().splitlines()
                s.rom[l] = [x.split() for x in devr[300:] if x.strip()]  # first 300 dev lines kept out (as in synth_lid)
            else:
                s.nat[l] = native_lines(l, "test", 100000)
                s.rom[l] = [x.split() for x in open(rp + "test.roman.txt", encoding="utf-8").read().splitlines() if x.strip()]
            print(split, l, "native", len(s.nat[l]), "roman", len(s.rom[l]), flush=True)
        bg = [json.loads(x) for x in open(BG1, encoding="utf-8")]
        en = [x["text"].split() for x in bg if x["src"] == "en"]
        s.en = en[:-3000] if split == "train" else en[-3000:]
        oth = []
        for d in sorted(glob.glob(os.path.expanduser("~/Downloads/data/wikipedia-other/*"))):
            fs = glob.glob(d + "/**/*.parquet", recursive=True)
            texts = pd.read_parquet(fs[0], columns=["text"]).text.head(3000).tolist()
            sents = [x.split() for t in texts for x in re.split(r"(?<=[.!?])\s+", t) if len(x.split()) >= 3][:2600]
            oth += sents[:2300] if split == "train" else sents[2300:2600]
        s.oth = oth
        # v2: English Wikipedia sentences (CC BY-SA) next to the short command-style en sentences (v1 English was too narrow:
        # COMI news words like Student, Leader, appellate were labelled hi)
        fs = glob.glob(os.path.expanduser("~/Downloads/data/wikipedia-en/**/*.parquet"), recursive=True)
        texts = pd.read_parquet(fs[0], columns=["text"]).text.head(20000).tolist()
        sents = [x.split() for t in texts for x in re.split(r"(?<=[.!?])\s+", t) if 4 <= len(x.split()) <= 40]
        s.enw = sents[:80000] if split == "train" else sents[-3000:]
        s.chat = {}
        if split == "train":  # v3: Ornith chat messages (MIT; jobs/lid-chat, lid-chat4) + Hinglish-Everyday-Conversations (MIT) for hi
            for f in os.environ.get("CHAT", "").split(","):
                if f:
                    for x in (json.loads(t) for t in open(f, encoding="utf-8")):
                        if x["lang"] in LANGS: s.chat.setdefault(x["lang"], []).append(x["roman"].split())
            s.chat.setdefault("hi", []).extend(x["text"].split() for x in bg if x["src"] == "hinglish-chat")
            print("chat segments", {k: len(v) for k, v in s.chat.items()}, flush=True)

    def native(s, l):
        return [(w, lab(w, l)) for w in win(s.r, s.r.choice(s.nat[l]))]

    def roman(s, l):
        return conv(s.roman0(l))

    def roman0(s, l):
        r = s.r
        if s.split == "train" and s.chat.get(l) and r.random() < 0.3:  # v3: real romanized chat (Ornith messages; Hinglish chat for hi)
            return [(chat(r, w) if r.random() < 0.5 else w, lab(w, l)) for w in win(r, r.choice(s.chat[l]))]
        if s.split == "test" or r.random() < 0.35:  # human romanizations (Dakshina dev for train, test for test)
            return [(chat(r, w) if s.split == "train" else w, lab(w, l)) for w in win(r, r.choice(s.rom[l]))]
        for _ in range(20):  # lexicon-synthesized romanization of a native sentence
            out, hit = [], 0; ws = win(r, r.choice(s.nat[l]))
            for w in ws:
                a, core, b = EDGE.match(w).groups()
                if core in s.lex[l]:
                    vs = s.lex[l][core]; out.append((chat(r, r.choices([v for v, _ in vs], [c for _, c in vs])[0]), l)); hit += 1
                elif core.isascii() and core:
                    out.append((core, "other" if OTHER_TOK.match(core) else "en"))  # Latin words inside native wiki text
            if hit >= 2 and hit >= 0.6 * len(ws): return out
        return [(w, lab(w, l)) for w in win(r, r.choice(s.rom[l]))]

    def english(s, lo=2, hi=12):
        src = s.enw if s.r.random() < 0.5 else s.en
        return [(w, lab(w, "en")) for w in win(s.r, s.r.choice(src), lo, hi)]

    def other(s):
        return [(w, lab(w, "other")) for w in win(s.r, s.r.choice(s.oth))]

    def row(s):
        r = s.r; l = r.choice(LANGS); kind = r.random(); roman = r.random() < 0.75
        seg = (lambda x: s.roman(x)) if roman else (lambda x: s.native(x))
        if kind < 0.25: return seg(l), "mono"
        if kind < 0.50:  # Hinglish style: English words/phrases inside an Indic sentence
            base = s.native(l) if r.random() < 0.5 else s.roman(l)  # v2: half native-script bases (news style, as in COMI)
            for _ in range(r.randint(1, 3)):
                i = r.randint(0, len(base)); base[i:i] = s.english(1, 3)
            return base, "indic+en-insert"
        if kind < 0.75:  # two Indic languages, e.g. Telugu then Hindi (the case that motivated this model)
            m = r.choice([x for x in LANGS if x != l]); a, b = seg(l), seg(m)
            return (a + b if r.random() < 0.5 else b + a), "indic+indic"
        if kind < 0.85: a, b = seg(l), s.english(); return (a + b if r.random() < 0.5 else b + a), "indic+en-seg"
        if kind < 0.92:
            m = r.choice([x for x in LANGS if x != l]); parts = [seg(l), seg(m), s.english(1, 4)]; r.shuffle(parts)
            return sum(parts, []), "3-way"
        return (s.english() if r.random() < 0.6 else s.other()), "en/other"


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for ws, kind in rows:
            ws = [(w, y) for w, y in ws if w]
            if ws: f.write(json.dumps({"words": [w for w, _ in ws], "labels": [y for _, y in ws], "kind": kind}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if os.environ.get("ENVOCAB"): EV = {k: set(v) for k, v in json.load(open(os.environ["ENVOCAB"])).items()}
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200000; os.makedirs(out, exist_ok=True)
    tr = Src("train", random.Random(1)); rows = [tr.row() for _ in range(n)]
    comi = os.environ.get("COMI_TRAIN")  # v2: COMI-LINGUA LID TRAIN split (human hi/en/ot labels; its test split stays held out)
    if comi:
        test_s = {" ".join(json.loads(l)["words"]) for l in open(os.environ["COMI_TEST"], encoding="utf-8")}
        c = [json.loads(l) for l in open(comi, encoding="utf-8")]
        c = [x for x in c if " ".join(x["words"]) not in test_s]
        rows += [([(w, g if g in ("hi", "en") else "other") for w, g in zip(x["words"], x["gold"])], "comi") for x in c]
        random.Random(3).shuffle(rows); print("COMI train rows", len(c))
    write(f"{out}/train.jsonl", rows[2000:]); write(f"{out}/dev.jsonl", rows[:2000])
    if not os.environ.get("NO_TEST"):
        te = Src("test", random.Random(2)); write(f"{out}/test-mix.jsonl", [te.row() for _ in range(3000)])
    print("kinds", collections.Counter(k for _, k in rows).most_common())
    print("labels", collections.Counter(y for ws, _ in rows for _, y in ws).most_common())
