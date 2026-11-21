"""Training/dev data for indic-lid. python synth_lid.py OUT_DIR [SEED=1]
Labels: <lang> (native script) and <lang>-Latn (romanized) for 12 Dakshina languages, plus en and other.
Sources and licences:
- Dakshina v1.0 (CC BY-SA 4.0): native wiki-filt TRAIN sentences (minus any sentence in the romanized dev/test sets),
  word lexicons TRAIN split (romanizations attested by native speakers) for synthetic romanization, romanized DEV
  sentences for the dev set only. The romanized TEST split is never read here.
- jobs/background-v1: hinglish-chat (MIT), hinglish-tr (Apache-2.0) as real romanized Hindi; en (Apache-2.0).
- wikimedia/wikipedia shards (CC BY-SA) for 27 Latin-script 'other' languages."""
import glob, gzip, json, os, random, re, sys
import pandas as pd

D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
LABELS = LANGS + [l + "-Latn" for l in LANGS] + ["en", "other"]
FILL = ["ok", "okay", "bro", "pls", "please", "sir", "thanks", "yaar", "haan", "acha", "lol", "na", "yes", "no", "sorry", "bye"]
ROM_SYN = int(os.environ.get("ROM_SYN", 15000))  # lexicon-synthesized romanized windows per language (v1: 30000)
KS = [2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 10, 12, 15, 20, 30]
EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
BG1 = os.path.expanduser("~/agent/rnd/jobs/background-v1/out/background.jsonl")


def window(r, words):
    k = r.choice(KS)
    if len(words) <= k: return words
    s = r.randrange(len(words) - k + 1); return words[s:s + k]


def lexicon(l):
    lex = {}
    for line in open(f"{D}/{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3: lex.setdefault(p[0], []).append((p[1], int(p[2])))
    return lex


def romanize(r, words, lex):
    out, hit = [], 0
    for w in words:
        a, core, b = EDGE.match(w).groups()
        if core in lex:
            vs = lex[core]; out.append(r.choices([v for v, _ in vs], [c for _, c in vs])[0]); hit += 1
        elif core.isascii() and core:
            out.append(core)
    return out, hit


def noise(r, words):
    out = []
    for w in words:
        if r.random() < 0.12 and len(w) > 3:  # chat-style vowel dropping: kar -> kr, rahe -> rhe
            w = w[0] + re.sub(r"[aeiou]", lambda m: "" if r.random() < 0.6 else m.group(), w[1:-1]) + w[-1]
        if r.random() < 0.03: w = w + w[-1] * r.randint(1, 3)
        out.append(w)
    if r.random() < 0.25: out.insert(r.randint(0, len(out)), r.choice(FILL))
    t = " ".join(out)
    if r.random() < 0.8: t = t.lower()
    if r.random() < 0.15: t = t.capitalize()
    return t


def native_lines(l, n, excl, skip=0):
    out = []
    with gzip.open(f"{D}/{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip: continue
            t = line.strip()
            if t and t not in excl and len(t.split()) >= 3: out.append(t)
            if len(out) >= n: break
    return out


def main():
    out_dir = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    r = random.Random(seed); os.makedirs(out_dir, exist_ok=True); train, dev = [], []
    bg = [json.loads(x) for x in open(BG1, encoding="utf-8")]
    for l in LANGS:
        rp = f"{D}/{l}/romanized/{l}.romanized.rejoined."
        excl = set()
        for sp in ("dev", "test"):
            excl |= {x.strip() for x in open(rp + sp + ".native.txt", encoding="utf-8")}
        lex = lexicon(l)
        for t in native_lines(l, 20000, excl):
            train.append((" ".join(window(r, t.split())), l))
        n_rom = 0
        for t in native_lines(l, 90000, excl, skip=25000):  # disjoint from the 20k native lines above (v1 skipped 200k: sd has fewer lines)
            win = window(r, t.split()); ws, hit = romanize(r, win, lex)
            if hit >= 2 and len(ws) >= 2 and hit >= 0.6 * len(win):
                train.append((noise(r, ws), l + "-Latn")); n_rom += 1
            if n_rom >= ROM_SYN: break
        devn = open(rp + "dev.native.txt", encoding="utf-8").read().splitlines()
        devr = open(rp + "dev.roman.txt", encoding="utf-8").read().splitlines()
        idx = list(range(len(devn))); r.shuffle(idx)
        for i in idx[:300]:  # held-out part of Dakshina DEV -> our dev set
            dev.append((" ".join(window(r, devn[i].split())), l)); dev.append((" ".join(window(r, devr[i].split())), l + "-Latn"))
        for i in idx[300:]:  # v2: the rest of Dakshina DEV (human romanizations) -> training, 6 windows each
            for _ in range(6):
                w = window(r, devr[i].split())
                train.append((" ".join(w).lower() if r.random() < 0.5 else " ".join(w), l + "-Latn"))
        print(l, "romanized", n_rom, "lexicon", len(lex), flush=True)
    chat = os.environ.get("CHAT")  # v3: Ornith chat pairs {native, roman, lang} (jobs/lid-chat); 5% held out for dev
    if chat:  # v4: comma-separated files; languages outside LANGS (out-of-scope Indic) are labelled "other"
        rows = [json.loads(x) for c in chat.split(",") for x in open(c, encoding="utf-8")]; r.shuffle(rows); nd = len(rows) // 20
        for i, x in enumerate(rows):
            rom = x["roman"].lower() if r.random() < 0.5 else x["roman"]
            ln, lr = (x["lang"], x["lang"] + "-Latn") if x["lang"] in LANGS else ("other", "other")
            pairs = [(x["native"], ln), (rom, lr)]
            if i < nd: dev += pairs
            else:
                train += pairs + [(" ".join(window(r, rom.split())), lr)]
                if ln == "other":  # v5: OOS_REP extra copies (short windows) of out-of-scope Indic chat
                    for _ in range(int(os.environ.get("OOS_REP", 0))):
                        train += [(x["native"], "other"), (" ".join(window(r, rom.split())), "other"), (" ".join(window(r, x["roman"].split())), "other")]
        print("chat", len(rows), "dev", nd, flush=True)
    hin = [x["text"] for x in bg if x["src"] in ("hinglish-chat", "hinglish-tr")]; r.shuffle(hin)
    train += [(" ".join(window(r, t.split())), "hi-Latn") for t in hin[:30000]]
    en = [x["text"] for x in bg if x["src"] == "en"]
    train += [(" ".join(window(r, t.split())), "en") for t in en for _ in range(2)]
    dev += [(t, "hi-Latn") for t in hin[30000:30300]] + [(" ".join(window(r, t.split())), "en") for t in r.sample(en, 300)]
    for d in sorted(glob.glob(os.path.expanduser("~/Downloads/data/wikipedia-other/*"))):
        fs = glob.glob(d + "/**/*.parquet", recursive=True)
        texts = pd.read_parquet(fs[0], columns=["text"]).text.head(3000).tolist()
        sents = [s.strip() for t in texts for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 3][:2600]
        r.shuffle(sents)
        train += [(" ".join(window(r, s.split())), "other") for s in sents[:2300]]
        dev += [(" ".join(window(r, s.split())), "other") for s in sents[2300:2340]]
    r.shuffle(train)
    for name, rows in (("train", train), ("dev", dev)):
        with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
            for t, y in rows:
                if t.strip(): f.write(json.dumps({"text": t, "label": y}, ensure_ascii=False) + "\n")
    import collections
    print("train", len(train), collections.Counter(y for _, y in train).most_common())
    print("dev", len(dev))


if __name__ == "__main__":
    main()
