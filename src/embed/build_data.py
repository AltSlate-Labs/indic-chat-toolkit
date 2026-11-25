"""Training inputs for indic-embed distillation (see notes/embed-plan.md). No text generation.
python build_data.py OUT_DIR [N_PER_LANG]
Rows: {"input", "target", "kind", "lang"}; the teacher (bge-m3) embeds "target", the student must map "input" there.
- native : Dakshina native Wikipedia TRAIN split (CC BY-SA 4.0); input = target = native sentence
- roman  : same sentence romanized word by word from aksharantar TRAIN pairs (CC0 / CC-BY 4.0), a random attested variant per word,
           lower-cased; sentences with >15% unknown words skipped (unknown words keep native script)
- mixed  : half the words romanized
- en     : English Wikipedia (one parquet shard, CC BY-SA)
- hinglish: findnitai/english-to-hinglish (Apache-2.0): Hinglish input → English target; English → English
Sentences that occur in any eval set (Dakshina rejoined dev/test natives) are dropped."""
import gzip, json, os, random, re, sys, zipfile, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_roman import romanize_word
D = os.path.expanduser("~/Downloads/data")
DK = f"{D}/dakshina/dakshina_dataset_v1.0"
LANGS = {"hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "ur": "urd"}
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60000; os.makedirs(out, exist_ok=True)
r = random.Random(0)
PUNCT = re.compile(r"^([\"'“‘(\[]*)(.*?)([\"'”’)\].,!?;:।॥۔،]*)$")
norm = lambda t: " ".join(t.split())

EVAL = set()
for l in LANGS:
    for sp in ("dev", "test"):
        p = f"{DK}/{l}/romanized/{l}.romanized.rejoined.{sp}.native.txt"
        EVAL |= {norm(x) for x in open(p, encoding="utf-8")}

def romanizer(l, code):
    """aksharantar TRAIN pairs + Dakshina romanization lexicon TRAIN (human romanizations); dev/test lexicons are not used."""
    d = collections.defaultdict(set)
    with zipfile.ZipFile(f"{D}/aksharantar/{code}.zip") as z:
        with z.open(f"{code}_train.json") as f:
            for line in f:
                x = json.loads(line); d[x["native word"]].add(x["english word"].lower())
    for line in open(f"{DK}/{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2: d[p[0]].add(p[1].lower())
    return {k: sorted(v) for k, v in d.items()}

def fallback(core, l):
    w = romanize_word(core, l)
    if r.random() < 0.5: w = w.replace("aa", "a").replace("ee", "i").replace("oo", "u")   # chat often writes long vowels short
    return w

def romanize(s, rd, frac, l):
    words, unk, out = s.split(), 0, []
    for w in words:
        a, core, b = PUNCT.match(w).groups()
        if r.random() >= frac or not core: out.append(w); continue
        v = rd.get(core)
        if v: out.append(a + r.choice(v) + b)
        else: unk += 1; out.append(a + fallback(core, l) + b)
    return " ".join(out), unk / max(1, len(words))

rows, stats = [], collections.Counter()
for l, code in LANGS.items():
    rd = romanizer(l, code); n = 0
    with gzip.open(f"{DK}/{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as g:
        for t in g:
            t = norm(t)
            if not 30 <= len(t) <= 300 or t in EVAL: stats["skip"] += 1; continue
            ro, u = romanize(t, rd, 1.0, l)
            if u > 0.3: stats["oov"] += 1; continue     # at least 70% of words romanized from human dictionaries; rest by rule_roman.py
            mx, _ = romanize(t, rd, 0.5, l)
            rows += [{"input": t, "target": t, "kind": "native", "lang": l}, {"input": ro, "target": t, "kind": "roman", "lang": l},
                     {"input": mx, "target": t, "kind": "mixed", "lang": l}]
            n += 1
            if n >= N: break
    stats[l] = n; print(l, n, "dict", len(rd), flush=True)

import pyarrow.parquet as pq
en = [norm(x) for x in pq.read_table(f"{D}/wikipedia-en/20231101.en/" + os.listdir(f"{D}/wikipedia-en/20231101.en")[0], columns=["text"]).column("text").to_pylist()]
sents = [s for doc in en for s in re.split(r"(?<=[.!?])\s+", doc) if 30 <= len(s) <= 300]
r.shuffle(sents)
for s in sents[:100000]: rows.append({"input": s, "target": s, "kind": "en", "lang": "en"})
for line in open(f"{D}/english-to-hinglish/hinglish_upload_v1.json", encoding="utf-8"):
    x = json.loads(line)["translation"]; e, h = norm(x.get("en", "")), norm(x.get("hi_ng", ""))
    if 10 <= len(e) <= 300 and 10 <= len(h) <= 300:
        rows += [{"input": h, "target": e, "kind": "hinglish", "lang": "hi-en"}, {"input": e, "target": e, "kind": "en-chat", "lang": "en"}]
r.shuffle(rows)
dev, train = rows[:5000], rows[5000:]
for name, rr in (("train", train), ("dev", dev)):
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rr: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("rows", len(rows), dict(stats), collections.Counter(x["kind"] for x in rows))
for k in ("roman", "mixed", "hinglish"):
    for x in [x for x in rows if x["kind"] == k][:2]: print(k, "|", x["input"][:120], "=>", x["target"][:80])
