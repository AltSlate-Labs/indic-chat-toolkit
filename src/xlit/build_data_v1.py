"""xlit-v1 training data = v0 recipe plus:
 (a) 20k lexicon words per language (v0: 8k);
 (b) LEXSENT: Dakshina native wiki TRAIN sentences whose every word is in the TRAIN lexicon, romanized word by word with
     attested variants sampled by count (exact targets; Dakshina test sentences excluded), N_LEXSENT per language;
 (c) keep-English mode: input "<lang>+en: ..." where English words stay in Latin script in the target. Built from chat and
     Dakshina-dev pairs whose roman and native word counts match: native word i is replaced by roman word i when that word is
     English (in /usr/share/dict/american-english, >=3 letters, seen >=3 times in our English corpus, and not >10x more frequent in
     romanized Indic than in English). Whole tokens are swapped (Indic combining marks are not \\w, so no edge splitting).
python build_data_v1.py OUT_DIR"""
import collections, csv, gzip, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0/")
R = os.path.expanduser("~/agent/rnd/jobs/")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
N_LEXSENT = int(os.environ.get("N_LEXSENT", 6000)); N_WORDS = int(os.environ.get("N_WORDS", 20000))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); r = random.Random(11)
W = re.compile(r"[a-z]+"); EDGE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
train, dev, en_rows = [], [], []


def add(rows, lang, roman, native, tag=""):
    roman, native = " ".join(roman.split()), " ".join(native.split())
    if roman and native: rows.append({"src": f"{lang}{tag}: " + (roman.lower() if r.random() < 0.5 else roman), "tgt": native, "lang": lang})


# English word test
en_c, ro_c = collections.Counter(), collections.Counter()
for l in open(R + "background-v1/out/background.jsonl", encoding="utf-8"):
    x = json.loads(l)
    if x["src"] == "en": en_c.update(W.findall(x["text"].lower()))
    elif x["src"] in ("hinglish-chat", "hinglish-tr"): ro_c.update(W.findall(x["text"].lower()))
chat = [json.loads(x) for f in ("lid-chat/out/chat-train.jsonl", "lid-chat4/out/chat-weak.jsonl") for x in open(R + f, encoding="utf-8")]
for x in chat: ro_c.update(W.findall(x["roman"].lower()))
lex = {}
for l in LANGS:
    lex[l] = collections.defaultdict(list)
    for row in csv.reader(open(f"{D}{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"), delimiter="\t"):
        if len(row) == 3: lex[l][row[0]].append((row[1], int(row[2]))); ro_c[row[1].lower()] += int(row[2])
DICT = {w.strip() for w in open("/usr/share/dict/american-english", encoding="utf-8", errors="ignore") if w.strip().isalpha() and w.strip().islower()}
E, RO = sum(en_c.values()), sum(ro_c.values())
# Homographs: romanizations of each language's 1,000 most frequent native words (e.g. men=में, the=थे, par=पर) are never English.
STOP = set()
for l in LANGS:
    fc = collections.Counter()
    with gzip.open(f"{D}{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 100000: break
            fc.update(EDGE.match(w).groups()[1] for w in line.split())
    for n, _ in fc.most_common(1000): STOP |= {v.lower() for v, _ in lex[l].get(n, [])}
STOP |= set("use are the men main par bus hum tab jab door din ban mat sun bar kar pan tum aap jan man mil log".split())
def is_en(w): return len(w) >= 3 and w in DICT and w not in STOP and en_c[w] >= 3 and (ro_c[w] / RO) <= 10 * (en_c[w] / E)

testA_native = {" ".join(json.loads(l)["native"].split()) for l in open(R + "xlit-gap/out/testA.jsonl", encoding="utf-8")}
testB = {" ".join(json.loads(l)["roman"].lower().split()) for l in open(R + "xlit-gap/out/testB.jsonl", encoding="utf-8")}
stats = collections.Counter()
for l in LANGS:
    rom = open(f"{D}{l}/romanized/{l}.romanized.rejoined.dev.roman.txt", encoding="utf-8").read().splitlines()
    nat = open(f"{D}{l}/romanized/{l}.romanized.rejoined.dev.native.txt", encoding="utf-8").read().splitlines()
    idx = list(range(len(rom))); r.shuffle(idx)
    for k, i in enumerate(idx):
        add(dev if k < 150 else train, l, rom[i], nat[i])
        if k >= 150: en_rows.append((l, rom[i], nat[i]))
    words = [(n, v) for n, vs in lex[l].items() for v, _ in vs]
    for n, v in r.sample(words, min(N_WORDS, len(words))): add(train, l, v, n)
    # (b) fully lexicon-covered native sentences, romanized word by word
    got = 0
    with gzip.open(f"{D}{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            t = " ".join(line.split())
            if not t or t in testA_native or not 3 <= len(t.split()) <= 25: continue
            parts = [EDGE.match(w).groups() for w in t.split()]
            if not all(c in lex[l] or c.isdigit() for _, c, _ in parts): continue
            ro = [a + (c if c.isdigit() else r.choices([v for v, _ in lex[l][c]], [n for _, n in lex[l][c]])[0]) + b for a, c, b in parts]
            add(train, l, " ".join(ro), t); got += 1
            if got >= N_LEXSENT: break
    stats["lexsent-" + l] = got
chat = [x for x in chat if x["lang"] in LANGS and " ".join(x["roman"].lower().split()) not in testB]; r.shuffle(chat)
for k, x in enumerate(chat):
    add(dev if k < len(chat) // 20 else train, x["lang"], x["roman"], x["native"])
    if k >= len(chat) // 20: en_rows.append((x["lang"], x["roman"], x["native"]))
# (c) keep-English variants
for l, ro, na in en_rows:
    rw, nw = ro.split(), na.split()
    if len(rw) != len(nw): continue
    keep = [EDGE.match(w).groups()[1].lower() for w in rw]
    if not any(is_en(k) for k in keep): continue
    tgt = " ".join(rw[i] if is_en(keep[i]) else nw[i] for i in range(len(nw)))
    add(train, l, ro, tgt, "+en"); stats["keep-en"] += 1
r.shuffle(train)
for name, rows in (("train", train), ("dev", dev)):
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("train", len(train), "dev", len(dev), dict(stats))
print("en-word examples:", [w for w in ("meeting", "phone", "bike", "main", "sab", "par", "the", "ready", "office", "kal", "hai", "tar", "for", "member", "laptop", "deadline", "men", "use", "are", "din", "door", "ban", "can", "card", "score") if is_en(w)])
