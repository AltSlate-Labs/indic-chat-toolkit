"""Training data for romanized -> native transliteration (input "<lang>: <roman>", target native).
- Dakshina DEV split sentence pairs (human romanizations, CC BY-SA 4.0); 150 per language held out as dev.
  (The Dakshina TEST split is test A and is never used here.)
- Dakshina lexicon TRAIN word pairs, 8k per language (short inputs).
- Ornith-1.5 chat pairs (MIT model; jobs/lid-chat + jobs/lid-chat4 weak langs), 5% to dev. 0 exact overlap with test B (checked in lid-v3/v4).
Half of the romanized inputs are lowercased. python build_data.py OUT_DIR"""
import json, os, random, sys, csv
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0/")
R = os.path.expanduser("~/agent/rnd/jobs/")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
out = sys.argv[1]; os.makedirs(out, exist_ok=True); r = random.Random(7)
train, dev = [], []


def add(rows, lang, roman, native):
    roman, native = " ".join(roman.split()), " ".join(native.split())
    if roman and native: rows.append({"src": f"{lang}: " + (roman.lower() if r.random() < 0.5 else roman), "tgt": native, "lang": lang})


testB = {" ".join(json.loads(l)["roman"].lower().split()) for l in open(R + "xlit-gap/out/testB.jsonl", encoding="utf-8")}
for l in LANGS:
    rom = open(f"{D}{l}/romanized/{l}.romanized.rejoined.dev.roman.txt", encoding="utf-8").read().splitlines()
    nat = open(f"{D}{l}/romanized/{l}.romanized.rejoined.dev.native.txt", encoding="utf-8").read().splitlines()
    idx = list(range(len(rom))); r.shuffle(idx)
    for k, i in enumerate(idx): add(dev if k < 150 else train, l, rom[i], nat[i])
    lex = [row for row in csv.reader(open(f"{D}{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"), delimiter="\t") if len(row) >= 2]
    for nat_w, rom_w, *_ in r.sample(lex, min(8000, len(lex))): add(train, l, rom_w, nat_w)
chat = [json.loads(x) for f in ("lid-chat/out/chat-train.jsonl", "lid-chat4/out/chat-weak.jsonl") for x in open(R + f, encoding="utf-8")]
chat = [x for x in chat if x["lang"] in LANGS and " ".join(x["roman"].lower().split()) not in testB]; r.shuffle(chat)
for k, x in enumerate(chat): add(dev if k < len(chat) // 20 else train, x["lang"], x["roman"], x["native"])
r.shuffle(train)
for name, rows in (("train", train), ("dev", dev)):
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("train", len(train), "dev", len(dev), "chat", len(chat))
