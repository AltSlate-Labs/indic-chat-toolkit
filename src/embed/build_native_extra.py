"""embed-v2 data: more native-script sentences (v1 had 60k/lang, all filtered to dictionary-romanizable ones; STS diagnostic
showed the student's Indic side loses graded similarity, notes in STATE.md). python build_native_extra.py V1_DATA OUT [N_PER_LANG] [REPLAY]
- native rows: Dakshina native Wikipedia TRAIN (CC BY-SA 4.0), sentences not used in v1, no romanization filter, <= N per language
- replay: REPLAY random rows of v1 train.jsonl (all kinds), so roman/mixed/en skills are kept
Dropped: Dakshina rejoined dev/test natives, every IndicGenBench FLORES sentence (all splits), every IndicCrosslingualSTS sentence.
dev.jsonl = v1 dev.jsonl."""
import glob, gzip, json, os, random, shutil, sys
import pyarrow.parquet as pq
v1, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 150000; REPLAY = int(sys.argv[4]) if len(sys.argv) > 4 else 500000
os.makedirs(out, exist_ok=True); r = random.Random(0); norm = lambda t: " ".join(t.split())
D = os.path.expanduser("~/Downloads/data"); DK = f"{D}/dakshina/dakshina_dataset_v1.0"
LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
EVAL = set()
for l in LANGS:
    for sp in ("dev", "test"):
        EVAL |= {norm(x) for x in open(f"{DK}/{l}/romanized/{l}.romanized.rejoined.{sp}.native.txt", encoding="utf-8")}
for f in glob.glob(f"{D}/IndicGenBenchFloresBitextMining/*/*.parquet"):
    for x in pq.read_table(f).to_pylist(): EVAL |= {norm(v) for v in x.values() if isinstance(v, str)}
for f in glob.glob(f"{D}/IndicCrosslingualSTS/*/*.parquet"):
    for x in pq.read_table(f).to_pylist(): EVAL |= {norm(v) for v in x.values() if isinstance(v, str)}
old = [json.loads(l) for l in open(f"{v1}/train.jsonl", encoding="utf-8")]
used = {x["target"] for x in old}
rows = r.sample(old, min(REPLAY, len(old))); del old
for l in LANGS:
    n = 0
    with gzip.open(f"{DK}/{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8") as g:
        for t in g:
            t = norm(t)
            if not 30 <= len(t) <= 300 or t in EVAL or t in used: continue
            used.add(t); rows.append({"input": t, "target": t, "kind": "native2", "lang": l}); n += 1
            if n >= N: break
    print(l, n, flush=True)
r.shuffle(rows)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
shutil.copy(f"{v1}/dev.jsonl", f"{out}/dev.jsonl")
print("train", len(rows), "eval-excluded", len(EVAL))
