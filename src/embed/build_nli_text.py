"""English caption / everyday sentences for the embedder from SNLI train (premises + hypotheses; stanfordnlp/snli, CC BY-SA 4.0;
premises are Flickr30k captions). Drops every sentence that occurs in any STS Benchmark split, IndicCrosslingualSTS or IndicGenBench
FLORES, as near-duplicates (lower-cased, punctuation and articles removed). Rows input = target = sentence, kind "en-nli". dev.jsonl copied from DEV_SRC.
python build_nli_text.py OUT DEV_SRC [N]"""
import glob, gzip, json, os, random, re, shutil, sys
import pyarrow.parquet as pq
out, dev_src = sys.argv[1:3]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 500000; os.makedirs(out, exist_ok=True)
D = os.path.expanduser("~/Downloads/data")
# near-duplicate key: lower-case, punctuation removed, articles a/an/the dropped (embed-v5 had 140 STSb dev sentences matching on this key)
norm = lambda t: " ".join(w for w in re.sub(r"[^\w\s]", " ", t.lower()).split() if w not in ("a", "an", "the"))
EVAL = set()
for f in glob.glob(f"{D}/stsbenchmark/*.jsonl.gz"):
    for l in gzip.open(f, "rt", encoding="utf-8"): x = json.loads(l); EVAL |= {norm(x["sentence1"]), norm(x["sentence2"])}
for f in glob.glob(f"{D}/IndicCrosslingualSTS/*/*.parquet") + glob.glob(f"{D}/IndicGenBenchFloresBitextMining/*/*.parquet"):
    for x in pq.read_table(f).to_pylist(): EVAL |= {norm(v) for v in x.values() if isinstance(v, str)}
t = pq.read_table(glob.glob(f"{D}/snli/plain_text/train-*.parquet")[0]).to_pylist()
sents = sorted({s.strip() for r in t for s in (r["premise"], r["hypothesis"]) if s and 10 <= len(s.strip()) <= 300})
keep = [s for s in sents if norm(s) not in EVAL]; print("unique", len(sents), "dropped(eval overlap)", len(sents) - len(keep), flush=True)
random.Random(0).shuffle(keep); keep = keep[:N]
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for s in keep: f.write(json.dumps({"input": s, "target": s, "kind": "en-nli", "lang": "en"}, ensure_ascii=False) + "\n")
shutil.copy(f"{dev_src}/dev.jsonl", f"{out}/dev.jsonl"); print("train", len(keep))
