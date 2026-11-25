"""Filter a streamed NLLB mined bitext (allenai/nllb, ODC-BY 1.0; CCMatrix/CommonCrawl-derived, upstream terms apply) prefix.
stdin: TSV rows (side A, side B, laser score, ..., url). python filter_nllb.py LANG EN_FIRST(0|1) N OUT.tsv
Keeps 30-300 char pairs, drops pairs whose English or Indic side is in any eval set (Dakshina dev/test natives, all FLORES,
all IndicCrosslingualSTS), dedupes each side, caps any one web domain at 2% of N, then random N. Output: en \t xx \t score \t url."""
import glob, os, random, sys, collections, urllib.parse
import pyarrow.parquet as pq
lang, en_first, N, out = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), sys.argv[4]
norm = lambda t: " ".join(t.split()); D = os.path.expanduser("~/Downloads/data"); EVAL = set()
for f in glob.glob(f"{D}/IndicGenBenchFloresBitextMining/*/*.parquet") + glob.glob(f"{D}/IndicCrosslingualSTS/*/*.parquet"):
    for x in pq.read_table(f).to_pylist(): EVAL |= {norm(v) for v in x.values() if isinstance(v, str)}
for sp in ("dev", "test"):
    EVAL |= {norm(x) for x in open(f"{D}/dakshina/dakshina_dataset_v1.0/{lang}/romanized/{lang}.romanized.rejoined.{sp}.native.txt", encoding="utf-8")}
seen_e, seen_x, rows = set(), set(), []
for line in sys.stdin.buffer:
    try: p = line.decode("utf-8").rstrip("\n").split("\t")
    except UnicodeDecodeError: continue
    if len(p) < 4: continue
    e, x = (norm(p[0]), norm(p[1])) if en_first else (norm(p[1]), norm(p[0]))
    if not (30 <= len(e) <= 300 and 30 <= len(x) <= 300) or e in EVAL or x in EVAL or e in seen_e or x in seen_x: continue
    seen_e.add(e); seen_x.add(x); rows.append((e, x, p[2], p[-1]))
r = random.Random(0); r.shuffle(rows); cap = max(1, N // 50); dom = collections.Counter(); keep = []
for e, x, s, u in rows:
    d = urllib.parse.urlparse(u).netloc
    if dom[d] >= cap: continue
    dom[d] += 1; keep.append((e, x, s, u))
    if len(keep) >= N: break
with open(out, "w", encoding="utf-8") as f:
    for row in keep: f.write("\t".join(row) + "\n")
print(lang, "candidates", len(rows), "kept", len(keep), "top domains", dom.most_common(3), flush=True)
