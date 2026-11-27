"""Round-0 training data for Hinglish/romanized -> English: findnitai/english-to-hinglish (hi_ng -> en; it aggregates Hinglish TOP,
CMU DoG, HinGE and PHINC) + PHINC train split. Removes every row whose source or target matches a PHINC dev/test source or
reference after normalization (lower-case, punctuation and @handles/#tags/URLs stripped). python build_data.py SPLIT_DIR OUT_DIR"""
import json, os, random, re, sys
sd, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
def key(t):
    t = re.sub(r"https?://\S+|[@#]\w+", " ", t.lower()); return " ".join(re.sub(r"[^\w\s]", " ", t).split())
held = set()
for sp in ("dev", "test"):
    for l in open(f"{sd}/{sp}.jsonl", encoding="utf-8"):
        r = json.loads(l); held.add(key(r["src"])); held.add(key(r["ref"]))
held.discard("")
rows, seen, drop, by = [], set(), 0, {}
def add(s, t, tag):
    global drop
    s, t = " ".join(s.split()), " ".join(t.split())
    if not s or not t or len(s) > 400 or len(t) > 400: return
    if key(s) in held or key(t) in held: drop += 1; return
    if (key(s), key(t)) in seen: return
    seen.add((key(s), key(t))); rows.append({"src": s, "tgt": t, "kind": tag}); by[tag] = by.get(tag, 0) + 1
for l in open(os.path.expanduser("~/Downloads/data/english-to-hinglish/hinglish_upload_v1.json"), encoding="utf-8"):
    r = json.loads(l)["translation"]; add(r["hi_ng"], r["en"], f"e2h-{r.get('source')}")
for l in open(f"{sd}/train.jsonl", encoding="utf-8"):
    r = json.loads(l); add(r["src"], r["ref"], "phinc")
random.Random(7).shuffle(rows)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rows", len(rows), by, "dropped (PHINC dev/test overlap)", drop)
