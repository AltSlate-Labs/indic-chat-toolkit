"""mt-v2 data = data1 (jobs/mt-data1/out/train.jsonl) + Ornith (training teacher) English for the Hinglish sources, x UP.
Ornith rows are filtered: target must look English (ASCII letters >= 97% of letters; English stopwords >= Spanish/French/German ones),
length ratio 0.4-2.5 (words), no 'translation:' preamble, no blank-line second paragraph
(Ornith sometimes appends 'Wait, let me reconsider…'), no refusals. Sources were already PHINC dev/test-filtered by build_data.py.
python build_mix_mt2.py DATA1_JSONL ORNITH_JSONL OUT_DIR UP"""
import json, os, random, re, sys
d1, orn, out, UP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
EN = set("the a an is are was were be i you he she it we they to of and in on for with that this not do does did have has my your me "
         "what why how when where will can just so but if or at from by all no yes".split())
def ok(s, h):
    if re.match(r"(?i)\s*(here is|translation|english)\b.*:", h) or "\n\n" in h: return False
    if re.search(r"(?i)\b(i can'?t|i cannot|i'm sorry|i am sorry|let me (re)?(consider|think|translate)|wait, let me|as an ai)\b", h): return False
    letters = [c for c in h if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.97: return False
    w = re.findall(r"[a-z]+", h.lower()); ws = len(s.split())
    if sum(x in FX for x in w) > sum(x in EN for x in w): return False   # Spanish/French/German/Portuguese stopwords win

    return 0.4 <= len(h.split()) / max(ws, 1) <= 2.5
FX = set("el la los las que y en del con por para es una lo se al como pero muy le les et est une avec pour dans der die das und "
         "ist nicht mit ein eine não com uma".split())
rows = [json.loads(l) for l in open(d1, encoding="utf-8")]
n0, keep, drop = len(rows), [], 0
for l in open(orn, encoding="utf-8"):
    r = json.loads(l)
    if ok(r["src"], r["hyp"]): keep.append({"src": r["src"], "tgt": r["hyp"].replace("\n", " "), "kind": "orn-" + r["kind"]})
    else: drop += 1
rows += keep * UP
random.Random(9).shuffle(rows)
os.makedirs(out, exist_ok=True)
with open(f"{out}/train.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"data1 {n0} ornith kept {len(keep)} dropped {drop} -> total {len(rows)}")
