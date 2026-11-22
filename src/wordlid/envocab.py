"""v3 labelling convention for Latin-script words inside romanized Indic text: English words are 'en' (as COMI-LINGUA annotators
label them), not the sentence's language (Dakshina human romanizations write loanwords like 'bike', 'phone' as part of the
Indic sentence, so v1/v2 learned to call them Indic).
python envocab.py COMI_TRAIN.jsonl OUT.json [INDIC_ROMAN_JSONL ...]   -> {"en": [...], "hi": [...], "top": [...]}
 en  : Latin words COMI annotators label en in >= 70% of >= 3 occurrences (train split only)
 hi  : Latin words COMI annotators label hi in >= 70% of >= 3 occurrences (main, hai, to, ... never become en)
 top : 30k most frequent English Wikipedia words (lowercase, alphabetic)"""
import collections, glob, json, os, re, sys
import pandas as pd
c = collections.defaultdict(collections.Counter)
for l in open(sys.argv[1], encoding="utf-8"):
    x = json.loads(l)
    for w, g in zip(x["words"], x["gold"]):
        k = re.sub(r"[^a-z]", "", w.lower())
        if k and re.fullmatch(r"[A-Za-z'.,!?;:()\"-]+", w): c[k][g] += 1
en = sorted(k for k, v in c.items() if sum(v.values()) >= 3 and v["en"] / sum(v.values()) >= 0.7)
hi = sorted(k for k, v in c.items() if sum(v.values()) >= 3 and v["hi"] / sum(v.values()) >= 0.7)
fs = glob.glob(os.path.expanduser("~/Downloads/data/wikipedia-en/**/*.parquet"), recursive=True)
wc = collections.Counter(w for t in pd.read_parquet(fs[0], columns=["text"]).text.head(20000) for w in re.findall(r"[a-z]+", t.lower()))
# v3b: a top-30k word counts as English only if it is >= 3x more frequent (relative) in English Wikipedia than in romanized Indic
# training text (Dakshina DEV romanizations of all 12 languages + Ornith chat + Hinglish chat); otherwise words like men (में),
# khan, singh, pradesh, kora (করা) became en.
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
ic = collections.Counter()
for f in glob.glob(f"{D}/*/romanized/*.romanized.rejoined.dev.roman.txt"): ic.update(re.findall(r"[a-z]+", open(f, encoding="utf-8").read().lower()))
for f in sys.argv[3:]:
    for t in open(f, encoding="utf-8"):
        x = json.loads(t); ic.update(re.findall(r"[a-z]+", (x.get("roman") or (x["text"] if x.get("src") == "hinglish-chat" else "")).lower()))
ne, ni = sum(wc.values()), sum(ic.values())
top = [w for w, _ in wc.most_common(30000) if wc[w] / ne >= 3 * ic[w] / ni]
raw = {w for w, _ in wc.most_common(30000)}
# v3d: chat loanwords (phone, bike, coffee, exam) that COMI annotators call en. Frequency cannot separate them from names (khan,
# singh, delhi: COMI also calls Latin-script names en), so keep only words of 4+ letters that English Wikipedia writes in lowercase
# >= 70% of the time (common nouns, not proper names); 'men' (Hindi में) fails the length test.
low = collections.Counter(w for t in pd.read_parquet(fs[0], columns=["text"]).text.head(20000) for w in re.findall(r"\b[a-z]+\b", t))
comi_en = [w for w in en if w in raw and w not in hi and len(w) >= 4 and low[w] >= 0.7 * wc[w]]
json.dump({"en": en, "hi": hi, "top": top, "comi_en": comi_en}, open(sys.argv[2], "w"))
print("en", len(en), "hi", len(hi), "top", len(top), "| hi sample", hi[:25], "| en sample", en[:25])
