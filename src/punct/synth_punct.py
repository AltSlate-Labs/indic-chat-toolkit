"""Punctuation restoration data: unpunctuated words -> the punctuation class after each word.
python synth_punct.py OUT_DIR   (envs/pii)
Classes: O, COMMA (, ،), PERIOD (। ॥ . ۔ !), QUESTION (? ؟). ; : and quotes/brackets are dropped from the input and not predicted.
Train: Dakshina native Wikipedia TRAIN sentences (CC BY-SA 4.0; rows of 1-4 random sentences of one language), Ornith chat messages
(native side, MIT; jobs/lid-chat, lid-chat4; the main source of questions), English Wikipedia (CC BY-SA). Half the rows lowercased
English (ASR-like). Test: test-wiki = Dakshina TEST-split native sentences (never read for training); test-chat = tn-gap test B messages
(written by Qwen3.8-27B, which wrote no training text)."""
import glob, gzip, json, os, random, re, sys
import pandas as pd
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
LABELS = ["O", "COMMA", "PERIOD", "QUESTION"]
CLS = {",": "COMMA", "،": "COMMA", "।": "PERIOD", "॥": "PERIOD", ".": "PERIOD", "۔": "PERIOD", "!": "PERIOD", "?": "QUESTION", "؟": "QUESTION"}
STRIP = ",;:?!।॥.۔،؟\"'“”‘’()[]{}«»"


def tag(text):
    """-> (words, labels): punctuation removed from words; label = class of the last punctuation mark after the word."""
    words, labels = [], []
    for raw in text.split():
        core = raw.strip(STRIP)
        trail = raw[len(raw.rstrip(STRIP)):]
        lab = next((CLS[c] for c in reversed(trail) if c in CLS), "O")
        if not core:  # a stand-alone mark attaches to the previous word
            if words and lab != "O": labels[-1] = lab
            continue
        if re.search(r"\d[.,]\d", core): core = core  # keep decimals/grouped numbers as they are
        words.append(core); labels.append(lab)
    return words, labels


def wiki(l, split, n, r):
    if split == "train":
        f = gzip.open(f"{D}/{l}/native_script_wikipedia/{l}.wiki-filt.train.text.shuf.txt.gz", "rt", encoding="utf-8")
        excl = {x.strip() for sp in ("dev", "test") for x in open(f"{D}/{l}/romanized/{l}.romanized.rejoined.{sp}.native.txt", encoding="utf-8")}
    else:
        f = open(f"{D}/{l}/romanized/{l}.romanized.rejoined.test.native.txt", encoding="utf-8"); excl = set()
    out = []
    for t in f:
        t = t.strip()
        if t and t not in excl and len(t.split()) >= 3: out.append(t)
        if len(out) >= n: break
    return out


def rows_from(sents, r, k, lang, kind, lower=False):
    rows = []
    for _ in range(k):
        t = " ".join(r.sample(sents, r.choice([1, 1, 2, 2, 3, 4])))
        w, y = tag(t.lower() if lower else t)
        if len(w) >= 3: rows.append({"words": w[:120], "labels": y[:120], "lang": lang, "kind": kind})
    return rows


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); r = random.Random(0); train, testw, ochat_test = [], [], []
    chat = {}
    for f in ("jobs/lid-chat/out/chat-train.jsonl", "jobs/lid-chat4/out/chat-weak.jsonl"):
        for x in (json.loads(t) for t in open(os.path.expanduser("~/agent/rnd/") + f, encoding="utf-8")):
            if x["lang"] in LANGS: chat.setdefault(x["lang"], []).append(x["native"])
    for l in LANGS:
        tr = wiki(l, "train", 40000, r); train += rows_from(tr, r, int(os.environ.get("WIKI_ROWS", 14000)), l, "wiki")
        if chat.get(l):  # last 10% of each language's Ornith chat messages held out as test-ochat (same generator as training chat)
            c = chat[l]; k = len(c) // 10; ochat_test += rows_from(c[-k:], random.Random("o" + l), 150, l, "ochat"); train += rows_from(c[:-k], r, 4000, l, "chat")
        testw += rows_from(wiki(l, "test", 100000, r), random.Random(l), 300, l, "wiki")
        print(l, "train wiki", len(tr), "chat", len(chat.get(l, [])), flush=True)
    fs = glob.glob(os.path.expanduser("~/Downloads/data/wikipedia-en/**/*.parquet"), recursive=True)
    texts = pd.read_parquet(fs[0], columns=["text"]).text.head(20000).tolist()
    en = [x for t in texts for x in re.split(r"(?<=[.!?])\s+", t) if 4 <= len(x.split()) <= 40][:60000]
    train += rows_from(en, r, 12000, "en", "wiki", lower=True)
    test_chat = []
    for t in open(os.path.expanduser("~/agent/rnd/jobs/tn-gap/out/testB.jsonl"), encoding="utf-8"):
        x = json.loads(t); w, y = tag(x["text"])
        if len(w) >= 3: test_chat.append({"words": w, "labels": y, "lang": x["lang"], "kind": "chat"})
    r.shuffle(train)
    for name, rows in (("train", train[2000:]), ("dev", train[:2000]), ("test-wiki", testw), ("test-chat", test_chat), ("test-ochat", ochat_test)):
        with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    import collections
    print("train", len(train), collections.Counter(y for x in train for y in x["labels"]))
    print("test-wiki", len(testw), "test-chat", len(test_chat), collections.Counter(y for x in test_chat for y in x["labels"]))
    print("test-ochat", len(ochat_test), collections.Counter(y for x in ochat_test for y in x["labels"]))
