"""Chat-style MT data from the Ornith-written chat messages of the LID/xlit projects ({native, roman, lang}; MIT teacher output).
  src OUT_DIR TEST_JSONL CHAT_JSONL...  -> OUT_DIR/src.jsonl (train sources: native text) + dev.jsonl (40/lang, roman + native,
                                           judged reference-free); drops any message whose native or roman text is in TEST_JSONL
  mix OUT_DIR DATA_JSONL ORNITH_JSONL UP -> OUT_DIR/train.jsonl = DATA + (roman->en, native->en) chat pairs x UP + native Telugu x2
Ornith targets filtered as in build_mix_mt2 (English-looking, no preamble/refusal/second paragraph), length ratio 0.4-3.5 words."""
import json, os, random, re, sys
LANGS = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"}
key = lambda t: " ".join(re.sub(r"[^\w\s]", " ", t.lower()).split())
mode, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
wj = lambda p, rows: open(p, "w", encoding="utf-8").writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
if mode == "src":
    held = set()
    for l in open(sys.argv[3], encoding="utf-8"): held.add(key(json.loads(l)["src"]))
    rows, seen, drop = [], set(), 0
    for p in sys.argv[4:]:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["lang"] not in LANGS or key(r["native"]) in seen: continue
            seen.add(key(r["native"]))
            if key(r["native"]) in held or key(r["roman"]) in held: drop += 1; continue
            rows.append({"src": r["native"], "roman": r["roman"], "lang": r["lang"]})
    random.Random(21).shuffle(rows)
    dev, train, n = [], [], {}
    for r in rows:
        n[r["lang"]] = n.get(r["lang"], 0) + 1
        if n[r["lang"]] <= 40: dev += [{"src": r["roman"], "ref": r["roman"], "lang": r["lang"], "kind": "roman"},
                                       {"src": r["src"], "ref": r["src"], "lang": r["lang"], "kind": "native"}]
        else: train.append(r)
    wj(f"{out}/src.jsonl", train); wj(f"{out}/dev.jsonl", dev)
    print("per lang", dict(sorted(n.items())), "test overlaps dropped", drop, "train", len(train), "dev rows", len(dev))
else:
    data, orn, UP = sys.argv[3], sys.argv[4], int(sys.argv[5])
    EN = set("the a an is are was were be i you he she it we they to of and in on for with that this not do does did have has my your me "
             "what why how when where will can just so but if or at from by all no yes".split())
    FX = set("el la los las que y en del con por para es una lo se al como pero muy le les et est une avec pour dans der die das und "
             "ist nicht mit ein eine não com uma".split())
    def ok(s, h):
        if re.match(r"(?i)\s*(here is|translation|english)\b.*:", h) or "\n\n" in h: return False
        if re.search(r"(?i)\b(i can'?t|i cannot|i'm sorry|i am sorry|let me (re)?(consider|think|translate)|wait, let me|as an ai)\b", h): return False
        letters = [c for c in h if c.isalpha()]
        if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.97: return False
        w = re.findall(r"[a-z]+", h.lower())
        if sum(x in FX for x in w) > sum(x in EN for x in w): return False
        return 0.4 <= len(h.split()) / max(len(s.split()), 1) <= 3.5
    rows = open(data, encoding="utf-8").readlines(); n0 = len(rows)
    te = [l for l in rows if any(0xC00 <= ord(c) <= 0xC7F for c in json.loads(l)["src"])]
    chat, drop = [], 0
    for l in open(orn, encoding="utf-8"):
        r = json.loads(l); h = r["hyp"].replace("\n", " ").strip()
        if not ok(r["roman"], h): drop += 1; continue
        chat += [{"src": r["roman"], "tgt": h, "kind": f"chat-roman-{r['lang']}"}, {"src": r["src"], "tgt": h, "kind": f"chat-{r['lang']}"}]
    out_rows = rows + te * 2 + [json.dumps(r, ensure_ascii=False) + "\n" for r in chat] * UP
    random.Random(9).shuffle(out_rows)
    open(f"{out}/train.jsonl", "w", encoding="utf-8").writelines(out_rows)
    print(f"data {n0} te-native extra {2 * len(te)} chat pairs {len(chat)} (dropped {drop} teacher outputs) x{UP} -> {len(out_rows)}")
