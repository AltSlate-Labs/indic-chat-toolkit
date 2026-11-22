"""COMI-LINGUA LID test (LingoIITGN, CC-BY-4.0; human hi/en/ot token labels on Hinglish news and social text) -> jsonl.
python comi_test.py LID_test.csv OUT.jsonl     rows: {"words": [...], "gold": [...]}  gold = majority of the 3 annotators
Metric fixed before any result (pass 116): PRIMARY = accuracy on Latin-script words whose gold is hi or en (romanized Hindi vs
English, ~32.8k words); SECONDARY = accuracy on all hi/en words. 'ot' words are not scored (our 'other' = digits/punctuation only,
COMI's 'ot' also has names). COMI labels by word origin: English loanwords in Devanagari (ट्रेन, स्टेशन) are 'en'."""
import ast, collections, csv, json, sys
out = open(sys.argv[2], "w", encoding="utf-8"); n = skip = 0
for x in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    A = [[t if isinstance(t, dict) else {"key": t[0], "value": {"e": "en", "h": "hi"}.get(t[1], "ot")} for t in ast.literal_eval(x[f"Annotated by: Annotator {i}"])]
         for i in (1, 2, 3)]  # 2 train rows use [word, 'e'/'h'] lists instead of dicts
    if not A[0] or not len(A[0]) == len(A[1]) == len(A[2]): skip += 1; continue
    words = [t["key"] for t in A[0]]
    gold = [collections.Counter(a["value"] for a in t).most_common(1)[0][0] for t in zip(*A)]
    out.write(json.dumps({"words": words, "gold": gold}, ensure_ascii=False) + "\n"); n += 1
print("rows", n, "skipped (annotator length mismatch)", skip)
