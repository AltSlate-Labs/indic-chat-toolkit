"""Audit judge NO verdicts: ask the same judge to name the error. python judge_why.py TEST.jsonl PRED.jsonl JUDGE.json LANG N OUT.jsonl"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8080/v1/chat/completions")
NAMES = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
test, preds, ok = [json.loads(l) for l in open(sys.argv[1])], [json.loads(l)["pred"] for l in open(sys.argv[2])], json.load(open(sys.argv[3]))
lang, n, out = sys.argv[4], int(sys.argv[5]), sys.argv[6]
items = [(x, p) for x, p, k in zip(test, preds, ok) if x["lang"] == lang and not k][:n]
def why(a):
    x, p = a; L = NAMES[lang]
    q = (f"Original {L} text:\n{x['text']}\n\nVersion for text-to-speech:\n{p}\n\nIs the second text a correct spoken version of the first? "
         f"It must write every number, symbol, unit, date, time and abbreviation as correct {L} words that a {L} speaker would say "
         f"(any natural reading is fine, e.g. a year read as a year), with the right values, and must not change, add or drop anything "
         f"else. First quote each word or phrase you consider wrong and say what it should be, then end with VERDICT: YES or VERDICT: NO.")
    body = json.dumps({"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 600,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    o = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
    return {"text": x["text"], "pred": p, "why": o["choices"][0]["message"]["content"]}
with ThreadPoolExecutor(8) as pool: rows = list(pool.map(why, items))
with open(out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(lang, "audited", len(rows), "still NO:", sum("VERDICT: NO" in r["why"].upper() for r in rows))
