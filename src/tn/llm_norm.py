"""Prompted-LLM baseline (OpenAI-compatible server): python llm_norm.py TEST.jsonl OUT.jsonl URL MODEL"""
import json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
NAMES = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
test, out, url, model = sys.argv[1:5]
rows = [json.loads(l) for l in open(test, encoding="utf-8")]


def run(x):
    L = NAMES[x["lang"]]
    p = (f"Rewrite this {L} text for a text-to-speech system: write every number, symbol, unit, date, time and abbreviation out in "
         f"{L} words in the native script, exactly as a {L} newsreader would read it aloud. Keep everything else unchanged. "
         f"Output only the rewritten text.\n\n{x['text']}")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 600,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        o = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=600))
        return " ".join(re.sub(r"<think>.*?</think>", "", o["choices"][0]["message"]["content"], flags=re.S).split())
    except Exception as e: return x["text"]


with ThreadPoolExecutor(16) as pool: preds = list(pool.map(run, rows))
with open(out, "w", encoding="utf-8") as f:
    for p in preds: f.write(json.dumps({"pred": p}, ensure_ascii=False) + "\n")
