"""Translate jsonl rows (field src; ref kept if present) with an OpenAI-compatible server. Resumable: rows already in OUT are skipped.
python gen_teacher_mt.py URL MODEL IN_JSONL OUT_JSONL [MAX_ROWS] [MAX_MINUTES]
Writes {src, hyp, ref?, kind?} per line in input order of completion. Used with Ornith (training teacher) only."""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, model, inp, out = sys.argv[1:5]
MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9; MIN = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9
P = ("Translate this message into natural English. It may be in an Indian language written in Latin or native script, or mix "
     "languages with English. Keep @handles, #hashtags, URLs and emoji unchanged. Reply with the translation only.\n\n")
rows = [json.loads(l) for l in open(inp, encoding="utf-8")][:MAX]
done = set()
if os.path.exists(out): done = {json.loads(l)["src"] for l in open(out, encoding="utf-8")}
todo = [r for r in rows if r["src"] not in done]
print(f"rows {len(rows)} done {len(done)} todo {len(todo)}", flush=True)
t0 = time.time(); errs = [0]
def tr(r):
    if time.time() - t0 > MIN * 60 or errs[0] >= 40: return None   # 40 consecutive errors = engine hung -> stop, caller restarts
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": P + r["src"]}], "temperature": 0, "max_tokens": 300,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    q = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    try: h = json.loads(urllib.request.urlopen(q, timeout=600).read())["choices"][0]["message"]["content"].strip()
    except Exception as e: errs[0] += 1; print("err", e, flush=True); return None
    errs[0] = 0; return dict(r, hyp=h)
n = 0
with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(48) as ex:
    for o in ex.map(tr, todo):
        if o is None: continue
        o.pop("tgt", None); f.write(json.dumps(o, ensure_ascii=False) + "\n"); n += 1
        if n % 2000 == 0: f.flush(); print(f"{n} rows {time.time() - t0:.0f}s", flush=True)
print(f"wrote {n} in {time.time() - t0:.0f}s", flush=True)
if errs[0] >= 40: sys.exit(3)
