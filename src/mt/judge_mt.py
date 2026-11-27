"""LLM-judge adequacy check (PHINC references are noisy). python judge_mt.py N OUT_JSON NAME=PRED_JSONL ...
The judge (Qwen3.8-27B via the OpenAI-compatible server at 127.0.0.1:8080; test judge only) sees the source and one candidate,
not the reference, and scores meaning 1-5. Same N rows (seed 3) for every system. Qwen judging its own output is biased toward it."""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
N, out = int(sys.argv[1]), sys.argv[2]
systems = dict(a.split("=", 1) for a in sys.argv[3:])
preds = {k: [json.loads(l) for l in open(v, encoding="utf-8")] for k, v in systems.items()}
n = min(len(v) for v in preds.values())
for k, v in preds.items(): assert all(v[i]["src"] == preds[next(iter(preds))][i]["src"] for i in range(n)), k
ix = sorted(random.Random(3).sample(range(n), N))
P = ("You are grading a translation of a social-media message (often Hindi-English code-mixed, in Latin script) into English.\n"
     "Rate how well the English conveys the full meaning of the source, ignoring minor style and capitalisation:\n"
     "5 = all meaning correct; 4 = small slip; 3 = main idea right, some meaning lost or wrong; 2 = mostly wrong or partly "
     "untranslated; 1 = wrong or not translated.\nReply with the number only.\n\nSource: {s}\nEnglish: {h}")
def judge(s, h):
    body = json.dumps({"messages": [{"role": "user", "content": P.format(s=s, h=h)}], "temperature": 0, "max_tokens": 5,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    q = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = json.loads(urllib.request.urlopen(q, timeout=600).read())["choices"][0]["message"]["content"]
    m = re.search(r"[1-5]", t); return int(m.group()) if m else 0
res = {}
for k, v in preds.items():
    with ThreadPoolExecutor(8) as ex: sc = list(ex.map(lambda i: judge(v[i]["src"], v[i]["hyp"]), ix))
    res[k] = sc
    print(f"judge {k:22} n={N} mean {sum(sc)/N:.2f} | >=4 {100*sum(x >= 4 for x in sc)/N:.1f}% | 5 {100*sum(x == 5 for x in sc)/N:.1f}%"
          f" | unparsed {sum(x == 0 for x in sc)}", flush=True)
json.dump({"ix": ix, "scores": res}, open(out, "w"))
