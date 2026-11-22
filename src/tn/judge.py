"""Score TTS-normalization outputs. python judge.py TEST.jsonl NAME=PRED.jsonl [...]   (judge server at $JUDGE_URL, default :8080)
 residue: output still contains a digit or one of ₹ % ° (a TTS model would have to read it raw)
 judge  : TEST teacher (Qwen3.8-27B) says the output reads every number/symbol correctly and changes nothing else (YES/NO)
 cer    : character error rate against the gold spoken form, when the test has one (test B; teacher-written, noisy)"""
import json, os, re, sys, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xlit"))
from metrics import ed, norm
def cer(r, h): r, h = norm(r), norm(h); return ed(r, h) / max(1, len(r))
NAMES = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
RES = re.compile(r"[0-9०-९০-৯੦-੯૦-૯௦-௯౦-౯೦-೯൦-൯۰-۹٠-٩₹%°]")
URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8080/v1/chat/completions")
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]


def judge(x, p):
    if RES.search(p): return False
    L = NAMES[x["lang"]]
    q = (f"Original {L} text:\n{x['text']}\n\nVersion for text-to-speech:\n{p}\n\nIs the second text a correct spoken version of the first? "
         f"It must write every number, symbol, unit, date, time and abbreviation as correct {L} words that a {L} speaker would say "
         f"(any natural reading is fine, e.g. a year read as a year), with the right values, and must not change, add or drop anything "
         f"else. Answer with one word: YES or NO.")
    body = json.dumps({"messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 8,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        o = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=600))
        return "YES" in o["choices"][0]["message"]["content"].upper()
    except Exception: return False


for arg in sys.argv[2:]:
    name, f = arg.split("=", 1)
    preds = [json.loads(l)["pred"] for l in open(f, encoding="utf-8")]
    with ThreadPoolExecutor(8) as pool: ok = list(pool.map(lambda a: judge(*a), zip(test, preds)))
    by = collections.defaultdict(list); res = collections.defaultdict(list); cr = collections.defaultdict(list)
    for x, p, k in zip(test, preds, ok):
        by[x["lang"]].append(k); res[x["lang"]].append(bool(RES.search(p)))
        if "spoken" in x: cr[x["lang"]].append(cer(" ".join(x["spoken"].split()), " ".join(p.split())))
    m = lambda d: sum(sum(v) / len(v) for v in d.values()) / len(d)
    print(f"{name:10s} judge-ok {m(by):.3f}  residue {m(res):.3f}" + (f"  CER {m(cr):.3f}" if cr else "") + " | " +
          " ".join(f"{l}:{sum(v)/len(v):.2f}" for l, v in sorted(by.items())), flush=True)
    with open(f + ".judge", "w") as g: json.dump(ok, g)
