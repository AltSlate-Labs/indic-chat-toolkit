"""Leakage check for embed-v5: how many STS Benchmark sentences have a near-duplicate in the SNLI training text (same words after
lower-casing, removing punctuation and the articles a/an/the)? Then STS on all pairs vs pairs where neither sentence has one.
python neardup_sts.py SPLIT(validation|test) TRAIN_JSONL MODEL_DIR..."""
import gzip, json, os, re, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
split, train = sys.argv[1:3]; D = os.path.expanduser("~/Downloads/data")
key = lambda t: " ".join(w for w in re.sub(r"[^\w\s]", " ", t.lower()).split() if w not in ("a", "an", "the"))
T = {key(json.loads(l)["input"]) for l in open(train, encoding="utf-8")}
rows = [json.loads(l) for l in gzip.open(f"{D}/stsbenchmark/{split}.jsonl.gz", "rt", encoding="utf-8")]
clean = [r for r in rows if key(r["sentence1"]) not in T and key(r["sentence2"]) not in T]
sent = {s for r in rows for s in (r["sentence1"], r["sentence2"])}
print(f"{split}: pairs {len(rows)}, sentences {len(sent)}, with near-dup in train {sum(key(s) in T for s in sent)}, clean pairs {len(clean)}")
def sp(a, b): return 100 * np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
for d in sys.argv[3:]:
    tok = AutoTokenizer.from_pretrained(d); m = AutoModel.from_pretrained(d, torch_dtype=torch.bfloat16).cuda().eval()
    sd = torch.load(f"{d}/head.pt", map_location="cuda"); h = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); h.load_state_dict(sd)
    @torch.no_grad()
    def E(x):
        b = tok(x, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda"); o = m(**b).last_hidden_state; k = b["attention_mask"][..., None]
        return torch.nn.functional.normalize(h(((o * k).sum(1) / k.sum(1)).float()), dim=-1)
    def score(rs):
        a = torch.cat([E([r["sentence1"] for r in rs[i:i + 128]]) for i in range(0, len(rs), 128)])
        b = torch.cat([E([r["sentence2"] for r in rs[i:i + 128]]) for i in range(0, len(rs), 128)])
        return sp((a * b).sum(-1).cpu().numpy(), np.array([r["score"] for r in rs]))
    print(f"  {d}: all {score(rows):.1f} | clean {score(clean):.1f}", flush=True)
