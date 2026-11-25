"""Held-out evaluation for Indic sentence embedders. python evaluate.py SPLIT MODEL...  (SPLIT: dev or test; MODEL as in probe.py:
name=dir:pooling[:prefix], pooling in mean | cls | cls_dense | mean_head)
1. dakshina : human-romanized <-> native recall@1, Dakshina rejoined SPLIT, 1000 sentences per language (10 langs)
2. flores   : English <-> Indic recall@1, IndicGenBench FLORES bitext (SPLIT test -> "test", dev -> "validation"), 10 langs
3. flores-ro: same, Indic side romanized by our dictionary+rule romanizer (synthetic romanization; same seed for every model)
4. sts      : IndicCrosslingualSTS (en-xx, test only), Spearman x 100, 10 langs
5. sts-en   : English STS Benchmark (mteb/stsbenchmark-sts; dev -> validation, test -> test), Spearman x 100
Each prints the mean and per-language scores. All sets are held out from training (training sentences are Dakshina TRAIN wiki,
English wiki, english-to-hinglish)."""
import glob, gzip, json, os, random, sys, numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from romanizer import Romanizer
D = os.path.expanduser("~/Downloads/data")
LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
ISO3 = {"hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "ur": "urd"}
split = sys.argv[1]
DEV = os.environ.get("EMBED_DEVICE", "cuda")   # cpu when the GPU is busy with someone else's job

def dakshina(l):
    b = f"{D}/dakshina/dakshina_dataset_v1.0/{l}/romanized/{l}.romanized.rejoined.{split}"
    seen, out = set(), []
    for a, n in zip(open(b + ".roman.txt", encoding="utf-8"), open(b + ".native.txt", encoding="utf-8")):
        a, n = a.strip(), n.strip()
        if a and n and n not in seen and len(n) > 20: seen.add(n); out.append((a, n))
    return out[:1000]
def flores(l):
    t = pq.read_table(f"{D}/IndicGenBenchFloresBitextMining/{ISO3[l]}-eng/{'test' if split == 'test' else 'validation'}-00000-of-00001.parquet")
    return [(r["sentence2"], r["sentence1"]) for r in t.to_pylist()]          # (English, Indic)
def sts(l):
    return pq.read_table(f"{D}/IndicCrosslingualSTS/en-{l}/test-00000-of-00001.parquet").to_pylist()
DATA = {l: (dakshina(l), flores(l), sts(l)) for l in LANGS}
STSB = [json.loads(x) for x in gzip.open(f"{D}/stsbenchmark/{'test' if split == 'test' else 'validation'}.jsonl.gz", "rt", encoding="utf-8")]
R = Romanizer(seed=7)
FRO = {l: [(e, R.sentence(i, l)) for e, i in DATA[l][1]] for l in LANGS}

class Enc:
    def __init__(s, spec):
        parts = spec.split(":"); s.repo, s.pool = parts[0], parts[1]; s.pre = parts[2] if len(parts) > 2 else ""
        s.tok = AutoTokenizer.from_pretrained(s.repo); s.m = AutoModel.from_pretrained(s.repo, torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        s.extra = None
        if s.pool == "mean_head":
            sd = torch.load(f"{s.repo}/head.pt", map_location=DEV); s.extra = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).to(DEV); s.extra.load_state_dict(sd)
        if s.pool == "cls_dense":
            from safetensors.torch import load_file
            w = load_file(f"{s.repo}/2_Dense/model.safetensors"); s.extra = torch.nn.Linear(w["linear.weight"].shape[1], w["linear.weight"].shape[0]).to(DEV)
            s.extra.weight.data, s.extra.bias.data = w["linear.weight"].float().to(DEV), w["linear.bias"].float().to(DEV)
    @torch.no_grad()
    def __call__(s, texts):
        out = []
        for i in range(0, len(texts), 64):
            b = s.tok([s.pre + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
            h = s.m(**b).last_hidden_state
            e = h[:, 0] if s.pool.startswith("cls") else (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            e = e.float()
            if s.pool == "cls_dense": e = torch.tanh(s.extra(e))
            if s.pool == "mean_head": e = s.extra(e)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)

def r1(enc, pairs):
    q = enc([a for a, _ in pairs]); c = enc([b for _, b in pairs]); s = q @ c.T; g = torch.arange(len(pairs), device=DEV)
    return ((s.argmax(1) == g).float().mean().item() + (s.argmax(0) == g).float().mean().item()) / 2
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]
def sts_score(enc, rows):
    a = enc([r["sentence1"] for r in rows]); b = enc([r["sentence2"] for r in rows])
    return 100 * spearman((a * b).sum(-1).cpu().numpy(), np.array([r["score"] for r in rows]))

for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1); enc = Enc(rest); res = {"dakshina": {}, "flores": {}, "flores-ro": {}, "sts": {}}
    for l in LANGS:
        dk, fl, st = DATA[l]
        res["dakshina"][l] = r1(enc, dk); res["flores"][l] = r1(enc, fl); res["flores-ro"][l] = r1(enc, FRO[l]); res["sts"][l] = sts_score(enc, st)
    res["sts-en"] = {"en": sts_score(enc, STSB)}
    for task, v in res.items():
        print(f"{name:26s} {task:10s} mean {np.mean(list(v.values())):6.3f} | " + " ".join(f"{l}:{x:.2f}" for l, x in v.items()), flush=True)
    del enc
    if DEV == "cuda": torch.cuda.empty_cache()
