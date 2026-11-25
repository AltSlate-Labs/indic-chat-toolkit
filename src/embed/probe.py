"""Cross-script retrieval probe for sentence embedders (no sentence-transformers needed).
python probe.py SPLIT N MODEL...   (MODEL = name=hf_repo_or_local_dir:pooling[:prefix])
Task per language: N human-romanized sentences (Dakshina rejoined SPLIT, roman side) as queries; their native-script originals as the
corpus; recall@1 of the right original. Also native→native-with-xlit? no: just roman→native and native→roman, averaged.
Pooling: mean (mean over tokens), cls (first token)."""
import os, sys, json, torch, numpy as np
from transformers import AutoTokenizer, AutoModel
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
split, n = sys.argv[1], int(sys.argv[2])
def data(l):
    r = [x.strip() for x in open(f"{D}/{l}/romanized/{l}.romanized.rejoined.{split}.roman.txt", encoding="utf-8")]
    t = [x.strip() for x in open(f"{D}/{l}/romanized/{l}.romanized.rejoined.{split}.native.txt", encoding="utf-8")]
    seen, out = set(), []
    for a, b in zip(r, t):
        if a and b and b not in seen and len(b) > 20: seen.add(b); out.append((a, b))
    return out[:n]
DATA = {l: data(l) for l in LANGS}
DENSE = None
HEAD = None
@torch.no_grad()
def enc(tok, m, texts, pool, pre):
    out = []
    for i in range(0, len(texts), 64):
        b = tok([pre + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        h = m(**b).last_hidden_state
        e = h[:, 0] if pool in ("cls", "cls_dense") else (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        if pool == "cls_dense": e = torch.tanh(DENSE(e.float()))  # LaBSE: CLS -> Dense(tanh) (sentence-transformers 2_Dense)
        if pool == "mean_head": e = HEAD(e.float())             # our student: mean pooling -> Linear(384 -> 1024)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1))
    return torch.cat(out)
for spec in sys.argv[3:]:
    name, rest = spec.split("=", 1); parts = rest.split(":"); repo, pool = parts[0], parts[1]; pre = parts[2] if len(parts) > 2 else ""
    tok = AutoTokenizer.from_pretrained(repo); m = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16).cuda().eval()
    if pool == "mean_head":
        sd = torch.load(f"{repo}/head.pt"); HEAD = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); HEAD.load_state_dict(sd)
    if pool == "cls_dense":
        from safetensors.torch import load_file
        w = load_file(f"{repo}/2_Dense/model.safetensors"); DENSE = torch.nn.Linear(w["linear.weight"].shape[1], w["linear.weight"].shape[0]).cuda()
        DENSE.weight.data, DENSE.bias.data = w["linear.weight"].float().cuda(), w["linear.bias"].float().cuda()
    res = {}
    for l, pairs in DATA.items():
        q = enc(tok, m, [a for a, _ in pairs], pool, pre); c = enc(tok, m, [b for _, b in pairs], pool, pre)
        s = q @ c.T; g = torch.arange(len(pairs), device="cuda")
        res[l] = round(((s.argmax(1) == g).float().mean().item() + (s.argmax(0) == g).float().mean().item()) / 2, 3)
    print(f"{name:28s} mean {np.mean(list(res.values())):.3f} | " + " ".join(f"{l}:{v:.2f}" for l, v in res.items()), flush=True)
    del m; torch.cuda.empty_cache()
