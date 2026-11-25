"""Embed the unique training targets with the teacher (BAAI/bge-m3 dense: CLS, L2-normalized; MIT).
python teacher.py DATA_DIR TEACHER_DIR [LIMIT]  -> DATA_DIR/targets.json (list of texts), DATA_DIR/targets.f16 (N x 1024 float16)
LIMIT > 0 = throughput probe on the first LIMIT texts (nothing written)."""
import json, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
d, tdir = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
texts = sorted({json.loads(l)["target"] for f in ("train", "dev") for l in open(f"{d}/{f}.jsonl", encoding="utf-8")})
texts.sort(key=len)                                   # length-sorted batches = less padding
if limit: texts = texts[:: max(1, len(texts) // limit)][:limit]
tok = AutoTokenizer.from_pretrained(tdir); m = AutoModel.from_pretrained(tdir, torch_dtype=torch.bfloat16).cuda().eval()
out = np.lib.format.open_memmap(f"{d}/targets.f16.npy", mode="w+", dtype=np.float16, shape=(len(texts), 1024)) if not limit else None
t0 = time.time(); bs = 256
with torch.no_grad():
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        e = torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0].float(), dim=-1)
        if out is not None: out[i:i + len(e)] = e.cpu().numpy().astype(np.float16)
        if i % (bs * 200) == 0: print(f"{i}/{len(texts)} {i / max(1e-9, time.time() - t0):.0f} texts/s", flush=True)
print(f"done {len(texts)} texts in {time.time() - t0:.0f} s ({len(texts) / (time.time() - t0):.0f}/s)")
if not limit:
    out.flush(); json.dump(texts, open(f"{d}/targets.json", "w", encoding="utf-8"), ensure_ascii=False)
