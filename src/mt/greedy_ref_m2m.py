# PyTorch greedy reference for the transformers.js check of an M2M100 MT model: 150 PHINC dev + 50 FLORES te native + 50 FLORES ta roman.
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
mdir, out = sys.argv[1], sys.argv[2]
te2kn = lambda t: "".join(chr(ord(c) + 0x80) if 0xC00 <= ord(c) <= 0xC7F else c for c in t)
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
fl = rd("jobs/mt-eval1/out/flores-dev.jsonl")
rows = [dict(r, lang="hi") for r in rd("jobs/mt-base/out/dev.jsonl")[:150]]
rows += [r for r in fl if r["lang"] == "te" and r["kind"] == "native"][:50] + [r for r in fl if r["lang"] == "ta" and r["kind"] == "roman"][:50]
tok = AutoTokenizer.from_pretrained(mdir); m = AutoModelForSeq2SeqLM.from_pretrained(mdir).cuda().eval()
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        tok.src_lang = "kn" if r["lang"] == "te" else r["lang"]
        x = tok(te2kn(r["src"]), return_tensors="pt").to("cuda")
        with torch.no_grad(): o = m.generate(**x, forced_bos_token_id=tok.get_lang_id("en"), num_beams=1, do_sample=False, max_new_tokens=128)
        f.write(json.dumps({"src": r["src"], "lang": r["lang"], "hyp": tok.decode(o[0], skip_special_tokens=True)}, ensure_ascii=False) + "\n")
print("wrote", len(rows))
