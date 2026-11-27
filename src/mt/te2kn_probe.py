# Probe: map native Telugu script to Kannada script (Unicode block offset +0x80) before an M2M100 model whose vocab lacks Telugu.
import json, sys, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
pred, mdir = sys.argv[1], sys.argv[2]
te2kn = lambda s: "".join(chr(ord(c) + 0x80) if 0xC00 <= ord(c) <= 0xC7F else c for s_ in [s] for c in s_)
rows = [json.loads(l) for l in open(pred)]
te = [r for r in rows if any(0xC00 <= ord(c) <= 0xC7F for c in r["src"])]
tok = AutoTokenizer.from_pretrained(mdir); tok.src_lang = "kn"
m = AutoModelForSeq2SeqLM.from_pretrained(mdir, torch_dtype=torch.bfloat16).cuda().eval()
src = [te2kn(r["src"]) for r in te]
ids = sum((tok(s).input_ids for s in src), [])
print("unk frac after map", sum(i == tok.unk_token_id for i in ids) / len(ids))
hyp = []
for i in range(0, len(src), 32):
    b = tok(src[i:i+32], return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        o = m.generate(**b, forced_bos_token_id=tok.get_lang_id("en"), num_beams=4, max_new_tokens=160)
    hyp += tok.batch_decode(o, skip_special_tokens=True)
refs = [r["ref"].lower() for r in te]; h = [x.lower() for x in hyp]
print("te native n=%d before chrF++ %.1f after chrF++ %.1f" % (len(te),
      sacrebleu.corpus_chrf([r["hyp"].lower() for r in te], [refs], word_order=2).score,
      sacrebleu.corpus_chrf(h, [refs], word_order=2).score))
for x in hyp[:3]: print(" ", x[:100])
