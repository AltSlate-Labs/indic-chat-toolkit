"""Baseline: UnimeType Transliteration-2.0-0.8B (Apache-2.0; Hindi is its only Indic language) on the Hindi rows.
Protocol unimetype-convert-short-v3: empty system prompt, JSON user message, temperature 0.
python run_unimetype.py TEST.jsonl OUT.jsonl MODE   (MODE: hi_en = enabled [english, hindi]; hi = [hindi] only)"""
import json, os, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M = os.path.expanduser("~/Downloads/model/UnimeType-Transliteration-2.0-0.8B")
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
test = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]; mode = sys.argv[3]
langs = ["english", "hindi"] if mode == "hi_en" else ["hindi"]
rows = [x for x in test if x["lang"] == "hi"]
prompts = [tok.apply_chat_template([{"role": "system", "content": ""}, {"role": "user", "content": json.dumps(
    {"operation": "convert", "input": x["roman"], "enabled_languages": langs, "primary_language": "hindi"}, ensure_ascii=False)}],
    tokenize=False, add_generation_prompt=True) for x in rows]
out = []
with torch.no_grad():
    for i in range(0, len(prompts), 32):
        enc = tok(prompts[i:i + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=256, do_sample=False)
        out += [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for s in g]
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for x, p in zip(rows, out): f.write(json.dumps({"lang": "hi", "roman": x["roman"], "native": x["native"], "pred": p}, ensure_ascii=False) + "\n")
print(out[:3])
