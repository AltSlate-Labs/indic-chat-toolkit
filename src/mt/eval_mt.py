"""Translate a jsonl split (src, ref) with a baseline and score chrF++ and BLEU, case-insensitive (PHINC references are mostly
lower-case). python eval_mt.py SPLIT_JSONL NAME KIND MODEL_DIR [N]
KIND: copy (output = input) | marian | m2m (source code = row lang, else hi) | nllb (source hin_Deva -> eng_Latn) | llm (OpenAI-compatible server at 127.0.0.1:8080,
MODEL_DIR unused). Rows with "kind"/"lang" also get per-group scores. Writes preds to jobs/mt-base/out/pred-NAME.jsonl."""
import json, os, sys, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
split, name, kind = sys.argv[1:4]; mdir = sys.argv[4] if len(sys.argv) > 4 else ""
N = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
rows = [json.loads(l) for l in open(split, encoding="utf-8")][:N]
src = [r["src"] for r in rows]; ref = [r["ref"] for r in rows]
if kind == "copy": hyp = src
elif kind == "llm":
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    P = ("Translate this message into natural English. It may be in an Indian language written in Latin or native script, or mix "
         "languages with English. Keep @handles, #hashtags, URLs and emoji unchanged. Reply with the translation only.\n\n")
    def tr(t):
        body = json.dumps({"messages": [{"role": "user", "content": P + t}], "temperature": 0, "max_tokens": 300,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        q = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(q, timeout=600).read())["choices"][0]["message"]["content"].strip()
    with ThreadPoolExecutor(8) as ex: hyp = list(ex.map(tr, src))
else:
    tok = AutoTokenizer.from_pretrained(mdir)
    m = AutoModelForSeq2SeqLM.from_pretrained(mdir, dtype=torch.bfloat16).cuda().eval()
    kw = {"forced_bos_token_id": tok.convert_tokens_to_ids("eng_Latn")} if kind == "nllb" else {}
    if kind == "m2m": kw = {"forced_bos_token_id": tok.get_lang_id("en")}
    # nllb: source-language code from the row's "lang" (FLORES), else Hindi (PHINC); batches never mix languages
    NL = dict(hi="hin_Deva", bn="ben_Beng", ta="tam_Taml", te="tel_Telu", mr="mar_Deva", gu="guj_Gujr", kn="kan_Knda",
              ml="mal_Mlym", pa="pan_Guru", ur="urd_Arab")
    hyp = [None] * len(src)
    te2kn = lambda t: "".join(chr(ord(c) + 0x80) if 0xC00 <= ord(c) <= 0xC7F else c for c in t)   # M2M100 lacks Telugu script
    for lg in sorted({r.get("lang") or "hi" for r in rows}):
        ix = [i for i, r in enumerate(rows) if (r.get("lang") or "hi") == lg]
        if kind == "nllb": tok.src_lang = NL[lg]
        if kind == "m2m": tok.src_lang = "kn" if lg == "te" else lg   # M2M100 has no Telugu code
        for j in range(0, len(ix), 64):
            b = tok([te2kn(src[i]) if kind == "m2m" else src[i] for i in ix[j:j + 64]], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
            with torch.no_grad(): o = m.generate(**b, num_beams=4, max_new_tokens=256, **kw)
            for i, h in zip(ix[j:j + 64], tok.batch_decode(o, skip_special_tokens=True)): hyp[i] = h
os.makedirs("jobs/mt-base/out", exist_ok=True)
with open(f"jobs/mt-base/out/pred-{name}.jsonl", "w", encoding="utf-8") as f:
    for s, h, r in zip(src, hyp, ref): f.write(json.dumps({"src": s, "hyp": h, "ref": r}, ensure_ascii=False) + "\n")
L = lambda xs: [x.lower() for x in xs]
chrf = sacrebleu.corpus_chrf(L(hyp), [L(ref)], word_order=2).score; bleu = sacrebleu.corpus_bleu(L(hyp), [L(ref)]).score
print(f"{name:28} {os.path.basename(split):12} n={len(rows)} chrF++ {chrf:.1f} BLEU {bleu:.1f}")
groups = sorted({(r.get("kind"), r.get("lang")) for r in rows if r.get("lang")})
for k in sorted({g[0] for g in groups}):
    out = []
    for g in [g for g in groups if g[0] == k]:
        ix = [i for i, r in enumerate(rows) if (r.get("kind"), r.get("lang")) == g]
        out.append(f"{g[1]}:{sacrebleu.corpus_chrf(L([hyp[i] for i in ix]), [L([ref[i] for i in ix])], word_order=2).score:.1f}")
    ix = [i for i, r in enumerate(rows) if r.get("kind") == k]
    print(f"  {k}: chrF++ {sacrebleu.corpus_chrf(L([hyp[i] for i in ix]), [L([ref[i] for i in ix])], word_order=2).score:.1f} | " + " ".join(out))
