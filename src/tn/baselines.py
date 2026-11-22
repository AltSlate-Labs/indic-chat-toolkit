"""Non-LLM baselines for TTS normalization: python baselines.py TEST.jsonl OUT_DIR   (envs/tn)
 copy  : input unchanged (what a TTS model sees today)
 rules : every number -> indic-numtowords words (Indian comma grouping removed, native digits mapped, decimals read digit by digit);
         symbols and formats (₹, %, :, /) left as they are
 itp   : indic-tts-preprocess 0.1.1 (hi, mr only; other languages = copy)"""
import json, re, sys
import indic_numtowords as n
from indic_tts_preprocess import preprocess
NAT = {c: str(i) for base in (0x966, 0x9E6, 0xA66, 0xAE6, 0xBE6, 0xC66, 0xCE6, 0xD66, 0x6F0, 0x660) for i, c in enumerate(chr(base + k) for k in range(10))}
NUM = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def words(num, lang):
    try: return n.num2words(int(num), lang=lang)
    except Exception: return num


def rules(t, lang):
    t = "".join(NAT.get(c, c) for c in t)
    def rep(m):
        s = m.group(0).replace(",", "")
        if "." in s:
            a, b = s.split(".", 1); return words(a, lang) + " " + " ".join(words(d, lang) for d in b)
        return words(s, lang)
    return NUM.sub(rep, t)


def itp(t, lang):
    try: return preprocess(t, lang) if lang in ("hi", "mr") else t
    except Exception: return t


test, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(test, encoding="utf-8")]
for name, f in (("copy", lambda t, l: t), ("rules", rules), ("itp", itp)):
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8") as g:
        for x in rows: g.write(json.dumps({"pred": f(x["text"], x["lang"])}, ensure_ascii=False) + "\n")
print("baselines done", test)
