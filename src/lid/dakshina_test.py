"""Build a LID test from Dakshina v1.0 TEST split (human romanizations, CC BY-SA 4.0).
python dakshina_test.py OUT.jsonl [N_PER_LANG=500]  -> native + roman rows per language (same sentences)."""
import json, os, random, sys

D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0")
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
r = random.Random(11)
with open(out, "w", encoding="utf-8") as f:
    for l in LANGS:
        p = f"{D}/{l}/romanized/{l}.romanized.rejoined.test."
        nat = open(p + "native.txt", encoding="utf-8").read().splitlines()
        rom = open(p + "roman.txt", encoding="utf-8").read().splitlines()
        assert len(nat) == len(rom), l
        idx = [i for i in range(len(nat)) if 3 <= len(rom[i].split()) <= 40]; r.shuffle(idx)
        for i in idx[:n]:
            f.write(json.dumps({"text": nat[i], "lang": l, "roman": False, "id": i}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"text": rom[i], "lang": l, "roman": True, "id": i}, ensure_ascii=False) + "\n")
print("ok")
