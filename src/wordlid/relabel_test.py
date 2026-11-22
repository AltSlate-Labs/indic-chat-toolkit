"""Apply the v3 English-word convention to an existing test-mix gold (Latin words with an Indic gold label that pass is_en -> en).
python relabel_test.py ENVOCAB.json IN.jsonl OUT.jsonl   (all model versions are rescored on OUT)"""
import json, sys
import synth_wordlid as S
S.EV = {k: set(v) for k, v in json.load(open(sys.argv[1])).items()}
n = ch = 0
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for l in open(sys.argv[2], encoding="utf-8"):
        x = json.loads(l); new = [y for _, y in S.conv(list(zip(x["words"], x["labels"])))]
        ch += sum(a != b for a, b in zip(new, x["labels"])); n += len(new); x["labels"] = new
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("relabelled", ch, "of", n, "words")
