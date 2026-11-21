#!/usr/bin/env bash
# lid-v4: v3 data + lid-chat4 (out-of-scope Indic chat as "other", more ur/pa/ta/ml/kn chat).
# Dakshina TEST and test B never used for training. Compare vs GlotLID on test A + B.
set -euo pipefail
cd ~/agent/rnd/src/lid
O=../../jobs/lid-v4/out; mkdir -p $O
echo "[lid-v4] start $(date -Is)"
until grep -q "\[lid-chat4\] end" ../../jobs/lid-chat4/log.txt; do sleep 60; done   # wait for chat data
python3 - <<PY
import json
tb={" ".join(json.loads(l)["text"].lower().split()) for l in open("../../jobs/lid-testb/out/testb.jsonl",encoding="utf-8")}
ch=[json.loads(l) for f in ("chat-oos","chat-weak") for l in open(f"../../jobs/lid-chat4/out/{f}.jsonl",encoding="utf-8")]
ov=sum(" ".join(x[k].lower().split()) in tb for x in ch for k in ("native","roman"))
print("chat pairs", len(ch), "exact overlap with test B", ov)
import os
ba={" ".join(x["romanized sentence"].lower().split()) for x in json.load(open(os.path.expanduser("~/Downloads/data/bhasha-abhijnaanam/bhasha-abhijnaanam.json")))["data"] if x["romanized sentence"]}
print("exact overlap with Bhasha-Abhijnaanam roman", sum(" ".join(x["roman"].lower().split()) in ba for x in ch))
PY
. ../../envs/pii/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
python fetch_other.py
[ -s $O/train.jsonl ] || CHAT=../../jobs/lid-chat/out/chat-train.jsonl,../../jobs/lid-chat4/out/chat-oos.jsonl,../../jobs/lid-chat4/out/chat-weak.jsonl python synth_lid.py $O 1
python - <<PY
import json,random
rows=[json.loads(l) for l in open("$O/train.jsonl",encoding="utf-8")]
r=random.Random(0)
for lab in ("ur-Latn","pa-Latn","other"):
    xs=[x["text"] for x in rows if x["label"]==lab]; print(lab, len(xs), "|", " || ".join(r.sample(xs,3)))
PY
[ -s $O/model/train_meta.json ] || python train_lid.py --model ~/Downloads/model/mmBERT-small --train $O/train.jsonl --dev $O/dev.jsonl --out $O/model --epochs 1
A=../../jobs/lid-gap/out/dakshina-test.jsonl; B=../../jobs/lid-testb/out/testb.jsonl
deactivate; . ../../envs/lid/bin/activate
cp ../../jobs/lid-v3/out/glot-cache.json $O/ 2>/dev/null || python compare.py glot $A $B $O/glot-cache.json > /dev/null
deactivate; . ../../envs/pii/bin/activate
python compare.py $O/model $A $B $O/glot-cache.json ../../jobs/lid-indiclid/out/indiclid-full-cache.json
python out_of_scope.py $O/model ~/Downloads/data/bhasha-abhijnaanam/bhasha-abhijnaanam.json
echo "[lid-v4] end $(date -Is)"
