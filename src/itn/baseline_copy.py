"""Copy baseline: python baseline_copy.py TEST.jsonl OUT.jsonl"""
import json, sys
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for l in open(sys.argv[1], encoding="utf-8"): f.write(json.dumps({"pred": json.loads(l)["src"].split(": ", 1)[1]}, ensure_ascii=False) + "\n")
