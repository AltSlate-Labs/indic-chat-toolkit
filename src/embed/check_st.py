"""Compare package_st.py's sentence-transformers embeddings (saved as ST_DIR/../st_check.npy) with the training-time encoder
(mean pool -> head -> L2) run in the training env. python check_st.py STUDENT_DIR ST_DIR"""
import os, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
src, st = sys.argv[1:3]
S = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_check.txt"), encoding="utf-8") if l.strip()]
a = torch.tensor(np.load(f"{st}/../st_check.npy"))
tok = AutoTokenizer.from_pretrained(src); m = AutoModel.from_pretrained(src).eval(); sd = torch.load(f"{src}/head.pt", map_location="cpu")
head = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]); head.load_state_dict(sd)
with torch.no_grad():
    b = tok(S, padding=True, truncation=True, max_length=128, return_tensors="pt"); h = m(**b).last_hidden_state
    e = torch.nn.functional.normalize(head((h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)), dim=-1)
c = (torch.nn.functional.normalize(a, dim=-1) * e).sum(-1); print("cos per sentence:", [round(x, 5) for x in c.tolist()])
assert c.min() > 0.9999, "MISMATCH"; print("OK")
