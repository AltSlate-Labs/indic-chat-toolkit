"""Which side loses STS? IndicCrosslingualSTS (en-xx): cosine of student vs teacher on each side, and Spearman for
teacher/teacher, student/student, teacher-en + student-xx, student-en + teacher-xx. python diag_sts.py STUDENT_DIR TEACHER_DIR"""
import os, sys, numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
D = os.path.expanduser("~/Downloads/data")
LANGS = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
def load(d, head):
    tok = AutoTokenizer.from_pretrained(d); m = AutoModel.from_pretrained(d, torch_dtype=torch.bfloat16).cuda().eval(); h = None
    if head:
        sd = torch.load(f"{d}/head.pt"); h = torch.nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).cuda(); h.load_state_dict(sd)
    @torch.no_grad()
    def f(texts):
        out = []
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            x = m(**b).last_hidden_state
            e = (x * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True) if head else x[:, 0]
            e = e.float(); e = h(e) if head else e
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)
    return f
S, T = load(sys.argv[1], True), load(sys.argv[2], False)
def sp(a, b): return 100 * np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
print("lang  cos(s,t)en cos(s,t)xx | TT   SS   Ten+Sxx Sen+Txx")
for l in LANGS:
    r = pq.read_table(f"{D}/IndicCrosslingualSTS/en-{l}/test-00000-of-00001.parquet").to_pylist(); g = np.array([x["score"] for x in r])
    a, b = [x["sentence1"] for x in r], [x["sentence2"] for x in r]
    sa, sb, ta, tb = S(a), S(b), T(a), T(b)
    c = lambda u, v: (u * v).sum(-1).cpu().numpy()
    print(f"{l:4s} {c(sa,ta).mean():.3f}      {c(sb,tb).mean():.3f}      | {sp(c(ta,tb),g):4.1f} {sp(c(sa,sb),g):4.1f} {sp(c(ta,sb),g):4.1f}    {sp(c(sa,tb),g):4.1f}", flush=True)
print("sample", a[0][:80], "||", b[0][:80])
