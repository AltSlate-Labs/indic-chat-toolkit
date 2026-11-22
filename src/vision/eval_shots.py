"""Screenshot PII redaction score. python eval_shots.py SHOTS_DIR MODEL   (envs/pii; MODEL = HF id or dir of a PII token classifier)
OCR lines (ocr.jsonl) are joined with newlines, the PII model finds character spans, each span becomes one box per OCR line
(line height x the x-range of its characters). A gold PII box counts as redacted if >= 60% of its area is covered by predicted boxes.
Reported for Latin/digit gold spans and native-script gold spans separately, plus a no-OCR upper bound (PII model on the original
text: a gold span counts as found if >= 60% of its characters are predicted as PII). False blur = predicted boxes that overlap gold
boxes by < 30% of their own area."""
import collections, json, re, sys
from transformers import AutoModelForTokenClassification, AutoTokenizer
sys.path.insert(0, __file__.rsplit("/", 2)[0])
from indic_pii.evaluate import predict_spans
d, mname = sys.argv[1], sys.argv[2]
import os; DUMP = open(os.environ["DUMP"], "w", encoding="utf-8") if os.environ.get("DUMP") else None; JOIN = os.environ.get("JOIN", "\n")  # separator between OCR lines (" " keeps entities that wrap across lines together)
tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForTokenClassification.from_pretrained(mname).cuda().eval()
shots = [json.loads(l) for l in open(f"{d}/shots.jsonl", encoding="utf-8")]; ocr = [json.loads(l) for l in open(f"{d}/ocr.jsonl", encoding="utf-8")]
area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
inter = lambda a, b: area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
native = lambda t: bool(re.search(r"[ऀ-෿]", t))
R = collections.defaultdict(lambda: [0, 0]); RA = [0, 0]; LG = collections.defaultdict(lambda: [0, 0]); U = collections.defaultdict(lambda: [0, 0]); fb = [0, 0]
for x, o in zip(shots, ocr):
    text, pos = "", []
    for li, L in enumerate(o["lines"]):
        for ci, c in enumerate(L["text"]): pos.append((li, ci))
        text += L["text"] + JOIN; pos.append(None)
    pred = []; spans = predict_spans(tok, model, text)
    for s, e, lab in spans:
        by = collections.defaultdict(list)
        for k in range(s, e):
            if k < len(pos) and pos[k] is not None: by[pos[k][0]].append(pos[k][1])
        for li, cs in by.items():
            L = o["lines"][li]; cs = [c for c in cs if c < len(L["chars"])]
            if cs: pred.append([L["chars"][min(cs)][0], L["box"][1], L["chars"][max(cs)][1], L["box"][3]])
    for b in x["boxes"]:
        m = x["msgs"][b["msg"]]; span_text = m["text"][b["span"][0]:b["span"][1]]
        grp = ("native " if native(span_text) else "latin ") + b["label"]
        g = b["xyxy"]; gh = g[3] - g[1]
        ok_area = sum(inter(g, p) for p in pred) >= 0.6 * area(g)  # v1 metric (gold box includes line spacing -> perfect blur ~0.58)
        vis = [p for p in pred if min(g[3], p[3]) - max(g[1], p[1]) >= 0.4 * gh]  # predicted boxes on this text line
        xs = sorted((max(g[0], p[0]), min(g[2], p[2])) for p in vis if min(g[2], p[2]) > max(g[0], p[0]))
        cov = 0; end = g[0]
        for a_, b_ in xs: cov += max(0, b_ - max(a_, end)); end = max(end, b_)
        ok = cov >= 0.8 * (g[2] - g[0])  # v2 metric: >= 80% of the span's width blurred on its line
        RA[0] += ok_area; RA[1] += 1
        if not ok and DUMP and not native(span_text):
            L = max(o["lines"], key=lambda L: inter(L["box"], b["xyxy"]), default={"text": "", "box": [0, 0, 0, 0]})
            cov = sum(inter(b["xyxy"], p) for p in pred) / max(1, area(b["xyxy"]))
            DUMP.write(json.dumps({"label": b["label"], "gold": span_text, "ocr_line": L["text"], "coverage": round(cov, 2),
                                   "pred_on_line": [text[s:e] for s, e, _ in spans if any(pos[k] and o["lines"][pos[k][0]] is L for k in range(s, min(e, len(pos))))]},
                                  ensure_ascii=False) + "\n")
        LG[m.get("lang") or "?"][0] += ok; LG[m.get("lang") or "?"][1] += 1
        R[grp][0] += ok; R[grp][1] += 1; R["native" if native(span_text) else "latin"][0] += ok; R["native" if native(span_text) else "latin"][1] += 1
    for p in pred:
        fb[1] += 1; fb[0] += sum(inter(p, b["xyxy"]) for b in x["boxes"]) < 0.3 * max(1, area(p))
    for m in x["msgs"]:  # no-OCR upper bound on the original text
        ps = {k for s, e, _ in predict_spans(tok, model, m["text"]) for k in range(s, e)}
        for s, e, lab in m["spans"]:
            g = "native" if native(m["text"][s:e]) else "latin"
            U[g][0] += sum(k in ps for k in range(s, e)) >= 0.6 * (e - s); U[g][1] += 1
print(f"redacted gold boxes: latin/digit {R['latin'][0]}/{R['latin'][1]} = {R['latin'][0]/max(1,R['latin'][1]):.3f} | "
      f"native script {R['native'][0]}/{R['native'][1]} = {R['native'][0]/max(1,R['native'][1]):.3f}")
print(f"(v1 area metric, all scripts: {RA[0]}/{RA[1]} = {RA[0]/max(1,RA[1]):.3f})")
print(f"no-OCR upper bound (spans found in the original text): latin {U['latin'][0]/max(1,U['latin'][1]):.3f} | native {U['native'][0]/max(1,U['native'][1]):.3f}")
print(f"false blur: {fb[0]}/{fb[1]} predicted boxes = {fb[0]/max(1,fb[1]):.3f}")
print("by message language: " + " ".join(f"{l}:{a/n:.2f}(n={n})" for l, (a, n) in sorted(LG.items())))
for g, (a, n) in sorted(R.items()):
    if " " in g and n >= 10: print(f"   {g:22s} {a/n:.2f} (n={n})")
