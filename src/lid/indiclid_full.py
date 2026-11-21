"""Full AI4Bharat IndicLID pipeline (FTN / FTR / BERT), following Inference/ai4bharat/IndicLID.py:
roman if share of [a-zA-Z0-9] among non-space, non-special chars > 0.5; FTR; if FTR score <= 0.6 -> IndicLID-BERT.
Step 1 (envs/lid):  python indiclid_full.py ft A.jsonl B.jsonl FT.json
Step 2 (envs/pii):  python indiclid_full.py bert FT.json ICACHE.json      (ICACHE in compare.py format)"""
import json, os, re, sys
M = os.path.expanduser("~/Downloads/model/IndicLID/")
ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
        "pan": "pa", "urd": "ur", "snd": "sd", "sin": "si"}
REV = ("asm ben brx guj hin kan kas kok mai mal mni mar nep ori pan san snd tam tel urd eng").split()  # 0..20 _Latn, 21 other


def roman_share(t):
    total = len(t) - len(re.findall(r'[@_!#$%^&*()<>?/\\|}{~:]', t)) - len(re.findall(r'\s', t))
    return len(re.findall('[a-zA-Z0-9]', t)) / total if total else 0


def ours(lab):
    iso, scr = (lab.split("_") + [""])[:2]; l = ISO3.get(iso)
    return l + ("-Latn" if scr == "Latn" else "") if l else ("en" if iso == "eng" else "other")


if sys.argv[1] == "ft":
    import fasttext
    ftr = fasttext.load_model(M + "IndicLID-FTR/model_baseline_roman.bin"); ftn = fasttext.load_model(M + "IndicLID-FTN/model_baseline_roman.bin")
    A = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]; out = {}
    sets = {f"A first {k}w" if k < 999 else "A full": [" ".join(x["text"].split()[:k]) for x in A] for k in (3, 5, 8, 999)}
    sets["B chat"] = [json.loads(l)["text"] for l in open(sys.argv[3], encoding="utf-8")]
    for name, texts in sets.items():
        rows = []
        for t in texts:
            t1 = t.replace("\n", " ")
            if roman_share(t) > 0.5:
                (lab,), (p,) = ftr.predict(t1); rows.append([t, lab[9:], float(p), "FTR"])
            else:
                (lab,), (p,) = ftn.predict(t1); rows.append([t, lab[9:], float(p), "FTN"])
        out[name] = rows
    json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False)
else:
    import torch, collections, transformers.models.bert.modeling_bert as mb, transformers.models.bert.configuration_bert as cb
    from transformers import AutoTokenizer, BertForSequenceClassification
    import transformers.activations as ta
    # allow-listed unpickling (globals inspected with pickletools: torch / transformers BERT classes only)
    safe = [mb.BertForSequenceClassification, mb.BertModel, mb.BertEmbeddings, mb.BertEncoder, mb.BertLayer, mb.BertAttention,
            mb.BertSelfOutput, mb.BertIntermediate, mb.BertOutput, mb.BertPooler, cb.BertConfig, ta.GELUActivation,
            torch.nn.Tanh, torch.nn.ModuleList, torch.nn.Dropout, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding,
            torch._C._nn.gelu, set, collections.OrderedDict]
    if hasattr(mb, "BertSelfAttention"): safe.append(mb.BertSelfAttention)
    torch.serialization.add_safe_globals(safe)
    try:
        old = torch.load(M + "IndicLID-BERT/basline_nn_simple.pt", map_location="cpu", weights_only=True)
        sd = old.state_dict()
        keep = ("vocab_size hidden_size num_hidden_layers num_attention_heads intermediate_size hidden_act max_position_embeddings "
                "type_vocab_size layer_norm_eps pad_token_id num_labels id2label label2id").split()
        cfg = cb.BertConfig(**{k: v for k, v in vars(old.config).items() if k in keep and k != "num_labels"})
        print("labels", cfg.num_labels, "layers", cfg.num_hidden_layers, "vocab", cfg.vocab_size)
    except Exception as e:  # class layout changed across transformers versions: read raw tensors instead
        print("module load failed:", type(e).__name__, str(e)[:200]); raise
    model = BertForSequenceClassification(cfg); print(model.load_state_dict(sd, strict=False)); model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained("ai4bharat/IndicBERTv2-MLM-only")
    ft = json.load(open(sys.argv[2])); cache = {}; stats = collections.Counter()
    for name, rows in ft.items():
        idx = [i for i, r in enumerate(rows) if r[3] == "FTR" and r[2] <= 0.6]; labs = [r[1] for r in rows]
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                enc = tok([rows[j][0] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
                pred = model(**enc).logits.argmax(-1).tolist()
                for j, p in zip(b, pred): labs[j] = REV[p] + "_Latn" if p < 21 else "other"
        stats[name] = len(idx)
        cache[name] = [ours(l) for l in labs]
    print("BERT-routed per set", dict(stats))
    json.dump(cache, open(sys.argv[3], "w"))
