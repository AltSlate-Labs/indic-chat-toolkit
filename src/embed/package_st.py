"""Package a trained student (train_embed.py output: HF encoder + head.pt) as a sentence-transformers model:
Transformer -> Pooling(mean) -> Dense(384->1024, identity, bias) -> Normalize. Saves its embeddings of src/embed/st_check.txt;
check_st.py (training env) compares them with the training-time encoder. python package_st.py STUDENT_DIR OUT_DIR   (envs/st, CPU is fine)"""
import json, os, shutil, sys, torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel
src, out = sys.argv[1:3]
# copy the encoder files; transformers v5 writes tokenizer_class "TokenizersBackend", which v4 cannot load -> PreTrainedTokenizerFast
os.makedirs(out, exist_ok=True)
for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"): shutil.copy(f"{src}/{f}", f"{out}/{f}")
cf = json.load(open(f"{out}/config.json")); rp = cf.get("rope_parameters") or {}   # v5 rope format; v4 reads *_rope_theta
if rp: cf["global_rope_theta"] = rp["full_attention"]["rope_theta"]; cf["local_rope_theta"] = rp["sliding_attention"]["rope_theta"]
json.dump(cf, open(f"{out}/config.json", "w"), indent=2)
tc = json.load(open(f"{out}/tokenizer_config.json")); tc["tokenizer_class"] = "PreTrainedTokenizerFast"
if isinstance(tc.get("extra_special_tokens"), list): tc["additional_special_tokens"] = tc.pop("extra_special_tokens")
for k in ("backend", "is_local", "local_files_only", "max_length", "stride", "truncation_strategy", "pad_to_multiple_of", "pad_token_type_id"):
    tc.pop(k, None)
json.dump(tc, open(f"{out}/tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
tr = models.Transformer(out, max_seq_length=128)
dim = tr.get_word_embedding_dimension()
pool = models.Pooling(dim, pooling_mode="mean")
sd = torch.load(f"{src}/head.pt", map_location="cpu")
dense = models.Dense(dim, sd["weight"].shape[0], bias=True, activation_function=torch.nn.Identity())
dense.linear.load_state_dict(sd)
st = SentenceTransformer(modules=[tr, pool, dense, models.Normalize()], device="cpu")
st.save(out)
import numpy as np
S = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_check.txt"), encoding="utf-8") if l.strip()]
a = SentenceTransformer(out, device="cpu").encode(S, normalize_embeddings=False)
np.save(f"{out}/../st_check.npy", a); print("saved", a.shape, "-> run check_st.py in the training env"); print((a[:5] @ a[:5].T).round(3))
