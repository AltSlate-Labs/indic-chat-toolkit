"""Make a transformers-v5-saved mmBERT cross-encoder loadable by transformers 4.x too (same fixes as src/embed/package_st.py:
*_rope_theta from rope_parameters; tokenizer_class PreTrainedTokenizerFast; list extra_special_tokens -> additional_special_tokens),
and add sentence-transformers CrossEncoder metadata. python package_ce.py SRC OUT"""
import json, os, shutil, sys
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"): shutil.copy(f"{src}/{f}", f"{out}/{f}")
cf = json.load(open(f"{out}/config.json")); rp = cf.get("rope_parameters") or {}
if rp: cf["global_rope_theta"] = rp["full_attention"]["rope_theta"]; cf["local_rope_theta"] = rp["sliding_attention"]["rope_theta"]
cf["sentence_transformers"] = {"activation_fn": "torch.nn.modules.activation.Sigmoid", "version": "5.1.2"}
cf["id2label"] = {"0": "LABEL_0"}; cf["label2id"] = {"LABEL_0": 0}
json.dump(cf, open(f"{out}/config.json", "w"), indent=2)
tc = json.load(open(f"{out}/tokenizer_config.json")); tc["tokenizer_class"] = "PreTrainedTokenizerFast"
if isinstance(tc.get("extra_special_tokens"), list): tc["additional_special_tokens"] = tc.pop("extra_special_tokens")
for k in ("backend", "is_local", "local_files_only", "max_length", "stride", "truncation_strategy", "pad_to_multiple_of", "pad_token_type_id"): tc.pop(k, None)
tc["model_max_length"] = 512
json.dump(tc, open(f"{out}/tokenizer_config.json", "w"), indent=2, ensure_ascii=False)
print("ok", out)
