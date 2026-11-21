"""Smallest parquet shard of wikimedia/wikipedia (CC BY-SA 3.0/4.0) for Latin-script 'other' languages -> ~/Downloads/data/wikipedia-other/<lang>/"""
import os
from huggingface_hub import HfApi, hf_hub_download
OTHER = "id ms jv su tl ceb et fi hu nl de fr es pt it pl tr vi sw so af ro sv cs ha yo uz".split()
a = HfApi()
for l in OTHER:
    d = os.path.expanduser(f"~/Downloads/data/wikipedia-other/{l}")
    if os.path.isdir(d) and any(f.endswith(".parquet") for _, _, fs in os.walk(d) for f in fs):
        continue
    fs = [f for f in a.list_repo_tree("wikimedia/wikipedia", repo_type="dataset", path_in_repo=f"20231101.{l}") if f.path.endswith(".parquet")]
    f = min(fs, key=lambda x: x.size)
    print(l, f.path, f.size // 2**20, "MB", flush=True)
    hf_hub_download("wikimedia/wikipedia", f.path, repo_type="dataset", local_dir=d)
