"""Quantize an exported seq2seq ONNX (encoder + merged decoder) to per-channel dynamic int8 with transformers.js names, then
compare CER of ONNX fp32 / int8 greedy generation with PyTorch predictions on a test. Runs in envs/export (optimum-onnx).
python onnx_check.py ONNX_DIR TEST.jsonl TORCH_PRED.jsonl [N=300]"""
import json, os, sys, time
from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score
d, test_f, ref_f = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
for f in ("encoder_model", "decoder_model_merged", "decoder_model", "decoder_with_past_model"):
    if not os.path.exists(f"{d}/{f}_quantized.onnx"):
        quantize_dynamic(f"{d}/{f}.onnx", f"{d}/{f}_quantized.onnx", weight_type=QuantType.QInt8, per_channel=True,
                         extra_options={"EnableSubgraph": True})  # merged decoder keeps its weights inside If-subgraphs
    print(f, round(os.path.getsize(f"{d}/{f}.onnx") / 1e6), "->", round(os.path.getsize(f"{d}/{f}_quantized.onnx") / 1e6), "MB", flush=True)
test = [json.loads(l) for l in open(test_f, encoding="utf-8")][:n]
torch_pred = [json.loads(l)["pred"] for l in open(ref_f, encoding="utf-8")][:n]
tok = AutoTokenizer.from_pretrained(d)
print("torch CER", round(score(test, torch_pred)[0], 4))
# use_cache=False: optimum 2.1 + transformers 4.57 feed past KV with the wrong layout; the no-past decoder has the same weights.
for tag, kw in (("fp32", {"decoder_file_name": "decoder_model.onnx"}), ("int8", {"encoder_file_name": "encoder_model_quantized.onnx", "decoder_file_name": "decoder_model_quantized.onnx"})):
    m = ORTModelForSeq2SeqLM.from_pretrained(d, use_cache=False, use_merged=False, **kw); pred = []; t0 = time.time()
    for x in test:
        enc = tok(f'{x["lang"]}: {x["roman"]}', return_tensors="pt")
        g = m.generate(**enc, max_new_tokens=4 * enc.input_ids.shape[1] + 32, do_sample=False, num_beams=1)
        pred.append(tok.decode(g[0], skip_special_tokens=True))
    same = sum(a == b for a, b in zip(pred, torch_pred)) / n
    print(tag, "CER", round(score(test, pred)[0], 4), "identical to torch", round(same, 3), f"{(time.time() - t0) / n * 1000:.0f} ms/msg (CPU)", flush=True)
