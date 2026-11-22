"""OCR every screenshot of a render_shots.py folder. python ocr_shots.py DET_DIR REC_DIR SHOTS_DIR   (envs/vision) -> SHOTS_DIR/ocr.jsonl"""
import json, sys, time
from PIL import Image
from ppocr import PPOCR
o = PPOCR(sys.argv[1], sys.argv[2].split(",")); d = sys.argv[3]; t0 = time.time(); n = 0  # REC_DIR may be a comma-separated list
with open(f"{d}/ocr.jsonl", "w", encoding="utf-8") as f:
    for l in open(f"{d}/shots.jsonl", encoding="utf-8"):
        x = json.loads(l); f.write(json.dumps({"img": x["img"], "lines": o(Image.open(f"{d}/{x['img']}"))}, ensure_ascii=False) + "\n"); n += 1
print("images", n, f"{(time.time() - t0) / n:.2f} s/image")
