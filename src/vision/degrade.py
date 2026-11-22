"""Degrade rendered screenshots the way sharing does (JPEG re-compression, downscaling), keeping gold boxes consistent.
python degrade.py SRC_DIR DST_DIR SCALE QUALITY   -> DST_DIR/NNNN.png (decoded JPEG) + shots.jsonl with boxes scaled."""
import io, json, os, sys
from PIL import Image
src, dst, scale, q = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); os.makedirs(dst, exist_ok=True)
with open(f"{dst}/shots.jsonl", "w", encoding="utf-8") as f:
    for l in open(f"{src}/shots.jsonl", encoding="utf-8"):
        x = json.loads(l); im = Image.open(f"{src}/{x['img']}").convert("RGB")
        if scale != 1: im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); Image.open(b).convert("RGB").save(f"{dst}/{x['img']}")
        for bx in x["boxes"]: bx["xyxy"] = [round(v * scale) for v in bx["xyxy"]]
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("degraded", dst)
