"""Minimal PP-OCRv5 (ONNX, Apache-2.0) inference with onnxruntime + numpy + Pillow only (no OpenCV / pyclipper: PyPI was unreachable
from the Spark). Made for screenshots with horizontal text: the detector's probability map is thresholded and split into line boxes
by row bands and column gaps; each line is recognized with CTC, and every output character keeps its approximate x-range, so text
spans can be mapped back to pixels.
    ocr = PPOCR(det_dir, rec_dir); lines = ocr(Image)  ->  [{"box": [x0,y0,x1,y1], "text": str, "chars": [[x0,x1], ...]}, ...]"""
import numpy as np, onnxruntime as ort, yaml
from PIL import Image


class PPOCR:
    def __init__(s, det_dir, rec_dirs, providers=("CPUExecutionProvider",)):
        """rec_dirs: one recognizer dir or a list; with several, each line keeps the reading with the highest mean confidence
        (the Chinese/Latin PP-OCRv5 model turns Devanagari into CJK garbage; the Devanagari model also reads Latin and digits)."""
        s.det = ort.InferenceSession(f"{det_dir}/inference.onnx", providers=list(providers)); s.recs = []
        for rd in ([rec_dirs] if isinstance(rec_dirs, str) else rec_dirs):
            cfg = yaml.safe_load(open(f"{rd}/inference.yml", encoding="utf-8"))
            s.recs.append((ort.InferenceSession(f"{rd}/inference.onnx", providers=list(providers)),
                           ["<blank>"] + [str(c) for c in cfg["PostProcess"]["character_dict"]] + [" "]))  # use_space_char

    def detect(s, img, thr=0.3, long_side=960):
        w, h = img.size; k = min(1.0, long_side / max(w, h))
        W, H = max(32, int(round(w * k / 32)) * 32), max(32, int(round(h * k / 32)) * 32)
        x = np.asarray(img.convert("RGB").resize((W, H)), dtype=np.float32)[:, :, ::-1] / 255.0  # BGR, as trained
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        p = s.det.run(None, {s.det.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        m = p > thr; sx, sy = w / W, h / H; boxes = []
        rows = m.any(1); y = 0
        while y < H:  # row bands
            if not rows[y]: y += 1; continue
            y0 = y
            while y < H and rows[y]: y += 1
            band = m[y0:y]; cols = band.any(0); bh = y - y0; x_ = 0
            while x_ < W:  # split a band where the horizontal gap is larger than ~1.2 line heights
                if not cols[x_]: x_ += 1; continue
                x0 = x_; gap = 0
                while x_ < W and gap <= 1.2 * bh:
                    gap = 0 if cols[x_] else gap + 1; x_ += 1
                x1 = x_ - gap
                if (x1 - x0) * bh > 20:
                    pad = 0.45 * bh  # DB maps shrink the text; grow the box (stands in for the unclip step)
                    boxes.append([max(0, (x0 - pad) * sx), max(0, (y0 - pad) * sy), min(w, (x1 + pad) * sx), min(h, (y + pad) * sy)])
        return boxes

    def recognize(s, img, box):
        best = None
        for sess, chars in s.recs:
            r = s._rec(img, box, sess, chars)
            if best is None or r[2] > best[2]: best = r
        return best[0], best[1]

    def _rec(s, img, box, sess, chars):
        x0, y0, x1, y1 = box; crop = img.convert("RGB").crop((x0, y0, x1, y1))
        H = 48; Wc = max(16, int(crop.width * H / max(1, crop.height)))
        a = np.asarray(crop.resize((Wc, H)), dtype=np.float32)[:, :, ::-1] / 255.0
        a = (a - 0.5) / 0.5
        out = sess.run(None, {sess.get_inputs()[0].name: a.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]  # [T, C] (softmaxed)
        ids = out.argmax(1); nb = ids != 0; conf = float(out.max(1)[nb].mean()) if nb.any() else 0.0; T = len(ids); text, spans, prev = [], [], 0; step = (x1 - x0) / T
        for t, c in enumerate(ids):
            if c != 0 and c != prev:
                text.append(chars[c] if c < len(chars) else ""); spans.append([x0 + t * step, x0 + (t + 1) * step])
            elif c != 0 and c == prev and spans:
                spans[-1][1] = x0 + (t + 1) * step  # the same character over several timesteps widens its range
            prev = c
        # CTC fires near each character's centre, so the ranges above are narrow and leave gaps: widen every character to the
        # midpoints between its neighbours, and the first/last character to the line box edges (probe v1 under-covered short spans).
        if spans:
            mid = [(a + b) / 2 for a, b in spans]
            spans = [[x0 if i == 0 else (mid[i - 1] + mid[i]) / 2, x1 if i == len(mid) - 1 else (mid[i] + mid[i + 1]) / 2]
                     for i in range(len(mid))]
        return "".join(text), spans, conf

    def __call__(s, img):
        out = []
        for b in s.detect(img):
            t, sp = s.recognize(img, b)
            if t.strip(): out.append({"box": [round(v) for v in b], "text": t, "chars": [[round(a), round(c)] for a, c in sp]})
        return out


if __name__ == "__main__":
    import sys, json
    o = PPOCR(sys.argv[1], sys.argv[2].split(","))
    for line in o(Image.open(sys.argv[3])): print(json.dumps(line, ensure_ascii=False)[:200])
