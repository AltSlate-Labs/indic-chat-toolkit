"""Render PII test messages (char-span gold) as chat-screenshot images with exact pixel boxes for every PII span.
python render_shots.py TEST.jsonl OUT_DIR N [SEED]   (envs/vision: Pillow; fonts in ~/Downloads/data/fonts/noto, OFL)
Each image: 1-4 messages as chat bubbles (left/right), light or dark theme, random font size; OUT_DIR/NNNN.png + shots.jsonl rows
{"img", "text" (messages joined by \\n), "boxes": [{"label", "xyxy", "msg", "span"}]}. A span that wraps gets one box per line.
The source messages are the held-out PII test sets (synthetic people; never used for training the PII models)."""
import json, os, random, sys
from PIL import Image, ImageDraw, ImageFont
F = os.path.expanduser("~/Downloads/data/fonts/noto")
FONTS = {"latin": f"{F}/NotoSans-Regular.ttf", "deva": f"{F}/NotoSansDevanagari-Regular.ttf", "beng": f"{F}/NotoSansBengali-Regular.ttf",
         "gujr": f"{F}/NotoSansGujarati-Regular.ttf", "taml": f"{F}/NotoSansTamil-Regular.ttf", "telu": f"{F}/NotoSansTelugu-Regular.ttf",
         "knda": f"{F}/NotoSansKannada-Regular.ttf"}
BLOCKS = [(0x0900, 0x097F, "deva"), (0x0980, 0x09FF, "beng"), (0x0A80, 0x0AFF, "gujr"), (0x0B80, 0x0BFF, "taml"),
          (0x0C00, 0x0C7F, "telu"), (0x0C80, 0x0CFF, "knda")]
THEMES = [dict(bg=(236, 229, 221), me=(220, 248, 198), them=(255, 255, 255), fg=(20, 20, 20)),   # light chat
          dict(bg=(11, 20, 26), me=(0, 92, 75), them=(32, 44, 51), fg=(233, 237, 239)),         # dark chat
          dict(bg=(255, 255, 255), me=(0, 132, 255), them=(228, 230, 235), fg=(15, 15, 15))]   # plain


class Fonts:
    """Per-run font fallback: Devanagari runs use Noto Sans Devanagari, everything else Noto Sans (the Devanagari font has no
    Latin glyphs, so 'GST' inside a Hindi message rendered as boxes)."""
    def __init__(s, size):
        s.f = {k: ImageFont.truetype(v, size, layout_engine=ImageFont.Layout.RAQM) for k, v in FONTS.items()}

    @staticmethod
    def runs(text):
        out = []
        for c in text:
            k = next((f for lo, hi, f in BLOCKS if lo <= ord(c) <= hi), None) or ("prev" if c in " .,-/:()'\"" and out else "latin")
            k = out[-1][0] if k == "prev" else k
            if out and out[-1][0] == k: out[-1][1] += c
            else: out.append([k, c])
        return out

    def length(s, d, text):
        return sum(d.textlength(t, font=s.f[k]) for k, t in s.runs(text))

    def draw(s, d, xy, text, fill):
        x, y = xy
        for k, t in s.runs(text):
            d.text((x, y), t, font=s.f[k], fill=fill); x += d.textlength(t, font=s.f[k])


def font_for(text, size):
    return Fonts(size)


def wrap(draw, text, font, width):
    """-> list of (start, end) char ranges per line, breaking at spaces."""
    lines, start, last_space = [], 0, None
    i = 0
    while i < len(text):
        if text[i] == " ": last_space = i
        if font.length(draw, text[start:i + 1]) > width and i > start:
            cut = last_space if last_space and last_space > start else i
            lines.append((start, cut)); start = cut + 1 if cut == last_space else cut; last_space = None
        i += 1
    lines.append((start, len(text)))
    return lines


def render(msgs, r, path):
    W = 720; th = r.choice(THEMES); size = r.choice([24, 26, 28, 30, 32]); pad, gap, maxw = 14, 14, int(W * 0.72)
    tmp = Image.new("RGB", (W, 10)); d = ImageDraw.Draw(tmp); blocks = []; y = 20
    for k, m in enumerate(msgs):
        font = font_for(m["text"], size); lines = wrap(d, m["text"], font, maxw - 2 * pad)
        lh = int(size * 1.45); bw = max(font.length(d, m["text"][a:b]) for a, b in lines) + 2 * pad
        side = r.random() < 0.5; x0 = W - 16 - bw if side else 16
        blocks.append((m, font, lines, lh, x0, y, bw, side)); y += len(lines) * lh + 2 * pad + gap
    img = Image.new("RGB", (W, y + 10), th["bg"]); d = ImageDraw.Draw(img); boxes = []
    for k, (m, font, lines, lh, x0, y0, bw, side) in enumerate(blocks):
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + len(lines) * lh + 2 * pad], 16, fill=th["me" if side else "them"])
        fg = (255, 255, 255) if th["me"] == (0, 132, 255) and side else th["fg"]
        for j, (a, b) in enumerate(lines):
            ty = y0 + pad + j * lh; font.draw(d, (x0 + pad, ty), m["text"][a:b], fg)
            for s, e, lab in m["spans"]:
                s2, e2 = max(s, a), min(e, b)
                if s2 >= e2: continue
                xa = x0 + pad + font.length(d, m["text"][a:s2]); xb = x0 + pad + font.length(d, m["text"][a:e2])
                boxes.append({"label": lab, "xyxy": [round(xa), ty, round(xb), ty + lh], "msg": k, "span": [s, e]})
    img.save(path)
    return boxes


if __name__ == "__main__":
    test, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); r = random.Random(sys.argv[4] if len(sys.argv) > 4 else 0)
    os.makedirs(out, exist_ok=True); rows = [json.loads(l) for l in open(test, encoding="utf-8")]; r.shuffle(rows); k = 0
    with open(f"{out}/shots.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            msgs = rows[k:k + r.choice([1, 2, 2, 3, 4])]; k += len(msgs)
            msgs = [dict(m, text=m["text"].replace("\n", " ")) for m in msgs]  # same length, so spans stay valid
            if not msgs: break
            p = f"{out}/{i:04d}.png"; boxes = render(msgs, r, p)
            f.write(json.dumps({"img": os.path.basename(p), "text": "\n".join(m["text"] for m in msgs),
                                "msgs": [{"text": m["text"], "spans": m["spans"], "lang": m.get("lang")} for m in msgs], "boxes": boxes},
                               ensure_ascii=False) + "\n")
    print("images", i + 1, "messages", k)
