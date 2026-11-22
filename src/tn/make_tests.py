"""TTS text-normalization tests (inputs are written text; a system must output the same text with every number/symbol spoken in words).
 A: Dakshina native wiki VALID-split sentences that contain digits (60/lang). Never use the valid split for training.
 B: messages from the TEST teacher (Qwen3.8-27B) that contain a given kind of number, with its spoken form (gold is the teacher's, noisy).
python make_tests.py OUT_DIR [N=60] [URL]"""
import gzip, json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
D = os.path.expanduser("~/Downloads/data/dakshina/dakshina_dataset_v1.0/")
LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
         "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
DIG = re.compile(r"[0-9०-९০-৯੦-੯૦-૯௦-௯౦-౯೦-೯൦-൯۰-۹٠-٩]")
KINDS = ["a price or amount in rupees written with the ₹ sign and Indian digit grouping (e.g. ₹1,25,000)",
         "a clock time such as 4:30 or 10:15 am", "a date written as DD/MM/YYYY or like '12 March 2024'", "a percentage with a decimal (e.g. 12.5%)",
         "a 10-digit mobile number", "a quantity with a unit (km, kg, ml, GB, °C)", "an ordinal such as 1st, 2nd or 3rd written in digits",
         "a year and an age", "a score in a cricket match (e.g. 245/6 in 50 overs)", "a flight or train number and a platform number",
         "a decimal number such as 3.75", "a range such as 10-15 or 2020-21"]


def ask(url, lang, r):
    k = r.choice(KINDS)
    p = (f"Write one realistic message or sentence (8 to 25 words) in {LANGS[lang]}, in its native script, that contains {k}, "
         f"written with digits and symbols as people normally write them. Then, on a second line starting with 'SPOKEN:', write the same "
         f"text exactly as a {LANGS[lang]} newsreader would read it aloud: every number, symbol, unit and abbreviation written out in "
         f"{LANGS[lang]} words in the native script, everything else unchanged. Output only the two lines.")
    body = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0.9, "max_tokens": 400,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        out = json.load(urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=600))
        out = re.sub(r"<think>.*?</think>", "", out["choices"][0]["message"]["content"], flags=re.S).strip()
    except Exception: return None
    m = re.search(r"^(.*?)\n+\s*SPOKEN:\s*(.+)$", out, flags=re.S)
    if not m: return None
    w, s = " ".join(m.group(1).strip().strip('"').split()), " ".join(m.group(2).strip().strip('"').split())
    return {"lang": lang, "kind": k.split(" (")[0], "text": w, "spoken": s} if DIG.search(w) and not DIG.search(s) else None


def main():
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080/v1/chat/completions"
    os.makedirs(out, exist_ok=True); r = random.Random(31)
    with open(f"{out}/testA.jsonl", "w", encoding="utf-8") as f:
        for l in LANGS:
            with gzip.open(f"{D}{l}/native_script_wikipedia/{l}.wiki-filt.valid.text.shuf.txt.gz", "rt", encoding="utf-8") as g:
                xs = [" ".join(x.split()) for x in g if DIG.search(x) and 5 <= len(x.split()) <= 30]
            for t in r.sample(xs, min(n, len(xs))): f.write(json.dumps({"lang": l, "text": t}, ensure_ascii=False) + "\n")
            print(l, "A pool", len(xs), flush=True)
    jobs = [(l, random.Random(r.random())) for l in LANGS for _ in range(int(n * 1.4))]
    with ThreadPoolExecutor(8) as pool: res = list(pool.map(lambda j: ask(url, *j), jobs))
    kept = {}
    with open(f"{out}/testB.jsonl", "w", encoding="utf-8") as f:
        for x in res:
            if x and kept.get(x["lang"], 0) < n:
                kept[x["lang"]] = kept.get(x["lang"], 0) + 1; f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("B kept", kept)


if __name__ == "__main__":
    main()
