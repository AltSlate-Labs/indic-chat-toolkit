"""LID test B: short chat messages from the TEST teacher (Qwen3.8-27B): native script first, then its own romanization.
Pairs are kept only if GlotLID agrees on the NATIVE text's language and the roman line is Latin-only.
python make_chat_test.py OUT.jsonl [N_PER_LANG=60] [URL]   (run in envs/lid; needs fasttext + GlotLID)"""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import fasttext

LANGS = {"hi": ("Hindi", "Devanagari"), "bn": ("Bengali", "Bengali"), "ta": ("Tamil", "Tamil"), "te": ("Telugu", "Telugu"),
         "mr": ("Marathi", "Devanagari"), "gu": ("Gujarati", "Gujarati"), "kn": ("Kannada", "Kannada"),
         "ml": ("Malayalam", "Malayalam"), "pa": ("Punjabi", "Gurmukhi"), "ur": ("Urdu", "Perso-Arabic"),
         "sd": ("Sindhi", "Perso-Arabic"), "si": ("Sinhala", "Sinhala")}
ISO3 = {"hin": "hi", "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn", "mal": "ml",
        "pan": "pa", "urd": "ur", "snd": "sd", "sin": "si"}
TOPICS = ["weekend plans with a friend", "a late food delivery", "a family group update", "asking a shop about a price",
          "a cricket match", "asking for directions", "a work deadline", "festival wishes", "a one-line movie review",
          "booking a doctor's appointment", "a late train", "exam results", "rain and traffic", "ordering food",
          "a phone recharge problem", "teasing a friend", "homework", "a wedding invitation", "a bus that didn't come",
          "a gym plan", "a broken fridge", "a cousin's new job", "a lost wallet", "cooking dinner"]
LATIN = re.compile(r"^[\x00-\x7F‘’“”…\s]+$")


def ask(url, lang, r):
    name, script = LANGS[lang]; n = r.choice(["3 to 6 words", "5 to 10 words", "8 to 16 words"])
    p = (f"Write one casual chat message of {n} in {name}, in {script} script, about: {r.choice(TOPICS)}. "
         f"Then, on a second line starting with 'ROMAN:', write exactly the same message the way {name} speakers type "
         f"it on phones in the Latin alphabet (informal spelling, no diacritics). Output only the two lines.")
    body = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0.9, "max_tokens": 200,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    m = re.search(r"^(.*?)\n+\s*ROMAN:\s*(.+)$", out, flags=re.S)
    return (m.group(1).strip().strip('"'), m.group(2).strip().strip('"')) if m else (None, None)


def main():
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080/v1/chat/completions"
    glot = fasttext.load_model(os.path.expanduser("~/Downloads/model/glotlid/model.bin"))
    r = random.Random(23); jobs = [(l, random.Random(r.random())) for l in LANGS for _ in range(n)]
    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(lambda j: ask(url, *j), jobs))
    kept = {}; rej = {}
    with open(out, "w", encoding="utf-8") as f:
        for (l, _), (nat, rom) in zip(jobs, res):
            ok = bool(nat and rom) and LATIN.match(rom) and not LATIN.match(nat)
            if ok:
                lab = glot.predict(" ".join(nat.split()))[0][0].replace("__label__", "")
                ok = ISO3.get(lab.split("_")[0]) == l
            if not ok:
                rej[l] = rej.get(l, 0) + 1; continue
            kept[l] = kept.get(l, 0) + 1
            for t, ro in ((nat, False), (rom, True)):
                f.write(json.dumps({"text": " ".join(t.split()), "lang": l, "roman": ro}, ensure_ascii=False) + "\n")
    print("kept", kept, "rejected", rej)


if __name__ == "__main__":
    main()
