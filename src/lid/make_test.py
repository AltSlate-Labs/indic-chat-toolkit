"""LID gap check: short chat messages per language and script, written by the TEST teacher (Qwen3.8-27B).
python make_test.py OUT.jsonl [N_PER_CELL=50] [URL]"""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
         "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
SCRIPT = {"hi": "Devanagari", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Devanagari", "gu": "Gujarati",
          "kn": "Kannada", "ml": "Malayalam", "pa": "Gurmukhi", "ur": "Urdu (Perso-Arabic)"}
TOPICS = ["asking a friend about weekend plans", "complaining about a late food delivery", "a family WhatsApp group update",
          "asking a shopkeeper about a price", "a cricket match reaction", "asking for directions", "a work deadline",
          "wishing someone on a festival", "a movie review in one line", "asking a doctor's clinic for an appointment",
          "a train running late", "a college exam result", "rain and traffic", "ordering at a restaurant",
          "a phone recharge problem", "a joke between friends", "asking a teacher about homework", "a wedding invitation"]


def prompt(r, lang, roman):
    t = r.choice(TOPICS); n = r.choice(["one short sentence", "two short sentences", "a short message of 8 to 20 words"])
    if lang == "en":
        return f"Write {n} that an Indian person might type in English in a chat: {t}. Plain English, no Hindi words."
    if roman:
        return (f"Write {n} in {LANGS[lang]} as people type it on phones in India, using only the Latin (English) "
                f"alphabet, no native script: {t}. Use natural romanized {LANGS[lang]} spelling; at most one or two English words.")
    return f"Write {n} in {LANGS[lang]}, in {SCRIPT[lang]} script, as a natural chat message: {t}."


def ask(url, p):
    body = json.dumps({"messages": [{"role": "user", "content": p + " Output only the message."}],
                       "temperature": 0.9, "max_tokens": 120, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"')


def main():
    out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080/v1/chat/completions"
    r = random.Random(7); cells = [("en", False)] + [(l, ro) for l in LANGS for ro in (False, True)]
    jobs = [(l, ro, prompt(r, l, ro)) for l, ro in cells for _ in range(n)]
    with ThreadPoolExecutor(8) as pool:
        texts = list(pool.map(lambda j: ask(url, j[2]), jobs))
    latin = re.compile(r"^[\x00-\x7F‘’“”…à-ÿĀ-ſ\s\U0001F000-\U0001FFFF☀-➿]+$")
    kept = 0
    with open(out, "w", encoding="utf-8") as f:
        for (l, ro, _), t in zip(jobs, texts):
            t = " ".join(t.split())
            if not t or (ro or l == "en") != bool(latin.match(t)):
                continue  # romanized must be Latin-only; native must contain non-Latin
            f.write(json.dumps({"text": t, "lang": l, "roman": ro}, ensure_ascii=False) + "\n"); kept += 1
    print("kept", kept, "of", len(jobs))


if __name__ == "__main__":
    main()
