"""Carrier sentences for TTS-normalization training, from Ornith (TRAINING teacher): native-script sentences with one typed placeholder.
Values are filled later by fill_data.py (written + spoken forms), so targets are exact. python gen_carriers.py URL OUT.jsonl [CALLS=30]"""
import json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
KINDS = {"AMOUNT": "an amount of money in rupees", "TIME": "a clock time", "DATE": "a full date (day, month and year)", "YEAR": "a year",
         "PERCENT": "a percentage", "DECIMAL": "a decimal number such as a rating or measurement", "PHONE": "a mobile phone number",
         "QUANTITY": "a quantity with its unit (kilometres, kilograms, litres, degrees Celsius, gigabytes, hours or minutes)",
         "ORDINAL": "an ordinal position such as first, 3rd or 12th, used before a noun", "RANGE": "a range of two numbers such as 10 to 15",
         "SCORE": "a cricket score given as runs for wickets", "NUMBER": "a count of people or things"}
TOPICS = ["shopping", "a news report", "railways and travel", "weather", "a school notice", "a bank or payment message", "sports",
          "health and a doctor visit", "a government scheme", "a wedding", "a recipe", "a phone bill", "a festival", "a delivery update",
          "an office email", "real estate", "farming", "a film release", "an exam result", "history", "science", "a hotel booking"]


def ask(url, lang, kind, seed):
    r = random.Random(seed); L = LANGS[lang]
    p = (f"Write 10 different sentences in {L}, in its native script, about {r.choice(TOPICS)} (vary style: news, chat, notices, "
         f"announcements). Each sentence must be 6 to 20 words long and contain the placeholder ⟦{kind}⟧ exactly once, standing for "
         f"{KINDS[kind]}. Do not write any digits. If a case ending or postposition follows the value, write it as a separate word after "
         f"the placeholder. Output one sentence per line, nothing else.")
    body = json.dumps({"model": "ornith", "messages": [{"role": "user", "content": p}], "temperature": 1.0, "max_tokens": 1200,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        o = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
        lines = o["choices"][0]["message"]["content"].splitlines()
    except Exception: return []
    out = []
    for s in lines:
        s = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", s).strip()
        if s.count(f"⟦{kind}⟧") == 1 and s.count("⟦") == 1 and not re.search(r"[0-9]", s) and 4 <= len(s.split()) <= 30:
            out.append({"lang": lang, "kind": kind, "carrier": " ".join(s.split())})
    return out


url, outf = sys.argv[1], sys.argv[2]; calls = int(sys.argv[3]) if len(sys.argv) > 3 else 30
jobs = [(l, k, i) for l in LANGS for k in KINDS for i in range(calls)]
with ThreadPoolExecutor(24) as pool: res = list(pool.map(lambda j: ask(url, *j), jobs))
seen = set(); n = 0
with open(outf, "w", encoding="utf-8") as f:
    for rows in res:
        for x in rows:
            if x["carrier"] in seen: continue
            seen.add(x["carrier"]); f.write(json.dumps(x, ensure_ascii=False) + "\n"); n += 1
print("carriers", n)
