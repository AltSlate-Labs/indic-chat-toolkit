"""Per-language TTS-normalization lexicon: CLDR parts (babel) + words from Ornith (the TRAINING teacher), asked 3 times at low temperature;
fields where the 3 answers disagree are flagged for review. python make_lexicon.py OUT_DIR URL   (envs/tn)"""
import json, os, re, sys, urllib.request
from babel import Locale
LANGS = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"}
UNITS = {"percent": "concentr-percent", "km": "length-kilometer", "m": "length-meter", "cm": "length-centimeter", "kg": "mass-kilogram",
         "g": "mass-gram", "l": "volume-liter", "ml": "volume-milliliter", "C": "temperature-celsius", "GB": "digital-gigabyte",
         "MB": "digital-megabyte", "hour": "duration-hour", "minute": "duration-minute"}
FIELDS = {
    "point": "the word read for the decimal point (as in 12.5)",
    "rupees": "the word for rupees after an amount (as in 500 rupees)",
    "paise": "the word for paise after an amount (as in 50 paise)",
    "and": "the word for 'and' used between rupees and paise",
    "range": "a pattern for reading a range like 10-15, using {a} and {b} for the two numbers",
    "time": "a pattern for reading a clock time like 4:30, using {h} and {m} for the hour and minute words",
    "time_oclock": "a pattern for reading a clock time like 4:00, using {h}",
    "am": "the word read before the hour for a morning time (AM)", "pm_afternoon": "the word for an afternoon time (PM, 12-4)",
    "pm_evening": "the word for an evening time (PM, 5-7)", "pm_night": "the word for a night time (PM, 8-11)",
    "score": "a pattern for reading a cricket score like 245/6 (runs/wickets), using {r} and {w}",
    "ordinals": "a JSON list of the ordinal words for 1st to 10th (first, second, ... tenth), in the form used before a noun",
    "ordinal_rule": "a pattern to form the ordinal of any number from its cardinal word, using {n} (e.g. for 11th, 25th)",
    "year_style": "how a year like 1984 is usually read aloud in news: answer 'hundreds' if it is read as nineteen hundred eighty-four, or 'cardinal' if read as one thousand nine hundred eighty-four",
    "zero": "the word for the digit 0 when reading out a phone number",
}


def ask(url, lang):
    L = LANGS[lang]
    spec = "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
    p = (f"You are helping build a text-to-speech front end for {L}. Return one JSON object with these keys, with all values written in "
         f"{L} in its native script (except year_style), exactly as a {L} newsreader would say them:\n{spec}\nOutput only the JSON.")
    body = json.dumps({"model": "ornith", "messages": [{"role": "user", "content": p}], "temperature": 0.3, "max_tokens": 1500,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    o = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
    t = o["choices"][0]["message"]["content"]; m = re.search(r"\{.*\}", t, re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None


out, url = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for lang in LANGS:
    loc = Locale.parse(lang); up = loc._data["unit_patterns"]
    def unit(k):
        try: d = up[k]["long"]; return d.get("other") or d.get("one")
        except Exception: return None
    lex = {"months": list(loc.months["format"]["wide"][i] for i in range(1, 13)), "inr_name": loc.currencies.get("INR"),
           "units": {k: unit(v) for k, v in UNITS.items()}}
    answers = [a for a in (ask(url, lang) for _ in range(3)) if a]
    flags = []
    for k in FIELDS:
        vals = [json.dumps(a.get(k), ensure_ascii=False) for a in answers if a.get(k) is not None]
        best = max(set(vals), key=vals.count) if vals else "null"
        lex[k] = json.loads(best)
        if len(set(vals)) > 1: flags.append(k); lex[k + "_alternatives"] = [json.loads(v) for v in set(vals)]
    lex["_flags"] = flags; lex["_answers"] = len(answers)
    json.dump(lex, open(f"{out}/{lang}.json", "w"), ensure_ascii=False, indent=1)
    print(lang, "answers", len(answers), "disagree on", flags, flush=True)
