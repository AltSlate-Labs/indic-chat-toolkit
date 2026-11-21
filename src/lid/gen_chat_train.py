"""indic-lid v3 TRAINING chat data from Ornith-1.5-35B-A3B (MIT; NOT the test teacher): native-script chat message first,
then its romanization. Kept if GlotLID agrees on the native text's language and the roman line is Latin-only.
Topics are disjoint from the test-B topic list. python gen_chat_train.py URL MODEL OUT.jsonl [N_PER_LANG=2000]  (envs/lid)
v4: GEN_LANGS=ur,pa,... picks languages (incl. the out-of-scope OOS ones, trained as 'other'); SEED_OFF shifts seeds."""
import json, os, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import fasttext
from make_chat_test import LANGS, ISO3, LATIN, TOPICS as TEST_TOPICS

TOPICS = ["paying the electricity bill", "a neighbour's dog", "planning a picnic", "a new phone", "an auto-rickshaw fare",
          "a cooking recipe", "a temple visit", "school admission", "a power cut", "a birthday party", "online shopping returns",
          "a job interview", "a sick grandparent", "a football match", "a bank queue", "a holiday trip", "exam preparation",
          "a water tanker", "a wedding outfit", "a missed call", "a vegetable price", "a TV serial", "office gossip",
          "a new baby in the family", "a car service", "a monsoon flood", "a house rent", "a college fest", "a parcel delivery",
          "a song recommendation", "an old friend's call", "a hospital visit", "a gas cylinder booking", "a bike puncture",
          "a kite festival", "a lost pet", "morning walk", "a cricket score", "a tuition class", "a broken laptop"]
assert not set(TOPICS) & set(TEST_TOPICS)
# v4: Indic languages indic-lid does NOT label; trained as "other" so they stop landing on a neighbour. GlotLID iso3 for the check.
OOS = {"as": ("Assamese", "Bengali", "asm"), "or": ("Odia", "Odia", "ory"), "ne": ("Nepali", "Devanagari", "npi"),
       "mai": ("Maithili", "Devanagari", "mai"), "sa": ("Sanskrit", "Devanagari", "san"), "kok": ("Konkani", "Devanagari", "gom"),
       "ks": ("Kashmiri", "Perso-Arabic", "kas"), "mni": ("Manipuri", "Bengali", "mni"), "brx": ("Bodo", "Devanagari", "brx")}
ALL = {**LANGS, **{k: v[:2] for k, v in OOS.items()}}
WANT = {**{iso: l for iso, l in ISO3.items()}, **{v[2]: k for k, v in OOS.items()}}
STYLE = ["very short, 2 to 5 words", "short, 4 to 9 words", "one sentence of 8 to 15 words", "two short sentences"]


def ask(url, model, lang, seed):
    r = random.Random(seed); name, script = ALL[lang]
    p = (f"Write one casual chat message ({r.choice(STYLE)}) in {name}, in {script} script, about: {r.choice(TOPICS)}. "
         f"Sound like a real person texting a {r.choice(['friend', 'sibling', 'parent', 'colleague', 'shopkeeper', 'cousin'])}. "
         f"Then, on a second line starting with 'ROMAN:', write the same message the way {name} speakers type it on phones "
         f"in the Latin alphabet: informal spelling, no diacritics, as in WhatsApp. Output only the two lines.")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}], "temperature": 1.0, "max_tokens": 160,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    try:
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"] or ""
    except Exception:
        return None, None
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
    m = re.search(r"^(.*?)\n+\s*ROMAN:\s*(.+)$", out, flags=re.S)
    return (m.group(1).strip().strip('"'), m.group(2).strip().strip('"')) if m else (None, None)


def main():
    url, model, out = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    glot = fasttext.load_model(os.path.expanduser("~/Downloads/model/glotlid/model.bin"))
    sel = os.environ.get("GEN_LANGS", ",".join(LANGS)).split(","); off = int(os.environ.get("SEED_OFF", 0))
    jobs = [(l, 1000003 * k + i + off) for k, l in enumerate(sel) for i in range(n)]
    kept, rej = {}, {}
    with open(out, "w", encoding="utf-8") as f, ThreadPoolExecutor(24) as pool:
        for (l, _), (nat, rom) in zip(jobs, pool.map(lambda j: ask(url, model, *j), jobs)):
            ok = bool(nat and rom) and bool(LATIN.match(rom)) and not LATIN.match(nat)
            if ok:
                lab = glot.predict(" ".join(nat.split()))[0][0].replace("__label__", "")
                ok = WANT.get(lab.split("_")[0]) == l
            if not ok:
                rej[l] = rej.get(l, 0) + 1; continue
            kept[l] = kept.get(l, 0) + 1
            f.write(json.dumps({"native": " ".join(nat.split()), "roman": " ".join(rom.split()), "lang": l}, ensure_ascii=False) + "\n")
            if sum(kept.values()) % 1000 == 0: print("kept", sum(kept.values()), flush=True)
    print("kept", kept, "rejected", rej)


if __name__ == "__main__":
    main()
