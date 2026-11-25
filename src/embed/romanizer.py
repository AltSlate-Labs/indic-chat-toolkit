"""Sentence romanizer used to build training data and the synthetic romanized eval: a random attested human romanization per word from
aksharantar TRAIN (CC0 / CC-BY 4.0) or the Dakshina romanization lexicon TRAIN (CC BY-SA 4.0); other words by rule_roman.py
(long vowels shortened half the time, as chat often does). Danda → "."."""
import collections, json, os, random, re, zipfile
from rule_roman import romanize_word
D = os.path.expanduser("~/Downloads/data")
DK = f"{D}/dakshina/dakshina_dataset_v1.0"
CODE = {"hi": "hin", "bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal", "pa": "pan", "ur": "urd"}
PUNCT = re.compile(r"^([\"'“‘(\[]*)(.*?)([\"'”’)\].,!?;:।॥۔،]*)$")


class Romanizer:
    def __init__(s, seed=0):
        s.r = random.Random(seed); s.d = {}

    def dict(s, l):
        if l not in s.d:
            d = collections.defaultdict(set)
            with zipfile.ZipFile(f"{D}/aksharantar/{CODE[l]}.zip") as z:
                with z.open(f"{CODE[l]}_train.json") as f:
                    for line in f:
                        x = json.loads(line); d[x["native word"]].add(x["english word"].lower())
            for line in open(f"{DK}/{l}/lexicons/{l}.translit.sampled.train.tsv", encoding="utf-8"):
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2: d[p[0]].add(p[1].lower())
            s.d[l] = {k: sorted(v) for k, v in d.items()}
        return s.d[l]

    def word(s, core, l):
        v = s.dict(l).get(core)
        if v: return s.r.choice(v), True
        w = romanize_word(core, l)
        if s.r.random() < 0.5: w = w.replace("aa", "a").replace("ee", "i").replace("oo", "u")
        return w, False

    def sentence(s, t, l, frac=1.0):
        out = []
        for w in t.split():
            a, core, b = PUNCT.match(w).groups()
            if core and s.r.random() < frac: w = a + s.word(core, l)[0] + b
            out.append(w)
        return " ".join(out).replace("।", ".").replace("॥", ".").replace("۔", ".")
