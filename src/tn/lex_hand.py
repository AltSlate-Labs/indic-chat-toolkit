"""Hand-written TTS-normalization lexicon (pass 83), from standard written/news usage; NOT checked by native speakers.
Replaces the Ornith lexicon (jobs/tn-lex/out/<l>.json), which was unreliable (e.g. kn decimal point "ಕನ್ನಡ ಬಿಂದು", invented te ordinals).
Cardinals come from indic-numtowords; units and month names from CLDR (babel). This file holds only the glue words and patterns."""
import random, re
import indic_numtowords as n


def W(v, lang):
    """indic-numtowords cardinal, with fixes for bugs we found: Kannada joins tens + a consonant-initial unit with a stray virama
    (43 ನಲವತ್ತ್ಮೂರು -> ನಲವತ್ತಮೂರು) and spells 400 ನಾನ್ನೂರು (-> ನಾನೂರು)."""
    v = int(v)
    if lang == "kn": return kn_words(v)
    if lang == "ur" and v >= 100: return big_words(v, "ur")
    w = n.num2words(v, lang=lang)
    if lang == "ta": w = w.replace("ஓன்", "ஒன்").replace("‌", "")
    return " ".join(FIX.get(lang, {}).get(t, t) for t in w.split())


# v4 (pass 95): misspellings in indic-numtowords found by eye in the 0-99 tables (tn-v3 learned them: pa 75 -> ਪਚਹੱਤ).
FIX = {"pa": {"ਉਨਾਹਟ": "ਉਨਾਹਠ", "ਤਹੇਹਰ": "ਤਿਹੱਤਰ", "ਪਚਹੱਤ": "ਪੰਝੱਤਰ", "ਇਕਆਸੀ": "ਇਕਾਸੀ"},
       "ur": {"پاںچ": "پانچ", "تےيس": "تئیس", "چھببیس": "چھبیس", "ستااس": "ستائیس", "اٹھااس": "اٹھائیس", "انتيس": "انتیس", "اكتيس": "اکتیس"},
       "bn": {"অষ্টআশি": "আটাশি"}, "mr": {"हज़ार": "हजार"},
       "ta": {"தொண்ணூற்றிரெண்டு": "தொண்ணூற்று இரண்டு", "தொண்ணூரெட்டு": "தொண்ணூற்று எட்டு"}}
BIG = {"ur": ("سو", "ہزار", "لاکھ", "کروڑ")}


def big_words(v, lang):
    """v4b: Urdu from the library's 0-99 words, composed here: the library says 550 ساڑھے پانچ سو, 325 سوا تین سو, 150 ڈیڑھ سو
    (fractional forms used for some numbers only), while other languages and the model's own output use the plain reading."""
    h, th, la, cr = BIG[lang]; out = []
    for size, word in ((10 ** 7, cr), (10 ** 5, la), (1000, th), (100, h)):
        if v >= size:
            q, v = divmod(v, size); out.append(f"{W(q, lang)} {word}")
    if v: out.append(W(v, lang))
    return " ".join(out)


KN_H = ["", "ನೂರು", "ಇನ್ನೂರು", "ಮುನ್ನೂರು", "ನಾನೂರು", "ಐನೂರು", "ಆರುನೂರು", "ಏಳುನೂರು", "ಎಂಟುನೂರು", "ಒಂಬೈನೂರು"]


def kn_words(v):
    """v4: the library truncates Kannada hundreds + tens (110 ನೂ, 210 ಇನ್ನೂ: 2,798 collisions below 100k), reads 83 as 82 and ends
    round thousands in the genitive (1000 ಒಂದು ಸಾವಿರದ). Built here from its 0-99 words (virama fix: 43 ನಲವತ್ತ್ಮೂರು -> ನಲವತ್ತಮೂರು)."""
    if v == 83: return "ಎಂಬತ್ತಮೂರು"
    if v < 100: return re.sub(r"ತ್ತ್(?=[ಕ-ಹ])", "ತ್ತ", n.num2words(v, lang="kn"))
    if v < 1000:
        h, r = divmod(v, 100)
        return KN_H[h] if not r else KN_H[h][:-1] + " " + kn_words(r)  # ನೂರು -> ನೂರ ಹತ್ತು
    for size, nom, gen in ((10 ** 7, "ಕೋಟಿ", "ಕೋಟಿ"), (10 ** 5, "ಲಕ್ಷ", "ಲಕ್ಷದ"), (1000, "ಸಾವಿರ", "ಸಾವಿರದ")):
        if v >= size:
            q, r = divmod(v, size)
            return f"{kn_words(q)} {gen} {kn_words(r)}" if r else f"{kn_words(q)} {nom}"

LEX = {
 "hi": dict(point="दशमलव", rupees="रुपये", paise="पैसे", and_="और", range="{a} से {b}", time="{h} बजकर {m} मिनट", oclock="{h} बजे",
            am="सुबह", noon="दोपहर", eve="शाम", night="रात", score="{r} रन पर {w} विकेट", year="hundreds"),
 "mr": dict(point="पूर्णांक", rupees="रुपये", paise="पैसे", and_="आणि", range="{a} ते {b}", time="{h} वाजून {m} मिनिटे", oclock="{h} वाजता",
            am="सकाळी", noon="दुपारी", eve="संध्याकाळी", night="रात्री", score="{w} बाद {r} धावा", year="cardinal"),
 "bn": dict(point="দশমিক", rupees="টাকা", paise="পয়সা", and_="", range="{a} থেকে {b}", time="{h}টা {m} মিনিট", oclock="{h}টা",
            am="সকাল", noon="দুপুর", eve="সন্ধ্যা", night="রাত", score="{w} উইকেটে {r} রান", year="cardinal"),
 "gu": dict(point="દશાંશ", rupees="રૂપિયા", paise="પૈસા", and_="અને", range="{a} થી {b}", time="{h} વાગીને {m} મિનિટ", oclock="{h} વાગ્યે",
            am="સવારે", noon="બપોરે", eve="સાંજે", night="રાત્રે", score="{w} વિકેટે {r} રન", year="cardinal"),
 "pa": dict(point="ਦਸ਼ਮਲਵ", rupees="ਰੁਪਏ", paise="ਪੈਸੇ", and_="ਅਤੇ", range="{a} ਤੋਂ {b}", time="{h} ਵੱਜ ਕੇ {m} ਮਿੰਟ", oclock="{h} ਵਜੇ",
            am="ਸਵੇਰੇ", noon="ਦੁਪਹਿਰ", eve="ਸ਼ਾਮ", night="ਰਾਤ", score="{w} ਵਿਕਟਾਂ 'ਤੇ {r} ਦੌੜਾਂ", year="hundreds"),
 "ur": dict(point="اعشاریہ", rupees="روپے", paise="پیسے", and_="اور", range="{a} سے {b}", time="{h} بج کر {m} منٹ", oclock="{h} بجے",
            am="صبح", noon="دوپہر", eve="شام", night="رات", score="{r} رنز پر {w} وکٹ", year="hundreds"),
 "ta": dict(point="புள்ளி", rupees="ரூபாய்", paise="பைசா", and_="", range="{a} முதல் {b}", time="{h} மணி {m} நிமிடம்", oclock="{h} மணி",
            am="காலை", noon="மதியம்", eve="மாலை", night="இரவு", score="{w} விக்கெட் இழப்புக்கு {r} ரன்கள்", year="cardinal"),
 "te": dict(point="దశాంశం", rupees="రూపాయలు", paise="పైసలు", and_="", range="{a} నుండి {b}", time="{h} గంటల {m} నిమిషాలు", oclock="{h} గంటలు",
            am="ఉదయం", noon="మధ్యాహ్నం", eve="సాయంత్రం", night="రాత్రి", score="{w} వికెట్ల నష్టానికి {r} పరుగులు", year="cardinal"),
 "kn": dict(point="ದಶಮಾಂಶ", rupees="ರೂಪಾಯಿ", paise="ಪೈಸೆ", and_="", range="{a} ರಿಂದ {b}", time="{h} ಗಂಟೆ {m} ನಿಮಿಷ", oclock="{h} ಗಂಟೆ",
            am="ಬೆಳಿಗ್ಗೆ", noon="ಮಧ್ಯಾಹ್ನ", eve="ಸಂಜೆ", night="ರಾತ್ರಿ", score="{w} ವಿಕೆಟ್ ನಷ್ಟಕ್ಕೆ {r} ರನ್", year="cardinal"),
 "ml": dict(percent="{0} ശതമാനം", point="ദശാംശം", rupees="രൂപ", paise="പൈസ", and_="", range="{a} മുതൽ {b} വരെ", time="{h} മണി {m} മിനിറ്റ്", oclock="{h} മണി",
            am="രാവിലെ", noon="ഉച്ചയ്ക്ക്", eve="വൈകുന്നേരം", night="രാത്രി", score="{w} വിക്കറ്റ് നഷ്ടത്തിൽ {r} റൺസ്", year="cardinal"),
}
# Irregular ordinals: n -> list of (written suffix after the digits, spoken word). Regular ones are built in ordinal().
IRR = {
 "hi": {1: [("ला", "पहला"), ("ली", "पहली"), ("ले", "पहले")], 2: [("रा", "दूसरा"), ("री", "दूसरी"), ("रे", "दूसरे")],
        3: [("रा", "तीसरा"), ("री", "तीसरी"), ("रे", "तीसरे")], 4: [("था", "चौथा"), ("थी", "चौथी"), ("थे", "चौथे")],
        6: [("ठा", "छठा"), ("ठी", "छठी"), ("ठे", "छठे")]},
 "mr": {1: [("ला", "पहिला"), ("ली", "पहिली"), ("ले", "पहिले")], 2: [("रा", "दुसरा"), ("री", "दुसरी"), ("रे", "दुसरे")],
        3: [("रा", "तिसरा"), ("री", "तिसरी"), ("रे", "तिसरे")], 4: [("था", "चौथा"), ("थी", "चौथी"), ("थे", "चौथे")]},
 "gu": {1: [("લો", "પહેલો"), ("લી", "પહેલી")], 2: [("જો", "બીજો"), ("જી", "બીજી")], 3: [("જો", "ત્રીજો"), ("જી", "ત્રીજી")],
        4: [("થો", "ચોથો"), ("થી", "ચોથી")], 6: [("ઠ્ઠો", "છઠ્ઠો"), ("ઠ્ઠી", "છઠ્ઠી")]},
 "pa": {1: [("ਲਾ", "ਪਹਿਲਾ"), ("ਲੀ", "ਪਹਿਲੀ")], 2: [("ਜਾ", "ਦੂਜਾ"), ("ਜੀ", "ਦੂਜੀ")], 3: [("ਜਾ", "ਤੀਜਾ"), ("ਜੀ", "ਤੀਜੀ")],
        4: [("ਥਾ", "ਚੌਥਾ"), ("ਥੀ", "ਚੌਥੀ")]},
 "ur": {1: [("st", "پہلا"), ("st", "پہلی")], 2: [("nd", "دوسرا"), ("nd", "دوسری")], 3: [("rd", "تیسرا"), ("rd", "تیسری")],
        4: [("th", "چوتھا"), ("th", "چوتھی")], 6: [("th", "چھٹا"), ("th", "چھٹی")]},
 "bn": {1: [("ম", "প্রথম")], 2: [("য়", "দ্বিতীয়")], 3: [("য়", "তৃতীয়")], 4: [("র্থ", "চতুর্থ")], 5: [("ম", "পঞ্চম")], 6: [("ষ্ঠ", "ষষ্ঠ")],
        7: [("ম", "সপ্তম")], 8: [("ম", "অষ্টম")], 9: [("ম", "নবম")], 10: [("ম", "দশম")]},
 "ta": {1: [("வது", "முதலாவது"), ("ஆம்", "முதலாம்")]},
 "te": {1: [("వ", "మొదటి")]},
 "kn": {1: [("ನೇ", "ಮೊದಲನೇ")]},
}
REG = {  # regular ordinals: list of (written suffix, function cardinal_words -> spoken), None if the form doesn't apply
 "hi": [("वाँ", lambda w: w + "वाँ"), ("वीं", lambda w: w + "वीं"), ("वें", lambda w: w + "वें")],
 "mr": [("वा", lambda w: w + "वा"), ("वी", lambda w: w + "वी"), ("वे", lambda w: w + "वे")],
 "gu": [("મો", lambda w: w + "મો"), ("મી", lambda w: w + "મી")],
 "pa": [("ਵਾਂ", lambda w: w + "ਵਾਂ"), ("ਵੀਂ", lambda w: w + "ਵੀਂ")],
 "ur": [("واں", lambda w: w + "واں"), ("ویں", lambda w: w + "ویں")],
 "bn": [("তম", lambda w: w + "তম")],
 "ta": [("வது", lambda w: w[:-1] + "ாவது" if w.endswith("ு") else None), ("ஆம்", lambda w: w[:-1] + "ாம்" if w.endswith("ு") else None)],
 "te": [("వ", lambda w: w[:-1] + "వ" if w[-1] in "ుి" else None)],
 "kn": [("ನೇ", lambda w: w[:-1] + "ನೇ" if w.endswith("ು") else None)],
 "ml": [("-ാം", lambda w: w[:-1] + "ാം" if w.endswith("്") else None), ("ാം", lambda w: w[:-1] + "ാം" if w.endswith("്") else None)],
}
EN = lambda v: "th" if 10 <= v % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")


def ordinal(lang, v, r: random.Random):
    """-> (written, spoken) or None. Written = digits + native suffix (or English suffix), spoken agrees with the suffix."""
    if v in IRR.get(lang, {}):
        k = r.randrange(len(IRR[lang][v])); suf, sp = IRR[lang][v][k]
        # v6: real text also writes irregular ordinals with the regular suffix (1ਵੀਂ, 1ویں, 3वीं) or English (1st, 3rd); tn-v5 garbled
        # these (ਪਹਿਵੀਂ, پہلیں). REG and IRR lists share gender order (m, f, oblique), so the same index keeps the gender.
        x = r.random()
        if x < 0.25 and len(REG.get(lang, [])) == len(IRR[lang][v]) and lang in ("hi", "mr", "gu", "pa", "ur"): suf = REG[lang][k][0]
        elif x < 0.4 and lang not in ("ur",): suf = EN(v)
        return f"{v}{suf}", sp
    if lang == "bn" and v <= 10: return None
    suf, f = r.choice(REG[lang]); sp = f(W(v, lang))
    if not sp: return None
    if r.random() < 0.15 and lang not in ("ur",): suf = EN(v)  # "5th" in native text, read as the default native ordinal
    return f"{v}{suf}", sp
