"""Score fixed (query, passage) pairs; run under two transformers versions and compare. python check_ce.py DIR OUT.json"""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
d, o = sys.argv[1:3]
P = [("भारत में कुल कितने राज्य है?", "भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं।"), ("bharat mein kitne rajya hain", "भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं।"),
     ("bharat mein kitne rajya hain", "क्रिकेट एक लोकप्रिय खेल है।"), ("রবীন্দ্রনাথ ঠাকুর কবে জন্মগ্রহণ করেন?", "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালের ৭ মে কলকাতায় জন্মগ্রহণ করেন।"),
     ("rabindranath thakur kobe jonmogrohon koren", "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালের ৭ মে কলকাতায় জন্মগ্রহণ করেন।"), ("who wrote the national anthem of india", "Jana Gana Mana was written by Rabindranath Tagore.")]
tok = AutoTokenizer.from_pretrained(d); m = AutoModelForSequenceClassification.from_pretrained(d).eval()
with torch.no_grad():
    b = tok([q for q, _ in P], [p for _, p in P], padding=True, truncation="only_second", max_length=320, return_tensors="pt")
    s = m(**b).logits[:, 0].tolist()
json.dump(s, open(o, "w")); print([round(x, 4) for x in s])
