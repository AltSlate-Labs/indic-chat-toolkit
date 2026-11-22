---
title: Screenshot PII Redactor
emoji: 🕶️
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
license: apache-2.0
models:
- altslate/indic-pii-redactor-small
- PaddlePaddle/PP-OCRv5_mobile_det_onnx
- PaddlePaddle/PP-OCRv5_mobile_rec_onnx
- PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx
short_description: Cover Aadhaar, PAN, UPI, phone numbers in screenshots
---

Covers Indian personal data in a screenshot (chat, app, document) **entirely in your browser**: the image never leaves your device.

1. Text is read with PaddlePaddle's PP-OCRv5 mobile models (ONNX, Apache-2.0) through onnxruntime-web: a text detector, a Chinese/Latin recognizer and a Devanagari recognizer; for every text line the more confident of the two readings is kept.
2. [altslate/indic-pii-redactor-small](https://huggingface.co/altslate/indic-pii-redactor-small) (transformers.js, int8) finds the personal data in the text: Aadhaar, PAN, UPI IDs, phone numbers, bank accounts, IFSC, GSTIN, emails, passport, driving licence, voter ID, vehicle numbers, PIN codes, names, addresses, dates of birth.
3. Each finding is mapped back to the pixels of its characters and covered with a solid box (not a blur: blurred digits can sometimes be recovered).

**How well it works.** On 300 chat screenshots rendered from the held-out English/Hindi/Hinglish PII test set (synthetic people; we do not use real screenshots), 96% of Latin-script personal data items and 90% of Devanagari ones were covered (a span counts as covered when at least 80% of its width is blacked out), and 1.6% of the boxes covered nothing personal. Given the true text instead of OCR, the PII model finds 99% of the items, so most of the remaining misses come from reading the image. Weakest: dates of birth (85%) and names (85–88%). Rendered screenshots are cleaner than real ones (no compression, photos, emoji or unusual fonts), so expect lower numbers on real images. **Always check the result before you share it.**

**Not covered yet:** scripts other than Latin and Devanagari (Bengali, Tamil, Telugu, …: the OCR here cannot read them), handwriting, photos of documents at an angle, and text inside images within the screenshot.

Code and evaluation: [github.com/AltSlate-Labs/indic-chat-toolkit](https://github.com/AltSlate-Labs/indic-chat-toolkit) (`src/vision/`).

Third-party: the OCR models are loaded from the PaddlePaddle repositories on Hugging Face; `dict-latin.json` and `dict-devanagari.json` are the character dictionaries from those models' `inference.yml` files (PaddleOCR, Apache-2.0, © PaddlePaddle Authors). The example screenshots were rendered by us from synthetic test messages.
