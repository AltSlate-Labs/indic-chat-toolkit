---
title: Indic Chat Transliteration
emoji: ✍️
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
license: apache-2.0
models:
- altslate/indic-chat-xlit-small
- altslate/indic-chat-lid-small
short_description: Romanized chat to native script, 12 South Asian languages
---

Demo of [altslate/indic-chat-xlit-small](https://huggingface.co/altslate/indic-chat-xlit-small): type romanized
Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu, Sindhi or Sinhala and get native
script. "Auto-detect" also loads [altslate/indic-chat-lid-small](https://huggingface.co/altslate/indic-chat-lid-small)
to pick the language. Everything runs in your browser with [transformers.js](https://huggingface.co/docs/transformers.js)
and int8 ONNX files (about 300 MB, plus about 180 MB for auto-detect, downloaded once and cached). Your text never leaves
your device. See the model cards for results and limitations.
