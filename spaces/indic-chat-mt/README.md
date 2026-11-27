---
title: Indic Chat Translator
emoji: 💬
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
license: mit
models:
- altslate/indic-chat-mt-small
- altslate/indic-chat-lid-small
short_description: Hinglish and romanized Indic chat to English, in-browser
---

Demo of [altslate/indic-chat-mt-small](https://huggingface.co/altslate/indic-chat-mt-small): type Hinglish, or a message
in Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi or Urdu (romanized or in native script),
and get English. "Auto-detect" also loads [altslate/indic-chat-lid-small](https://huggingface.co/altslate/indic-chat-lid-small)
to pick the language. Everything runs in your browser with [transformers.js](https://huggingface.co/docs/transformers.js)
and int8 ONNX files (about 730 MB, plus about 140 MB for auto-detect, downloaded once and cached). Your text never leaves
your device. See the model card for results and limitations.
