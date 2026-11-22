---
title: Indic TTS Normalizer and ITN
emoji: 🔢
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
license: apache-2.0
models:
- altslate/indic-tts-normalizer-small
- altslate/indic-itn-small
- altslate/indic-punctuation-base
short_description: Numbers, ₹ and dates to spoken words, 10 Indic languages
---

Demo of [altslate/indic-tts-normalizer-small](https://huggingface.co/altslate/indic-tts-normalizer-small): paste Hindi,
Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi or Urdu text with numbers and get the spoken form to
feed a text-to-speech model. Everything runs in your browser with [transformers.js](https://huggingface.co/docs/transformers.js)
and int8 ONNX files (about 300 MB, downloaded once and cached). Your text never leaves your device. See the model card for
results and limitations. A second mode runs [altslate/indic-itn-small](https://huggingface.co/altslate/indic-itn-small),
which does the reverse for speech-recognition output (number words → digits).
