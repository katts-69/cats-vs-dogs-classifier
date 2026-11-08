---
title: Cat or Dog
emoji: 🐱
colorFrom: pink
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: Cat vs dog classifier that runs in your browser
models:
  - bebird/cats-vs-dogs-efficientnet-b0
---

# Cat or Dog?

Upload a photo and an EfficientNet-B0 model tells you whether it shows a cat or a dog.
It was fine-tuned on 19,793 photos and scored 98.95% on 2,474 test photos it never saw during training.

The model runs in your browser with ONNX Runtime Web, so photos never leave your device.

- Model: [bebird/cats-vs-dogs-efficientnet-b0](https://huggingface.co/bebird/cats-vs-dogs-efficientnet-b0)
- Code, training and evaluation: [github.com/katts-69/cats-vs-dogs-classifier](https://github.com/katts-69/cats-vs-dogs-classifier)
