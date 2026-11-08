---
library_name: pytorch
pipeline_tag: image-classification
tags:
  - image-classification
  - efficientnet
  - cats-vs-dogs
metrics:
  - accuracy
---

# Cats vs Dogs: EfficientNet-B0

A cat/dog photo classifier: torchvision's EfficientNet-B0, pretrained on ImageNet, fine-tuned on the
Microsoft/Kaggle "Cats and Dogs" dataset.

**Try it in your browser:** [bebird/cat-or-dog](https://huggingface.co/spaces/bebird/cat-or-dog) ·
**Code:** [github.com/katts-69/cats-vs-dogs-classifier](https://github.com/katts-69/cats-vs-dogs-classifier)

## Results

| Split | Photos | Accuracy |
|---|---|---|
| Validation (used to pick the best epoch) | 2,474 | 98.99% |
| Test (never used for training or choosing) | 2,474 | **98.95%** |

On the test set, 10 cats were predicted as dogs and 16 dogs as cats. Several of those 26 are
problems with the data, not the model: a beagle labelled as a cat, a "No Photo Available"
placeholder, and photos showing both animals.

## Training

- Data: 24,741 usable photos after removing 259 broken, fake or duplicate files, split 80/10/10
  (stratified, seed 42), so no photo appears in more than one split.
- Phase 1: only the new 2-class head trained, 3 epochs, AdamW, learning rate 1e-3.
- Phase 2: whole network fine-tuned, learning rate 1e-4; the best validation epoch was kept.
- Augmentation: random resized crop and horizontal flip. Mixed precision on an RTX 3050.

## Files

| File | What it is |
|---|---|
| `best_model.pt` | PyTorch `state_dict` of the fine-tuned model; outputs raw scores |
| `model.onnx` | ONNX version with softmax added, so it outputs probabilities. Input `image` (N x 3 x 224 x 224), output `probabilities` (N x 2, order Cat, Dog). Used by the web demo through ONNX Runtime Web |

The ONNX export was checked against PyTorch on all 2,474 test photos: same answer on every photo,
probabilities within 0.000003. In the browser, image resizing differs slightly from Python's, which
changed the answer on 5 borderline photos; the browser version also scores 98.95%.

## How to use

Load `best_model.pt` with the code from the GitHub repo:

```python
from huggingface_hub import hf_hub_download
from src.predict import Predictor

predictor = Predictor(hf_hub_download("bebird/cats-vs-dogs-efficientnet-b0", "best_model.pt"))
predictor.predict("photo.jpg")  # {"Cat": 0.02, "Dog": 0.98}
```

Images are resized to 256, centre-cropped to 224 and normalised with ImageNet mean and std.

## Limitations

The model only knows two classes, so any other photo is still labelled as a cat or a dog. It is
least reliable on dark, blurry or very small photos and on photos with both animals in them.
