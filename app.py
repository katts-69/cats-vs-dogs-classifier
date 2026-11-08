"""Web app: upload a photo and the model says whether it shows a cat or a dog.

Run from the project folder:
    python app.py
then open http://127.0.0.1:7860 in your browser.
"""
from pathlib import Path

import gradio as gr

from src.predict import Predictor

MODEL_PATH = Path("models/best_model.pt")

# Load the model once when the app starts, not on every upload
predictor = Predictor(MODEL_PATH)


def classify(image):
    """Gradio calls this with the uploaded photo and shows what it returns."""
    if image is None:
        return None
    return predictor.predict(image)  # e.g. {"Cat": 0.02, "Dog": 0.98}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Photo of a cat or a dog"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Cat or Dog?",
    description=(
        "Upload a photo and an EfficientNet-B0 model will tell you whether it shows a cat or a dog. "
        "It was fine-tuned on 19,793 photos and scored 98.95% on 2,474 test photos it never saw during training."
    ),
    article=(
        "**Limitations:** the model only knows cats and dogs, so any other photo will still be labelled "
        "as one of the two. It is least reliable on dark, blurry or very small photos, and on photos "
        "with both animals in them."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
