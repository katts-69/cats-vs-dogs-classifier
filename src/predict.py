"""Predict whether images show a cat or a dog.

Run from the project folder:
    python -m src.predict path/to/photo.jpg another_photo.png

Or use it from other code, such as the web app:
    predictor = Predictor("models/best_model.pt")
    predictor.predict(image)  # {"Cat": 0.02, "Dog": 0.98}
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data import eval_transform
from src.model import CLASS_NAMES, load_model

DEFAULT_MODEL_PATH = Path("models/best_model.pt")
# Where the trained model is published, for machines that don't have it locally
MODEL_REPO = "bebird/cats-vs-dogs-efficientnet-b0"


def find_model(path=DEFAULT_MODEL_PATH):
    """Return the local model file if it exists (after training); otherwise download it."""
    path = Path(path)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download  # only needed when downloading

    return Path(hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pt"))


class Predictor:
    """Loads the trained model once, then classifies as many images as you give it."""

    def __init__(self, model_path="models/best_model.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(model_path, self.device)

    @torch.no_grad()
    def predict(self, image):
        """Classify one image (a PIL image or a file path). Returns each class's probability."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        # Same preparation as the validation and test images, or predictions silently get worse
        batch = eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=1)[0].tolist()
        return dict(zip(CLASS_NAMES, probabilities))


def main():
    parser = argparse.ArgumentParser(description="Predict cat or dog for one or more images.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    args = parser.parse_args()

    predictor = Predictor(args.model)
    for path in args.images:
        probabilities = predictor.predict(path)
        label = max(probabilities, key=probabilities.get)
        print(f"{path.name}: {label} ({probabilities[label]:.1%} confident)")


if __name__ == "__main__":
    main()
