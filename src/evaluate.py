"""Measure the trained model on the test set: images never used for training or choosing the model.

Run from the project folder:
    python -m src.evaluate
    python -m src.evaluate --model models/best_model.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.data import load_datasets, make_loader
from src.model import CLASS_NAMES, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the classifier on the test set.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/catsanddogs"))
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    """Return (predicted labels, true labels) for every image in the loader."""
    predictions, labels = [], []
    for images, batch_labels in tqdm(loader, desc="test", leave=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            outputs = model(images.to(device, non_blocking=True))
        predictions.append(outputs.argmax(1).cpu())
        labels.append(batch_labels)
    return torch.cat(predictions).numpy(), torch.cat(labels).numpy()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = load_datasets(args.data_dir, splits=("test",))["test"]
    model = load_model(args.model, device)
    predictions, labels = predict_all(model, make_loader(test_set, args.batch_size), device)

    correct = int((predictions == labels).sum())
    print(f"Test accuracy: {correct / len(labels):.2%} ({correct} of {len(labels)} correct)\n")
    print(classification_report(labels, predictions, target_names=CLASS_NAMES, digits=4))

    # Rows are the real animal, columns are what the model said
    matrix = confusion_matrix(labels, predictions)
    width = max(len(name) for name in CLASS_NAMES) + 2
    print("Confusion matrix (rows: actual, columns: predicted)")
    print(" " * width + "".join(f"{name:>{width}}" for name in CLASS_NAMES))
    for name, row in zip(CLASS_NAMES, matrix):
        print(f"{name:<{width}}" + "".join(f"{count:>{width}}" for count in row))

    wrong = np.flatnonzero(predictions != labels)
    print(f"\n{len(wrong)} misclassified test images")


if __name__ == "__main__":
    main()
