"""Convert the trained PyTorch model to ONNX, so it can run in a web browser, then check it.

Run from the project folder:
    python deploy/export_onnx.py

ONNX is a standard file format for neural networks. The web page loads models/model.onnx with
ONNX Runtime Web and classifies photos on the visitor's own device, so no Python server is needed.

The check runs every test photo through both the PyTorch model and the ONNX model and compares
them, so we know the conversion didn't change any answers.
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from tqdm import tqdm

from src.data import load_datasets, make_loader
from src.model import CLASS_NAMES, load_model


def export(model_path, onnx_path):
    # Add softmax to the model, so it outputs probabilities and the web page doesn't have to
    model = nn.Sequential(load_model(model_path, "cpu"), nn.Softmax(dim=1)).eval()
    example = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_shapes=({0: torch.export.Dim("batch")},),  # any number of images at once
        opset_version=18,
        external_data=False,  # keep the weights inside the one file
        dynamo=True,
        verbose=False,  # its progress messages contain emoji that crash the Windows console
    )
    return model


@torch.no_grad()
def compare(model, onnx_path, data_dir):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_set = load_datasets(data_dir, splits=("test",))["test"]
    torch_probs, onnx_probs, labels = [], [], []
    for images, batch_labels in tqdm(make_loader(test_set, batch_size=64), desc="test", leave=False):
        torch_probs.append(model(images).numpy())
        onnx_probs.append(session.run(None, {"image": images.numpy()})[0])
        labels.append(batch_labels.numpy())
    torch_probs, onnx_probs, labels = map(np.concatenate, (torch_probs, onnx_probs, labels))

    torch_pred, onnx_pred = torch_probs.argmax(1), onnx_probs.argmax(1)
    print(f"PyTorch test accuracy: {(torch_pred == labels).mean():.2%}")
    print(f"ONNX test accuracy:    {(onnx_pred == labels).mean():.2%}")
    print(f"Photos where they disagree: {int((torch_pred != onnx_pred).sum())} of {len(labels)}")
    print(f"Largest probability difference: {np.abs(torch_probs - onnx_probs).max():.2e}")


def main():
    parser = argparse.ArgumentParser(description="Export the model to ONNX and check it on the test set.")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/model.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/catsanddogs"))
    args = parser.parse_args()

    model = export(args.model, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB), classes {CLASS_NAMES}")
    compare(model, args.output, args.data_dir)


if __name__ == "__main__":
    main()
