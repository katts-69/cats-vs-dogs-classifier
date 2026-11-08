"""Train the cat/dog classifier and save the version with the best validation accuracy.

Run from the project folder:
    python -m src.train
    python -m src.train --head-epochs 3 --fine-tune-epochs 2 --output models/best_model.pt
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from src.data import load_datasets, make_loader
from src.model import CLASS_NAMES, build_model, set_backbone_trainable


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cat/dog classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/catsanddogs"))
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--head-epochs", type=int, default=3,
                        help="epochs training only the new Cat/Dog layer")
    parser.add_argument("--fine-tune-epochs", type=int, default=2,
                        help="epochs training the whole model (the best result came after 1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"  # mixed precision: faster on a GPU
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="train" if training else "validate", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    datasets_ = load_datasets(args.data_dir, splits=("train", "validation"))
    found = datasets_["train"].classes
    if found != CLASS_NAMES:
        raise SystemExit(f"Expected class folders {CLASS_NAMES} in {args.data_dir}, found {found}")
    train_loader = make_loader(datasets_["train"], args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(datasets_["validation"], args.batch_size, seed=args.seed)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Phase 1 trains only the new layer at a normal learning rate.
    # Phase 2 unfreezes everything with a 10x lower rate, so pretrained knowledge is
    # adjusted rather than overwritten.
    phases = [
        ("head only", args.head_epochs, False, 1e-3),
        ("fine-tune", args.fine_tune_epochs, True, 1e-4),
    ]

    best_val_acc = 0.0
    history = []
    for phase, epochs, unfreeze_backbone, learning_rate in phases:
        set_backbone_trainable(model, unfreeze_backbone)
        optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            history.append({"phase": phase, "train_loss": train_loss, "train_acc": train_acc,
                            "val_loss": val_loss, "val_acc": val_acc})

            saved = val_acc > best_val_acc
            if saved:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.output)
            print(f"[{phase}] epoch {epoch}/{epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.2%} | "
                  f"{time.time() - start:.0f}s" + (" | saved best" if saved else ""))

    history_path = args.output.with_name(args.output.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {args.output}, training history to {history_path}")


if __name__ == "__main__":
    main()
