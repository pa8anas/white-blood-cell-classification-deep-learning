from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .data import build_dataloaders
from .losses import FocalLoss
from .models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WBC image classifiers")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=["custom_cnn", "vgg16", "resnet18"], default="resnet18")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=args.val_size,
        seed=args.seed,
        num_workers=args.num_workers,
        balanced_sampling=args.balanced_sampling,
    )

    model = build_model(
        args.model,
        num_classes=len(loaders.class_names),
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
    ).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    default_lr = 1e-4 if args.model != "resnet18" else 3e-5
    lr = args.lr if args.lr is not None else default_lr
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_parameters, lr=lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.model}_best.pth"
    history_path = output_dir / f"{args.model}_history.json"

    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    print(f"Device: {device}")
    print(f"Classes: {loaders.class_names}")
    print(f"Model: {args.model} | Loss: {args.loss} | LR: {lr}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(loaders.train, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(loaders.train), 1)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loaders.val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= max(len(loaders.val), 1)
        val_accuracy = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_names": loaders.class_names,
                    "image_size": args.image_size,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
