from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import eval_transform
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained WBC classifier")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", choices=["custom_cnn", "vgg16", "resnet18"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path(args.data_dir) / "TEST"
    dataset = datasets.ImageFolder(test_dir, transform=eval_transform(args.image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        args.model,
        num_classes=len(dataset.classes),
        image_size=checkpoint.get("image_size", args.image_size) if isinstance(checkpoint, dict) else args.image_size,
        pretrained=False,
    ).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_preds)
    y_score = np.asarray(all_probs)
    classes = dataset.classes

    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    metrics["roc_auc_macro"] = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
    metrics["average_precision_macro"] = average_precision_score(y_bin, y_score, average="macro")

    confidence = y_score.max(axis=1)
    correct = (y_pred == y_true).astype(int)
    metrics["brier_correctness"] = brier_score_loss(correct, confidence)
    metrics["ece"] = expected_calibration_error(confidence, correct)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_normalized.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 7))
    for i, name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 7))
    for i, name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_bin[:, i], y_score[:, i]):.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=200)
    plt.close()

    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(confidence[correct == 1], bins=20, alpha=0.65, label="Correct")
    if np.any(correct == 0):
        plt.hist(confidence[correct == 0], bins=20, alpha=0.65, label="Incorrect")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of samples")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
