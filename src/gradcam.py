from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .data import eval_transform
from .models import build_model


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad

    def generate(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

    def close(self) -> None:
        self.handle.remove()


def target_layer_for(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if model_name == "custom_cnn":
        return model.features[-3]
    if model_name == "vgg16":
        return model.features[29]
    if model_name == "resnet18":
        return model.layer4[-1].conv2
    raise ValueError(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for one WBC image")
    parser.add_argument("--model", choices=["custom_cnn", "vgg16", "resnet18"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam.png")
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"])
    image_size = checkpoint.get("image_size", args.image_size)

    model = build_model(args.model, num_classes=len(class_names), image_size=image_size, pretrained=False).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    pil_img = Image.open(args.image).convert("RGB")
    tensor = eval_transform(image_size)(pil_img).unsqueeze(0).to(device)

    cam = GradCAM(model, target_layer_for(args.model, model))
    heatmap = cam.generate(tensor, args.class_index)
    cam.close()

    original = np.asarray(pil_img.resize((image_size, image_size)))
    heatmap = cv2.resize(heatmap, (image_size, image_size))
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original.astype(np.uint8), 0.6, heatmap_rgb.astype(np.uint8), 0.4, 0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
