# Methodology

## Task

Multi-class classification of microscopic white blood cell images into:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

## Dataset handling

The project expects the Kaggle Blood Cell Images dataset with separate `TRAIN` and `TEST` directories.

The repository deliberately separates **training augmentation** from **validation/test preprocessing**.

### Important methodological rule

The train/validation split is created from the original `TRAIN` image list first. Any balancing or random augmentation is applied only after the split and only to the training subset.

This avoids a common leakage pattern where repeated paths are created before splitting, allowing different augmented views of the same original image to appear in both train and validation.

## Preprocessing

All images are resized to 224 × 224 and normalized with ImageNet statistics.

## Augmentation

The default training augmentation is intentionally moderate:

- horizontal flip
- rotation up to 10 degrees
- small affine translation / scale changes
- mild brightness, contrast and saturation jitter

These transforms are used as regularization and should be validated experimentally. In microscopy, aggressive transformations can modify morphology or staining characteristics in ways that may not be biologically meaningful.

For paper-quality experiments, an ablation comparing no augmentation vs conservative augmentation is recommended.

## Class balancing

The repository uses `WeightedRandomSampler` as an optional balancing strategy. This balances training exposure without physically copying image paths before the train/validation split.

## Models

### Custom CNN

A convolutional network trained from scratch with four feature-extraction stages and fully connected classification layers.

### VGG16

ImageNet-pretrained VGG16. Earlier convolutional layers are frozen while the final convolutional block and classifier remain trainable.

### ResNet18

ImageNet-pretrained ResNet18 with the final fully connected layer replaced for four-class classification.

## Loss functions

Two losses are available:

- Cross-Entropy Loss
- Focal Loss

Cross-entropy should be treated as the baseline. Focal loss should only be claimed as beneficial when a controlled comparison demonstrates improvement.

## Early stopping

Training monitors validation loss and saves the checkpoint with the lowest observed validation loss. Training stops after a configurable number of non-improving epochs.

## Final evaluation

The independent `TEST` directory should be used only after model and hyperparameter selection are complete.

Recommended reported metrics:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class Precision / Recall / F1
- ROC-AUC
- Average Precision / PR curves
- Confusion matrix
- Calibration metrics

## Interpretability

Grad-CAM visualizations can indicate which spatial regions influence a prediction. They should be interpreted as model-behavior explanations, not as proof of clinical or biological validity.

## Leakage checks

Run:

```bash
python -m src.check_duplicates --data-dir /path/to/images
```

This detects exact duplicate files across `TRAIN` and `TEST` using MD5 hashes. For stronger validation, near-duplicate / perceptual similarity checks and patient- or slide-level grouping should be considered if source metadata are available.
