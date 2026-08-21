# Experimental Plan

This document describes experiments that strengthen the project for an academic report, portfolio, or paper-style evaluation.

## 1. Baseline comparison

Train all models using the same split, deterministic validation preprocessing, and evaluation protocol.

| Experiment | Architecture | Pretrained | Loss | Balanced sampling | Augmentation |
|---|---|---:|---|---:|---:|
| A | Custom CNN | No | CE | No | No |
| B | Custom CNN | No | CE | No | Yes |
| C | VGG16 | Yes | CE | No | Yes |
| D | ResNet18 | Yes | CE | No | Yes |

The comparison should use the independent test set for final results.

## 2. Augmentation ablation

Because microscopy augmentation must preserve meaningful morphology, compare:

1. no augmentation;
2. flips only;
3. flips + small rotations;
4. conservative full augmentation;
5. aggressive augmentation, only as an ablation rather than a default.

Report test accuracy and macro F1.

## 3. Loss ablation

Compare:

- Cross-Entropy
- Focal Loss

Keep every other setting fixed.

## 4. Balancing ablation

Compare:

- original sampling;
- `WeightedRandomSampler`;
- optional class-weighted loss.

Avoid copying image paths before splitting.

## 5. Transfer-learning ablation

For VGG16 / ResNet18 compare:

- ImageNet pretrained;
- training from scratch;
- partial freezing;
- full fine-tuning.

## 6. Model comparison table

Recommended final table:

| Model | Accuracy | Macro F1 | Macro ROC-AUC | Parameters | Inference ms/image |
|---|---:|---:|---:|---:|---:|
| Custom CNN | TBD | TBD | TBD | TBD | TBD |
| VGG16 | TBD | TBD | TBD | TBD | TBD |
| ResNet18 | TBD | TBD | TBD | TBD | TBD |

## 7. Error analysis

Include:

- normalized confusion matrix;
- most common confusion pair;
- misclassified-image gallery;
- confidence distribution for correct vs incorrect predictions.

## 8. Calibration

Report:

- reliability diagram;
- ECE;
- Brier score.

For a medical-image classification project, calibrated confidence is useful alongside raw discrimination metrics.

## 9. Representation analysis

Use t-SNE or UMAP on penultimate-layer embeddings as an exploratory visualization. Avoid treating visual cluster separation as a quantitative generalization result.

## 10. Explainability

Generate Grad-CAM examples for:

- one or more correctly classified samples per class;
- representative errors;
- low-confidence samples.

Interpret heatmaps cautiously.

## 11. Stronger reproducibility

For a stronger academic result, run multiple random seeds and report mean ± standard deviation, for example:

```text
Accuracy: 0.9821 ± 0.0034
Macro F1: 0.9815 ± 0.0038
```

If patient/slide/source grouping is available, use group-aware splitting instead of image-level random splitting.

## 12. Suggested final figures

A compact paper-quality figure set could contain:

1. class distribution;
2. representative images / augmentation examples;
3. learning curves;
4. normalized confusion matrices;
5. combined ROC curves;
6. combined PR curves;
7. reliability diagram;
8. t-SNE embeddings;
9. Grad-CAM examples;
10. model-comparison chart.

More plots are not automatically better; prioritize figures that answer a methodological or performance question.
