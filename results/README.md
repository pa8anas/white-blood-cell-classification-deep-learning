# Results

Store generated evaluation artifacts in model-specific subdirectories, for example:

```text
results/
├── custom_cnn/
├── vgg16/
└── resnet18/
```

Recommended files for each model:

```text
classification_report.csv
metrics.json
confusion_matrix.png
confusion_matrix_normalized.png
roc_curves.png
precision_recall_curves.png
reliability_diagram.png
confidence_distribution.png
gradcam_*.png
```

Generated images and tables are ignored by default in `.gitignore` so large experimental outputs do not accidentally bloat the repository. Remove the relevant ignore rules if you want selected final figures committed for presentation.
