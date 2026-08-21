# White Blood Cell Classification with Deep Learning

Deep learning project for classifying microscopic white blood cell (WBC) images into four categories:

- **Eosinophil**
- **Lymphocyte**
- **Monocyte**
- **Neutrophil**

The project compares three convolutional neural-network approaches:

1. **Custom CNN** trained from scratch
2. **VGG16** with ImageNet transfer learning
3. **ResNet18** with ImageNet transfer learning

The repository includes the original experiment notebook and paper together with reusable PyTorch modules for leakage-safe data handling, training, independent test-set evaluation, calibration analysis, ROC/Precision-Recall analysis, and Grad-CAM interpretation.

## Main project files

- [`deep_learning_bloodcells.ipynb`](deep_learning_bloodcells.ipynb) — complete experimental notebook
- [`paper.pdf`](paper.pdf) — project report / paper
- [`src/`](src/) — reusable training and evaluation code
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — methodology and evaluation decisions
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — recommended paper-oriented ablations and experiments

## Dataset

The experiments use the **Blood Cell Images** dataset from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Expected local structure:

```text
DATA_DIR/
└── dataset2-master/
    └── dataset2-master/
        └── images/
            ├── TRAIN/
            │   ├── EOSINOPHIL/
            │   ├── LYMPHOCYTE/
            │   ├── MONOCYTE/
            │   └── NEUTROPHIL/
            └── TEST/
                ├── EOSINOPHIL/
                ├── LYMPHOCYTE/
                ├── MONOCYTE/
                └── NEUTROPHIL/
```

> The dataset itself is **not** stored in this repository.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── deep_learning_bloodcells.ipynb
├── paper.pdf
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── check_duplicates.py
├── notebooks/
│   └── README.md
├── results/
│   └── README.md
├── docs/
│   ├── METHODOLOGY.md
│   └── EXPERIMENTS.md
└── .github/
    └── workflows/
        └── python-check.yml
```

## Methodology

### Preprocessing

All images are resized to **224 × 224** and normalized with ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Data augmentation

Training-time augmentation includes conservative geometric and photometric transformations. Validation and test images use a **deterministic** transform only.

A key design decision in the reusable pipeline is that the train/validation split is created **before** any class balancing or repeated sampling. This avoids the same original image being represented in both training and validation through duplicated paths.

The `src/` pipeline therefore applies augmentation **only to the training subset** and supports class balancing with `WeightedRandomSampler` rather than duplicating image paths before splitting.

### Models

#### Custom CNN

Four convolutional stages with increasing feature depth and a fully connected classifier with dropout.

#### VGG16

ImageNet-pretrained VGG16 with the final classifier replaced for the four WBC classes. Earlier convolutional features are frozen while the final convolutional block and classifier can be fine-tuned.

#### ResNet18

ImageNet-pretrained ResNet18 with its final fully connected layer replaced by a four-class classifier.

### Loss functions

The reusable code supports both:

- Cross-Entropy Loss
- Focal Loss

Cross-entropy is recommended as the baseline. Focal loss should be treated as an experimental choice and compared under otherwise identical settings.

## Installation

```bash
git clone https://github.com/pa8anas/white-blood-cell-classification-deep-learning.git
cd white-blood-cell-classification-deep-learning
pip install -r requirements.txt
```

## Training

Custom CNN:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model custom_cnn \
  --epochs 20 \
  --batch-size 16
```

VGG16:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model vgg16 \
  --epochs 20 \
  --batch-size 16
```

ResNet18:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model resnet18 \
  --epochs 20 \
  --batch-size 32
```

Add `--balanced-sampling` to balance classes using only the training subset.

Use focal loss with:

```bash
python -m src.train \
  --data-dir /path/to/images \
  --model resnet18 \
  --loss focal
```

## Independent test-set evaluation

```bash
python -m src.evaluate \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model resnet18 \
  --checkpoint outputs/resnet18_best.pth \
  --output-dir results/resnet18
```

Generated outputs include:

- classification report
- confusion matrix
- normalized confusion matrix
- ROC curves
- Precision-Recall curves
- accuracy / macro precision / macro recall / macro F1
- macro ROC-AUC and average precision
- reliability diagram
- Brier score
- Expected Calibration Error (ECE)
- confidence distribution

## Data-leakage check

Before reporting final results, check for exact duplicate files between TRAIN and TEST:

```bash
python -m src.check_duplicates \
  --data-dir /path/to/dataset2-master/dataset2-master/images
```

This performs an exact MD5-based duplicate check. Near-duplicate images can still require perceptual or source-level analysis.

## Grad-CAM

```bash
python -m src.gradcam \
  --model resnet18 \
  --checkpoint outputs/resnet18_best.pth \
  --image /path/to/image.jpeg \
  --output results/resnet18/gradcam.png
```

Grad-CAM visualizations are model-interpretation aids and should not by themselves be treated as proof of clinical or biological validity.

## Experimental results

Earlier project experiments produced very high validation performance, with ResNet18 reaching approximately **99.95% validation accuracy** in one run. These values should be interpreted in the context of the exact split and preprocessing pipeline used in that run.

For paper-quality reporting, the repository recommends:

- deterministic validation/test preprocessing;
- no augmentation on validation or test data;
- splitting original images before oversampling;
- using the independent `TEST` folder only for final evaluation;
- checking exact and near duplicates across splits;
- reporting macro F1 in addition to accuracy;
- comparing augmentation and loss choices through controlled ablations;
- reporting repeated-seed or cross-validation uncertainty where possible.

If patient, slide, or acquisition-source metadata are available, group-aware splitting should be preferred over image-level random splitting.

## Paper-oriented analysis

Useful analyses for the report include:

- class distribution
- augmentation examples
- learning curves
- normalized confusion matrices
- per-class precision / recall / F1
- ROC and Precision-Recall curves
- calibration analysis
- t-SNE or UMAP embeddings
- confidence/error analysis
- Grad-CAM visualizations
- model-comparison tables including parameter count and inference time

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for a structured experiment plan.

## Reproducibility

Training accepts a fixed random seed and uses stratified splitting:

```bash
python -m src.train \
  --data-dir /path/to/images \
  --model resnet18 \
  --seed 42
```

CUDA is used automatically when available.

## Authors

Academic deep-learning project developed at the **University of Piraeus**.

## License

This repository is released under the [MIT License](LICENSE). The Blood Cell Images dataset is distributed separately and remains subject to its original terms.
