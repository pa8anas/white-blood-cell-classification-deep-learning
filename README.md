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

The repository also includes model evaluation and interpretability utilities for confusion matrices, ROC and Precision-Recall curves, calibration analysis, t-SNE feature visualization, and Grad-CAM/Grad-CAM++ style explanations.

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
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
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

A key design decision in this repository is that the train/validation split is created **before** any class balancing or repeated sampling. This avoids the same original image being represented in both training and validation through duplicated paths.

The repository therefore uses augmentation **only for the training subset** and supports class balancing with `WeightedRandomSampler` rather than duplicating image paths before splitting.

### Models

#### Custom CNN

Four convolutional blocks with increasing channel depth:

```text
3 → 32 → 64 → 128 → 256
```

Each block uses convolution, batch normalization, ReLU activation, and max pooling. The classifier contains fully connected layers with dropout.

#### VGG16

ImageNet-pretrained VGG16 with the final classifier replaced for four WBC classes. The last convolutional layers can be fine-tuned while earlier feature layers remain frozen.

#### ResNet18

ImageNet-pretrained ResNet18 with the final fully connected layer replaced by a four-class classifier.

### Loss

Both standard cross-entropy and focal loss are available. Focal loss can be useful when experiments intentionally use class-weighted training, but should be validated against an unweighted baseline.

## Installation

```bash
git clone https://github.com/pa8anas/white-blood-cell-classification-deep-learning.git
cd white-blood-cell-classification-deep-learning
pip install -r requirements.txt
```

## Training

Train the Custom CNN:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model custom_cnn \
  --epochs 20 \
  --batch-size 16
```

Train VGG16:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model vgg16 \
  --epochs 20 \
  --batch-size 16
```

Train ResNet18:

```bash
python -m src.train \
  --data-dir /path/to/dataset2-master/dataset2-master/images \
  --model resnet18 \
  --epochs 20 \
  --batch-size 32
```

Add `--balanced-sampling` to balance classes using only the training subset.

To use focal loss:

```bash
python -m src.train \
  --data-dir /path/to/images \
  --model resnet18 \
  --loss focal
```

## Evaluation

Evaluate a trained checkpoint on the independent `TEST` directory:

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
- overall accuracy / macro precision / macro recall / macro F1
- reliability diagram
- Brier score
- Expected Calibration Error (ECE)
- confidence distribution

## Grad-CAM

```bash
python -m src.gradcam \
  --model resnet18 \
  --checkpoint outputs/resnet18_best.pth \
  --image /path/to/image.jpeg \
  --output results/resnet18/gradcam.png
```

Grad-CAM visualizations are intended as **model interpretation aids**, not as proof that a model has learned clinically valid biological features.

## Experimental results

Earlier project experiments produced very high validation performance, with ResNet18 reaching approximately **99.95% validation accuracy** in one run. These values should be interpreted in the context of the exact split and preprocessing pipeline used for that experiment.

For reproducible reporting, this repository recommends:

- deterministic validation/test preprocessing;
- no augmentation on validation or test data;
- splitting original images before oversampling;
- using the independent `TEST` folder only for final evaluation;
- checking exact and near-duplicate images across splits;
- reporting macro F1 in addition to accuracy;
- reporting uncertainty or repeated-split / cross-validation results where possible.

Because microscopy datasets can contain images from related acquisition sources, patient- or slide-level splitting should be preferred whenever such metadata are available.

## Paper-oriented analysis

The repository supports figures commonly useful in a technical report or paper:

- learning curves
- confusion matrices
- per-class metrics
- ROC and Precision-Recall curves
- calibration analysis
- t-SNE embeddings
- confidence/error analysis
- Grad-CAM visualizations

Useful additional experiments are described in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Reproducibility

Training accepts a random seed and uses stratified splitting. GPU acceleration is used automatically when CUDA is available.

Example:

```bash
python -m src.train \
  --data-dir /path/to/images \
  --model resnet18 \
  --seed 42
```

## Authors

Academic project developed at the **University of Piraeus** for deep learning / artificial intelligence coursework.

## License

This repository is released under the [MIT License](LICENSE). The Blood Cell Images dataset is distributed separately and remains subject to its original terms.
