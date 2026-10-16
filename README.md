# White Blood Cell Classification using Custom Inception-Style CNN

Classifies microscopic blood cell images into 4 WBC subtypes — **Neutrophil, Eosinophil, Monocyte, and Lymphocyte** — using a custom-built Inception-style convolutional neural network.

## Architectures

This project includes two distinct model architectures, both built from scratch without transfer learning:

### 1. Custom Inception-Style CNN (`train.py`)

Multi-branch architecture inspired by GoogLeNet/Inception:

- **Stem**: BatchNorm → 7×7 Conv (stride 2) → MaxPool → 3×3 Conv → MaxPool
- **Inception Blocks** (×4): Each block runs 4 parallel branches:
  - 1×1 → 1×1 convolution
  - 1×1 → 3×3 → 3×3 convolution
  - 1×1 → 5×5 → 5×5 convolution
  - MaxPool → 1×1 convolution
- **Head**: MaxPool → Flatten → Dense (softmax, 5 classes)

All branch outputs are concatenated depth-wise, capturing features at multiple spatial scales.

### 2. CNN + Bidirectional LSTM Hybrid (`train_cnn_rnn.py`)

A hybrid architecture that merges spatial and sequential features:

- **CNN branch**: Conv2D layers extract spatial features from the full RGB image
- **RNN branch**: Image is converted to grayscale, and each row is processed as a sequence step through Bidirectional LSTM layers, capturing spatial patterns as sequences
- **Merge**: CNN and RNN outputs concatenated, followed by Dense classification head with Dropout
- **Training**: Uses data augmentation (rotation, zoom, shifts, flips) via ImageDataGenerator

This approach is uncommon in WBC classification literature, where most projects rely purely on CNNs or transfer learning.

## Results

- **Inception CNN**: 93%+ validation accuracy (best configuration)
- **CNN+RNN Hybrid**: Trained for 70 epochs with data augmentation
- **Classes**: Neutrophil, Eosinophil, Monocyte, Lymphocyte

See `RESULTS.md` for detailed training logs.

## Dataset

[Kaggle Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) by Paul Timothy Mooney.

- ~12,500 augmented images across 4 WBC subtypes
- Pre-split into TRAIN and TEST directories

### Setup

1. Download the dataset from Kaggle
2. Extract `dataset2-master` into the project root:

```
wbc-classification/
├── dataset2-master/
│   └── images/
│       ├── TRAIN/
│       │   ├── NEUTROPHIL/
│       │   ├── EOSINOPHIL/
│       │   ├── MONOCYTE/
│       │   └── LYMPHOCYTE/
│       └── TEST/
│           ├── NEUTROPHIL/
│           ├── EOSINOPHIL/
│           ├── MONOCYTE/
│           └── LYMPHOCYTE/
├── train.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/wbc-classification.git
cd wbc-classification

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and extract to ./dataset2-master/

# Train Inception-style model
python train.py

# Train CNN+RNN hybrid model
python train_cnn_rnn.py
```

The best model weights are saved to `./checkpoints/best_model.keras` and training curves are exported to `training_history.png`.

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- OpenCV
- NumPy
- Matplotlib
- tqdm

See `requirements.txt` for exact versions.

## Key Design Decisions

- **No transfer learning**: Both models are trained entirely from scratch to demonstrate understanding of CNN and RNN architecture design rather than fine-tuning pre-built models.
- **Multi-scale feature extraction** (Inception model): Parallel branches (1×1, 3×3, 5×5) capture both fine-grained cellular details and broader structural patterns.
- **Cross-modal fusion** (CNN+RNN model): Merging CNN spatial features with LSTM sequential features provides complementary representations of blood cell morphology.
- **Data augmentation** (CNN+RNN model): Rotation, zoom, shifts, and flips to reduce overfitting and improve generalization.
- **Binary sub-classification**: Both pipelines support a secondary binary task (Mononuclear vs. Polynuclear) for multi-task learning experiments.


