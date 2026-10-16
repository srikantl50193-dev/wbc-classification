"""
White Blood Cell Classification using CNN + Bidirectional LSTM Hybrid

A hybrid architecture that merges spatial features from a CNN with
sequential features from a Bidirectional LSTM for WBC subtype
classification. Includes data augmentation, confusion matrix
visualization, and classification reporting.

Architecture:
- CNN branch: Conv2D layers extract spatial features from blood cell images
- RNN branch: Bidirectional LSTM processes grayscale image rows as sequences
- Merge: CNN and RNN outputs concatenated, followed by dense classification head

Dataset: Kaggle Blood Cells Dataset
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Author: Srikant Lakshminarayan
"""

import os
import itertools
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D,
    BatchNormalization, Lambda, Bidirectional, LSTM, concatenate,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DIR = "./dataset2-master/images/TRAIN/"
TEST_DIR = "./dataset2-master/images/TEST/"
ANNOTATION_DIR = "./dataset-master/Annotations/"
IMAGE_DIR = "./dataset-master/JPEGImages/"
LABELS_CSV = "./dataset-master/labels.csv"

IMG_ROWS, IMG_COLS = 60, 80
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)
BATCH_SIZE = 32
EPOCHS = 70
NUM_CLASSES = 5

CELL_TYPES = {1: "Neutrophil", 2: "Eosinophil", 3: "Monocyte", 4: "Lymphocyte"}
BINARY_TYPES = {0: "Mononuclear", 1: "Polynuclear"}

CHECKPOINT_PATH = "./checkpoints/best_cnn_rnn_model.keras"
METRICS_PATH = "./logs.npy"


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_data(folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess blood cell images from the given directory.

    Images are resized to (IMG_COLS, IMG_ROWS) and returned as numpy arrays
    alongside multiclass (4-type) and binary (mono/polynuclear) labels.

    Args:
        folder: Path to the image directory (TRAIN or TEST).

    Returns:
        Tuple of (images, multiclass_labels, binary_labels).
    """
    label_map = {
        "NEUTROPHIL": (1, 1),
        "EOSINOPHIL": (2, 1),
        "MONOCYTE": (3, 0),
        "LYMPHOCYTE": (4, 0),
    }

    images, labels, binary = [], [], []

    for cell_type in os.listdir(folder):
        if cell_type.startswith(".") or cell_type not in label_map:
            continue

        label, bin_label = label_map[cell_type]
        cell_dir = os.path.join(folder, cell_type)

        for filename in tqdm(os.listdir(cell_dir), desc=f"Loading {cell_type}"):
            img = cv2.imread(os.path.join(cell_dir, filename))
            if img is not None:
                img = cv2.resize(img, (IMG_COLS, IMG_ROWS), interpolation=cv2.INTER_CUBIC)
                images.append(np.asarray(img))
                labels.append(label)
                binary.append(bin_label)

    return np.asarray(images), np.asarray(labels), np.asarray(binary)


def normalize_images(X: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1] range."""
    return X.astype("float32") / 255.0


# ---------------------------------------------------------------------------
# Visualization Utilities
# ---------------------------------------------------------------------------

def plot_rgb_histogram(image: np.ndarray, title: str = "RGB Histogram") -> None:
    """Plot an image alongside its RGB pixel intensity histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.max() > 1 else image)
    axes[0].axis("off")
    axes[0].set_title("Sample Image")

    for channel, color in enumerate(["r", "g", "b"]):
        axes[1].hist(image[:, :, channel].flatten(), bins=30, lw=0, color=color, alpha=0.5)
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title(title)

    plt.tight_layout()
    plt.show()


def plot_learning_curve(history) -> None:
    """Plot training and validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("cnn_rnn_training_history.png", dpi=150)
    plt.show()
    print("Training history saved to cnn_rnn_training_history.png")


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    normalize: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """Plot a confusion matrix with annotations."""
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved to confusion_matrix.png")


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def rgb_to_grayscale(x):
    """Convert RGB tensor to grayscale by averaging channels."""
    return tf.reduce_mean(x, axis=3)


def build_cnn_rnn_model(num_classes: int = NUM_CLASSES, stride: int = 1) -> Model:
    """
    Build a hybrid CNN + Bidirectional LSTM model for WBC classification.

    The CNN branch extracts spatial features from the full RGB image.
    The RNN branch converts the image to grayscale and processes each row
    as a step in a sequence, capturing spatial patterns via Bidirectional LSTM.
    Both branches are merged via concatenation before the classification head.

    Args:
        num_classes: Number of output classes.
        stride: Stride for the first convolutional layer.

    Returns:
        Compiled Keras Model.
    """
    input_tensor = Input(shape=INPUT_SHAPE)

    # CNN branch
    cnn = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE, strides=stride),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
    ])
    cnn_output = cnn(input_tensor)

    # RNN branch (grayscale image rows as sequence)
    rnn_input = Lambda(
        rgb_to_grayscale,
        output_shape=INPUT_SHAPE[:-1],
    )(input_tensor)
    rnn = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.35))(rnn_input)
    rnn = Dropout(0.35)(rnn)
    rnn_output = Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.35))(rnn)
    rnn_output = Dropout(0.25)(rnn_output)

    # Merge branches
    merged = concatenate([cnn_output, rnn_output])
    x = Dense(128, activation="relu")(merged)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class MetricsCheckpoint(Callback):
    """Callback that saves metrics history after each epoch."""

    def __init__(self, savepath: str):
        super().__init__()
        self.savepath = savepath
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)


def create_data_augmentation() -> ImageDataGenerator:
    """Create an ImageDataGenerator with augmentation settings."""
    return ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
    )


def main():
    """Load data, build model, train, evaluate, and visualize results."""

    # Load data
    print("Loading training data...")
    X_train, y_train, z_train = load_data(TRAIN_DIR)
    print(f"Training set: {X_train.shape[0]} images")

    print("Loading test data...")
    X_test, y_test, z_test = load_data(TEST_DIR)
    print(f"Test set: {X_test.shape[0]} images")

    # Normalize
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)

    # One-hot encode
    y_train_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_hot = to_categorical(y_test, num_classes=NUM_CLASSES)

    # Visualize sample
    print("Sample image histogram (pre-normalized):")
    plot_rgb_histogram(X_train[0], title="Normalized RGB Histogram")

    # Build model
    model = build_cnn_rnn_model(num_classes=NUM_CLASSES, stride=1)
    model.summary()

    # Data augmentation
    datagen = create_data_augmentation()

    # Callbacks
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    callbacks = [
        MetricsCheckpoint(METRICS_PATH),
        ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train_hot, batch_size=BATCH_SIZE, shuffle=True),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_hot),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    score = model.evaluate(X_test, y_test_hot, verbose=0)
    print(f"\nCNN+RNN Test Accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)")

    # Classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_hot, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=list(CELL_TYPES.values()),
    ))

    # Visualizations
    plot_learning_curve(history)

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plot_confusion_matrix(cm, classes=list(CELL_TYPES.values()))


if __name__ == "__main__":
    main()
