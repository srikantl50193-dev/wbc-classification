"""
White Blood Cell Classification using Custom Inception-Style CNN

Classifies microscopic blood cell images into 4 WBC subtypes:
Neutrophil, Eosinophil, Monocyte, and Lymphocyte.

Architecture: Custom multi-branch Inception-style CNN with parallel
convolution paths (1x1, 3x3, 5x5) and max-pooling, concatenated
at each block. Built from scratch without transfer learning.

Dataset: Kaggle Blood Cells Dataset
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

Author: Srikant Lakshminarayan
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization,
    MaxPool2D, Flatten, concatenate
)
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DIR = "./dataset2-master/images/TRAIN/"
TEST_DIR = "./dataset2-master/images/TEST/"
IMG_SIZE = (60, 80, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 5
INCEPTION_BLOCKS = 4
CHECKPOINT_PATH = "./checkpoints/best_model.keras"

CELL_TYPES = {
    "NEUTROPHIL": 1,
    "EOSINOPHIL": 2,
    "MONOCYTE": 3,
    "LYMPHOCYTE": 4,
}

CELL_TYPE_NAMES = {1: "Neutrophil", 2: "Eosinophil", 3: "Monocyte", 4: "Lymphocyte"}
BINARY_LABELS = {
    "NEUTROPHIL": 1,   # Polynuclear
    "EOSINOPHIL": 1,   # Polynuclear
    "MONOCYTE": 0,     # Mononuclear
    "LYMPHOCYTE": 0,   # Mononuclear
}
BINARY_NAMES = {0: "Mononuclear", 1: "Polynuclear"}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(folder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess blood cell images from the given directory.

    Each subdirectory should be named after a WBC type (e.g., NEUTROPHIL).
    Images are resized to IMG_SIZE and returned as numpy arrays alongside
    their multiclass and binary labels.

    Args:
        folder: Path to the image directory (TRAIN or TEST).

    Returns:
        Tuple of (images, multiclass_labels, binary_labels) as numpy arrays.
    """
    images, labels, binary = [], [], []

    for cell_type in os.listdir(folder):
        if cell_type.startswith(".") or cell_type not in CELL_TYPES:
            continue

        label = CELL_TYPES[cell_type]
        bin_label = BINARY_LABELS[cell_type]
        cell_dir = os.path.join(folder, cell_type)

        for filename in tqdm(os.listdir(cell_dir), desc=f"Loading {cell_type}"):
            filepath = os.path.join(cell_dir, filename)
            img = cv2.imread(filepath)

            if img is not None:
                img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
                images.append(np.asarray(img))
                labels.append(label)
                binary.append(bin_label)

    return np.asarray(images), np.asarray(labels), np.asarray(binary)


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def build_inception_model(num_blocks: int = INCEPTION_BLOCKS) -> Model:
    """
    Build a custom Inception-style CNN for WBC classification.

    The architecture uses parallel convolution branches (1x1, 3x3, 5x5)
    alongside a max-pooling branch at each Inception block. Outputs from
    all branches are concatenated depth-wise before passing to the next
    block. This allows the model to capture features at multiple spatial
    scales simultaneously.

    Args:
        num_blocks: Number of Inception blocks to stack.

    Returns:
        Compiled Keras Model.
    """
    inp = Input(shape=IMG_SIZE)

    # Stem layers
    x = BatchNormalization()(inp)
    x = Conv2D(32, (7, 7), padding="same", activation="relu", strides=(2, 2))(x)
    x = MaxPool2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu", strides=(1, 1))(x)
    x = MaxPool2D(pool_size=(3, 3), padding="same", strides=(2, 2))(x)

    # Inception blocks
    for _ in range(num_blocks):
        branches = []

        # Branch 1: 1x1 convolution
        b1 = Conv2D(32, (1, 1), padding="same", activation="relu")(x)
        b1 = Conv2D(32, (1, 1), padding="same", activation="relu")(b1)
        branches.append(b1)

        # Branch 2: 1x1 -> 3x3 convolution
        b2 = Conv2D(32, (1, 1), padding="same", activation="relu")(x)
        b2 = Conv2D(32, (3, 3), padding="same", activation="relu")(b2)
        b2 = Conv2D(32, (3, 3), padding="same", activation="relu")(b2)
        branches.append(b2)

        # Branch 3: 1x1 -> 5x5 convolution
        b3 = Conv2D(32, (1, 1), padding="same", activation="relu")(x)
        b3 = Conv2D(32, (5, 5), padding="same", activation="relu")(b3)
        b3 = Conv2D(32, (5, 5), padding="same", activation="relu")(b3)
        branches.append(b3)

        # Branch 4: Max pooling -> 1x1 convolution
        b4 = MaxPool2D(pool_size=(2, 2), padding="same", strides=(1, 1))(x)
        b4 = Conv2D(32, (1, 1), padding="same", activation="relu")(b4)
        branches.append(b4)

        x = concatenate(branches, axis=-1)

    # Classification head
    x = MaxPool2D(pool_size=(7, 7), padding="same", strides=(2, 2))(x)
    x = Flatten()(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=inp, outputs=output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, decay=WEIGHT_DECAY),
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Training & Visualization
# ---------------------------------------------------------------------------

def plot_training_history(history) -> None:
    """Plot training and validation loss curves."""
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history.history["loss"], label="Training Loss")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history.history["accuracy"], label="Training Accuracy")
    axes[1].plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Training history saved to training_history.png")


def main():
    """Load data, build model, train, and evaluate."""

    # Load data
    print("Loading training data...")
    X_train, y_train, z_train = load_data(TRAIN_DIR)
    print(f"Training set: {X_train.shape[0]} images")

    print("Loading test data...")
    X_test, y_test, z_test = load_data(TEST_DIR)
    print(f"Test set: {X_test.shape[0]} images")

    # One-hot encode labels
    y_train_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test_hot = to_categorical(y_test, num_classes=NUM_CLASSES)

    # Build model
    model = build_inception_model(num_blocks=INCEPTION_BLOCKS)
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    # Train
    history = model.fit(
        X_train,
        y_train_hot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test_hot),
        callbacks=[checkpoint],
        verbose=1,
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_hot, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Plot
    plot_training_history(history)


if __name__ == "__main__":
    main()
