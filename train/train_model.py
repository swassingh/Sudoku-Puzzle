"""
Train a CNN digit recognizer for Sudoku cells.

Usage:
    python train/train_model.py

This will prepare the data (MNIST + blanks + augmentation), train the model,
and save it to model/digit_model.keras.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import prepare_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")


def build_model(num_classes: int = 10) -> keras.Model:
    """
    Build a compact CNN for 28x28 grayscale digit classification.

    Classes: 0 (blank) and 1-9 (digits).
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def train(epochs: int = 10, batch_size: int = 128):
    """Run the full training pipeline."""
    print("=" * 60)
    print("  Sudoku Digit Recognizer — Training")
    print("=" * 60)

    print("\n[1/4] Preparing data...")
    (x_train, y_train), (x_test, y_test) = prepare_data(augment=True)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    print(f"       Train: {x_train.shape[0]} samples")
    print(f"       Test:  {x_test.shape[0]} samples")

    print("\n[2/4] Building model...")
    model = build_model(num_classes=10)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print(f"\n[3/4] Training for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    print("\n[4/4] Evaluating...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"       Test loss:     {loss:.4f}")
    print(f"       Test accuracy: {accuracy:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    train()
