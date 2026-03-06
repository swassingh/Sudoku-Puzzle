"""
Download MNIST and prepare augmented training data for the Sudoku digit recognizer.

Classes:
    0 = blank/empty cell
    1-9 = digits

The MNIST dataset contains digits 0-9.  We remap MNIST digit-0 samples to
become additional "blank" examples and keep digits 1-9 as-is.  We also
generate synthetic blank cell images (uniform noise / empty) to strengthen
class 0.
"""

import os
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_and_remap_mnist():
    """
    Load MNIST data and remap for Sudoku use:
      - Original MNIST class 0 images -> class 0 (blank proxy)
      - Original MNIST classes 1-9 -> classes 1-9

    Returns:
        (x_train, y_train), (x_test, y_test)  with class 0 = blank, 1-9 = digits
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def generate_blank_samples(n: int = 6000, img_size: int = 28) -> np.ndarray:
    """Create synthetic blank cell images (low-intensity noise)."""
    blanks = np.random.randint(0, 30, size=(n, img_size, img_size), dtype=np.uint8)
    return blanks


def augment_dataset(x: np.ndarray, y: np.ndarray, factor: int = 2):
    """
    Apply random augmentations (rotation, shift, zoom) to expand the dataset.
    Returns augmented arrays appended to the originals.
    """
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    x_4d = x.reshape(-1, 28, 28, 1)
    aug_images = []
    aug_labels = []

    batches = 0
    target = len(x) * (factor - 1)
    for x_batch, y_batch in datagen.flow(x_4d, y, batch_size=256, seed=42):
        aug_images.append(x_batch.reshape(-1, 28, 28))
        aug_labels.append(y_batch)
        batches += x_batch.shape[0]
        if batches >= target:
            break

    x_aug = np.concatenate([x] + aug_images, axis=0)
    y_aug = np.concatenate([y] + aug_labels, axis=0)
    return x_aug, y_aug


def prepare_data(augment: bool = True):
    """
    Full data preparation pipeline.

    Returns:
        (x_train, y_train), (x_test, y_test) — normalized float32 arrays.
    """
    (x_train, y_train), (x_test, y_test) = load_and_remap_mnist()

    blank_train = generate_blank_samples(n=6000)
    blank_test = generate_blank_samples(n=1000)
    x_train = np.concatenate([x_train, blank_train], axis=0)
    y_train = np.concatenate([y_train, np.zeros(len(blank_train), dtype=y_train.dtype)])
    x_test = np.concatenate([x_test, blank_test], axis=0)
    y_test = np.concatenate([y_test, np.zeros(len(blank_test), dtype=y_test.dtype)])

    shuffle_train = np.random.permutation(len(x_train))
    x_train, y_train = x_train[shuffle_train], y_train[shuffle_train]
    shuffle_test = np.random.permutation(len(x_test))
    x_test, y_test = x_test[shuffle_test], y_test[shuffle_test]

    if augment:
        x_train, y_train = augment_dataset(x_train, y_train, factor=2)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    print("Preparing data...")
    (x_train, y_train), (x_test, y_test) = prepare_data()
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    print(f"Classes:          {sorted(set(y_train))}")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_DIR, "sudoku_digits.npz"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Saved to {DATA_DIR}/sudoku_digits.npz")
