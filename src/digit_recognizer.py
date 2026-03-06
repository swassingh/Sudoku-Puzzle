"""
Digit recognition for extracted Sudoku cells using a trained CNN.
"""

import os
import numpy as np
import cv2
from tensorflow import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "digit_model.keras")

_model = None


def load_model(path: str = MODEL_PATH) -> keras.Model:
    """Load the trained digit recognition model (cached after first call)."""
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Trained model not found at {path}. "
                "Run 'python train/train_model.py' first."
            )
        _model = keras.models.load_model(path)
    return _model


def predict_digit(cell_image: np.ndarray, confidence_threshold: float = 0.8) -> tuple:
    """
    Predict the digit in a single cell image.

    Args:
        cell_image: 28x28 grayscale uint8 image (already cleaned).
        confidence_threshold: Minimum confidence to accept a prediction.
            If below threshold the cell is treated as blank (0).

    Returns:
        (digit, confidence) where digit is 0 for blank, 1-9 for digits.
    """
    model = load_model()

    img = cell_image.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    probs = model.predict(img, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    if confidence < confidence_threshold or digit == 0:
        return 0, confidence

    return digit, confidence


def is_cell_empty(cell_image: np.ndarray, pixel_threshold: int = 3) -> bool:
    """
    Quick heuristic: if the cell has almost no foreground pixels it's
    certainly empty.  The threshold is kept very low (3 %) so that thin
    digits like 1 or 7 are never accidentally filtered out.
    """
    total_pixels = cell_image.size
    white_pixels = np.count_nonzero(cell_image > 128)
    ratio = white_pixels / total_pixels
    return ratio < (pixel_threshold / 100.0)


def recognize_grid(cells: list, confidence_threshold: float = 0.8) -> tuple:
    """
    Recognize all 81 cells and return the 9x9 digit grid.

    Args:
        cells: List of 81 cell images (28x28 grayscale) in row-major order.
        confidence_threshold: Minimum confidence for digit predictions.

    Returns:
        (board, confidences) where board is a 9x9 numpy array (0=empty, 1-9=digits)
        and confidences is a 9x9 array of prediction confidence values.
    """
    board = np.zeros((9, 9), dtype=int)
    confidences = np.zeros((9, 9), dtype=float)

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)

        if is_cell_empty(cell):
            board[r, c] = 0
            confidences[r, c] = 1.0
            continue

        digit, conf = predict_digit(cell, confidence_threshold)
        board[r, c] = digit
        confidences[r, c] = conf

    return board, confidences
