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


def predict_digit(cell_image: np.ndarray, min_confidence: float = 0.0) -> tuple:
    """
    Predict the digit in a single cell image (always the model's best guess).

    Args:
        cell_image: 28x28 grayscale uint8 image (already cleaned).
        min_confidence: If > 0, predictions below this confidence are returned as 0 (blank).
            Use 0 to always return the model's best prediction for max correctness.

    Returns:
        (digit, confidence) where digit is 0 for blank, 1-9 for digits.
    """
    model = load_model()

    img = cell_image.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    probs = model.predict(img, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit])

    if digit == 0:
        return 0, confidence
    if min_confidence > 0 and confidence < min_confidence:
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


def recognize_grid(cells: list, grid_fill_fraction: float = 0.8) -> tuple:
    """
    Recognize all 81 cells and return the 9x9 digit grid. The model always
    predicts its best guess per cell; grid_fill_fraction controls how many
    of the cells (by confidence order) are filled in.

    Args:
        cells: List of 81 cell images (28x28 grayscale) in row-major order.
        grid_fill_fraction: Fraction of the grid (0–1) to fill with predictions.
            Cells are ranked by confidence; only the top fraction are filled;
            the rest are left as 0 for manual entry.

    Returns:
        (board, confidences) where board is a 9x9 numpy array (0=empty, 1-9=digits)
        and confidences is a 9x9 array of prediction confidence values.
    """
    board = np.zeros((9, 9), dtype=int)
    confidences = np.zeros((9, 9), dtype=float)
    # Always use model's best guess (no confidence filter) for correctness
    predictions = []  # (linear_idx, digit, confidence)

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)

        if is_cell_empty(cell):
            confidences[r, c] = 1.0
            predictions.append((idx, 0, 1.0))
            continue

        digit, conf = predict_digit(cell, min_confidence=0.0)
        confidences[r, c] = conf
        predictions.append((idx, digit, conf))

    # Fill only the top grid_fill_fraction of cells by confidence
    n_fill = max(0, min(81, int(round(81 * grid_fill_fraction))))
    by_confidence = sorted(predictions, key=lambda x: -x[2])
    for i in range(n_fill):
        idx, digit, _ = by_confidence[i]
        r, c = divmod(idx, 9)
        board[r, c] = digit

    return board, confidences
