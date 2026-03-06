# Sudoku Image Solver

A Python application that extracts a Sudoku puzzle from a photograph, recognizes the digits using a custom-trained CNN, solves the puzzle with a backtracking algorithm, and presents the results through an interactive Streamlit web interface.

## Features

- **Image Processing** — Detects the Sudoku grid boundary, applies a perspective warp, and extracts individual cells using OpenCV.
- **Custom CNN** — A convolutional neural network trained from scratch on MNIST (+ augmentations and blank-cell samples) to classify digits 0–9 where 0 represents an empty cell.
- **Backtracking Solver** — Validates the board and solves the puzzle using constraint checking with backtracking.
- **Web Interface** — Upload an image, review/edit the recognized digits, and view the solution overlaid on the original photo.

## Project Structure

```
Sudoku-Puzzle/
├── app.py                    # Streamlit web app
├── requirements.txt
├── src/
│   ├── image_processor.py    # Grid detection and cell extraction
│   ├── digit_recognizer.py   # CNN inference
│   ├── solver.py             # Sudoku solver
│   └── utils.py              # Drawing/overlay helpers
├── train/
│   ├── prepare_data.py       # Data preparation (MNIST + augmentation)
│   └── train_model.py        # CNN training script
├── model/                    # Saved model weights (created after training)
└── samples/                  # Sample images for testing
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the digit recognition model

This downloads MNIST, augments it with blank-cell samples, trains a CNN, and saves the weights to `model/digit_model.keras`:

```bash
python train/train_model.py
```

Training takes roughly 5–10 minutes on a modern CPU (faster with a GPU).

### 3. Run the web app

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

## Usage

1. **Upload** a clear photo of a Sudoku puzzle using the sidebar.
2. **Review** the recognized digits in the editable 9×9 grid. Correct any misrecognized cells (set to `0` for empty).
3. **Adjust** the confidence threshold slider if too many or too few digits are detected.
4. Click **Solve Puzzle** to compute and display the solution.
5. The solution is shown both as a styled grid and overlaid on the original image.

## Tips for Best Results

- Use a well-lit, straight-on photo of the puzzle for best grid detection.
- Printed puzzles work better than handwritten ones (the model is trained on MNIST handwritten digits but augmented for general use).
- If the grid isn't detected, try cropping the image closer to the puzzle border.
