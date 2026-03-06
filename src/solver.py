import numpy as np


def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing `num` at (row, col) violates Sudoku constraints."""
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    if num in board[box_r:box_r + 3, box_c:box_c + 3]:
        return False
    return True


def solve(board: np.ndarray) -> bool:
    """
    Solve the Sudoku puzzle in-place using backtracking.

    Args:
        board: 9x9 numpy array where 0 represents an empty cell.

    Returns:
        True if a solution was found (board is modified in-place),
        False if the puzzle is unsolvable.
    """
    empty = find_empty(board)
    if empty is None:
        return True

    row, col = empty
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row, col] = num
            if solve(board):
                return True
            board[row, col] = 0

    return False


def find_empty(board: np.ndarray):
    """Return (row, col) of the first empty cell, or None if the board is full."""
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return (r, c)
    return None


def is_valid_board(board: np.ndarray) -> bool:
    """
    Validate that the initial board state has no duplicate digits
    in any row, column, or 3x3 box.
    """
    for i in range(9):
        row_vals = board[i, board[i] != 0]
        if len(row_vals) != len(set(row_vals)):
            return False
        col_vals = board[:, i][board[:, i] != 0]
        if len(col_vals) != len(set(col_vals)):
            return False

    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            box = board[box_r:box_r + 3, box_c:box_c + 3].flatten()
            box_vals = box[box != 0]
            if len(box_vals) != len(set(box_vals)):
                return False
    return True


def solve_puzzle(board: np.ndarray):
    """
    High-level API: validate and solve the puzzle.

    Args:
        board: 9x9 numpy array (0 = empty cell).

    Returns:
        Solved 9x9 numpy array, or None if unsolvable / invalid.
    """
    if board.shape != (9, 9):
        return None
    if not is_valid_board(board):
        return None

    puzzle = board.copy()
    if solve(puzzle):
        return puzzle
    return None
