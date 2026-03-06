"""
Drawing / overlay utilities and HTML renderers for the Streamlit UI.
"""

import cv2
import numpy as np
from .image_processor import order_points


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def draw_grid_contour(
    image: np.ndarray,
    corners: np.ndarray,
    color=(0, 255, 0),
    thickness=3,
) -> np.ndarray:
    """Draw the detected grid boundary on the image."""
    vis = image.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    return vis


def draw_solution_on_warped(
    warped: np.ndarray,
    original_board: np.ndarray,
    solved_board: np.ndarray,
    cell_size: int | None = None,
) -> np.ndarray:
    """Overlay solved digits (green) onto the warped grid image."""
    vis = warped.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if cell_size is None:
        cell_size = vis.shape[0] // 9

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = cell_size / 50.0
    thick = max(1, int(cell_size / 25))

    for r in range(9):
        for c in range(9):
            if original_board[r, c] == 0 and solved_board[r, c] != 0:
                digit = str(solved_board[r, c])
                sz = cv2.getTextSize(digit, font, font_scale, thick)[0]
                x = c * cell_size + (cell_size - sz[0]) // 2
                y = r * cell_size + (cell_size + sz[1]) // 2
                cv2.putText(vis, digit, (x, y), font, font_scale, (0, 200, 0), thick)
    return vis


def overlay_solution_on_original(
    original: np.ndarray,
    corners: np.ndarray,
    original_board: np.ndarray,
    solved_board: np.ndarray,
    grid_size: int = 450,
) -> np.ndarray:
    """Warp solved digits back onto the original photograph."""
    blank = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    solution_overlay = draw_solution_on_warped(blank, original_board, solved_board)

    ordered = order_points(corners.astype(np.float32))
    src_pts = np.array([
        [0, 0], [grid_size - 1, 0],
        [grid_size - 1, grid_size - 1], [0, grid_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, ordered)
    warped_back = cv2.warpPerspective(
        solution_overlay, M, (original.shape[1], original.shape[0]),
    )

    mask = cv2.cvtColor(warped_back, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    result = original.copy()
    result[mask > 0] = warped_back[mask > 0]
    return result


# ---------------------------------------------------------------------------
# Plain-text board
# ---------------------------------------------------------------------------

def board_to_string(board: np.ndarray) -> str:
    """Format a 9x9 board as a human-readable string with box separators."""
    lines = []
    for r in range(9):
        if r % 3 == 0 and r != 0:
            lines.append("------+-------+------")
        parts = []
        for c in range(9):
            if c % 3 == 0 and c != 0:
                parts.append("|")
            v = board[r, c]
            parts.append(str(v) if v != 0 else ".")
        lines.append(" ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML grid renderers (used by Streamlit)
# ---------------------------------------------------------------------------

_GRID_CSS = """
<style>
.sdk-grid {
    display: inline-grid;
    grid-template-columns: repeat(9, 1fr);
    gap: 0;
    border: 3px solid #1e293b;
    border-radius: 6px;
    overflow: hidden;
    max-width: 420px;
    width: 100%;
    aspect-ratio: 1;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
.sdk-grid .c {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 1;
    font-size: clamp(14px, 2.4vw, 24px);
    font-weight: 600;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #1e293b;
    transition: background 0.15s;
}
.sdk-grid .c.solved  { color: #16a34a; background: #f0fdf4; }
.sdk-grid .c.empty   { color: #e2e8f0; }
.sdk-grid .c.low-conf { background: #fef9c3; }

/* 3x3 box borders */
.sdk-grid .c:nth-child(9n+4) { border-left: 3px solid #1e293b; }
.sdk-grid .c:nth-child(9n+7) { border-left: 3px solid #1e293b; }

/* Horizontal thick borders after rows 3 and 6 (children 19-27 are row 3, 46-54 are row 6) */
.sdk-grid .c:nth-child(n+19):nth-child(-n+27) { border-top: 3px solid #1e293b; }
.sdk-grid .c:nth-child(n+46):nth-child(-n+54) { border-top: 3px solid #1e293b; }
</style>
"""


def render_board_html(
    board: np.ndarray,
    original: np.ndarray | None = None,
    confidences: np.ndarray | None = None,
    conf_warn: float = 0.85,
) -> str:
    """
    Render a 9x9 board as a styled HTML grid.

    *original*    – if supplied, cells that were 0 in the original are shown in green.
    *confidences* – if supplied, cells below *conf_warn* get a yellow highlight.
    """
    html = _GRID_CSS + '<div class="sdk-grid">'
    for r in range(9):
        for c in range(9):
            val = int(board[r, c])
            classes = ["c"]

            if val == 0:
                classes.append("empty")
                display = "&middot;"
            elif original is not None and int(original[r, c]) == 0:
                classes.append("solved")
                display = str(val)
            else:
                display = str(val)

            if (
                confidences is not None
                and val != 0
                and (original is None or int(original[r, c]) != 0)
                and confidences[r, c] < conf_warn
            ):
                classes.append("low-conf")

            html += f'<div class="{" ".join(classes)}">{display}</div>'
    html += "</div>"
    return html
