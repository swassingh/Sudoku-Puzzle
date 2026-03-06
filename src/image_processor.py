import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Produce a binary (inverted) image optimised for grid-line detection.

    Tries adaptive threshold first, then a colour-saturation approach for
    screenshots with coloured grid lines (e.g. blue-on-white Sudoku apps).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2,
    )

    if _has_strong_grid(thresh):
        return thresh

    enhanced = _preprocess_color(image)
    if enhanced is not None and _has_strong_grid(enhanced):
        return enhanced

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(thresh, kernel, iterations=1)


def _preprocess_color(image: np.ndarray) -> np.ndarray | None:
    """Isolate saturated (coloured) pixels → binary mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    if np.mean(s) < 10:
        return None
    blurred = cv2.GaussianBlur(s, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _has_strong_grid(binary: np.ndarray, min_ratio: float = 0.02) -> bool:
    return np.count_nonzero(binary) / binary.size > min_ratio


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------

def find_grid_contour(thresh: np.ndarray, image_bgr: np.ndarray | None = None):
    """
    Find the 9x9 grid boundary.

    Strategy order:
      1. Largest approximately-square contour (filters out non-square UI chrome).
      2. Hough-line fallback (derives corners from detected grid lines).
      3. Largest contour with relaxed squareness.

    Returns (4, 2) corner array or None.
    """
    corners = _find_square_contour(thresh)
    if corners is not None:
        return corners

    if image_bgr is not None:
        corners = _find_grid_via_lines(image_bgr)
        if corners is not None:
            return corners

    return _find_square_contour(thresh, max_aspect=1.4)


def _find_square_contour(thresh: np.ndarray, max_aspect: float = 1.15):
    """
    Search for the largest approximately-square quadrilateral contour.
    Uses both RETR_EXTERNAL and RETR_TREE to catch nested grids (e.g.
    a grid inside an app widget).
    """
    img_area = thresh.shape[0] * thresh.shape[1]
    best = None
    best_area = 0

    for mode in (cv2.RETR_EXTERNAL, cv2.RETR_TREE):
        contours, _ = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:15]:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.05 or area <= best_area:
                continue

            pts = _approx_quad(cnt)
            if pts is None:
                pts = _quad_from_minrect(cnt)
            if pts is None:
                continue

            if _quad_aspect(pts) > max_aspect:
                continue

            best = pts
            best_area = area

    return best


def _approx_quad(contour):
    """Try progressively larger epsilon to approximate to exactly 4 points."""
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def _quad_from_minrect(contour):
    """Minimum-area rotated rectangle as a fallback quad."""
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w == 0 or h == 0:
        return None
    return np.intp(cv2.boxPoints(rect)).reshape(4, 2)


def _quad_aspect(pts: np.ndarray) -> float:
    """Width/height aspect ratio of a quadrilateral (>= 1.0)."""
    ordered = order_points(pts.astype(np.float32))
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    w = (w_top + w_bot) / 2
    h = (h_left + h_right) / 2
    return max(w, h) / max(min(w, h), 1)


# ---------------------------------------------------------------------------
# Hough-line fallback
# ---------------------------------------------------------------------------

def _find_grid_via_lines(image_bgr: np.ndarray):
    """
    Detect horizontal/vertical line segments and infer the grid's four
    corners from the outermost lines.  Works well for clean digital
    screenshots where contour detection may pick up the wrong border.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_len = max(gray.shape) // 4
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                            minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return None

    h_lines = []  # roughly horizontal
    v_lines = []  # roughly vertical
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 10 or angle > 170:
            h_lines.append((y1 + y2) / 2)
        elif 80 < angle < 100:
            v_lines.append((x1 + x2) / 2)

    h_clusters = _cluster_1d(sorted(h_lines), min_gap=gray.shape[0] * 0.03)
    v_clusters = _cluster_1d(sorted(v_lines), min_gap=gray.shape[1] * 0.03)

    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return None

    top, bottom = h_clusters[0], h_clusters[-1]
    left, right = v_clusters[0], v_clusters[-1]

    grid_h = bottom - top
    grid_w = right - left
    if grid_h == 0 or grid_w == 0:
        return None
    if max(grid_h, grid_w) / min(grid_h, grid_w) > 1.2:
        return None

    corners = np.array([
        [left, top], [right, top],
        [right, bottom], [left, bottom],
    ], dtype=np.float32)
    return corners.astype(int)


def _cluster_1d(values: list, min_gap: float) -> list:
    """Cluster sorted 1-D values; return the mean of each cluster."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [np.mean(c) for c in clusters]


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def perspective_warp(image: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Perspective-transform the grid region into a *size x size* square."""
    ordered = order_points(corners.astype(np.float32))
    dst = np.array([
        [0, 0], [size - 1, 0],
        [size - 1, size - 1], [0, size - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, M, (size, size))


# ---------------------------------------------------------------------------
# Cell extraction & cleaning
# ---------------------------------------------------------------------------

def extract_cells(warped: np.ndarray) -> list:
    """Split the warped grid into 81 cell images (row-major)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    cell_size = gray.shape[0] // 9
    cells = []
    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            cell = gray[y1:y2, x1:x2]
            cells.append(clean_cell(cell))
    return cells


def clean_cell(cell: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Clean a single cell: strip border artefacts, binarise with a
    background-relative threshold (handles coloured digits), centre
    the digit, and resize to *target_size x target_size*.
    """
    h, w = cell.shape[:2]
    margin = max(2, int(min(h, w) * 0.08))
    cropped = cell[margin:h - margin, margin:w - margin]

    binary = _threshold_relative(cropped)

    binary = _flood_fill_border(binary)

    return _centre_digit(binary, target_size)


def _threshold_relative(cell_gray: np.ndarray) -> np.ndarray:
    """
    Threshold by comparing each pixel to the local background brightness.
    Much more reliable than global OTSU for coloured/thin digits on a
    bright background.
    """
    bg = float(np.percentile(cell_gray, 85))

    if bg < 50:
        _, out = cv2.threshold(cell_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return out

    diff = bg - cell_gray.astype(np.float32)
    threshold = max(bg * 0.30, 25)
    out = np.where(diff > threshold, 255, 0).astype(np.uint8)
    return out


def _flood_fill_border(binary: np.ndarray) -> np.ndarray:
    """Flood-fill from edges to remove grid-line fragments."""
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    out = binary.copy()
    for x in range(w):
        if out[0, x] == 255:
            cv2.floodFill(out, mask, (x, 0), 0)
        if out[h - 1, x] == 255:
            cv2.floodFill(out, mask, (x, h - 1), 0)
    for y in range(h):
        if out[y, 0] == 255:
            cv2.floodFill(out, mask, (0, y), 0)
        if out[y, w - 1] == 255:
            cv2.floodFill(out, mask, (w - 1, y), 0)
    return out


def _centre_digit(binary: np.ndarray, target_size: int = 28) -> np.ndarray:
    """Centre the digit on a square canvas and resize."""
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    x, y, bw, bh = cv2.boundingRect(coords)
    digit = binary[y:y + bh, x:x + bw]

    pad = 4
    canvas_size = max(bw, bh) + 2 * pad
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    oy = (canvas_size - bh) // 2
    ox = (canvas_size - bw) // 2
    canvas[oy:oy + bh, ox:ox + bw] = digit

    return cv2.resize(canvas, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def extract_grid_from_image(image: np.ndarray):
    """
    Full pipeline: image -> (cells, warped_image, corners).

    Returns (None, None, None) when the grid cannot be found.
    """
    thresh = preprocess(image)
    corners = find_grid_contour(thresh, image_bgr=image)
    if corners is None:
        return None, None, None

    warped = perspective_warp(image, corners)
    cells = extract_cells(warped)
    return cells, warped, corners
