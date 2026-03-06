"""
Sudoku Image Solver — Streamlit Web App
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2

from src.image_processor import extract_grid_from_image
from src.digit_recognizer import recognize_grid, load_model
from src.solver import solve_puzzle, is_valid_board
from src.utils import (
    draw_grid_contour,
    draw_solution_on_warped,
    overlay_solution_on_original,
    board_to_string,
    render_board_html,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sudoku Solver", page_icon="🧩", layout="wide")

st.markdown(
    """
    <style>
    /* tighten sidebar */
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

    /* compact data-editor: hide row indices & toolbar */
    [data-testid="stDataEditor"] [data-testid="glide-display-column"] { display: none !important; }
    [data-testid="stDataEditor"] header { display: none !important; }

    /* step cards */
    .step-card { background: #f8fafc; border: 1px solid #e2e8f0;
                 border-radius: 12px; padding: 1.2rem 1rem; text-align: center; }
    .step-card h4 { margin: 0 0 0.4rem; color: #334155; }
    .step-card p  { margin: 0; color: #64748b; font-size: 0.92rem; }

    /* status badge */
    .badge { display: inline-block; padding: 0.2em 0.7em; border-radius: 999px;
             font-size: 0.82rem; font-weight: 600; }
    .badge-ok   { background: #dcfce7; color: #166534; }
    .badge-warn { background: #fef9c3; color: #854d0e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("# 🧩 Sudoku Image Solver")
st.caption("Upload a photo of a Sudoku puzzle, review the detected digits, and solve it instantly.")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader(
        "Sudoku image", type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    st.divider()
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.3, max_value=1.0, value=0.80, step=0.05,
        help="Lower → more digits detected (may include errors). Higher → fewer but safer.",
    )
    show_overlay = st.checkbox("Overlay solution on original", value=True)

# ── Landing page (no image yet) ─────────────────────────────────────────────

if uploaded_file is None:
    st.info("👈  Upload a Sudoku image in the sidebar to get started.")
    cols = st.columns(3, gap="medium")
    steps = [
        ("1 · Upload", "Take or screenshot a Sudoku puzzle and upload it."),
        ("2 · Review", "Check the recognised digits — fix any mistakes in the editable grid."),
        ("3 · Solve", "Hit **Solve** and see the completed puzzle in seconds."),
    ]
    for col, (title, desc) in zip(cols, steps):
        col.markdown(f'<div class="step-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
    st.stop()

# ── Read the uploaded image ──────────────────────────────────────────────────

file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is None:
    st.error("Could not decode the uploaded image. Please try a different file.")
    st.stop()

# ── Step 1: Grid detection ───────────────────────────────────────────────────

with st.status("Detecting grid…", expanded=True) as status:
    cells, warped, corners = extract_grid_from_image(image)
    if cells is None:
        status.update(label="Grid detection failed", state="error", expanded=True)
        st.error(
            "Could not find a Sudoku grid in the image. "
            "Ensure the full grid border is visible and try again."
        )
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=400)
        st.stop()
    status.update(label="Grid detected ✓", state="complete", expanded=False)

# ── Step 2: Digit recognition ────────────────────────────────────────────────

try:
    load_model()
except FileNotFoundError:
    st.error("Trained model not found — run `python train/train_model.py` first.")
    st.stop()

board, confidences = recognize_grid(cells, confidence_threshold)
low_conf_count = int(np.sum((confidences < 0.85) & (board != 0)))

# ── Layout: image column  |  grid column ─────────────────────────────────────

col_img, col_grid = st.columns([5, 7], gap="large")

with col_img:
    tab_orig, tab_detect, tab_warped = st.tabs(["Original", "Detected grid", "Warped"])
    with tab_orig:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with tab_detect:
        vis = draw_grid_contour(image, corners)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    with tab_warped:
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), use_container_width=True)

# ── Editable grid (data_editor) ──────────────────────────────────────────────

with col_grid:
    st.subheader("Recognised Puzzle")

    if low_conf_count:
        st.markdown(
            f'<span class="badge badge-warn">{low_conf_count} low-confidence cell(s)</span> '
            "— highlighted in yellow below. Double-check these.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-ok">All digits recognised confidently</span>',
            unsafe_allow_html=True,
        )

    st.caption("Edit any cell directly (0 = empty). Columns and rows are labelled 1–9.")

    upload_key = uploaded_file.name + str(confidence_threshold)
    if st.session_state.get("_upload_key") != upload_key:
        st.session_state._upload_key = upload_key
        st.session_state.board_df = pd.DataFrame(
            board,
            columns=[str(i) for i in range(1, 10)],
            index=[str(i) for i in range(1, 10)],
        )

    edited_df = st.data_editor(
        st.session_state.board_df,
        use_container_width=True,
        hide_index=False,
        num_rows="fixed",
        key="grid_editor",
        column_config={
            str(c): st.column_config.NumberColumn(
                min_value=0, max_value=9, step=1, format="%d",
            )
            for c in range(1, 10)
        },
    )

    edited_board = edited_df.to_numpy(dtype=int)

    # ── Confidence mini-heatmap ──────────────────────────────────────────
    with st.expander("Confidence details"):
        st.markdown(render_board_html(board, confidences=confidences), unsafe_allow_html=True)
        st.caption("Yellow cells had a CNN confidence below 85 %.")

    # ── Solve button ─────────────────────────────────────────────────────

    solve_btn = st.button("Solve Puzzle", type="primary", use_container_width=True)

# ── Step 3: Solve ─────────────────────────────────────────────────────────────

if solve_btn:
    if not is_valid_board(edited_board):
        st.error("The board has duplicates in a row, column, or box. Fix the grid and try again.")
        st.stop()

    with st.spinner("Solving…"):
        solution = solve_puzzle(edited_board)

    if solution is None:
        st.error("No valid solution exists — the puzzle may be incorrect.")
        st.stop()

    st.divider()
    st.subheader("Solution")
    sol_left, sol_right = st.columns(2, gap="large")

    with sol_left:
        st.markdown(render_board_html(solution, original=edited_board), unsafe_allow_html=True)
        with st.expander("Plain text"):
            st.code(board_to_string(solution))

    with sol_right:
        if show_overlay and corners is not None:
            overlay = overlay_solution_on_original(image, corners, edited_board, solution)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="Solution overlaid on original")
        else:
            warped_sol = draw_solution_on_warped(warped, edited_board, solution)
            st.image(cv2.cvtColor(warped_sol, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="Solution on warped grid")
