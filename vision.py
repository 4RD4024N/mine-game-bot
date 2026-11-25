"""Computer vision for the minesweeper bot: finds the board in a screenshot and reads every cell.

Nothing here knows the board size or position in advance. The grid is found from the
screenshot itself (square cell contours on a darker background, clustered into rows and
columns), and each cell is classified from its pixels (background brightness, foreground
pixels and their colour in Lab space).
"""
from dataclasses import dataclass

import cv2
import numpy as np

COVERED, EMPTY, FLAG, MINE = "#", "0", "F", "*"   # numbers are "1".."8"

# The number legend (what a player learns: 1 is blue, 2 green, 3 red ...), as RGB.
NUMBER_COLORS = {
    1: (74, 168, 255), 2: (79, 196, 107), 3: (255, 90, 82), 4: (176, 124, 255),
    5: (255, 154, 60), 6: (54, 214, 200), 7: (242, 242, 242), 8: (154, 164, 174),
}
_NUMBER_LAB = {n: cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float) for n, rgb in NUMBER_COLORS.items()}


@dataclass
class Grid:
    x0: float      # centre of the top-left cell, in screenshot pixels
    y0: float
    pitch_x: float  # distance between neighbouring cell centres
    pitch_y: float
    cell: float     # cell side in pixels
    cols: int
    rows: int
    xs: tuple = ()  # measured centre of every column / row (no drift on wide boards)
    ys: tuple = ()

    def center(self, c, r):
        x = self.xs[c] if self.xs else self.x0 + c * self.pitch_x
        y = self.ys[r] if self.ys else self.y0 + r * self.pitch_y
        return x, y


def _cluster(values, tol):
    """1-D clustering: sorted values closer than tol fall in one group; returns group means."""
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups], [len(g) for g in groups]


def find_grid(img):
    """Locate the cell lattice in a BGR screenshot. Returns a Grid or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Cells (covered or open) are brighter than the thin gaps between them.
    bg = np.percentile(gray, 5)
    mask = (gray > bg + 14).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 12 or w > 90 or h > 90:
            continue
        if not 0.85 < w / h < 1.18:
            continue
        if cv2.contourArea(c) < 0.8 * w * h:   # a filled square, not an outline or a glyph
            continue
        boxes.append((x + w / 2, y + h / 2, (w + h) / 2))
    if len(boxes) < 9:
        return None
    # The cells are the largest family of equally sized squares.
    sizes = np.array([b[2] for b in boxes])
    hist_vals, hist_counts = _cluster(sizes, 2.0)
    size = hist_vals[int(np.argmax(hist_counts))]
    cells = [b for b in boxes if abs(b[2] - size) <= max(2.5, size * 0.08)]
    xs, xn = _cluster([b[0] for b in cells], size / 3)
    ys, yn = _cluster([b[1] for b in cells], size / 3)
    # Keep columns/rows that are actually populated and evenly spaced.
    xs = [x for x, n in zip(xs, xn) if n >= 2]
    ys = [y for y, n in zip(ys, yn) if n >= 2]
    if len(xs) < 3 or len(ys) < 3:
        return None
    px = float(np.median(np.diff(xs)))
    py = float(np.median(np.diff(ys)))
    cols = int(round((xs[-1] - xs[0]) / px)) + 1
    rows = int(round((ys[-1] - ys[0]) / py)) + 1

    def lattice(found, start, pitch, n):
        out, last = [], start - pitch
        for i in range(n):
            guess = last + pitch
            near = [f for f in found if abs(f - guess) < pitch / 3]
            last = min(near, key=lambda f: abs(f - guess)) if near else guess
            out.append(last)
        return tuple(out)

    return Grid(xs[0], ys[0], px, py, size, cols, rows, lattice(xs, xs[0], px, cols), lattice(ys, ys[0], py, rows))


def classify_cell(patch):
    """Classify one cell image (BGR)."""
    h, w = patch.shape[:2]
    m = int(min(h, w) * 0.14)
    inner = patch[m:h - m, m:w - m]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(float)
    s = hsv[..., 1].astype(float)
    hue = hsv[..., 0].astype(float)

    # Exploded mine: a strongly red cell.
    red_bg = ((hue < 8) | (hue > 172)) & (s > 120) & (v > 110)
    if red_bg.mean() > 0.55:
        return MINE

    bg_v = float(np.median(v))
    if bg_v > 72:  # raised, light tile: still covered (maybe flagged)
        red = ((hue < 10) | (hue > 170)) & (s > 140) & (v > 150)
        return FLAG if red.mean() > 0.04 else COVERED

    # Open cell: anything that differs clearly from its flat background is a glyph.
    lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB).astype(float)
    bg = np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    fg = dist > 38
    if fg.mean() < 0.03:
        return EMPTY
    # A mine: a black disc with a light ring.
    if ((v < 28) & fg).mean() > 0.06:
        return MINE
    glyph = fg & (v > bg_v + 25)   # digits are brighter than the cell; ignore darker gap pixels
    col = lab[glyph if glyph.sum() >= 6 else fg].mean(axis=0)
    n = min(_NUMBER_LAB, key=lambda k: np.linalg.norm(_NUMBER_LAB[k] - col))
    return str(n)


def read_board(img, grid):
    """Read all cells -> list of rows of labels."""
    half = grid.cell / 2
    rows = []
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            cx, cy = grid.center(c, r)
            x1, y1 = int(round(cx - half)), int(round(cy - half))
            x2, y2 = int(round(cx + half)), int(round(cy + half))
            row.append(classify_cell(img[max(0, y1):y2, max(0, x1):x2]))
        rows.append(row)
    return rows


def draw_overlay(img, grid, board, highlight=None):
    """Debug view: the detected lattice and the label read for every cell."""
    out = img.copy()
    half = grid.cell / 2
    colors = {COVERED: (150, 150, 150), FLAG: (60, 60, 255), MINE: (0, 0, 255), EMPTY: (90, 90, 90)}
    for r, row in enumerate(board):
        for c, lab in enumerate(row):
            cx, cy = grid.center(c, r)
            p1 = (int(cx - half), int(cy - half)); p2 = (int(cx + half), int(cy + half))
            color = colors.get(lab, (40, 220, 255))
            cv2.rectangle(out, p1, p2, color, 1)
            if lab not in (COVERED, EMPTY):
                cv2.putText(out, lab, (int(cx - half * 0.45), int(cy + half * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, half / 22, color, 2, cv2.LINE_AA)
    if highlight:
        for (c, r), color in highlight.items():
            cx, cy = grid.center(c, r)
            cv2.rectangle(out, (int(cx - half - 2), int(cy - half - 2)), (int(cx + half + 2), int(cy + half + 2)), color, 3)
    return out
