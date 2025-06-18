import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import time

# --- Template Matching için yüklenen görseller ---
def load_templates():
    templates = {}
    for val in ['covered_tile', '1', '2']:
        path = f'templates/{val}.png'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            templates[val] = img
    return templates

# --- Grid Tanıma ---
def get_game_grid():
    print("[INFO] Select top-left corner of grid in 3 seconds...")
    time.sleep(3)
    top_left = pyautogui.position()
    print(f"[DEBUG] Top-left: {top_left}")

    print("[INFO] Select bottom-right corner of grid in 3 seconds...")
    time.sleep(3)
    bottom_right = pyautogui.position()
    print(f"[DEBUG] Bottom-right: {bottom_right}")

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    screenshot_np = np.array(screenshot)

    cell_size = (width // 9, height // 9)
    return screenshot_np, top_left, cell_size

def get_grid_state(screenshot, top_left, cell_size):
    templates = load_templates()
    grid = []

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    cw, ch = cell_size
    rows, cols = 9, 9

    for row in range(rows):
        row_data = []
        for col in range(cols):
            x = col * cw
            y = row * ch
            cell_img = gray[y:y+ch, x:x+cw]

            matched_val = 'covered_tile'
            max_score = 0.0

            for label, template in templates.items():
                res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > max_score and score > 0.75:
                    matched_val = label
                    max_score = score

            row_data.append(matched_val)
        grid.append(row_data)
    return grid

# --- Mantıksal Hamle Kararları ---
def analyze_and_get_moves(grid):
    rows = len(grid)
    cols = len(grid[0])
    moves = []

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val in ['1', '2']:
                num = int(val)
                neighbors = get_neighbors(grid, r, c)
                covered = [(x, y) for (x, y) in neighbors if grid[x][y] == 'covered_tile']

                if len(covered) == num:
                    for (x, y) in covered:
                        moves.append({'x': y, 'y': x, 'type': 'flag'})
                elif count_flagged(grid, neighbors) == num:
                    for (x, y) in covered:
                        if not is_flagged(grid, x, y):
                            moves.append({'x': y, 'y': x, 'type': 'click'})
    return moves

def get_neighbors(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors

def count_flagged(grid, neighbors):
    return sum(1 for (x, y) in neighbors if grid[x][y] == 'flag')

def is_flagged(grid, x, y):
    return grid[x][y] == 'flag'

# --- Ekrana Tıklama ---
def perform_click(x, y, click_type, top_left, cell_size):
    ox, oy = top_left
    cw, ch = cell_size
    cx = ox + x * cw + cw // 2
    cy = oy + y * ch + ch // 2

    if click_type == 'click':
        pyautogui.click(cx, cy, button='left')
    elif click_type == 'flag':
        pyautogui.click(cx, cy, button='right')

# --- Ana Döngü ---
def main():
    print("[INFO] Minesweeper Bot Starting...")
    screenshot, top_left, cell_size = get_game_grid()

    print("[INFO] Detecting grid...")
    grid = get_grid_state(screenshot, top_left, cell_size)

    print("[INFO] Applying logic...")
    moves = analyze_and_get_moves(grid)

    print(f"[INFO] Executing {len(moves)} moves...")
    for move in moves:
        perform_click(move['x'], move['y'], move['type'], top_left, cell_size)
        time.sleep(0.1)

    print("[DONE] Bot Finished.")

if __name__ == "__main__":
    main()
