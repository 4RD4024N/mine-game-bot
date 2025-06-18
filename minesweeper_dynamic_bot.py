import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
import time

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = 'tesseract'  # sistemine göre tam path gerekebilir

def get_game_grid():
    print("[INFO] Select top-left corner of grid in 3 seconds...")
    time.sleep(3)
    top_left = pyautogui.position()

    print("[INFO] Select bottom-right corner of grid in 3 seconds...")
    time.sleep(3)
    bottom_right = pyautogui.position()

    x1, y1 = top_left
    x2, y2 = bottom_right
    width = x2 - x1
    height = y2 - y1

    screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    screenshot_np = np.array(screenshot)

    rows, cols = 9, 9
    cell_size = (width // cols, height // rows)

    return screenshot_np, top_left, cell_size

def classify_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 50:
        return 'mine'
    elif avg_brightness > 180:
        return 'covered_tile'
    elif 100 < avg_brightness < 180:
        text = pytesseract.image_to_string(gray, config='--psm 10 -c tessedit_char_whitelist=12345678')
        text = text.strip()
        if text.isdigit():
            return text
        else:
            return 'empty'
    else:
        return 'unknown'

def get_grid_state(screenshot, cell_size):
    rows, cols = 9, 9
    cw, ch = cell_size
    grid = []

    for row in range(rows):
        row_data = []
        for col in range(cols):
            x = col * cw
            y = row * ch
            cell_img = screenshot[y:y+ch, x:x+cw]
            label = classify_cell(cell_img)
            row_data.append(label)
        grid.append(row_data)
    return grid

def get_neighbors(grid, r, c):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append((nr, nc))
    return neighbors

def count_flagged(grid, neighbors):
    return sum(1 for (x, y) in neighbors if grid[x][y] == 'flag')

def is_flagged(grid, x, y):
    return grid[x][y] == 'flag'

def analyze_and_get_moves(grid):
    moves = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
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

def perform_click(x, y, click_type, top_left, cell_size):
    ox, oy = top_left
    cw, ch = cell_size
    cx = ox + x * cw + cw // 2
    cy = oy + y * ch + ch // 2

    if click_type == 'click':
        pyautogui.click(cx, cy, button='left')
    elif click_type == 'flag':
        pyautogui.click(cx, cy, button='right')

def main():
    print("[INFO] Minesweeper Bot Starting...")
    screenshot, top_left, cell_size = get_game_grid()

    print("[INFO] Detecting grid state...")
    grid = get_grid_state(screenshot, cell_size)

    print("[INFO] Grid:")
    for row in grid:
        print(row)

    moves = analyze_and_get_moves(grid)
    print(f"[INFO] Executing {len(moves)} moves...")

    for move in moves:
        perform_click(move['x'], move['y'], move['type'], top_left, cell_size)
        time.sleep(0.1)

    print("[DONE] Bot finished.")

if __name__ == "__main__":
    main()
