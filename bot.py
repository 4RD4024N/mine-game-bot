"""Minesweeper bot: sees the game only through screenshots and plays it with the real mouse.

    python bot.py                      # finds the window titled "Mayın Tarlası"
    python bot.py --title "Minesweeper" --mines 40 --debug out/

Loop: capture the window -> find the grid (first frame) -> read every cell (vision.py)
-> certain moves, or the safest guess (solver.py) -> click -> repeat until won or lost.
"""
import argparse
import ctypes
import os
import time
from ctypes import wintypes

import cv2
import mss
import numpy as np
import pyautogui

import solver
import vision

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True          # slam the mouse into a screen corner to stop the bot
STANDARD_MINES = {(9, 9): 10, (16, 16): 40, (30, 16): 99, (16, 30): 99}

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # physical pixels for capture and clicks
except Exception:
    user32.SetProcessDPIAware()


def find_window(title_part):
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part.lower() in buf.value.lower():
                found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return None
    hwnd = found[0]
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    user32.SetForegroundWindow(hwnd)
    return {"left": rect.left, "top": rect.top, "width": rect.right - rect.left, "height": rect.bottom - rect.top}


def grab(sct, region):
    return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)


def click(region, grid, cell, button="left", speed=0.06):
    x, y = grid.center(*cell)
    pyautogui.moveTo(region["left"] + x, region["top"] + y, duration=speed)
    pyautogui.click(button=button)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Mayın Tarlası")
    ap.add_argument("--mines", type=int, help="total mines (default: standard count for the detected size)")
    ap.add_argument("--speed", type=float, default=0.06, help="mouse travel time per move (s)")
    ap.add_argument("--delay", type=float, default=0.12, help="pause after each turn (s); raise it to watch the bot think")
    ap.add_argument("--debug", help="folder to save what the bot sees")
    args = ap.parse_args()

    region = find_window(args.title)
    if region is None:
        raise SystemExit(f"'{args.title}' başlıklı pencere bulunamadı")
    time.sleep(0.3)
    with mss.MSS() if hasattr(mss, 'MSS') else mss.mss() as sct:
        img = grab(sct, region)
        grid = vision.find_grid(img)
        if grid is None:
            raise SystemExit("Tahta bulunamadı")
        total = args.mines or STANDARD_MINES.get((grid.cols, grid.rows), round(grid.cols * grid.rows * 0.16))
        print(f"[görüntü] tahta {grid.cols}x{grid.rows}, hücre {grid.cell:.0f}px, {total} mayın")
        if args.debug:
            os.makedirs(args.debug, exist_ok=True)

        turn, guesses, t0, retries = 0, 0, time.time(), 0
        while True:
            img = grab(sct, region)
            board = vision.read_board(img, grid)
            state = solver.status(board, total)
            if state != "playing":
                print(f"[sonuç] {'KAZANDI' if state == 'won' else 'KAYBETTİ'} · {turn} tur, {guesses} tahmin, {time.time() - t0:.1f} sn")
                if args.debug:
                    cv2.imencode(".png", vision.draw_overlay(img, grid, board))[1].tofile(os.path.join(args.debug, "son.png"))
                return state
            turn += 1
            flat = [x for row in board for x in row]
            if flat.count(vision.COVERED) == grid.cols * grid.rows:
                safe, mines, note = {(grid.cols // 2, grid.rows // 2)}, set(), "ilk hamle: ortadan"
            else:
                safe, mines = solver.certain_moves(board)
                note = f"kesin: {len(safe)} güvenli, {len(mines)} mayın"
                if not safe and not mines:
                    cell, p = solver.best_guess(board, total)
                    if cell is None and retries < 5:
                        # an inconsistent read (frame captured mid-repaint or something over the window): look again
                        retries += 1; time.sleep(0.4); turn -= 1
                        continue
                    if cell is None:
                        print("[uyarı] hamle yok")
                        if args.debug:
                            cv2.imencode(".png", vision.draw_overlay(img, grid, board))[1].tofile(os.path.join(args.debug, "takildi.png"))
                            cv2.imencode(".png", img)[1].tofile(os.path.join(args.debug, "takildi-ham.png"))
                        return "stuck"
                    safe, guesses = {cell}, guesses + 1
                    note = f"tahmin: {cell} mayın olasılığı %{p * 100:.1f}"
            retries = 0
            print(f"[tur {turn}] {note}")
            if args.debug and turn in (1, 3, 8, 15, 30):
                hl = {**{c: (80, 220, 80) for c in safe}, **{c: (60, 60, 255) for c in mines}}
                cv2.imencode(".png", vision.draw_overlay(img, grid, board, hl))[1].tofile(os.path.join(args.debug, f"tur-{turn:02d}.png"))
            for cell in sorted(mines):
                click(region, grid, cell, "right", args.speed)
            for cell in sorted(safe, key=lambda c: (c[1], c[0])):
                click(region, grid, cell, "left", args.speed)
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
