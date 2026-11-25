# Minesweeper Bot

A bot that plays Minesweeper **by looking at the screen**, like a person would. It never reads the
game's code or memory: it takes a screenshot of the window, finds the board, reads every cell
with computer vision, reasons about where the mines are, and clicks with the real mouse.

## How it works

1. **Find the board** (`vision.find_grid`): cells are separated from the darker gaps with a
   threshold, square contours of equal size are collected, and their centres are clustered into
   columns and rows. The board size (9×9, 16×16, 30×16 …) and position are detected, not given.
2. **Read the cells** (`vision.classify_cell`): a raised light tile is *covered* (a red glyph on it
   means *flag*); on an open cell, pixels that differ from the flat background in Lab colour space
   are the glyph, and its colour is matched to the number legend (1 blue, 2 green, 3 red …).
   Black discs are mines, a red cell is the exploded one.
3. **Reason** (`solver.py`): certain moves from single constraints and from subset relations
   between pairs of constraints. When nothing is certain, exact mine probabilities are computed for
   the frontier (each connected group is enumerated and weighted by the mines left for the rest of
   the board) and compared with the interior density; the safest cell is opened.
4. **Act** (`bot.py`): flags certain mines, opens safe cells, captures again, until it wins or loses.

On the hard board guessing is sometimes unavoidable, so it doesn't win every game; in testing it
won all easy and medium games and roughly 40% of hard ones (all losses were forced guesses).

## Try it

```bash
pip install -r requirements.txt
# open game/index.html in a browser window (Kolay / Orta / Zor), then:
python bot.py                 # finds the window titled "Mayın Tarlası"
python bot.py --delay 0.4 --debug out/   # slower, and saves what the bot sees
```

Move the mouse into a screen corner to stop it (pyautogui fail-safe).

`game/` is a small original Minesweeper made for the demo (first click is always safe).
`main.py` and `minesweeper_dynamic_bot.py` are the first versions (template matching on a fixed
9×9 board, corners picked by hand).
