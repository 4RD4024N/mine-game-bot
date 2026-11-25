"""Minesweeper reasoning on a board read by vision.py (no access to the game's state).

1. Certain moves: single-constraint rules and subset rules between pairs of constraints.
2. When nothing is certain: exact mine probabilities on the frontier (each connected group of
   frontier cells is enumerated with backtracking, weighted by how many mines are left for
   the rest of the board), compared with the density of the unseen interior. The safest cell
   is opened.
"""
from itertools import combinations
from math import comb

from vision import COVERED, FLAG, MINE

MAX_COMPONENT = 22   # larger frontier groups fall back to a local estimate


def neighbors(c, r, cols, rows):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dc or dr) and 0 <= c + dc < cols and 0 <= r + dr < rows:
                yield c + dc, r + dr


def constraints(board):
    rows, cols = len(board), len(board[0])
    out = []
    for r in range(rows):
        for c in range(cols):
            lab = board[r][c]
            if lab.isdigit() and lab != "0":
                unknown, flagged = [], 0
                for nc, nr in neighbors(c, r, cols, rows):
                    if board[nr][nc] == COVERED:
                        unknown.append((nc, nr))
                    elif board[nr][nc] == FLAG:
                        flagged += 1
                if unknown:
                    out.append((frozenset(unknown), int(lab) - flagged))
    return out


def certain_moves(board):
    """Returns (safe cells, mine cells) that follow with certainty."""
    cons = constraints(board)
    safe, mines = set(), set()
    for cells, need in cons:
        if need == 0:
            safe |= cells
        elif need == len(cells):
            mines |= cells
    if safe or mines:
        return safe, mines
    uniq = list(set(cons))
    for (a, na), (b, nb) in combinations(uniq, 2):
        for (s, ns), (t, nt) in (((a, na), (b, nb)), ((b, nb), (a, na))):
            if s < t:
                rest, k = t - s, nt - ns
                if k == 0:
                    safe |= rest
                elif k == len(rest):
                    mines |= rest
    return safe, mines


def _components(cons):
    cells = {}
    for i, (cs, _) in enumerate(cons):
        for cell in cs:
            cells.setdefault(cell, []).append(i)
    seen, comps = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, comp_cells, comp_cons = [start], set(), set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp_cells.add(x)
            for ci in cells[x]:
                comp_cons.add(ci)
                stack.extend(cons[ci][0])
        comps.append((sorted(comp_cells), [cons[i] for i in comp_cons]))
    return comps


def _enumerate(cells, cons):
    """For one frontier group: {mine count: (number of solutions, per-cell mine counts)}."""
    index = {c: i for i, c in enumerate(cells)}
    cons_i = [([index[c] for c in cs], need) for cs, need in cons]
    by_cell = [[] for _ in cells]
    for k, (cs, _) in enumerate(cons_i):
        for i in cs:
            by_cell[i].append(k)
    assign = [None] * len(cells)
    result = {}

    def ok(k):
        cs, need = cons_i[k]
        m = sum(1 for i in cs if assign[i] == 1)
        u = sum(1 for i in cs if assign[i] is None)
        return m <= need <= m + u

    def rec(i):
        if i == len(cells):
            m = sum(assign)
            cnt, per = result.get(m, (0, [0] * len(cells)))
            result[m] = (cnt + 1, [p + a for p, a in zip(per, assign)])
            return
        for v in (0, 1):
            assign[i] = v
            if all(ok(k) for k in by_cell[i]):
                rec(i + 1)
        assign[i] = None

    rec(0)
    return result


def probabilities(board, total_mines):
    """Mine probability for every covered cell (dict), using the remaining mine count."""
    rows, cols = len(board), len(board[0])
    covered = [(c, r) for r in range(rows) for c in range(cols) if board[r][c] == COVERED]
    flags = sum(row.count(FLAG) for row in board)
    left = max(0, total_mines - flags)
    cons = constraints(board)
    frontier = {cell for cs, _ in cons for cell in cs}
    interior = [x for x in covered if x not in frontier]
    probs = {}

    exact, rough = [], []
    for cells, ccons in _components(cons):
        (exact if len(cells) <= MAX_COMPONENT else rough).append((cells, ccons))
    # Rough groups: local ratio need/unknown averaged per cell.
    for cells, ccons in rough:
        for cell in cells:
            vals = [need / len(cs) for cs, need in ccons if cell in cs]
            probs[cell] = sum(vals) / len(vals)
    rough_mines = sum(probs.values())

    # Exact groups: combine their solution counts with the interior via binomial weights.
    tables = [(_enumerate(cells, ccons), cells) for cells, ccons in exact]
    dist = {0: 1}  # total frontier mines -> number of ways (convolution over groups)
    for table, _ in tables:
        new = {}
        for a, wa in dist.items():
            for m, (cnt, _) in table.items():
                new[a + m] = new.get(a + m, 0) + wa * cnt
        dist = new
    n_int = len(interior)
    budget = left - round(rough_mines)
    weight = {k: w * comb(n_int, budget - k) for k, w in dist.items() if 0 <= budget - k <= n_int}
    total_w = sum(weight.values()) or 1
    for table, cells in tables:
        # probability for each cell = sum over (this group's m, rest) of weights
        others = {0: 1}
        for t2, c2 in tables:
            if c2 is cells:
                continue
            new = {}
            for a, wa in others.items():
                for m, (cnt, _) in t2.items():
                    new[a + m] = new.get(a + m, 0) + wa * cnt
            others = new
        acc = [0.0] * len(cells)
        for m, (cnt, per) in table.items():
            w_rest = sum(wo * comb(n_int, budget - m - o) for o, wo in others.items() if 0 <= budget - m - o <= n_int)
            for i, p in enumerate(per):
                acc[i] += p * w_rest
        for i, cell in enumerate(cells):
            probs[cell] = acc[i] / total_w
    if interior:
        exp_frontier = sum(k * w for k, w in weight.items()) / total_w
        p_int = max(0.0, min(1.0, (budget - exp_frontier) / n_int))
        for cell in interior:
            probs[cell] = p_int
    return probs


def best_guess(board, total_mines):
    probs = probabilities(board, total_mines)
    if not probs:
        return None, None
    rows, cols = len(board), len(board[0])

    def key(cell):
        c, r = cell
        corner = (c in (0, cols - 1)) + (r in (0, rows - 1))   # corners open up more often
        return (round(probs[cell], 4), -corner)

    cell = min(probs, key=key)
    return cell, probs[cell]


def status(board, total_mines):
    flat = [x for row in board for x in row]
    if MINE in flat:
        return "lost"
    if flat.count(COVERED) + flat.count(FLAG) == total_mines:
        return "won"
    return "playing"
