#!/usr/bin/env python3
# stabbing_gap_maximizer.py
# Maximize ILP/LP integrality gap for rectangle "stabbing" by horizontal/vertical lines.

from __future__ import annotations
import argparse
import math
import os
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

Rect = Tuple[int, int, int, int]  # (x1, x2, y1, y2), inclusive integer coordinates


# -----------------------
# Utility / RNG helpers
# -----------------------
def set_seed(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    return rng


# -----------------------
# Instance generators
# -----------------------
def gen_unit_squares(
    n: int,
    width: int,
    coord_max: int,
    rng: random.Random,
    mode: str = "stagger",
    grid_step: Optional[int] = None,
) -> List[Rect]:
    """
    Generate n axis-aligned unit squares (side = width) within [0, coord_max] x [0, coord_max].

    modes:
      - "random": place each square uniformly on a grid (optionally grid_step); clamp inside box.
      - "grid"  : place on approx sqrt(n) x sqrt(n) lattice, spaced to fit tightly.
      - "stagger": staircase/diagonal layout to force both orientations to matter.

    Coordinates are integers. Squares are (x1,x1+width,y1,y1+width).
    """
    rects: List[Rect] = []

    def place(x, y):
        x1 = max(0, min(coord_max - width, x))
        y1 = max(0, min(coord_max - width, y))
        return (x1, x1 + width, y1, y1 + width)

    if mode == "grid":
        k = int(math.ceil(math.sqrt(n)))
        gap = max(1, (coord_max - width) // max(1, k - 1))
        pts = []
        for i in range(k):
            for j in range(k):
                if len(pts) >= n:
                    break
                x = i * gap
                y = j * gap
                pts.append((x, y))
            if len(pts) >= n:
                break
        for (x, y) in pts:
            rects.append(place(x, y))

    elif mode == "stagger":
        # A diagonal staircase that tends to require a mix of H and V lines.
        # Spread along the diagonal with jitter.
        stride = max(1, (coord_max - width) // max(1, n - 1))
        for i in range(n):
            base = i * stride
            jx = rng.randint(0, max(0, stride // 3)) if stride > 2 else 0
            jy = rng.randint(0, max(0, stride // 3)) if stride > 2 else 0
            rects.append(place(base + jx, base + jy))

    elif mode == "random":
        if grid_step is None:
            grid_step = max(1, width // 2)
        xs = list(range(0, max(1, coord_max - width + 1), grid_step))
        ys = list(range(0, max(1, coord_max - width + 1), grid_step))
        for _ in range(n):
            x = rng.choice(xs)
            y = rng.choice(ys)
            rects.append(place(x, y))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return rects


def jitter_instance(
    rects: List[Rect],
    coord_max: int,
    width: int,
    rng: random.Random,
    per_square_jitter: int = 2,
    p_big_move: float = 0.1,
) -> List[Rect]:
    """
    Produce a neighbor instance by randomly moving a few squares by a small integer delta.
    """
    out = list(rects)
    n = len(out)
    k = max(1, n // 4)  # move ~25% of squares
    idxs = rng.sample(range(n), k=k)
    for i in idxs:
        (x1, x2, y1, y2) = out[i]
        dx = rng.randint(-per_square_jitter, per_square_jitter)
        dy = rng.randint(-per_square_jitter, per_square_jitter)
        if rng.random() < p_big_move:
            # occasional larger move
            dx += rng.randint(-2 * per_square_jitter, 2 * per_square_jitter)
            dy += rng.randint(-2 * per_square_jitter, 2 * per_square_jitter)
        nx1 = max(0, min(coord_max - width, x1 + dx))
        ny1 = max(0, min(coord_max - width, y1 + dy))
        out[i] = (nx1, nx1 + width, ny1, ny1 + width)
    return out


def swap_two(rects: List[Rect], rng: random.Random) -> List[Rect]:
    """Swap positions of two squares (center swap), for variety."""
    out = list(rects)
    if len(out) < 2:
        return out
    i, j = rng.sample(range(len(out)), 2)
    (x1, x2, y1, y2) = out[i]
    (X1, X2, Y1, Y2) = out[j]
    w = x2 - x1
    W = X2 - X1
    # swap upper-left corners but keep widths
    out[i] = (X1, X1 + w, Y1, Y1 + w)
    out[j] = (x1, x1 + W, y1, y1 + W)
    return out


# -----------------------
# Stabbing LP/ILP
# -----------------------
def candidate_lines(rects: List[Rect]) -> Tuple[List[int], List[int]]:
    """
    Choose candidate vertical and horizontal lines.
    Safe and simple choice: ALL unique x1 and x2 (same for y1,y2).
    This ensures feasibility (there's always a line on each rectangle boundary).
    """
    xs: set[int] = set()
    ys: set[int] = set()
    for (x1, x2, y1, y2) in rects:
        xs.add(x1)
        xs.add(x2)
        ys.add(y1)
        ys.add(y2)
    return sorted(xs), sorted(ys)


def solve_stabbing_ilp_lp(
    rects: List[Rect],
    threads: int = 0,
    relax_lp_from_ilp: bool = True,
) -> Tuple[float, float]:
    """
    Solve the covering (stabbing) problem for the given rectangles:

      - Variables: V_i for vertical lines at x=xs[i], H_j for horizontal at y=ys[j].
      - Constraint per rect r: sum(V_i for x1<=xs[i]<=x2) + sum(H_j for y1<=ys[j]<=y2) >= 1
      - Objective: minimize sum(V_i) + sum(H_j)

    Returns (ILP_obj_value, LP_obj_value).
    """
    xs, ys = candidate_lines(rects)

    # ILP
    m_ilp = gp.Model("stab_ilp")
    m_ilp.setParam("OutputFlag", 0)
    if threads > 0:
        m_ilp.setParam("Threads", threads)

    V = m_ilp.addVars(len(xs), vtype=GRB.BINARY, name="V")
    H = m_ilp.addVars(len(ys), vtype=GRB.BINARY, name="H")
    for (x1, x2, y1, y2) in rects:
        x_idx = [i for i, x in enumerate(xs) if x1 <= x <= x2]
        y_idx = [j for j, y in enumerate(ys) if y1 <= y <= y2]
        if not x_idx and not y_idx:
            # If this happens, our candidate lines missed the rect -> add its right/top edges ad hoc
            # (Shouldn't happen with our set, but safe-guard anyway)
            if x2 not in xs:
                xs.append(x2)
            if y2 not in ys:
                ys.append(y2)
            xs.sort()
            ys.sort()
            return solve_stabbing_ilp_lp(rects, threads=threads, relax_lp_from_ilp=relax_lp_from_ilp)
        m_ilp.addConstr(gp.quicksum(V[i] for i in x_idx) + gp.quicksum(H[j] for j in y_idx) >= 1)
    m_ilp.setObjective(V.sum() + H.sum(), GRB.MINIMIZE)
    m_ilp.optimize()
    if m_ilp.status != GRB.OPTIMAL:
        return (0.0, 0.0)
    ilp = float(m_ilp.objVal)

    # LP
    if relax_lp_from_ilp:
        m_lp = m_ilp.relax()
        # Carry over Threads to safety:
        if threads > 0:
            try:
                m_lp.setParam("Threads", threads)
            except gp.GurobiError:
                pass
    else:
        m_lp = gp.Model("stab_lp")
        m_lp.setParam("OutputFlag", 0)
        if threads > 0:
            m_lp.setParam("Threads", threads)
        Vc = m_lp.addVars(len(xs), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="V")
        Hc = m_lp.addVars(len(ys), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="H")
        for (x1, x2, y1, y2) in rects:
            x_idx = [i for i, x in enumerate(xs) if x1 <= x <= x2]
            y_idx = [j for j, y in enumerate(ys) if y1 <= y <= y2]
            m_lp.addConstr(gp.quicksum(Vc[i] for i in x_idx) + gp.quicksum(Hc[j] for j in y_idx) >= 1)
        m_lp.setObjective(Vc.sum() + Hc.sum(), GRB.MINIMIZE)

    m_lp.setParam("OutputFlag", 0)
    m_lp.optimize()
    if m_lp.status != GRB.OPTIMAL:
        return (ilp, 0.0)
    lp = float(m_lp.objVal)
    return (ilp, lp)


def ilp_lp_ratio(rects: List[Rect], threads: int = 0) -> float:
    ilp, lp = solve_stabbing_ilp_lp(rects, threads=threads)
    if lp > 1e-12:
        return ilp / lp
    return 0.0


# -----------------------
# Local search loop
# -----------------------
def search_max_gap(
    init_rects: List[Rect],
    threads: int,
    rng: random.Random,
    time_budget_s: float = 30.0,
    neighbor_k: int = 40,
    per_square_jitter: int = 2,
) -> Tuple[float, List[Rect]]:
    """
    Simple stochastic local search with greedy + occasional accept-equal policy.
    """
    start = time.time()
    best_rects = list(init_rects)
    best_score = ilp_lp_ratio(best_rects, threads=threads)

    cur_rects = list(best_rects)
    cur_score = best_score

    while time.time() - start < time_budget_s:
        improved = False
        # propose a batch of neighbors
        for _ in range(neighbor_k):
            if rng.random() < 0.15:
                nbr = swap_two(cur_rects, rng)
            else:
                nbr = jitter_instance(cur_rects, coord_max=max(r[1] for r in cur_rects), width=cur_rects[0][1]-cur_rects[0][0], rng=rng,
                                      per_square_jitter=per_square_jitter, p_big_move=0.15)
            score = ilp_lp_ratio(nbr, threads=threads)
            if score > best_score + 1e-9:
                best_score = score
                best_rects = nbr
                cur_rects = nbr
                cur_score = score
                improved = True
            elif score > cur_score - 1e-12:
                # side-step (keeps exploration going)
                cur_rects = nbr
                cur_score = score
        if not improved:
            # small random restart around best
            cur_rects = jitter_instance(best_rects, coord_max=max(r[1] for r in best_rects), width=best_rects[0][1]-best_rects[0][0],
                                        rng=rng, per_square_jitter=per_square_jitter, p_big_move=0.25)
            cur_score = ilp_lp_ratio(cur_rects, threads=threads)
    return best_score, best_rects


# -----------------------
# Pretty-print & I/O
# -----------------------
def summarize(rects: List[Rect]) -> str:
    return "; ".join(f"({x1},{y1})–({x2},{y2})" for (x1, x2, y1, y2) in rects)


def save_instance(rects: List[Rect], path: str):
    import json
    data = [{"x1":x1, "x2":x2, "y1":y1, "y2":y2} for (x1,x2,y1,y2) in rects]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_instance(path: str) -> List[Rect]:
    import json
    with open(path, "r") as f:
        data = json.load(f)
    rects: List[Rect] = []
    for d in data:
        rects.append((int(d["x1"]), int(d["x2"]), int(d["y1"]), int(d["y2"])))
    return rects


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Maximize ILP/LP gap for rectangle stabbing.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n", type=int, default=6, help="number of unit squares")
    ap.add_argument("--width", type=int, default=6, help="square side length (integer)")
    ap.add_argument("--coord_max", type=int, default=None,
                    help="max coordinate (inclusive) for lower-left corner + width; default: (n+1)*n - width")
    ap.add_argument("--mode", type=str, default="stagger", choices=["stagger","grid","random"])
    ap.add_argument("--grid_step", type=int, default=None, help="step for random mode grid")
    ap.add_argument("--threads", type=int, default=8, help="Gurobi threads")
    ap.add_argument("--time_s", type=float, default=30.0, help="search time budget (seconds)")
    ap.add_argument("--neighbor_k", type=int, default=40, help="neighbors per iteration")
    ap.add_argument("--jitter", type=int, default=2, help="per-square jitter in coords")
    ap.add_argument("--save_best", type=str, default="", help="optional path to save best instance JSON")
    ap.add_argument("--load_instance", type=str, default="", help="optional path to load an instance instead of generating")
    args = ap.parse_args()

    rng = set_seed(args.seed)

    if args.coord_max is None:
        # Default from your Keras script: COORD_MAX = (n+1)*n - SQUARE_WIDTH
        coord_max = (args.n + 1) * args.n - args.width
    else:
        coord_max = args.coord_max

    if args.load_instance:
        rects = load_instance(args.load_instance)
    else:
        rects = gen_unit_squares(n=args.n, width=args.width, coord_max=coord_max, rng=rng,
                                 mode=args.mode, grid_step=args.grid_step)

    # Baseline ratio
    ilp0, lp0 = solve_stabbing_ilp_lp(rects, threads=args.threads)
    ratio0 = ilp0 / lp0 if lp0 > 1e-12 else 0.0
    print(f"[init] ILP={ilp0:.3f}  LP={lp0:.3f}  ratio={ratio0:.4f}")
    print(f"[init] rects: {summarize(rects)}")

    # Search
    best_ratio, best_rects = search_max_gap(
        init_rects=rects,
        threads=args.threads,
        rng=rng,
        time_budget_s=args.time_s,
        neighbor_k=args.neighbor_k,
        per_square_jitter=args.jitter,
    )
    ilpB, lpB = solve_stabbing_ilp_lp(best_rects, threads=args.threads)
    print("\n=== BEST FOUND ===")
    print(f"ILP={ilpB:.3f}  LP={lpB:.3f}  ratio={best_ratio:.6f}")
    print(f"rects: {summarize(best_rects)}")

    if args.save_best:
        save_instance(best_rects, args.save_best)
        print(f"Saved best instance to {args.save_best}")


if __name__ == "__main__":
    main()
