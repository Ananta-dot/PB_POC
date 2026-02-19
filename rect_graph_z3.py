#!/usr/bin/env python3
# Robust rectangle-realization from an adjacency matrix using Z3.
# Phase 1: solve pairwise constraints only (edges => positive-area overlap, non-edges => disjoint).
# Phase 2: iteratively forbid only the triple-covered sample points actually violated.
# Optional: MaxSAT soft "no triple" constraints.

from __future__ import annotations
import argparse, json, math, sys, random
from typing import List, Tuple
import matplotlib.pyplot as plt
from z3 import Solver, Optimize, Int, And, Or, If, sat, Sum

# ======= default matrix (20x20) =======
ADJ_FLAT = """
0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0
"""

def parse_adj(flat: str) -> List[List[int]]:
    nums = [int(x) for x in flat.replace("\n", " ").split(",") if x.strip()]
    n = int(math.isqrt(len(nums)))
    if n * n != len(nums):
        raise ValueError("Adjacency length is not a perfect square")
    M = [nums[i*n:(i+1)*n] for i in range(n)]
    # make symmetric, clear diag
    for i in range(n):
        M[i][i] = 0
        for j in range(i+1, n):
            v = 1 if (M[i][j] or M[j][i]) else 0
            M[i][j] = M[j][i] = v
    return M

def build_vars(n, grid):
    x1 = [Int(f"x1_{i}") for i in range(n)]
    x2 = [Int(f"x2_{i}") for i in range(n)]
    y1 = [Int(f"y1_{i}") for i in range(n)]
    y2 = [Int(f"y2_{i}") for i in range(n)]
    return x1, x2, y1, y2

def base_constraints(slv, A, x1, x2, y1, y2, grid, min_wh, max_wh):
    n = len(A)

    # bounds & size
    for i in range(n):
        slv.add(0 <= x1[i], x1[i] < x2[i], x2[i] <= grid)
        slv.add(0 <= y1[i], y1[i] < y2[i], y2[i] <= grid)
        slv.add(x2[i] - x1[i] >= min_wh, x2[i] - x1[i] <= max_wh)
        slv.add(y2[i] - y1[i] >= min_wh, y2[i] - y1[i] <= max_wh)

    # Pairwise relations
    def intersects(i, j):
        return And(x1[i] < x2[j], x1[j] < x2[i],
                   y1[i] < y2[j], y1[j] < y2[i])  # positive area

    def disjoint(i, j):
        # separated on at least one axis (touching boundary allowed)
        return Or(x2[i] <= x1[j], x2[j] <= x1[i],
                  y2[i] <= y1[j], y2[j] <= y1[i])

    for i in range(n):
        for j in range(i+1, n):
            if A[i][j] == 1:
                slv.add(intersects(i, j))
            else:
                slv.add(disjoint(i, j))

def sampled_points(grid: int, samples_per_axis: int) -> List[Tuple[int,int]]:
    # uniform grid of sample points (cell centers-ish)
    if samples_per_axis < 2:
        samples_per_axis = 2
    step = grid // samples_per_axis
    if step < 1: step = 1
    half = max(1, step // 2)
    pts = [(x, y) for x in range(half, grid, step)
                  for y in range(half, grid, step)]
    return pts

def add_no_triple_soft(opt: Optimize, x1, x2, y1, y2, grid, samples_per_axis, weight="1"):
    pts = sampled_points(grid, samples_per_axis)
    for (gx, gy) in pts:
        cover = Sum([If(And(x1[i] <= gx, gx < x2[i], y1[i] <= gy, gy < y2[i]), 1, 0)
                     for i in range(len(x1))])
        opt.add_soft(cover <= 2, weight=weight)  # penalize triple+ covers
    return len(pts)

def draw(rects, out_png="rectangles.png"):
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(7,7))
    for idx, (x1, x2, y1, y2) in enumerate(rects, start=1):
        w, h = x2-x1, y2-y1
        ax.add_patch(patches.Rectangle((x1, y1), w, h, fill=False, linewidth=2, alpha=0.8))
        ax.text(x1+w/2, y1+h/2, str(idx), ha="center", va="center", fontsize=9)
    xs = [v for (x1,x2,y1,y2) in rects for v in (x1, x2)]
    ys = [v for (x1,x2,y1,y2) in rects for v in (y1, y2)]
    ax.set_xlim(min(xs)-2, max(xs)+2); ax.set_ylim(min(ys)-2, max(ys)+2)
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("Rectangle realization")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); print(f"saved: {out_png}")
    plt.show()

def extract_model(m, x1, x2, y1, y2):
    return [(m[x1[i]].as_long(), m[x2[i]].as_long(), m[y1[i]].as_long(), m[y2[i]].as_long())
            for i in range(len(x1))]

def count_triple_points(rects, grid, samples_per_axis):
    pts = sampled_points(grid, samples_per_axis)
    def inside(r, gx, gy):
        x1,x2,y1,y2 = r
        return (x1 <= gx < x2) and (y1 <= gy < y2)
    bad = []
    for p in pts:
        c = sum(1 for r in rects if inside(r, *p))
        if c >= 3: bad.append((p, c))
    return bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_file", type=str, default="",
                    help="optional CSV/flat file with n*n entries")
    ap.add_argument("--grid", type=int, default=120,
                    help="coordinate range [0..grid]")
    ap.add_argument("--min_wh", type=int, default=6,
                    help="min width/height")
    ap.add_argument("--max_wh", type=int, default=40,
                    help="max width/height")
    ap.add_argument("--samples", type=int, default=24,
                    help="samples per axis (for triple checks)")
    ap.add_argument("--max_iters", type=int, default=30,
                    help="max iterations of cutting triple points")
    ap.add_argument("--soft_triples", action="store_true",
                    help="use MaxSAT soft no-triple constraints instead of iterative cutting")
    ap.add_argument("--out", type=str, default="rectangles.png")
    ap.add_argument("--json", type=str, default="rectangles.json")
    args = ap.parse_args()

    # Parse matrix
    if args.matrix_file:
        with open(args.matrix_file, "r") as f:
            A = parse_adj(f.read())
    else:
        A = parse_adj(ADJ_FLAT)
    n = len(A)

    # -------- Phase 1: pairwise only --------
    x1, x2, y1, y2 = build_vars(n, args.grid)

    if args.soft_triples:
        opt = Optimize()
        base_constraints(opt, A, x1, x2, y1, y2, args.grid, args.min_wh, args.max_wh)
        add_no_triple_soft(opt, x1, x2, y1, y2, args.grid, args.samples, weight="1")
        if opt.check() != sat:
            print("UNSAT even with soft triple constraints. Try increasing --grid or --max_wh.")
            sys.exit(2)
        m = opt.model()
        rects = extract_model(m, x1, x2, y1, y2)

    else:
        s = Solver()
        base_constraints(s, A, x1, x2, y1, y2, args.grid, args.min_wh, args.max_wh)
        if s.check() != sat:
            print("UNSAT on pairwise constraints. Increase --grid or widen --max_wh, or enable --soft_triples.")
            sys.exit(2)
        m = s.model()
        rects = extract_model(m, x1, x2, y1, y2)

        # -------- Phase 2: iterative triple cutting --------
        it = 0
        while it < args.max_iters:
            bad = count_triple_points(rects, args.grid, args.samples)
            if not bad:
                break
            # add cuts only for the offending sample points
            for (gx, gy), _cnt in bad:
                cov = Sum([If(And(x1[i] <= gx, gx < x2[i], y1[i] <= gy, gy < y2[i]), 1, 0)
                           for i in range(n)])
                s.add(cov <= 2)
            if s.check() != sat:
                print(f"UNSAT after adding triple cuts on iter {it}. "
                      f"Try --grid bigger or use --soft_triples.")
                sys.exit(2)
            m = s.model()
            rects = extract_model(m, x1, x2, y1, y2)
            it += 1
        if it >= args.max_iters:
            print("Reached max iterations; remaining triple samples may exist.")

    # Save & plot
    import json as _json
    with open(args.json, "w") as f:
        _json.dump({i+1: {"x1":r[0], "x2":r[1], "y1":r[2], "y2":r[3]} for i,r in enumerate(rects)},
                   f, indent=2)
    print(f"saved: {args.json}")
    draw(rects, out_png=args.out)

if __name__ == "__main__":
    main()
