#!/usr/bin/env python3
"""
Z3 rectangle realization with depth ≤ 2 from a target adjacency list.

- Rectangles i have integer coords (x1[i] < x2[i], y1[i] < y2[i]) within a grid [0..GRID].
- Intersection(i,j) means strict interior overlap on both axes:
      (x1[i] < x2[j] and x1[j] < x2[i]) and (y1[i] < y2[j] and y1[j] < y2[i]).
- Enforce adjacency: I(i,j) == (j in adj[i]).
- Enforce depth ≤ 2 by forbidding any triple i<j<k with pairwise intersections:
      not( I(i,j) and I(i,k) and I(j,k) )
  (Sufficient for rectangles due to Helly number 2 for axis-parallel boxes.)

Usage examples:
  # Use generator (n must be multiple of 4, e.g., 8, 12, 16):
  python3 z3_rect_depth2.py --n 16 --use_generated --plot --save rects_n16.png

  # Use an external JSON file with adjacency dict { "1":[...], "2":[...], ... }:
  python3 z3_rect_depth2.py --adj_json my_adj.json --plot

Requirements: z3-solver, matplotlib
"""

from __future__ import annotations
import argparse, json, math
from typing import Dict, List, Tuple, Set
from z3 import Solver, Int, And, Or, Not, If, sat
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Adj = Dict[int, List[int]]

# -----------------------------
# Optional generator (same family you used earlier)
# -----------------------------
def make_ring_chords_graph(n: int) -> Adj:
    """
    Ring edges + quarter-chords + a staggered chord family for n>=12.
    Requires n % 4 == 0.
    """
    if n % 4 != 0:
        raise ValueError("n must be a multiple of 4 for the built-in generator.")
    adj: Dict[int, Set[int]] = {i: set() for i in range(1, n+1)}

    def add(u, v):
        if u == v: return
        adj[u].add(v)
        adj[v].add(u)

    # ring
    for i in range(1, n+1):
        add(i, (i % n) + 1)

    # quarter-chords
    skip = n // 4
    for i in range(1, n+1):
        add(i, ((i - 1 + skip) % n) + 1)

    # second family for n>=12
    if n >= 12:
        skip2 = skip + 1
        for i in range(1, n+1):
            add(i, ((i - 1 + skip2) % n) + 1)

    return {u: sorted(vs) for u, vs in adj.items()}

# -----------------------------
# Utilities
# -----------------------------
def load_adjacency_from_json(path: str) -> Adj:
    with open(path, "r") as f:
        raw = json.load(f)
    # keys may be strings; normalize to int->sorted list[int]
    adj: Adj = {}
    for k, vs in raw.items():
        u = int(k)
        adj[u] = sorted(int(x) for x in vs if int(x) != u)
    # symmetrize
    nodes = sorted(adj.keys())
    for u in nodes:
        for v in adj[u]:
            if u not in adj.get(v, []):
                adj.setdefault(v, []).append(u)
    for u in adj:
        adj[u] = sorted(set(adj[u]))
    return adj

def normalize_adjacency(adj: Adj) -> Adj:
    # Ensure 1..n continuous labels and symmetric neighbors.
    nodes = sorted(adj.keys())
    n = max(nodes)
    for i in range(1, n+1):
        adj.setdefault(i, [])
    for u in range(1, n+1):
        adj[u] = sorted(set(v for v in adj[u] if v != u))
    for u in range(1, n+1):
        for v in adj[u]:
            if u not in adj[v]:
                adj[v].append(u)
                adj[v] = sorted(set(adj[v]))
    return adj

# -----------------------------
# Z3 model builder
# -----------------------------
def build_and_solve(adj: Adj,
                    grid: int = 50,
                    min_size: int = 1,
                    symbreak: bool = True):
    """
    adj: dict {1:[...], 2:[...], ...} (symmetric)
    grid: coordinates in [0..grid]
    min_size: minimum integer width/height per rectangle
    symbreak: optional symmetry breaking (x1[1] < x1[2] < ... < x1[n])
    """
    adj = normalize_adjacency(adj)
    n = max(adj.keys())
    S = Solver()

    # Variables
    x1 = {i: Int(f"x1_{i}") for i in range(1, n+1)}
    x2 = {i: Int(f"x2_{i}") for i in range(1, n+1)}
    y1 = {i: Int(f"y1_{i}") for i in range(1, n+1)}
    y2 = {i: Int(f"y2_{i}") for i in range(1, n+1)}

    for i in range(1, n+1):
        S.add(0 <= x1[i], x1[i] < x2[i], x2[i] <= grid)
        S.add(0 <= y1[i], y1[i] < y2[i], y2[i] <= grid)
        S.add(x2[i] - x1[i] >= min_size)
        S.add(y2[i] - y1[i] >= min_size)

    # Symmetry breaking: order by x1
    if symbreak:
        for i in range(1, n):
            S.add(x1[i] < x1[i+1])

    # Helper: strict 1D overlap
    def overlap_1d(a1, a2, b1, b2):
        # (a1 < b2) and (b1 < a2)
        return And(a1 < b2, b1 < a2)

    # Intersection predicate (formula, not a free Bool)
    def I(i, j):
        return And(
            overlap_1d(x1[i], x2[i], x1[j], x2[j]),
            overlap_1d(y1[i], y2[i], y1[j], y2[j]),
        )

    # Enforce the adjacency exactly
    for i in range(1, n+1):
        nbrs = set(adj[i])
        for j in range(1, n+1):
            if i >= j:
                continue
            if j in nbrs:
                # must intersect
                S.add(I(i, j))
            else:
                # must NOT intersect (non-overlap on at least one axis)
                S.add(Or(
                    x2[i] <= x1[j],
                    x2[j] <= x1[i],
                    y2[i] <= y1[j],
                    y2[j] <= y1[i],
                ))

    # Depth ≤ 2: forbid triple with pairwise intersections
    # For all i<j<k: NOT( I(i,j) & I(i,k) & I(j,k) )
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            for k in range(j+1, n+1):
                S.add(Not(And(I(i, j), I(i, k), I(j, k))))

    # Solve
    if S.check() != sat:
        return None

    M = S.model()
    rects = {}
    for i in range(1, n+1):
        rects[i] = (
            M[x1[i]].as_long(),
            M[y1[i]].as_long(),
            M[x2[i]].as_long(),
            M[y2[i]].as_long(),
        )
    return rects

# -----------------------------
# Plotting (optional)
# -----------------------------
def plot_rectangles(rects: Dict[int, Tuple[int,int,int,int]],
                    title: str = "",
                    save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (x1, y1, x2, y2) in rects.items():
        w = x2 - x1
        h = y2 - y1
        patch = patches.Rectangle((x1, y1), w, h, fill=False, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x1 + w/2, y1 + h/2, str(i), ha="center", va="center", fontsize=9)

    # bounds
    xs = [p[0] for p in rects.values()] + [p[2] for p in rects.values()]
    ys = [p[1] for p in rects.values()] + [p[3] for p in rects.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(1, int(0.05 * max(xmax - xmin, ymax - ymin)))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.grid(True, linestyle=':', linewidth=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj_json", type=str, default="",
                    help="Path to JSON adjacency dict { '1':[...], '2':[...], ... }")
    ap.add_argument("--n", type=int, default=16,
                    help="n for built-in generator (must be multiple of 4)")
    ap.add_argument("--use_generated", action="store_true",
                    help="Use the built-in ring+chords generator (ignores --adj_json)")
    ap.add_argument("--grid", type=int, default=50, help="grid size upper bound")
    ap.add_argument("--min_size", type=int, default=1, help="min width/height of rectangles")
    ap.add_argument("--no_symbreak", action="store_true", help="disable symmetry breaking")
    ap.add_argument("--plot", action="store_true", help="plot a solution if SAT")
    ap.add_argument("--save", type=str, default="", help="PNG path for the plot")
    args = ap.parse_args()

    if args.use_generated:
        adj = make_ring_chords_graph(args.n)
    elif args.adj_json:
        adj = load_adjacency_from_json(args.adj_json)
    else:
        # Minimal example to show format (edit as needed)
        # A tiny 8-node demo adjacency (must be symmetric):
        adj = {
            1:[2,8,3], 2:[1,3,4], 3:[2,4,1], 4:[3,5,6],
            5:[4,6,7], 6:[5,7,4], 7:[6,8,5], 8:[7,1]
        }

    print("Adjacency (normalized):")
    adj = normalize_adjacency(adj)
    for u in sorted(adj.keys()):
        print(f"{u:2d}: {adj[u]}")

    rects = build_and_solve(adj,
                            grid=args.grid,
                            min_size=args.min_size,
                            symbreak=not args.no_symbreak)

    if rects is None:
        print("\nUNSAT: No rectangle realization with depth ≤ 2 satisfies this adjacency.")
        return

    print("\nSAT: Rectangle coordinates (x1,y1,x2,y2):")
    for i in sorted(rects.keys()):
        print(f"{i:2d}: {rects[i]}")

    if args.plot:
        title = f"Rectangle realization (depth ≤ 2), n={len(rects)}"
        plot_rectangles(rects, title=title, save_path=(args.save or None))

if __name__ == "__main__":
    main()
