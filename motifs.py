#!/usr/bin/env python3
# visualize_motifs.py
# Visualize MISR motif-induced rectangle sets with Matplotlib.

import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

Seq  = List[int]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2)) with x1<=x2, y1<=y2

# ---------------------------
# Motifs
# ---------------------------
def motif_rainbow(n: int) -> Seq:
    # [1,2,...,n, n,...,2,1]
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    # [1,1,2,2,...,n,n]
    out=[]
    for i in range(1, n+1):
        out += [i,i]
    return out

def motif_interleave(n: int) -> Seq:
    # [1,2,1,2, 3,4,3,4, ...] then trimmed to two of each label
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    # ensure exactly two per label
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_zipper(n: int) -> Seq:
    # [1,n,1,n, 2,n-1,2,n-1, ...]
    out=[]
    for i in range(1, (n//2)+1):
        j = n - i + 1
        out += [i, j, i, j]
    if n % 2 == 1:
        k = (n//2)+1
        out += [k,k]
    return out[:2*n]

def motif_ladder(n: int) -> Seq:
    # spreads endpoints to create corner stacks; then trimmed to 2 of each
    out=[]
    a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1
        b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}
    fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_repeat(n: int) -> Seq:
    # [1,2,...,n, 1,2,...,n]
    return list(range(1, n+1)) + list(range(1, n+1))

def motif_corner_combo(n: int):
    # H=rainbow, V=doubled (and the swapped version)
    return motif_rainbow(n), motif_doubled(n)

# ---------------------------
# Seq -> Rectangles utilities
# ---------------------------
def seq_spans(seq: Seq) -> List[Tuple[int,int]]:
    """Return [(l_i, r_i)] for labels i=1..n (0-based indices of occurrences)."""
    first = {}
    spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n+1)]

def build_rects(H: Seq, V: Seq) -> List[Rect]:
    X = seq_spans(H); Y = seq_spans(V)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

# ---------------------------
# Plotting
# ---------------------------
def plot_rects(ax, rects: List[Rect], title: str, label_fontsize: int = 8):
    # draw rectangles 1..n
    for i, ((x1,x2),(y1,y2)) in enumerate(rects, start=1):
        w = (x2 - x1); h = (y2 - y1)
        # outline rectangle
        r = plt.Rectangle((x1, y1), w, h, fill=False, linewidth=1.2)
        ax.add_patch(r)
        # label near center
        cx = x1 + w/2.0; cy = y1 + h/2.0
        ax.text(cx, cy, str(i), ha='center', va='center', fontsize=label_fontsize)

    # bounds & cosmetics
    xs = [x for r in rects for x in (r[0][0], r[0][1])]
    ys = [y for r in rects for y in (r[1][0], r[1][1])]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("H index", fontsize=9)
    ax.set_ylabel("V index", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle=':', linewidth=0.5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14, help="number of rectangles (labels 1..n)")
    ap.add_argument("--outfile", type=str, default="motifs_grid.png", help="output image file")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    n = args.n

    # Build motif pairs to visualize (H,V)
    motifs = [
        ("Rainbow / Rainbow",         motif_rainbow(n),    motif_rainbow(n)),
        ("Doubled / Doubled",         motif_doubled(n),    motif_doubled(n)),
        ("Interleave / Interleave",   motif_interleave(n), motif_interleave(n)),
        ("Zipper / Zipper",           motif_zipper(n),     motif_zipper(n)),
        ("Ladder / Ladder",           motif_ladder(n),     motif_ladder(n)),
        ("Repeat / Repeat",           motif_repeat(n),     motif_repeat(n)),
    ]

    # Corner-focused combos
    RH, RV = motif_corner_combo(n)
    motifs += [
        ("Corner A: Rainbow vs Doubled", RH, RV),
        ("Corner B: Doubled vs Rainbow", RV, RH),
    ]

    # Figure layout: 4 x 2 grid (8 panels)
    fig, axes = plt.subplots(4, 2, figsize=(10, 14))
    axes = axes.ravel()

    for ax, (name, H, V) in zip(axes, motifs):
        rects = build_rects(H, V)
        plot_rects(ax, rects, name)

    # If fewer panels than axes, hide extras (defensive)
    for k in range(len(motifs), len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=args.dpi)
    print(f"Saved: {args.outfile}")
    plt.show()

if __name__ == "__main__":
    main()
