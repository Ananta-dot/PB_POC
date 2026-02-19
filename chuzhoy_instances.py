#!/usr/bin/env python3
# Build I_j per Appendix C step:
# I_1 = your 5-rect base.
# I_{j+1}: rotate(I_j) 90° CW about center, then add three rectangles:
#   R^{j+1}_1 (left):  intersects R^j_1 and R^j_2, and does NOT intersect R^j_v
#   R^{j+1}_2 (right): intersects R^j_3, and does NOT intersect R^j_v
#   R^{j+1}_3 (cap):   intersects ONLY R^{j+1}_1 and R^{j+1}_2 (no old rects)
#
# Requires: pip install z3-solver
# Optional: --plot to visualize.
from __future__ import annotations
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from z3 import Solver, Real, And, Or, Not, sat

Rect = Tuple[float,float,float,float]  # (x,y,w,h)

# ---------------- Base case (exactly as you drew) ----------------
def base_case() -> List[Rect]:
    rects = []
    # 1: left vertical
    rects.append((0.1, 0.1, 0.2, 0.6))
    # 2: top horizontal
    rects.append((0.2, 0.6, 0.6, 0.2))
    # 3: right vertical
    rects.append((0.7, 0.1, 0.2, 0.6))
    # 4: bottom-left horizontal
    rects.append((0.2, 0.2, 0.35, 0.2))
    # 5: bottom-right horizontal
    rects.append((0.45, 0.25, 0.35, 0.2))
    return rects

# ---------------- Geometry helpers ----------------
def bounds(R: Rect):
    x,y,w,h = R
    return (x, x+w, y, y+h)

def bbox(rects: List[Rect]):
    xs, ys = [], []
    for R in rects:
        x1,x2,y1,y2 = bounds(R)
        xs += [x1,x2]; ys += [y1,y2]
    return (min(xs), max(xs), min(ys), max(ys))

def rotate_90cw_about_center(R: Rect, cx=0.5, cy=0.5) -> Rect:
    x,y,w,h = R
    pts = [(x,y), (x+w,y), (x,y+h), (x+w,y+h)]
    rot = []
    for (px,py) in pts:
        dx, dy = px-cx, py-cy
        rx = cx + dy       # 90 CW
        ry = cy - dx
        rot.append((rx,ry))
    xs = [p[0] for p in rot]; ys = [p[1] for p in rot]
    x1,x2 = min(xs), max(xs); y1,y2 = min(ys), max(ys)
    return (x1, y1, x2-x1, y2-y1)

def rotate_all(rects: List[Rect], cx=0.5, cy=0.5):
    return [rotate_90cw_about_center(R, cx, cy) for R in rects]

def rects_intersect(A: Rect, B: Rect) -> bool:
    ax1,ax2,ay1,ay2 = bounds(A)
    bx1,bx2,by1,by2 = bounds(B)
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

# ---------------- Z3 helpers ----------------
def z3_box(prefix: str):
    x1 = Real(prefix+"_x1")
    y1 = Real(prefix+"_y1")
    w  = Real(prefix+"_w")
    h  = Real(prefix+"_h")
    x2 = Real(prefix+"_x2")
    y2 = Real(prefix+"_y2")
    return (x1,y1,w,h,x2,y2)

def z3_pack(x1,y1,w,h,x2,y2):
    return (x1,x2,y1,y2)

def Z3_intersects(a, b):
    ax1,ax2,ay1,ay2 = a
    bx1,bx2,by1,by2 = b
    return And(ax1 < bx2, bx1 < ax2, ay1 < by2, by1 < ay2)

def Z3_disjoint(a, b):
    return Not(Z3_intersects(a,b))

# ---------------- Add the 3 rectangles per paper ----------------
def add_three_paper(rects: List[Rect], rv_index: int) -> List[Rect]:
    """
    Given rotated I_j, add R1_new (left), R2_new (right), R3_new (cap) with:
      - R1_new intersects R1_old and R2_old; not Rv_old
      - R2_new intersects R3_old; not Rv_old
      - R3_new intersects only R1_new & R2_new (no old rects)
    """
    # old = current rotated I_j
    old = rects
    # indices (1-based) of the three special old rects
    R1 = old[0]; R2 = old[1]; R3 = old[2]
    Rv = old[rv_index-1]  # rv_index is 4 or 5 typically

    xmin,xmax,ymin,ymax = bbox(old)
    W = xmax - xmin
    H = ymax - ymin
    cx = 0.5*(xmin+xmax)

    # Z3 variables for new rectangles
    L = z3_box("L")     # R^{j+1}_1 (left post)
    R = z3_box("R")     # R^{j+1}_2 (right post)
    C = z3_box("C")     # R^{j+1}_3 (cap)
    (Lx1,Ly1,Lw,Lh,Lx2,Ly2) = L
    (Rx1,Ry1,Rw,Rh,Rx2,Ry2) = R
    (Cx1,Cy1,Cw,Ch,Cx2,Cy2) = C

    s = Solver()

    # define x2,y2
    for (x1,y1,w,h,x2,y2) in [L,R,C]:
        s.add(w > 0, h > 0, x2 == x1 + w, y2 == y1 + h)

    # keep in expanded canvas
    pad = 1.0
    s.add(Lx1 >= xmin - pad*W, Ly1 >= ymin - pad*H, Lx2 <= xmax + pad*W, Ly2 <= ymax + pad*H)
    s.add(Rx1 >= xmin - pad*W, Ry1 >= ymin - pad*H, Rx2 <= xmax + pad*W, Ry2 <= ymax + pad*H)
    s.add(Cx1 >= xmin - pad*W, Cy1 >= ymin - pad*H, Cx2 <= xmax + pad*W, Cy2 <= ymax + pad*H)

    # size hints
    s.add(And(Lw >= 0.06*W, Lw <= 0.25*W, Lh >= 0.35*H, Lh <= 1.20*H))
    s.add(And(Rw >= 0.06*W, Rw <= 0.25*W, Rh >= 0.35*H, Rh <= 1.20*H))
    s.add(And(Cw >= 0.40*W, Cw <= 1.40*W, Ch >= 0.08*H, Ch <= 0.40*H))

    # "left" and "right" placement that can still intersect old rects
    # Let posts slightly intrude into the old bbox so they can intersect.
    s.add(Lx1 <= xmin + 0.10*W, Lx2 <= xmin + 0.40*W)
    s.add(Rx1 >= xmax - 0.40*W, Rx2 >= xmax - 0.10*W)

    # cap near top to be "above" I_j but still able to intersect posts
    s.add(Cy1 >= ymax - 0.25*H)

    # Pack old rects for Z3
    def pack_old(Rr: Rect):
        x1,x2,y1,y2 = bounds(Rr)
        from z3 import RealVal
        return (RealVal(x1), RealVal(x2), RealVal(y1), RealVal(y2))
    R1o, R2o, R3o, Rvo = map(pack_old, [R1,R2,R3,Rv])

    Lb = (Lx1,Lx2,Ly1,Ly2)
    Rb = (Rx1,Rx2,Ry1,Ry2)
    Cb = (Cx1,Cx2,Cy1,Cy2)

    # Required intersections/non-intersections per paper:
    # Left post
    s.add(Z3_intersects(Lb, R1o))
    s.add(Z3_intersects(Lb, R2o))
    s.add(Z3_disjoint(Lb, Rvo))
    # Right post
    s.add(Z3_intersects(Rb, R3o))
    s.add(Z3_disjoint(Rb, Rvo))
    # Cap only with the two posts (no old rectangles)
    s.add(Z3_intersects(Cb, Lb))
    s.add(Z3_intersects(Cb, Rb))
    for Rold in [R1o,R2o,R3o,Rvo]:
        s.add(Z3_disjoint(Cb, Rold))

    # Ensure L and R do NOT intersect each other directly (chain goes via cap)
    s.add(Z3_disjoint(Lb, Rb))

    # Solve
    if s.check() != sat:
        raise RuntimeError("UNSAT under paper constraints (try different --rv: 4 or 5).")
    m = s.model()
    def val(v): return float(m[v].as_decimal(20).replace("?",""))
    L_rect = (val(Lx1), val(Ly1), val(Lw), val(Lh))
    R_rect = (val(Rx1), val(Ry1), val(Rw), val(Rh))
    C_rect = (val(Cx1), val(Cy1), val(Cw), val(Ch))
    return rects + [L_rect, R_rect, C_rect]

# ---------------- Builder ----------------
def build_Ij(j: int, rv_index: int) -> List[Rect]:
    if j < 1:
        raise ValueError("j must be >= 1")
    rects = base_case()
    if j == 1:
        return rects
    for _ in range(2, j+1):
        rects = rotate_all(rects, cx=0.5, cy=0.5)
        rects = add_three_paper(rects, rv_index=rv_index)
    return rects

# ---------------- Plot ----------------
def plot_rects(rects: List[Rect], title=""):
    fig, ax = plt.subplots(figsize=(9,8))
    for i,R in enumerate(rects, start=1):
        x,y,w,h = R
        ax.add_patch(patches.Rectangle((x,y), w,h, fill=False, linewidth=2))
        ax.text(x+0.5*w, y+0.5*h, f"{i}", ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.85))
    xs=[]; ys=[]
    for R in rects:
        x1,x2,y1,y2 = bounds(R); xs += [x1,x2]; ys += [y1,y2]
    dx = max(0.02, 0.06*(max(xs)-min(xs)))
    dy = max(0.02, 0.06*(max(ys)-min(ys)))
    ax.set_xlim(min(xs)-dx, max(xs)+dx)
    ax.set_ylim(min(ys)-dy, max(ys)+dy)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.set_title(title or "I_j per paper"); ax.set_xlabel("X"); ax.set_ylabel("Y")
    plt.tight_layout(); plt.show()

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", type=int, default=2, help="Build I_j (j>=1)")
    ap.add_argument("--rv", type=int, default=4, help="Index v in I_j that new posts must avoid (4 or 5 for I1)")
    ap.add_argument("--plot", action="store_true", help="Show the drawing")
    args = ap.parse_args()

    rects = build_Ij(args.j, rv_index=args.rv)
    if args.plot:
        plot_rects(rects, title=f"I_{args.j} (paper constraints; rv={args.rv})")

if __name__ == "__main__":
    main()
