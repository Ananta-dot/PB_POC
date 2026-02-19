#!/usr/bin/env python3
# motif_chuzhoy.py
# Deterministic Chuzhoy-style instances with Z3.
# Encoding: (x, y, w, h) where (x,y)=bottom-left, all in [0,1].

from __future__ import annotations
from typing import Dict, Tuple, List
import argparse

from z3 import (
    Solver, Real, RealVal, And, Or, Not, sat, BoolRef
)

Rect = Tuple[Real, Real, Real, Real]                # (x,y,w,h)
RVal = Tuple[float, float, float, float]            # concrete values

# ---------- primitives ----------
def mk_rect(name: str, min_w=0.06, min_h=0.06) -> Tuple[Rect, List[BoolRef]]:
    x, y, w, h = Real(f"{name}_x"), Real(f"{name}_y"), Real(f"{name}_w"), Real(f"{name}_h")
    cons = [x >= 0, y >= 0, w >= min_w, h >= min_h, x + w <= 1, y + h <= 1]
    return (x, y, w, h), cons

def in_range(r: Rect, xr: Tuple[float,float]|None=None, yr: Tuple[float,float]|None=None,
             wr: Tuple[float,float]|None=None, hr: Tuple[float,float]|None=None) -> List[BoolRef]:
    x,y,w,h = r
    out: List[BoolRef] = []
    if xr: out += [x >= xr[0], x <= xr[1]]
    if yr: out += [y >= yr[0], y <= yr[1]]
    if wr: out += [w >= wr[0], w <= wr[1]]
    if hr: out += [h >= hr[0], h <= hr[1]]
    return out

def intersect(a: Rect, b: Rect) -> BoolRef:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return And(ax < bx + bw, bx < ax + aw, ay < by + bh, by < ay + ah)

def disjoint(a: Rect, b: Rect) -> BoolRef:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return Or(ax >= bx + bw, bx >= ax + aw, ay >= by + bh, by >= ay + ah)

def center_x(a: Rect): x,y,w,h = a; return x + w/2
def center_y(a: Rect): x,y,w,h = a; return y + h/2

def to_left(a: Rect, b: Rect) -> BoolRef:  return center_x(a) < center_x(b)
def to_right(a: Rect, b: Rect) -> BoolRef: return center_x(a) > center_x(b)

def top_of_group(a: Rect, group: List[Rect]) -> BoolRef:
    ay1, ay2 = a[1], a[1] + a[3]
    return And(*[And(ay1 > g[1], ay2 > g[1] + g[3]) for g in group])

def const_rect(vals: RVal) -> Rect:
    x,y,w,h = vals; return (RealVal(x), RealVal(y), RealVal(w), RealVal(h))

def model_as_tuple(m, r: Rect) -> RVal:
    return tuple(float(m[v].as_decimal(12).replace("?", "")) for v in r)  # type: ignore

def rotate90_about_center(v: RVal, pivot=(0.5,0.5)) -> RVal:
    x,y,w,h = v
    cx, cy = pivot
    pts = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
    rot = []
    for px,py in pts:
        dx,dy = px-cx, py-cy
        rx, ry = cx+dy, cy-dx  # 90° CW
        rot.append((rx,ry))
    xs = [p[0] for p in rot]; ys=[p[1] for p in rot]
    nx, ny = min(xs), min(ys)
    nw, nh = max(xs)-nx, max(ys)-ny
    return (nx, ny, nw, nh)

# ---------- I1 : 5-rectangle base loop ----------
# 1: leftmost vertical, 2: top horizontal, 3: rightmost vertical,
# 4: between 2 and 5; intersects only {2,5}
# 5: between 2 and 3; intersects only {3,4}
def solve_I1() -> Dict[int, RVal]:
    s = Solver()
    R: Dict[int, Rect] = {}
    C: List[BoolRef] = []

    R[1], c1 = mk_rect("r1", min_w=0.07, min_h=0.45)
    R[2], c2 = mk_rect("r2", min_w=0.50, min_h=0.10)
    R[3], c3 = mk_rect("r3", min_w=0.07, min_h=0.45)
    R[4], c4 = mk_rect("r4", min_w=0.18, min_h=0.10)
    R[5], c5 = mk_rect("r5", min_w=0.18, min_h=0.10)
    C += c1 + c2 + c3 + c4 + c5

    # Loose ranges to make the pattern easy to satisfy
    C += in_range(R[1], xr=(0.05,0.15), yr=(0.10,0.20), wr=(0.07,0.12), hr=(0.55,0.70))
    C += in_range(R[2], xr=(0.20,0.30), yr=(0.75,0.85), wr=(0.55,0.70), hr=(0.10,0.15))
    C += in_range(R[3], xr=(0.80,0.90), yr=(0.10,0.20), wr=(0.07,0.12), hr=(0.55,0.70))
    C += in_range(R[4], xr=(0.35,0.55), yr=(0.30,0.45), wr=(0.20,0.28), hr=(0.10,0.16))
    C += in_range(R[5], xr=(0.55,0.72), yr=(0.32,0.48), wr=(0.20,0.28), hr=(0.10,0.16))

    # Incidences:
    # 4 intersects only {2,5}
    C += [intersect(R[4], R[2]), intersect(R[4], R[5])]
    for k in [1,3]:
        C += [disjoint(R[4], R[k])]

    # 5 intersects only {3,4}
    C += [intersect(R[5], R[3]), intersect(R[5], R[4])]
    for k in [1,2]:
        C += [disjoint(R[5], R[k])]

    # Leftmost / rightmost / top relations
    C += [to_left(R[1], R[2]), to_left(R[1], R[3])]
    C += [to_right(R[3], R[1]), to_right(R[3], R[2])]
    C += [top_of_group(R[2], [R[1], R[3], R[4], R[5]])]

    s.add(*C)
    assert s.check() == sat, "I1 UNSAT with the given specification."
    m = s.model()
    return {i: model_as_tuple(m, R[i]) for i in range(1,6)}

# ---------- add (6,7,8) after flip ----------
# 6 intersects {1,3,7}; disjoint from {2,5,4,8}; to_left(6,7)
# 7 intersects only {6,8} and is top above 6 & 8
# 8 intersects {2,7} only; to_right(8,2)
def solve_add_6_7_8_after_flip(I1: Dict[int,RVal]) -> Dict[int,RVal]:
    R1 = {i: rotate90_about_center(I1[i]) for i in I1}  # rotate all five
    s = Solver()

    # constants
    RC: Dict[int, Rect] = {i: const_rect(R1[i]) for i in R1}

    # new
    r6, c6 = mk_rect("r6", min_w=0.08, min_h=0.12)
    r7, c7 = mk_rect("r7", min_w=0.10, min_h=0.18)
    r8, c8 = mk_rect("r8", min_w=0.08, min_h=0.12)
    s.add(*(c6 + c7 + c8))

    # coarse placement ranges to help satisfiability
    s.add(*in_range(r7, yr=(0.70,0.88)))      # 7 high
    s.add(*in_range(r6, xr=(0.10,0.35)))      # 6 more to the left
    s.add(*in_range(r8, xr=(0.70,0.92)))      # 8 more to the right

    # constraints from your spec
    s.add(intersect(r6, RC[1]))
    s.add(intersect(r6, RC[3]))
    s.add(intersect(r6, r7))
    for k in [2,4,5,8]:  # ensure no unintended
        pass
    s.add(disjoint(r6, RC[2]))
    s.add(disjoint(r6, RC[4]))
    s.add(disjoint(r6, RC[5]))
    s.add(disjoint(r6, r8))
    s.add(to_left(r6, r7))

    # 7: only intersects 6 and 8; and is above both 6 & 8
    s.add(intersect(r7, r6))
    s.add(intersect(r7, r8))
    for k in [1,2,3,4,5]:
        s.add(disjoint(r7, RC[k]))
    s.add(Not(intersect(r7, r6)) == False)  # keep Z3 honest (already true)
    s.add(Not(intersect(r7, r8)) == False)
    s.add(And(r7[1] > r6[1], r7[1] + r7[3] > r6[1] + r6[3]))
    s.add(And(r7[1] > r8[1], r7[1] + r7[3] > r8[1] + r8[3]))

    # 8: intersects {2,7} only; to_right(8,2)
    s.add(intersect(r8, RC[2]))
    s.add(intersect(r8, r7))
    s.add(to_right(r8, RC[2]))
    for k in [1,3,4,5]:  # and not with 6 already ensured above
        s.add(disjoint(r8, RC[k]))

    assert s.check() == sat, "I2 (flip + add 6,7,8) UNSAT."
    m = s.model()

    out = {i: R1[i] for i in R1}
    out[6] = model_as_tuple(m, r6)
    out[7] = model_as_tuple(m, r7)
    out[8] = model_as_tuple(m, r8)
    return out

# ---------- add (9,10,11) after another flip ----------
# 9 intersects {6,8} and is to their right; not others
# 10 is top above {9,11}; intersects 11; not others
# 11 intersects {8,10}; to_right(11,8); not others
def solve_add_9_10_11_after_flip(I2: Dict[int,RVal]) -> Dict[int,RVal]:
    R2 = {i: rotate90_about_center(I2[i]) for i in I2}
    s = Solver()
    RC: Dict[int, Rect] = {i: const_rect(R2[i]) for i in R2}

    r9,  c9  = mk_rect("r9",  min_w=0.08, min_h=0.12)
    r10, c10 = mk_rect("r10", min_w=0.10, min_h=0.16)
    r11, c11 = mk_rect("r11", min_w=0.08, min_h=0.12)
    s.add(*(c9 + c10 + c11))

    # ranges to encourage layout
    s.add(*in_range(r9,  xr=(0.65,0.92)))   # 9 to the right
    s.add(*in_range(r10, yr=(0.70,0.90)))   # 10 high
    s.add(*in_range(r11, xr=(0.60,0.88)))   # 11 also on right-ish

    # 9: intersects 6 and 8; to their right; not others
    s.add(intersect(r9, RC[6]))
    s.add(intersect(r9, RC[8]))
    s.add(to_right(r9, RC[6]))
    s.add(to_right(r9, RC[8]))
    for k in [1,2,3,4,5,7]:
        s.add(disjoint(r9, RC[k]))
    # keep away from 10 & 11 (as per your phrasing)
    s.add(disjoint(r9, r10))
    s.add(disjoint(r9, r11))

    # 11: intersects 8 and 10; to_right(11,8); not others
    s.add(intersect(r11, RC[8]))
    s.add(to_right(r11, RC[8]))
    s.add(intersect(r11, r10))
    for k in [1,2,3,4,5,6,7]:
        s.add(disjoint(r11, RC[k]))

    # 10: top above 9 and 11; intersects 11; not others
    s.add(intersect(r10, r11))
    s.add(top_of_group(r10, [r9, r11]))
    for k in [1,2,3,4,5,6,7,8]:
        s.add(disjoint(r10, RC[k]))

    assert s.check() == sat, "I3 (flip + add 9,10,11) UNSAT."
    m = s.model()
    out = {i: R2[i] for i in R2}
    out[9]  = model_as_tuple(m, r9)
    out[10] = model_as_tuple(m, r10)
    out[11] = model_as_tuple(m, r11)
    return out

# ---------- driver ----------
def build_Ij(j: int) -> Dict[int,RVal]:
    if j < 1 or j > 3:
        raise ValueError("Supported j: 1, 2, 3")
    I1 = solve_I1()
    if j == 1:
        return I1
    I2 = solve_add_6_7_8_after_flip(I1)
    if j == 2:
        return I2
    I3 = solve_add_9_10_11_after_flip(I2)
    return I3

# ---------- plot ----------
def plot_rects(rects: Dict[int,RVal], title: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(6,6))
    for i in sorted(rects):
        x,y,w,h = rects[i]
        ax.add_patch(patches.Rectangle((x,y), w, h, fill=False, linewidth=2))
        ax.text(x + w/2, y + h/2, str(i), ha='center', va='center')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.35)
    ax.set_title(title)
    plt.tight_layout(); plt.show()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j", type=int, default=1, help="Build instance I_j (1..3)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rects = build_Ij(args.j)
    for i in sorted(rects):
        x,y,w,h = rects[i]
        print(f"{i:2d}: ({x:.6f}, {y:.6f}, {w:.6f}, {h:.6f})")
    if args.plot:
        plot_rects(rects, f"I_{args.j}")

if __name__ == "__main__":
    main()
2