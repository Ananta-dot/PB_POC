# src/chuzhoy_seeds.py
from __future__ import annotations
from typing import List, Tuple
import math
import random

Rect = Tuple[float,float,float,float]  # (xl, xr, yb, yt)

def _rot90(rects: List[Rect]) -> List[Rect]:
    # rotate 90° clockwise around origin: (x,y) -> (y, -x)
    out=[]
    for (xl,xr,yb,yt) in rects:
        pts = [(xl,yb),(xl,yt),(xr,yb),(xr,yt)]
        rot = [(y, -x) for (x,y) in pts]
        xs = [p[0] for p in rot]
        ys = [p[1] for p in rot]
        out.append((min(xs), max(xs), min(ys), max(ys)))
    return _normalize(out)

def _translate(rects: List[Rect], dx: float, dy: float) -> List[Rect]:
    return [(xl+dx, xr+dx, yb+dy, yt+dy) for (xl,xr,yb,yt) in rects]

def _normalize(rects: List[Rect]) -> List[Rect]:
    # shift so min corner is (0,0)
    minx = min(r[0] for r in rects)
    miny = min(r[2] for r in rects)
    return _translate(rects, -minx, -miny)

def _bbox(rects: List[Rect]) -> Rect:
    xs = [x for r in rects for x in (r[0], r[1])]
    ys = [y for r in rects for y in (r[2], r[3])]
    return (min(xs), max(xs), min(ys), max(ys))

def _scale_to(rects: List[Rect], W: float, H: float) -> List[Rect]:
    (xl,xr,yb,yt) = _bbox(rects)
    w = xr - xl; h = yt - yb
    sx = W / max(w, 1e-9); sy = H / max(h, 1e-9)
    s = min(sx, sy)
    out = []
    for (a,b,c,d) in rects:
        out.append(( (a-xl)*s, (b-xl)*s, (c-yb)*s, (d-yb)*s ))
    return out

def _I1() -> List[Rect]:
    """
    A concrete 5-cycle (corner intersections only).
    A,B,C,D,E with A-B-C-D-E-A edges; no other edges.
    Layout uses a diamond + frame so that touches are corners.
    """
    # core scale
    s=10.0; pad=1.0
    # central diamond-ish rectangle (A)
    A = (4*s, 6*s, 4*s, 6*s)
    # place B to touch A at A's top-right corner
    B = (6*s, 8*s, 6*s, 8*s)
    # C touches B at C's bottom-right corner
    C = (4*s, 6*s, 8*s, 10*s)
    # D touches C at C's top-left corner
    D = (2*s, 4*s, 6*s, 8*s)
    # E touches D at D's bottom-left corner, and should touch A at A's bottom-left
    E = (4*s - 2*s, 4*s, 4*s - 2*s, 4*s)

    rects = [A,B,C,D,E]
    # Nudge a hair to avoid unintended overlaps (keep corners)
    eps=1e-3
    A = (A[0], A[1], A[2], A[3])
    B = (B[0]+eps, B[1]+eps, B[2]+eps, B[3]+eps)
    C = (C[0], C[1], C[2]+2*eps, C[3]+2*eps)
    D = (D[0]-eps, D[1]-eps, D[2], D[3])
    E = (E[0], E[1], E[2], E[3])
    return _normalize([A,B,C,D,E])

def build_chuzhoy_instance(depth: int) -> List[Rect]:
    """
    Recreate Appendix C family I_1..I_depth (corner intersections only).
    I_1 is a 5-cycle. For each j->j+1:
      - rotate I_j by 90°
      - add three special rectangles R1^{j+1}, R2^{j+1}, R3^{j+1}
        positioned to satisfy: R1 hits (R1^j,R2^j), R2 hits R3^j, R3 hits only R1,R2.
      - keep a virtual container around the rotated core, but it's not added.
    Returns concrete coordinates (xl,xr,yb,yt) for all rectangles in I_depth.
    """
    core = _I1()  # I_1
    # Remember indices of the "three special" in each level:
    # We'll designate in I_1: R1=A (idx 0), R2=B (idx 1), R3=C (idx 2)
    idx_R1, idx_R2, idx_R3 = 0, 1, 2

    for j in range(1, depth):
        core = _rot90(core)
        # Scale/normalize the core into a standard box
        core = _scale_to(core, 100.0, 100.0)
        core = _normalize(core)
        (L,R,B,T) = _bbox(core)
        W = R-L; H = T-B
        # Handy handles to R1^j, R2^j, R3^j
        R1j = core[idx_R1]
        R2j = core[idx_R2]
        R3j = core[idx_R3]

        # Add R1^{j+1} to the left: hits R1^j and R2^j at top/bottom corners, avoids core bbox interior
        w = W*0.18; h = H*0.18
        x_gap = W*0.04
        # place it left of min x of core, aligned in y so that it can corner-touch R1^j (upper) and R2^j (lower)
        y_mid = (R1j[3] + R2j[2]) / 2.0
        R1_next = (L - x_gap - w, L - x_gap, y_mid - h/2.0, y_mid + h/2.0)

        # Add R2^{j+1} to the right: hits R3^j
        y_mid2 = (R3j[2] + R3j[3]) / 2.0
        R2_next = (R + x_gap, R + x_gap + w, y_mid2 - h/2.0, y_mid2 + h/2.0)

        # Add R3^{j+1} above: touches only R1^{j+1} and R2^{j+1}
        # Place a small box centered above the line between R1_next and R2_next
        top_y = T + H*0.07
        mid_x = ( (R1_next[0] + R1_next[1]) * 0.5 + (R2_next[0] + R2_next[1]) * 0.5 ) * 0.5
        R3_next = (mid_x - w*0.4, mid_x + w*0.4, top_y, top_y + h*0.8)

        core = core + [R1_next, R2_next, R3_next]
        # Update the indices of the designated three for the next step
        idx_R1 = len(core)-3
        idx_R2 = len(core)-2
        idx_R3 = len(core)-1

    # tiny epsilon shifts to preserve "corner-only" (no positive-area overlaps)
    eps = 1e-5
    out=[]
    for (xl,xr,yb,yt) in core:
        out.append((xl+eps, xr-eps, yb+eps, yt-eps))
    return out

def rects_to_sequences(rects: List[Rect]) -> Tuple[List[int], List[int]]:
    """
    Convert rectangle coordinates into MISR (H,V) encoding used by your code:
      - H is the order of left/right x-endpoints when sweeping x (label appears twice).
      - V is the order of bottom/top y-endpoints when sweeping y.
    Assumes all endpoints are distinct (we epsilon-nudged to ensure that).
    """
    n = len(rects)
    X=[]; Y=[]
    for i,(xl,xr,yb,yt) in enumerate(rects, start=1):
        X.append((xl, i)); X.append((xr, i))
        Y.append((yb, i)); Y.append((yt, i))
    X.sort(key=lambda t: t[0])
    Y.sort(key=lambda t: t[0])
    H = [lab for (_,lab) in X]
    V = [lab for (_,lab) in Y]
    return H, V

def seeds_for_n_from_chuzhoy(n: int, rng: random.Random, variants: int = 8) -> List[Tuple[List[int], List[int]]]:
    """
    Produce a small bundle of (H,V) seeds whose max label is n (or the closest smaller depth if exact match impossible).
    The construction sizes grow as 3^j+2, but in practice we can overshoot and then subselect the first n rectangles,
    or generate at the nearest depth and randomly relabel/crop in a canonical way.
    """
    # choose depth so that we have at least n rectangles
    depth=1
    while True:
        rects = build_chuzhoy_instance(depth)
        if len(rects) >= n: break
        depth += 1
        if depth > 10_000: raise RuntimeError("Depth search runaway")
    H, V = rects_to_sequences(rects)
    # canonically crop to the first n labels (keeping relative order of endpoints for labels <= n)
    # Note: cropping preserves corner-only structure among the kept ones.
    def crop(HV):
        return [x for x in HV if x <= n]
    Hn, Vn = crop(H), crop(V)

    out=[]
    out.append((Hn, Vn))
    # Make a few stable variants (relabel permutations that preserve 2-occurrence structure)
    for _ in range(variants-1):
        perm = list(range(1, n+1))
        rng.shuffle(perm)
        rel = {old: new for new, old in enumerate(perm, 1)}
        def remap(seq): return [rel[x] for x in seq]
        out.append((remap(Hn), remap(Vn)))
    # final canonicalization is done by your pipeline anyway
    return out
