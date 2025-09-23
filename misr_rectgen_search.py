#!/usr/bin/env python3
# Adversarial MISR instance generator + LP/ILP ratio hill-climber (MPS-optional)

import argparse, math, random, pickle, sys
from typing import List, Tuple
import gurobipy as gp
from gurobipy import GRB

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

Rect = Tuple[Tuple[float,float], Tuple[float,float]]  # ((x1,x2),(y1,y2))

# -------------------------
# utilities
# -------------------------
def set_seed(s):
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s % (2**32-1))
    except Exception:
        pass

def uniq_sorted(vals, jitter=0.0):
    out = []
    for v in vals:
        v2 = float(v)
        if jitter:
            v2 += (random.random()-0.5)*jitter
        out.append(v2)
    out = sorted(out)
    # de-duplicate by tiny push if equal
    eps = 1e-6
    for i in range(1, len(out)):
        if abs(out[i] - out[i-1]) < eps:
            out[i] += eps*(i+1)
    return out

def general_position_rects(rects: List[Rect], eps=1e-5):
    # Ensure no identical x or y endpoints across different rectangles (helps avoid giant cliques)
    xs = []
    ys = []
    for ((x1,x2),(y1,y2)) in rects:
        xs += [x1, x2]; ys += [y1, y2]
    # small jitter map
    seenx = {}
    seeny = {}
    def jit(v, seen):
        if v in seen:
            seen[v] += 1
            return v + eps*seen[v]
        else:
            seen[v] = 0
            return v
    out=[]
    for ((x1,x2),(y1,y2)) in rects:
        xx1, xx2 = jit(x1, seenx), jit(x2, seenx)
        yy1, yy2 = jit(y1, seeny), jit(y2, seeny)
        if xx1 > xx2: xx1, xx2 = xx2, xx1
        if yy1 > yy2: yy1, yy2 = yy2, yy1
        # enforce min size
        if xx2-xx1 < eps: xx2 = xx1 + eps
        if yy2-yy1 < eps: yy2 = yy1 + eps
        out.append(((xx1,xx2),(yy1,yy2)))
    return out

# -------------------------
# generators
# -------------------------
def gen_staggered_fence(n: int, seed: int) -> List[Rect]:
    """
    Two interleaved bands of thin rectangles. Designed to minimize triple points.
    n should be even; if odd we add one at the end.
    """
    set_seed(seed)
    k = n//2
    W = 1000.0
    # vertical band around x in [100, 900], horizontal band around y in [100, 900]
    xs = uniq_sorted([random.uniform(120, 880) for _ in range(k)])
    ys = uniq_sorted([random.uniform(120, 880) for _ in range(k)])
    rects = []
    # thin horizontals with slightly staggered x-ranges
    for i, y in enumerate(ys):
        h = random.uniform(18, 28)
        x1 = 60 + 10*math.sin(0.23*i) + random.uniform(0,10)
        x2 = 940 - 10*math.cos(0.19*i) - random.uniform(0,10)
        rects.append(((x1, x2), (y - h/2, y + h/2)))
    # thin verticals, staggered y-ranges
    for j, x in enumerate(xs):
        w = random.uniform(18, 28)
        y1 = 60 + 10*math.cos(0.17*j) + random.uniform(0,10)
        y2 = 940 - 10*math.sin(0.21*j) - random.uniform(0,10)
        rects.append(((x - w/2, x + w/2), (y1, y2)))
    if len(rects) < n:
        # add one slanted-ish axis-aligned rectangle bridging a gap
        rects.append(((200, 820), (480, 520)))
    return general_position_rects(rects)[:n]

def gen_comb(n: int, seed: int) -> List[Rect]:
    """
    Comb family (teeth vs gums), but offsets avoid massive cliques.
    """
    set_seed(seed)
    base = []
    teeth = max(3, n//3)
    gums  = n - teeth
    # gums: wide short slabs
    for g in range(gums):
        y = 70 + (g+1)* (850.0/(gums+1))
        h = random.uniform(26, 36)
        x1 = 80 + random.uniform(0, 40)
        x2 = 920 - random.uniform(0, 40)
        base.append(((x1, x2), (y - h/2, y + h/2)))
    # teeth: tall thin slabs, slightly jittered, but not lining up to share exact corners
    for t in range(teeth):
        x = 90 + (t+1) * (820.0/(teeth+1)) + random.uniform(-8, 8)
        w = random.uniform(16, 24)
        y1 = 80 + random.uniform(0, 60)
        y2 = 920 - random.uniform(0, 60)
        base.append(((x - w/2, x + w/2), (y1, y2)))
    random.shuffle(base)
    return general_position_rects(base)[:n]

def gen_laminated(n: int, seed: int) -> List[Rect]:
    """
    Laminated slabs but with deliberate offsets so triple intersections are rarer.
    """
    set_seed(seed)
    m = max(3, n//2)
    rects=[]
    # vertical-ish
    for i in range(m//2):
        x = 100 + i*(760.0/max(1,(m//2)-1)) + random.uniform(-6, 6)
        w = random.uniform(20, 30)
        y1 = 100 + random.uniform(0, 120)
        y2 = 900 - random.uniform(0, 120)
        rects.append(((x-w/2, x+w/2), (y1, y2)))
    # horizontal-ish
    for j in range(n - len(rects)):
        y = 100 + j*(760.0/max(1,(n-len(rects))-1)) + random.uniform(-6, 6)
        h = random.uniform(20, 30)
        x1 = 100 + random.uniform(0, 120)
        x2 = 900 - random.uniform(0, 120)
        rects.append(((x1, x2), (y-h/2, y+h/2)))
    random.shuffle(rects)
    return general_position_rects(rects)[:n]

# -------------------------
# clique points
# -------------------------
def grid_points_endpoint(rects: List[Rect]):
    xs = sorted({a for ((x1,x2),(y1,y2)) in rects for a in (x1,x2)})
    ys = sorted({b for ((x1,x2),(y1,y2)) in rects for b in (y1,y2)})
    return [(x,y) for x in xs for y in ys]

def grid_points_cell_centers(rects: List[Rect]):
    xs = sorted({a for ((x1,x2),(y1,y2)) in rects for a in (x1,x2)})
    ys = sorted({b for ((x1,x2),(y1,y2)) in rects for b in (y1,y2)})
    xmid = [(xs[i]+xs[i+1])/2.0 for i in range(len(xs)-1)]
    ymid = [(ys[j]+ys[j+1])/2.0 for j in range(len(ys)-1)]
    return [(x,y) for x in xmid for y in ymid]

def build_covers(rects: List[Rect], pts, use_gpu=False):
    n = len(rects)
    P = len(pts)
    if TORCH_OK and use_gpu:
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        X1 = torch.tensor([r[0][0] for r in rects], device=dev).view(1,n)
        X2 = torch.tensor([r[0][1] for r in rects], device=dev).view(1,n)
        Y1 = torch.tensor([r[1][0] for r in rects], device=dev).view(1,n)
        Y2 = torch.tensor([r[1][1] for r in rects], device=dev).view(1,n)
        XP = torch.tensor([p[0] for p in pts], device=dev).view(P,1)
        YP = torch.tensor([p[1] for p in pts], device=dev).view(P,1)
        mask = (X1 <= XP) & (XP <= X2) & (Y1 <= YP) & (YP <= Y2)
        covers = []
        mask_cpu = mask.cpu()
        for p in range(P):
            S = torch.nonzero(mask_cpu[p], as_tuple=False).view(-1).tolist()
            covers.append([int(i) for i in S])
        return covers
    else:
        covers=[]
        for (x,y) in pts:
            S=[]
            for i,((x1,x2),(y1,y2)) in enumerate(rects):
                if (x1 <= x <= x2) and (y1 <= y <= y2):
                    S.append(i)
            covers.append(S)
        return covers

# -------------------------
# LP / ILP
# -------------------------
def solve_lp_ilp(rects: List[Rect], covers: List[List[int]], grb_threads=0):
    n = len(rects)
    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads>0: m_lp.setParam('Threads', grb_threads)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1.0)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads>0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1.0)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0

    return lp, ilp

# -------------------------
# hill climb
# -------------------------
def nudge_rect(r: Rect, scale=4.0):
    (x1,x2),(y1,y2) = r
    # choose a side and move a little (keeping ordering)
    which = random.randint(0,3)
    d = (random.random()-0.5)*scale
    if which==0:
        x1 += d; x1 = min(x1, x2-1e-5)
    elif which==1:
        x2 += d; x2 = max(x2, x1+1e-5)
    elif which==2:
        y1 += d; y1 = min(y1, y2-1e-5)
    else:
        y2 += d; y2 = max(y2, y1+1e-5)
    return ((x1,x2),(y1,y2))

def hill_climb(rects: List[Rect], covers_builder, use_gpu, grb_threads, steps=60, scale=6.0):
    best = list(rects)
    pts = covers_builder(best)
    cov = build_covers(best, pts, use_gpu)
    blp, bilp = solve_lp_ilp(best, cov, grb_threads)
    if bilp <= 0: return best, blp, bilp
    best_ratio = blp / bilp

    for _ in range(steps):
        i = random.randrange(len(best))
        trial = list(best)
        trial[i] = nudge_rect(trial[i], scale)
        pts2 = covers_builder(trial)
        cov2 = build_covers(trial, pts2, use_gpu)
        lp2, ilp2 = solve_lp_ilp(trial, cov2, grb_threads)
        if ilp2 > 0:
            r2 = lp2/ilp2
            if r2 > best_ratio + 1e-9:
                best, blp, bilp, best_ratio = trial, lp2, ilp2, r2
    return best, blp, bilp

# -------------------------
# main search
# -------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--sizes", type=int, nargs="+", default=[14,17,20])
    pa.add_argument("--seeds", type=int, default=128)
    pa.add_argument("--family", type=str, default="staggered", choices=["staggered","comb","laminated"])
    pa.add_argument("--gpu", action="store_true", help="use MPS (for cover building only)")
    pa.add_argument("--threads", type=int, default=0)
    pa.add_argument("--hill", type=int, default=60, help="hill-climb steps per seed (0=disabled)")
    pa.add_argument("--interior_only", action="store_true",
                    help="use cell centers (ignore corner-only touches as cliques)")
    pa.add_argument("--outpkl", type=str, default="misr_elites_adv.pkl")
    args = pa.parse_args()

    if args.gpu and not (TORCH_OK and (torch.backends.mps.is_available())):
        print("Note: --gpu requested but MPS not available; falling back to CPU.", file=sys.stderr)
        args.gpu = False

    gen_map = {
        "staggered": gen_staggered_fence,
        "comb": gen_comb,
        "laminated": gen_laminated
    }

    elites = []
    for n in args.sizes:
        print(f"\n=== SIZE n={n} ({args.seeds} seeds) ===")
        best_ratio = 0.0
        for s in range(args.seeds):
            rects = gen_map[args.family](n, seed=s)
            if args.interior_only:
                pts_fn = lambda R: grid_points_cell_centers(R)
            else:
                pts_fn = lambda R: grid_points_endpoint(R)
            pts = pts_fn(rects)
            covers = build_covers(rects, pts, use_gpu=args.gpu)
            lp, ilp = solve_lp_ilp(rects, covers, grb_threads=args.threads)
            ratio = (lp/ilp) if ilp > 0 else 0.0

            # hill climb
            if args.hill > 0 and ilp > 0:
                rects2, lp2, ilp2 = hill_climb(rects, pts_fn, args.gpu, args.threads, steps=args.hill, scale=6.0)
                if ilp2 > 0 and (lp2/ilp2) > ratio + 1e-9:
                    rects, lp, ilp = rects2, lp2, ilp2
                    ratio = lp/ilp

            if ratio > 0.0:
                elites.append((ratio, rects, (lp, ilp)))
            if (s % 16) == 0:
                print(f"[seed {s:03d}] LP={lp:.3f} ILP={ilp:.3f} ratio={ratio:.4f}  best_so_far={max(best_ratio, ratio):.4f}")

            best_ratio = max(best_ratio, ratio)

        # report section top-10
        topn = sorted([e for e in elites if len(e[1])==n], key=lambda z: -z[0])[:10]
        print("\n=== BEST ELITES ===")
        for i,(r,rects,(lp,ilp)) in enumerate(topn, start=1):
            print(f"#{i} ratio={r:.4f}  n={n} (LP={lp:.3f}, ILP={ilp:.3f})")
    # save all
    elites_sorted = sorted(elites, key=lambda z: -z[0])[:64]
    with open(args.outpkl, "wb") as f:
        pickle.dump(elites_sorted, f)
    print(f"Saved top elites to {args.outpkl}")

if __name__ == "__main__":
    main()
