#!/usr/bin/env python3
# z3_generate_candidates_gap2.py
#
# Enumerate K new rectangles on (H',V') subject to:
#   - endpoints must land on zero slots (or optional bounded extension)
#   - EVERY grid point is covered by <= 2 rectangles (existing + new)
# and maximize a surrogate objective that correlates with a larger LP/ILP gap.
#
# Output: top-N candidate augmentations (endpoint pairs) in a .cands.json file.
#
# Usage example:
#   python z3_generate_candidates_gap2.py \
#     --in_json graphs/n12/misr_filtered_000.json \
#     --K 6 --topN 128 --allow_extend --extend_x 6 --extend_y 6

import json, argparse
from typing import List, Tuple, Dict
from z3 import Int, Bool, Optimize, And, Or, Not, If, Distinct, Sum, sat

# ---------- base helpers ----------
def seq_zero_positions(seq: List[int]) -> List[int]:
    return [i for i,v in enumerate(seq) if v == 0]

def build_existing_rects(Hp: List[int], Vp: List[int]) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    firstH, spansH = {}, {}
    firstV, spansV = {}, {}
    for i,v in enumerate(Hp):
        if v>0:
            if v not in firstH: firstH[v]=i
            else: spansH[v]=(firstH[v], i)
    for i,v in enumerate(Vp):
        if v>0:
            if v not in firstV: firstV[v]=i
            else: spansV[v]=(firstV[v], i)
    n = max(spansH) if spansH else 0
    rects=[]
    for lab in range(1, n+1):
        (x1,x2),(y1,y2) = spansH[lab], spansV[lab]
        if x1>x2: x1,x2 = x2,x1
        if y1>y2: y1,y2 = y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def build_grid(Hp: List[int], Vp: List[int], extend_x: int, extend_y: int, allow_extend: bool):
    X = sorted({i for i,v in enumerate(Hp) if v>0})
    Y = sorted({i for i,v in enumerate(Vp) if v>0})
    # include zero-slot indices too, since grid points live at all endpoints
    X = sorted(set(X) | set(range(len(Hp))))
    Y = sorted(set(Y) | set(range(len(Vp))))
    if allow_extend:
        X = list(range(0, len(Hp)+extend_x))
        Y = list(range(0, len(Vp)+extend_y))
    return X, Y

# overlap kind booleans between a (x1,x2,y1,y2) and fixed (a1,a2,b1,b2)
def overlap_kind_bools(x1,x2,y1,y2, a1,a2,b1,b2):
    # ox_pos = min(x2,a2) > max(x1,a1)
    ox_pos  = (If(x2<a2, x2, a2) - If(x1>a1, x1, a1)) > 0
    oy_pos  = (If(y2<b2, y2, b2) - If(y1>b1, y1, b1)) > 0
    touch_x = (If(x2<a2, x2, a2) - If(x1>a1, x1, a1)) == 0
    touch_y = (If(y2<b2, y2, b2) - If(y1>b1, y1, b1)) == 0
    proper  = And(ox_pos, oy_pos)
    edge    = Or(And(ox_pos, touch_y), And(oy_pos, touch_x))
    return proper, edge

# coverage boolean: grid point (gx,gy) lies inside closed rectangle box
def cov_bool(x1,x2,y1,y2, gx,gy):
    return And(x1 <= gx, gx <= x2, y1 <= gy, gy <= y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--topN", type=int, default=64)
    ap.add_argument("--allow_extend", action="store_true")
    ap.add_argument("--extend_x", type=int, default=4)
    ap.add_argument("--extend_y", type=int, default=4)

    # weights (surrogate objective)
    ap.add_argument("--w_exact2", type=int, default=5, help="reward grid points covered by exactly 2 rects (tight clique)")
    ap.add_argument("--w_prop",   type=int, default=3, help="reward proper overlaps with existing rects")
    ap.add_argument("--w_edge",   type=int, default=2, help="reward edge touches with existing rects")
    ap.add_argument("--w_span",   type=int, default=1, help="lightly reward span sizes to hit more grid points")
    ap.add_argument("--w_nndis",  type=int, default=1, help="penalty per new-new disjoint pair (set via --penalize_nndis)")
    ap.add_argument("--penalize_nndis", action="store_true")

    args = ap.parse_args()

    base = json.load(open(args.in_json))
    Hp, Vp = base["H_prime"], base["V_prime"]
    rects0 = build_existing_rects(Hp, Vp)

    # fixed grid (indices) where coverage is evaluated
    X, Y = build_grid(Hp, Vp, args.extend_x, args.extend_y, args.allow_extend)
    G = [(gx,gy) for gx in X for gy in Y]

    # precompute existing-coverage at each grid point (constants)
    exist_cov = [0]*len(G)
    exist_cov_matrix = []  # per existing rect cov at each point
    for ((a1,a2),(b1,b2)) in rects0:
        col=[]
        for gi,(gx,gy) in enumerate(G):
            inside = (a1 <= gx <= a2) and (b1 <= gy <= b2)
            col.append(1 if inside else 0)
            if inside:
                exist_cov[gi] += 1
        exist_cov_matrix.append(col)

    # endpoint zero slots
    Zx = seq_zero_positions(Hp)
    Zy = seq_zero_positions(Vp)

    # decision vars for new rectangles
    hx1 = [Int(f"hx1_{i}") for i in range(args.K)]
    hx2 = [Int(f"hx2_{i}") for i in range(args.K)]
    vy1 = [Int(f"vy1_{i}") for i in range(args.K)]
    vy2 = [Int(f"vy2_{i}") for i in range(args.K)]

    opt = Optimize()

    # domains: either from zero slots or from bounded [0, Lx-1]/[0, Ly-1] if allow_extend
    Lx = len(Hp) + (args.extend_x if args.allow_extend else 0)
    Ly = len(Vp) + (args.extend_y if args.allow_extend else 0)

    def dom_x(expr):
        if args.allow_extend:
            return And(expr >= 0, expr < Lx)
        return Or(*[expr == z for z in Zx]) if Zx else False

    def dom_y(expr):
        if args.allow_extend:
            return And(expr >= 0, expr < Ly)
        return Or(*[expr == z for z in Zy]) if Zy else False

    used_x = []
    used_y = []
    for i in range(args.K):
        opt.add(dom_x(hx1[i]), dom_x(hx2[i]), dom_y(vy1[i]), dom_y(vy2[i]))
        opt.add(hx1[i] < hx2[i], vy1[i] < vy2[i])
        used_x += [hx1[i], hx2[i]]
        used_y += [vy1[i], vy2[i]]
    # new endpoints cannot reuse the same slot
    if used_x: opt.add(Distinct(*used_x))
    if used_y: opt.add(Distinct(*used_y))

    # coverage booleans for new rects at each grid point
    cov_new = [[Bool(f"cov_new_{i}_{gi}") for gi in range(len(G))] for i in range(args.K)]
    for i in range(args.K):
        for gi,(gx,gy) in enumerate(G):
            opt.add(cov_new[i][gi] == cov_bool(hx1[i],hx2[i],vy1[i],vy2[i], gx,gy))

    # HARD CONSTRAINT: At most 2 rectangles through ANY grid point (existing + new)
    for gi in range(len(G)):
        tot = exist_cov[gi] + Sum([If(cov_new[i][gi], 1, 0) for i in range(args.K)])
        opt.add(tot <= 2)

    # GAP-ORIENTED SURROGATE OBJECTIVE
    score_terms = []

    # 1) reward grid points with EXACTLY 2 coverage (tight cliques)
    for gi in range(len(G)):
        tot = exist_cov[gi] + Sum([If(cov_new[i][gi], 1, 0) for i in range(args.K)])
        score_terms.append(If(tot == 2, args.w_exact2, 0))

    # 2) reward proper/edge overlaps with existing rects
    for i in range(args.K):
        for ((a1,a2),(b1,b2)) in rects0:
            proper, edge = overlap_kind_bools(hx1[i],hx2[i],vy1[i],vy2[i], a1,a2,b1,b2)
            score_terms.append(If(proper, args.w_prop, 0))
            score_terms.append(If(edge,   args.w_edge, 0))

    # 3) lightly reward larger spans (hit more grid points)
    for i in range(args.K):
        score_terms.append(args.w_span * ((hx2[i]-hx1[i]) + (vy2[i]-vy1[i])))

    # 4) (optional) penalize new-new disjoint placements
    if args.penalize_nndis and args.w_nndis > 0 and args.K >= 2:
        for i in range(args.K):
            for j in range(i+1, args.K):
                # disjoint if either x-intervals don't meet or y-intervals don't meet (strictly outside)
                x_disj = Or(hx2[i] < hx1[j], hx2[j] < hx1[i])
                y_disj = Or(vy2[i] < vy1[j], vy2[j] < vy1[i])
                disjoint = Or(x_disj, y_disj)
                score_terms.append(If(disjoint, -args.w_nndis, 0))

    opt.maximize(Sum(score_terms))

    # enumerate top-N distinct solutions (block exact repetition)
    sols=[]
    for _ in range(args.topN):
        if opt.check() != sat:
            break
        m = opt.model()
        rects=[]
        blockers=[]
        for i in range(args.K):
            x1 = m[hx1[i]].as_long(); x2 = m[hx2[i]].as_long()
            y1 = m[vy1[i]].as_long(); y2 = m[vy2[i]].as_long()
            rects.append((x1,x2,y1,y2))
            blockers += [hx1[i]==x1, hx2[i]==x2, vy1[i]==y1, vy2[i]==y2]
        sols.append(rects)
        opt.add(Not(And(*blockers)))

    out = {
        "source": args.in_json,
        "K": args.K,
        "grid_sizes": {"|X|": len(X), "|Y|": len(Y), "|G|": len(G)},
        "topN": len(sols),
        "candidates": [{"rects": [(int(x1),int(x2),int(y1),int(y2)) for (x1,x2,y1,y2) in s]} for s in sols]
    }
    out_path = args.in_json.replace(".json", f"_z3gap2_K{args.K}_N{len(sols)}.cands.json")
    json.dump(out, open(out_path,"w"), indent=2)
    print(f"[OK] wrote {out_path} with {len(sols)} candidates; grid |G|={len(G)}")

if __name__ == "__main__":
    main()
