#!/usr/bin/env python3
# mcts_augment_primes_parallel.py
#
# MCTS augmentation on (H', V') with:
#  - CPU parallel rollouts (ProcessPoolExecutor)
#  - Optional Metal GPU (MPS) heuristics via PyTorch
#  - Final Gurobi verification of the best ratio
#
# Requires: gurobipy, (optional) torch

import os, json, math, random, argparse
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============== Optional Torch for MPS/CUDA heuristics ==============
def get_torch_device(use_mps: bool, use_cuda: bool):
    try:
        import torch
    except Exception:
        return None, None
    dev = None
    if use_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
    elif use_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return torch, dev

# ==================== Prime encoding helpers =======================
def seq_spans_prime(seq: List[int]) -> List[Tuple[int,int]]:
    first, spans = {}, {}
    for idx, lab in enumerate(seq):
        if lab == 0: continue
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    k = max(spans) if spans else 0
    return [spans[i] for i in range(1, k+1)]

def build_rects_from_primes(Hp: List[int], Vp: List[int]):
    X = seq_spans_prime(Hp)
    Y = seq_spans_prime(Vp)
    rects = []
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects):
    xs = sorted({x for r in rects for x in (r[0][0], r[0][1])})
    ys = sorted({y for r in rects for y in (r[1][0], r[1][1])})
    return [(x,y) for x in xs for y in ys]

def covers_grid_closed(rects, pts):
    C=[]
    for (x,y) in pts:
        S=[]
        for i,((x1,x2),(y1,y2)) in enumerate(rects):
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                S.append(i)
        C.append(S)
    return C

# ==================== Gurobi LP/ILP evaluator ======================
def solve_lp_ilp_with_vars(rects, threads=0, output_flag=0):
    import gurobipy as gp
    from gurobipy import GRB

    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    n = len(rects)

    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', output_flag)
    if threads>0: m_lp.setParam('Threads', threads)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0

    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', output_flag)
    if threads>0: m_ilp.setParam('Threads', threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    return lp, ilp

def evaluate_ratio(rects, threads=0, output_flag=0) -> Tuple[float,float,float]:
    if not rects:
        return 0.0, 0.0, 0.0
    lp, ilp = solve_lp_ilp_with_vars(rects, threads=threads, output_flag=output_flag)
    ratio = (lp/ilp) if ilp>0 else (float('inf') if lp>0 else 0.0)
    return lp, ilp, ratio

# ==================== State ops & candidates ========================
def zero_positions(seq: List[int]) -> List[int]:
    return [i for i,v in enumerate(seq) if v==0]

def _ensure_len(seq, upto_idx: int):
    """Extend seq with zeros so that seq[upto_idx] is a valid index."""
    if upto_idx >= len(seq):
        seq.extend([0] * (upto_idx - len(seq) + 1))

def add_rect_to_primes(Hp: List[int], Vp: List[int],
                       hx1: int, hx2: int, vy1: int, vy2: int) -> Tuple[List[int], List[int]]:
    """
    Insert a new rectangle by writing a fresh label k+1 at the two x-endpoints and two y-endpoints.
    If indices exceed current lengths (because allow_extend generated actions beyond the original
    axis), auto-extend with zeros first to preserve prime-encoding semantics.
    """
    Hp2 = Hp[:]  # copy
    Vp2 = Vp[:]
    # ensure capacity
    _ensure_len(Hp2, max(hx1, hx2))
    _ensure_len(Vp2, max(vy1, vy2))

    k = max(Hp2 + Vp2) if (Hp2 or Vp2) else 0
    new_id = k + 1

    # sanity: the target slots should be holes (0); if not, skip/overwrite depending on your policy
    # Here we strictly require holes; if occupied, raise to signal a bad action.
    if not (Hp2[hx1] == Hp2[hx2] == 0 and Vp2[vy1] == Vp2[vy2] == 0):
        raise IndexError("add_rect_to_primes received non-zero endpoint positions")

    Hp2[hx1] = new_id; Hp2[hx2] = new_id
    Vp2[vy1] = new_id; Vp2[vy2] = new_id
    return Hp2, Vp2

def maybe_extend(seq: List[int], n_new_positions: int) -> List[int]:
    return seq + [0]*n_new_positions if n_new_positions>0 else seq

def _rectangle_overlap_score_cpu(r, rects) -> int:
    (x1,x2),(y1,y2) = r
    score = 0
    for ((a1,a2),(b1,b2)) in rects:
        ox = max(0, min(x2,a2)-max(x1,a1))
        oy = max(0, min(y2,b2)-max(y1,y1, b1))  # typo guard
        oy = max(0, min(y2,b2)-max(y1,b1))
        if ox>0 and oy>0: score += 3
        elif (ox>0 and (y1==b2 or y2==b1)) or (oy>0 and (x1==a2 or x2==a1)): score += 2
        elif (x1 in (a1,a2)) and (y1 in (b1,b2)): score += 1
    return score

def _rectangle_overlap_score_torch(r, rects, torch, device) -> int:
    # Vectorized scoring on CPU/MPS/CUDA
    (x1,x2),(y1,y2) = r
    if not rects:
        return 0
    import numpy as np
    a1 = torch.tensor([rc[0][0] for rc in rects], device=device)
    a2 = torch.tensor([rc[0][1] for rc in rects], device=device)
    b1 = torch.tensor([rc[1][0] for rc in rects], device=device)
    b2 = torch.tensor([rc[1][1] for rc in rects], device=device)

    X1 = torch.full_like(a1, x1); X2 = torch.full_like(a2, x2)
    Y1 = torch.full_like(b1, y1); Y2 = torch.full_like(b2, y2)

    ox = torch.clamp(torch.minimum(X2, a2) - torch.maximum(X1, a1), min=0)
    oy = torch.clamp(torch.minimum(Y2, b2) - torch.maximum(Y1, b1), min=0)

    # proper overlap
    proper = (ox > 0) & (oy > 0)

    # edge-touch
    touch_x = (torch.minimum(X2, a2) - torch.maximum(X1, a1) == 0)
    touch_y = (torch.minimum(Y2, b2) - torch.maximum(Y1, b1) == 0)
    edge = ((ox > 0) & touch_y) | ((oy > 0) & touch_x)

    # corner (coarse proxy)
    corner = ((X1 == a1) | (X1 == a2)) & ((Y1 == b1) | (Y1 == b2))

    score = (proper.int()*3 + edge.int()*2 + corner.int()*1).sum().item()
    return int(score)

def candidate_spans(zeros: List[int], max_pairs: int, min_len: int = 1) -> List[Tuple[int,int]]:
    if len(zeros) < 2: return []
    all_pairs = []
    for i in range(len(zeros)):
        for j in range(i+1, len(zeros)):
            if zeros[j] - zeros[i] >= min_len:
                all_pairs.append((zeros[i], zeros[j]))
    if not all_pairs: return []
    random.shuffle(all_pairs)
    return all_pairs[:max_pairs]

def generate_actions(Hp: List[int], Vp: List[int],
                     rects_now, max_cands: int,
                     allow_extend: bool,
                     torch_dev=None) -> List[Tuple[int,int,int,int]]:
    zx = zero_positions(Hp)
    zy = zero_positions(Vp)
    need_extend = (len(zx) < 2 or len(zy) < 2)
    if need_extend and allow_extend:
        Hp = maybe_extend(Hp, 2)
        Vp = maybe_extend(Vp, 2)
        zx = zero_positions(Hp); zy = zero_positions(Vp)
    if len(zx) < 2 or len(zy) < 2:
        return []

    x_pairs = candidate_spans(zx, max_pairs=min(max_cands, 64))
    y_pairs = candidate_spans(zy, max_pairs=min(max_cands, 64))
    if not x_pairs or not y_pairs:
        return []

    cands = []
    for (x1,x2) in x_pairs:
        for (y1,y2) in y_pairs:
            r = ((x1,x2),(y1,y2))
            if torch_dev and torch_dev[0] is not None:
                sc = _rectangle_overlap_score_torch(r, rects_now, torch_dev[0], torch_dev[1])
            else:
                sc = _rectangle_overlap_score_cpu(r, rects_now)
            cands.append((sc, x1,x2,y1,y2))

    cands.sort(key=lambda t: t[0], reverse=True)
    top = cands[:max_cands]
    random.shuffle(top)
    return [(x1,x2,y1,y2) for (_,x1,x2,y1,y2) in top]

# ======================== MCTS Core ================================
class Node:
    def __init__(self, Hp, Vp, depth, K):
        self.Hp = Hp; self.Vp = Vp
        self.depth = depth; self.K = K
        self.visits = 0
        self.value = 0.0  # best ratio seen under this node
        self.children = []  # (action, child)
        self.untried_actions = None

def uct_select(node: Node, c_uct: float):
    best, best_u = None, -1e9
    for (a, ch) in node.children:
        if ch.visits == 0:
            u = 1e9
        else:
            u = ch.value + c_uct * math.sqrt(math.log(node.visits + 1) / ch.visits)
        if u > best_u:
            best_u, best = u, (a, ch)
    return best

def add_many(Hp, Vp, actions: List[Tuple[int,int,int,int]]):
    Hp2, Vp2 = Hp[:], Vp[:]
    for (x1,x2,y1,y2) in actions:
        Hp2, Vp2 = add_rect_to_primes(Hp2, Vp2, x1,x2,y1,y2)
    return Hp2, Vp2

# -------- single rollout (worker-safe) ----------
def rollout_once(Hp, Vp, depth_left, allow_extend, max_cands,
                 use_mps=False, use_cuda=False,
                 worker_gurobi_threads=1, output_flag=0, seed=None) -> Tuple[float, Tuple[List[int],List[int],float,float]]:
    if seed is not None:
        random.seed(seed ^ (sum(Hp)+sum(Vp)))
    torch_dev = get_torch_device(use_mps, use_cuda)
    Hp_r, Vp_r = Hp[:], Vp[:]
    for _ in range(depth_left):
        rects_now = build_rects_from_primes(Hp_r, Vp_r)
        acts = generate_actions(Hp_r, Vp_r, rects_now, max_cands=max_cands, allow_extend=allow_extend, torch_dev=torch_dev)
        if not acts: break
        # epsilon-greedy
        if random.random() < 0.15:
            x1,x2,y1,y2 = random.choice(acts)
        else:
            x1,x2,y1,y2 = acts[0]
        Hp_r, Vp_r = add_rect_to_primes(Hp_r, Vp_r, x1,x2,y1,y2)
    rects = build_rects_from_primes(Hp_r, Vp_r)
    lp, ilp, ratio = evaluate_ratio(rects, threads=worker_gurobi_threads, output_flag=output_flag)
    return ratio, (Hp_r, Vp_r, lp, ilp)

def parallel_rollouts(Hp, Vp, depth_left, n_rollouts, workers,
                      allow_extend, max_cands,
                      use_mps=False, use_cuda=False,
                      worker_gurobi_threads=1, output_flag=0, seed=None):
    """Run multiple independent rollouts in parallel and return the best outcome."""
    if n_rollouts <= 1 or workers <= 1:
        return rollout_once(Hp, Vp, depth_left, allow_extend, max_cands,
                            use_mps, use_cuda, worker_gurobi_threads, output_flag, seed)
    best_ratio = -1.0
    best_payload = None
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for j in range(n_rollouts):
            futs.append(ex.submit(
                rollout_once, Hp, Vp, depth_left, allow_extend, max_cands,
                use_mps, use_cuda, worker_gurobi_threads, output_flag, (seed or 0) + j + 12345
            ))
        for ft in as_completed(futs):
            ratio, payload = ft.result()
            if ratio > best_ratio:
                best_ratio, best_payload = ratio, payload
    return best_ratio, best_payload

def extract_added(Hp0, Vp0, Hp1, Vp1) -> List[Dict[str,int]]:
    k0 = max(Hp0+Vp0) if (Hp0 or Vp0) else 0
    added = []
    for lab in range(k0+1, max(Hp1+Vp1)+1):
        hx = [i for i,v in enumerate(Hp1) if v==lab]
        vy = [i for i,v in enumerate(Vp1) if v==lab]
        if len(hx)==2 and len(vy)==2:
            added.append({"id": lab, "hx1": hx[0], "hx2": hx[1], "vy1": vy[0], "vy2": vy[1]})
    return added

def mcts_search(Hp0, Vp0, K, iters=1000, c_uct=1.0,
                allow_extend=False, max_cands=64, seed=42,
                # parallel controls
                workers=1, rollouts_per_iter=4, worker_gurobi_threads=1,
                # device/solver outputs
                use_mps=False, use_cuda=False, output_flag=0,
                target_ratio=1.5):

    random.seed(seed)
    root = Node(Hp0[:], Vp0[:], depth=0, K=K)
    best_global = {"ratio": -1.0, "Hp": Hp0[:], "Vp": Vp0[:], "lp": 0.0, "ilp": 0.0, "added": []}

    for t in range(iters):
        # SELECTION
        node = root
        path = [node]
        while node.depth < K and node.children and (node.untried_actions is not None and len(node.untried_actions)==0):
            _, node = uct_select(node, c_uct)
            path.append(node)

        # EXPANSION
        if node.depth < K:
            if node.untried_actions is None:
                rects_now = build_rects_from_primes(node.Hp, node.Vp)
                # candidates use local torch device for scoring
                torch_dev = get_torch_device(use_mps, use_cuda)
                node.untried_actions = generate_actions(node.Hp, node.Vp, rects_now,
                                                        max_cands=max_cands,
                                                        allow_extend=allow_extend,
                                                        torch_dev=torch_dev)
            if node.untried_actions:
                act = node.untried_actions.pop()
                Hp2, Vp2 = add_rect_to_primes(node.Hp, node.Vp, *act)
                child = Node(Hp2, Vp2, node.depth+1, K)
                node.children.append((act, child))
                node = child
                path.append(node)

        # ROLLOUT (parallel batch)
        depth_left = K - node.depth
        ratio, (Hp_r, Vp_r, lp, ilp) = parallel_rollouts(
            node.Hp, node.Vp, depth_left,
            n_rollouts=max(1, rollouts_per_iter),
            workers=max(1, workers),
            allow_extend=allow_extend,
            max_cands=max_cands,
            use_mps=use_mps, use_cuda=use_cuda,
            worker_gurobi_threads=max(1, worker_gurobi_threads),
            output_flag=output_flag,
            seed=seed + t*131
        )

        # BACKUP
        for nd in path:
            nd.visits += 1
            if ratio > nd.value:
                nd.value = ratio

        # track global best
        if ratio > best_global["ratio"]:
            added = extract_added(root.Hp, root.Vp, Hp_r, Vp_r)
            best_global = {"ratio": ratio, "Hp": Hp_r, "Vp": Vp_r, "lp": lp, "ilp": ilp, "added": added}

        if target_ratio != float('inf') and ratio >= target_ratio:
            break

    return best_global

# ======================== CLI / MAIN ===============================
def main():
    ap = argparse.ArgumentParser(description="MCTS augmentation with CPU parallelism, MPS heuristics, and Gurobi verification.")
    ap.add_argument("--in_json", required=True, help="Path to misr_filtered_XXX.json with H_prime, V_prime")
    ap.add_argument("--K", type=int, default=6, help="Max rectangles to add")
    ap.add_argument("--iters", type=int, default=1500, help="MCTS iterations")
    ap.add_argument("--c_uct", type=float, default=1.0, help="UCT exploration constant")
    ap.add_argument("--max_cands", type=int, default=64, help="Action candidates per level")
    ap.add_argument("--allow_extend", action="store_true", help="Allow appending zero endpoints when holes run out")
    ap.add_argument("--target_ratio", type=float, default=1.5)

    # Parallelism
    ap.add_argument("--workers", type=int, default=1, help="Parallel rollout workers (processes)")
    ap.add_argument("--rollouts_per_iter", type=int, default=4, help="Rollouts per MCTS iteration (batched)")
    ap.add_argument("--worker_gurobi_threads", type=int, default=1, help="Threads per Gurobi model inside workers to avoid oversubscription")

    # Devices/solver flags
    ap.add_argument("--use_mps", action="store_true", help="Use Metal (MPS) for heuristic scoring if available")
    ap.add_argument("--use_cuda", action="store_true", help="Use CUDA for heuristic scoring if available")
    ap.add_argument("--output_flag", type=int, default=0, help="Gurobi OutputFlag (0: silent)")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.in_json, "r") as f:
        data = json.load(f)
    Hp0 = data.get("H_prime", [])
    Vp0 = data.get("V_prime", [])
    if not isinstance(Hp0, list) or not isinstance(Vp0, list):
        raise ValueError("Input JSON missing H_prime/V_prime lists")

    # Run MCTS
    best = mcts_search(
        Hp0, Vp0, K=args.K, iters=args.iters, c_uct=args.c_uct,
        allow_extend=args.allow_extend, max_cands=args.max_cands, seed=args.seed,
        workers=args.workers, rollouts_per_iter=args.rollouts_per_iter,
        worker_gurobi_threads=args.worker_gurobi_threads,
        use_mps=args.use_mps, use_cuda=args.use_cuda, output_flag=args.output_flag,
        target_ratio=args.target_ratio
    )

    # --------- Final verification (fresh Gurobi run) ----------
    rects_best = build_rects_from_primes(best["Hp"], best["Vp"])
    v_lp, v_ilp, v_ratio = evaluate_ratio(rects_best, threads=max(1, args.worker_gurobi_threads), output_flag=args.output_flag)

    # Save payload
    base, _ = os.path.splitext(args.in_json)
    out_path = f"{base}_augK{args.K}_best_parallel.json"
    out_payload = {
        "source": args.in_json,
        "Hp_aug": best["Hp"],
        "Vp_aug": best["Vp"],
        "added_rects": best["added"],
        "lp_search": best["lp"],
        "ilp_search": best["ilp"],
        "ratio_search": best["ratio"],
        "lp_verify": v_lp,
        "ilp_verify": v_ilp,
        "ratio_verify": v_ratio,
        "params": {
            "K": args.K, "iters": args.iters, "c_uct": args.c_uct,
            "max_cands": args.max_cands, "allow_extend": args.allow_extend,
            "seed": args.seed, "target_ratio": args.target_ratio,
            "workers": args.workers, "rollouts_per_iter": args.rollouts_per_iter,
            "worker_gurobi_threads": args.worker_gurobi_threads,
            "use_mps": args.use_mps, "use_cuda": args.use_cuda,
            "output_flag": args.output_flag
        }
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[OK] Wrote best augmentation to: {out_path}")
    print(f"Search   -> LP={best['lp']:.3f}  ILP={best['ilp']:.3f}  ratio={best['ratio']:.3f}")
    print(f"Verified -> LP={v_lp:.3f}  ILP={v_ilp:.3f}  ratio={v_ratio:.3f}")
    if v_ratio >= args.target_ratio:
        print("✅ Target ratio met or exceeded (verified).")
    else:
        print("⚠️ Target ratio not met on verification; consider increasing --iters/--K, enabling --allow_extend, or raising parallelism.")

if __name__ == "__main__":
    main()
