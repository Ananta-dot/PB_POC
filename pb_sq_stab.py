from __future__ import annotations
import argparse, os, sys, math, time, json, random, pickle, warnings, hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gurobipy as gp
from gurobipy import GRB

warnings.filterwarnings("ignore", message=r".*TF32 behavior.*", category=UserWarning)

# ========= device =========
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass

# ========= types / constants =========
Sq = Tuple[int, int]     # (x_bin, y_bin) lower-left corner on a discrete grid
Squares = List[Sq]

SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3  # start of coordinate-token ids
MAX_COORD = 512  # safety ceiling for vocabulary
MAX_N = 256      # safety ceiling on number of squares

# ========= helpers (I/O, hashing, dirs) =========
def instance_key(sqs: Squares) -> str:
    # sqs assumed canonical (sorted). Hash to dedupe/tabu.
    s = ';'.join(f"{x},{y}" for (x,y) in sqs)
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def make_run_dirs(out_root: str, seed: int, n_list: List[int]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_root, f"stab_run-{ts}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for n in n_list:
        os.makedirs(os.path.join(run_dir, f"n{n:02d}"), exist_ok=True)
    return run_dir

def ns_sequence(n_start: int, n_target: int, step: int) -> List[int]:
    out=[]; n=n_start
    while n<=n_target:
        out.append(n); n+=step
    return out

# ========= canonicalization =========
def canonicalize_squares(sqs: Squares) -> Squares:
    """
    Sort squares by (x,y) so permutations map to a unique canonical form.
    """
    return sorted(sqs)

# ========= tokenization (TinyGPT sees a stream of coordinate bins) =========
def squares_to_tokens(sqs: Squares) -> List[int]:
    """
    Sequence layout: [BOS], x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, [EOS]
    Each coordinate token is BASE_VOCAB + bin (0..grid_bins-1).
    """
    toks = [SPECIAL["BOS"]]
    for (x,y) in sqs:
        toks.append(BASE_VOCAB + x)
        toks.append(BASE_VOCAB + y)
    toks.append(SPECIAL["EOS"])
    return toks

def tokens_to_squares(tokens: List[int]) -> Squares:
    """
    Inverse of squares_to_tokens (assume valid layout).
    """
    # strip BOS/EOS if present
    if tokens and tokens[0] == SPECIAL["BOS"]:
        tokens = tokens[1:]
    if tokens and tokens[-1] == SPECIAL["EOS"]:
        tokens = tokens[:-1]
    if len(tokens) % 2 != 0:
        return []
    sqs=[]
    for i in range(0, len(tokens), 2):
        x_tok, y_tok = tokens[i], tokens[i+1]
        x = x_tok - BASE_VOCAB
        y = y_tok - BASE_VOCAB
        sqs.append((x,y))
    return canonicalize_squares(sqs)

# ========= geometry / stabbing LP+ILP =========
def squares_to_segments(sqs: Squares, side: int) -> List[Tuple[int,int,int,int]]:
    """
    Convert squares (x,y) bins with fixed 'side' into axis-aligned boxes in integer grid:
    returns list of (x1,x2,y1,y2) inclusive coordinates (x2=x+side, y2=y+side).
    """
    out=[]
    for (x,y) in sqs:
        x1=x; x2=x+side
        y1=y; y2=y+side
        out.append((x1,x2,y1,y2))
    return out

def solve_stabbing_lp_ilp(sqs: Squares,
                          side: int,
                          gurobi_threads: int = 0,
                          output_flag: int = 0,
                          candidates: str = "right_top") -> Tuple[float,float]:
    """
    Build ILP and LP for stabbing squares by horizontal/vertical lines.
    candidates:
      - "right_top": vertical lines at x2, horizontal at y2 (your baseline)
      - "ends": use both left/right & bottom/top edges (bigger candidate set)
    Returns (ilp, lp) objective values (minimization).
    """
    rects = squares_to_segments(sqs, side)

    # Build candidate lines
    xs, ys = set(), set()
    if candidates == "right_top":
        for (x1,x2,y1,y2) in rects:
            xs.add(x2); ys.add(y2)
    else:  # "ends"
        for (x1,x2,y1,y2) in rects:
            xs.add(x1); xs.add(x2)
            ys.add(y1); ys.add(y2)
    xs = sorted(xs); ys = sorted(ys)

    # Map: for each rect r, which candidate indices stab it
    stabX = []  # list of lists of candidate indices for vertical lines
    stabY = []
    for (x1,x2,y1,y2) in rects:
        vx = [i for i,xc in enumerate(xs) if x1 <= xc <= x2]
        hy = [j for j,yc in enumerate(ys) if y1 <= yc <= y2]
        stabX.append(vx); stabY.append(hy)

    # ILP (min lines)
    m_ilp = gp.Model("square_stab_ilp")
    m_ilp.setParam("OutputFlag", output_flag)
    if gurobi_threads > 0: m_ilp.setParam("Threads", gurobi_threads)
    V = m_ilp.addVars(len(xs), vtype=GRB.BINARY, name="V")
    H = m_ilp.addVars(len(ys), vtype=GRB.BINARY, name="H")
    for r in range(len(rects)):
        m_ilp.addConstr(gp.quicksum(V[i] for i in stabX[r]) +
                        gp.quicksum(H[j] for j in stabY[r]) >= 1)
    m_ilp.setObjective(V.sum('*') + H.sum('*'), GRB.MINIMIZE)
    m_ilp.optimize()
    if m_ilp.status != GRB.OPTIMAL:
        return (0.0, 0.0)
    ilp = float(m_ilp.objVal)

    # LP relaxation
    m_lp = m_ilp.relax()
    m_lp.setParam("OutputFlag", output_flag)
    m_lp.optimize()
    if m_lp.status != GRB.OPTIMAL:
        return (ilp, 0.0)
    lp = float(m_lp.objVal)
    return ilp, lp

# ========= scoring =========
def score_ratio(sqs: Squares,
                side: int,
                alpha_ilp: float = 0.0,
                beta_lp: float = 0.0,
                gurobi_threads: int = 0) -> Tuple[float,float,float,float]:
    """
    Returns (ilp, lp, ratio, blended) where ratio = ilp / max(lp,eps).
    blended allows shaping (more ilp good, bigger lp bad).
    """
    ilp, lp = solve_stabbing_lp_ilp(sqs, side, gurobi_threads=gurobi_threads)
    eps = 1e-9
    ratio = (ilp / max(lp, eps)) if ilp > 0 else 0.0
    n = len(sqs) if sqs else 1
    blended = ratio + alpha_ilp * (ilp / n) - beta_lp * (lp / n)
    return ilp, lp, ratio, blended

# ========= seed builders (motifs + random) =========
def clip_square(x: int, y: int, side: int, grid: int) -> Tuple[int,int]:
    x = max(0, min(x, grid - side))
    y = max(0, min(y, grid - side))
    return x, y

def random_square(rng: random.Random, side: int, grid: int) -> Sq:
    x = rng.randrange(0, grid - side + 1)
    y = rng.randrange(0, grid - side + 1)
    return (x,y)

def random_squares(n: int, rng: random.Random, side: int, grid: int) -> Squares:
    return canonicalize_squares([random_square(rng, side, grid) for _ in range(n)])

def motif_grid(n: int, side: int, grid: int) -> Squares:
    # lay squares on a coarse lattice as a hardish seed
    step = max(side, grid // max(2, int(math.sqrt(n))))
    pts=[]
    x=0; y=0
    while len(pts) < n and y <= grid - side:
        pts.append((x,y))
        x += step
        if x > grid - side:
            x = 0; y += step
    return canonicalize_squares(pts[:n])

def motif_diagonal(n: int, side: int, grid: int) -> Squares:
    pts=[]
    step = max(1, side//2)
    x=0; y=0
    for _ in range(n):
        pts.append(clip_square(x,y,side,grid))
        x+=step; y+=step
    return canonicalize_squares(pts)

def motif_two_clusters(n: int, side: int, grid: int) -> Squares:
    pts=[]
    k = n//2
    c1=(side, side)
    c2=(grid//2, grid//2)
    for _ in range(k):
        pts.append(clip_square(c1[0]+_, c1[1]+_//2, side, grid))
    for _ in range(n-k):
        pts.append(clip_square(c2[0]+_//2, c2[1]+_, side, grid))
    return canonicalize_squares(pts[:n])

def seeded_pool(n: int, rng: random.Random, base_count: int,
                side: int, grid: int, motifs_on: bool=True) -> List[Squares]:
    seeds=[]
    if motifs_on:
        motif_builders = [motif_grid, motif_diagonal, motif_two_clusters]
        for mb in motif_builders:
            seeds.append(mb(n, side, grid))
    while len(seeds) < base_count:
        seeds.append(random_squares(n, rng, side, grid))
    return seeds[:base_count]

# ========= neighbors (local search) =========
def neighbors(sqs: Squares, rng: random.Random, k: int,
              side: int, grid: int) -> List[Squares]:
    """
    Generate k neighbor instances via simple geometric moves.
    Moves:
      - jitter: move one square by small dx,dy
      - swapXY: swap x coords of two squares or y coords
      - reflect: reflect a subset across midlines
      - block_shift: pick a block, shift together
      - reinit_one: replace one square randomly
    """
    out=[]
    n = len(sqs)
    for _ in range(k):
        which = rng.choice(["jitter","swapX","swapY","reflect","block","reinit"])
        A = list(sqs)
        if which == "jitter" and n>0:
            i = rng.randrange(n)
            dx = rng.randint(-max(1,side//2), max(1,side//2))
            dy = rng.randint(-max(1,side//2), max(1,side//2))
            x,y = A[i]
            A[i] = clip_square(x+dx, y+dy, side, grid)
        elif which == "swapX" and n>=2:
            i,j = rng.sample(range(n),2)
            xi,_ = A[i]; xj,_ = A[j]
            A[i] = (xj, A[i][1]); A[j] = (xi, A[j][1])
        elif which == "swapY" and n>=2:
            i,j = rng.sample(range(n),2)
            _,yi = A[i]; _,yj = A[j]
            A[i] = (A[i][0], yj); A[j] = (A[j][0], yi)
        elif which == "reflect" and n>0:
            # reflect half squares across midlines
            sel = rng.sample(range(n), k=max(1, n//3))
            for i in sel:
                x,y = A[i]
                x2 = (grid - side) - x
                y2 = (grid - side) - y
                # reflect either x or y or both
                pick = rng.choice([0,1,2])  # 0:x,1:y,2:both
                if pick==0: A[i]=(x2,y)
                elif pick==1: A[i]=(x,y2)
                else: A[i]=(x2,y2)
        elif which == "block" and n>=2:
            i,j = sorted(rng.sample(range(n),2))
            dx = rng.randint(-side, side)
            dy = rng.randint(-side, side)
            for t in range(i,j+1):
                x,y = A[t]
                A[t] = clip_square(x+dx, y+dy, side, grid)
        else:  # reinit
            if n>0:
                i = rng.randrange(n)
                A[i] = random_square(rng, side, grid)
        out.append(canonicalize_squares(A))
    return out

# ========= local search =========
def local_search(seed: Squares,
                 time_budget_s: float,
                 rng: random.Random,
                 side: int,
                 alpha_ilp: float,
                 beta_lp: float,
                 gurobi_threads: int,
                 tabu_seconds: float = 20.0,
                 elite_size: int = 64,
                 neighbor_k: int = 96):
    """
    Similar skeleton to your MISR local search. Returns (elites_sorted, best_ratio).
    """
    start = time.time()
    S = canonicalize_squares(seed)
    seen: Dict[str, Tuple[float,float,float,float]] = {}  # key -> (ilp,lp,ratio,blended)
    elites: List[Tuple[float, Squares]] = []  # (ratio, squares)
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, ss: Squares):
        elites.append((score, list(ss)))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    while time.time() - start < time_budget_s:
        key = instance_key(S); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, S = rng.choice(elites)
            else: S = S[::-1]
            continue

        if key not in seen:
            ilp, lp, ratio, blended = score_ratio(S, side, alpha_ilp=alpha_ilp, beta_lp=beta_lp, gurobi_threads=gurobi_threads)
            seen[key] = (ilp, lp, ratio, blended)
            push(ratio, S)
            best = max(best, ratio)
        else:
            ilp, lp, ratio, blended = seen[key]

        cand = neighbors(S, rng, neighbor_k, side=side, grid=SIDEGRID['grid'])
        best_nb=None; best_sc=-1e9
        for S2 in cand:
            k2 = instance_key(S2)
            if k2 in seen:
                ilp2, lp2, r2, b2 = seen[k2]
            else:
                ilp2, lp2, r2, b2 = score_ratio(S2, side, alpha_ilp=alpha_ilp, beta_lp=beta_lp, gurobi_threads=gurobi_threads)
                seen[k2] = (ilp2, lp2, r2, b2)
                push(r2, S2)
            if b2 > best_sc:
                best_sc = b2; best_nb=(S2, ilp2, lp2, r2, b2)

        if best_nb:
            _, ilp2, lp2, r2, b2 = best_nb
            if b2 >= blended:
                S = best_nb[0]
            else:
                delta = b2 - blended
                T = 0.03
                if math.exp(delta/max(T,1e-6)) > random.random():
                    S = best_nb[0]
                else:
                    tabu[key] = now
                    if elites: _, S = random.choice(elites)
                    else: S = S[::-1]
            best = max(best, r2)

    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

# ========= TinyGPT (same architecture, different tokens) =========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]

class TinyGPT(nn.Module):
    """
    Decoder-only transformer over coordinate tokens with 'n' conditioning and 'grid' conditioning.
    Vocab size = BASE_VOCAB + grid_bins (we mask valid bins at sampling time).
    """
    def __init__(self, d=192, nhead=6, nlayers=3, dropout=0.1, max_grid=MAX_COORD):
        super().__init__()
        self.label_embed = nn.Embedding(BASE_VOCAB + max_grid, d)  # tokens
        self.n_embed     = nn.Embedding(MAX_N + 1, d)              # n conditioning
        self.g_embed     = nn.Embedding(MAX_COORD + 1, d)          # grid conditioning
        self.pos         = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4*d,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d, BASE_VOCAB + max_grid)

    def forward(self, tokens, n_scalar, g_scalar):
        tok_emb = self.label_embed(tokens)            # [B, L, d]
        n_emb = self.n_embed(n_scalar).unsqueeze(1)   # [B,1,d]
        g_emb = self.g_embed(g_scalar).unsqueeze(1)   # [B,1,d]
        cond  = n_emb + g_emb
        cond  = cond.expand(-1, tok_emb.size(1), -1)  # [B,L,d]
        x = self.pos(tok_emb + cond)
        L = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.enc(x, mask=causal)
        return self.out(h)                             # [B,L,vocab]

# ========= batching / training =========
@dataclass
class Batch:
    tokens: torch.Tensor    # [B, L] long
    n_scalar: torch.Tensor  # [B] long
    g_scalar: torch.Tensor  # [B] long
    targets: torch.Tensor   # [B, L] long

def make_batch(elites: List[Tuple[float, Squares]],
               B: int,
               rng: random.Random,
               grid_bins: int) -> Batch:
    tlist=[]; tglist=[]; ns=[]; gs=[]
    for _ in range(B):
        _, S = rng.choice(elites)
        tok = squares_to_tokens(S)
        tgt = tok[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
        ns.append(len(S))
        gs.append(grid_bins)
    L = max(len(t) for t in tlist)
    pad = SPECIAL["EOS"]
    tokens = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)] = t
        targets[i,:len(tt)] = tt
    return Batch(tokens=tokens.to(DEVICE),
                 n_scalar=torch.tensor(ns, dtype=torch.long, device=DEVICE),
                 g_scalar=torch.tensor(gs, dtype=torch.long, device=DEVICE),
                 targets=targets.to(DEVICE))

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch: Batch):
    model.train()
    logits = model(batch.tokens, batch.n_scalar, batch.g_scalar)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.targets.reshape(-1),
        ignore_index=SPECIAL["EOS"]
    )
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())

# ========= sampling from TinyGPT =========
@torch.no_grad()
def sample_model(model: TinyGPT,
                 n: int,
                 grid_bins: int,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_len: int = 4096) -> Squares:
    model.eval()
    toks = [SPECIAL["BOS"]]

    def step(mask_valid: List[bool]) -> int:
        inp = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
        nvec = torch.tensor([n], dtype=torch.long, device=DEVICE)
        gvec = torch.tensor([grid_bins], dtype=torch.long, device=DEVICE)
        logits = model(inp, nvec, gvec)[0, -1]  # [V]
        mask = torch.tensor(mask_valid, device=DEVICE)
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits/temperature, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        csum = torch.cumsum(sorted_probs, dim=-1)
        keep = csum <= top_p
        if not torch.any(keep): keep[0] = True
        p = torch.zeros_like(probs).scatter(0, idx[keep], sorted_probs[keep])
        p = p / p.sum()
        return int(torch.multinomial(p, 1).item())

    # Generate 2n tokens (x,y pairs)
    total_coord_tokens = 2*n
    while len(toks) - 1 < total_coord_tokens and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + MAX_COORD)
        # only allow coordinate bins in [0..grid_bins-1]
        for b in range(grid_bins):
            mask[BASE_VOCAB + b] = True
        toks.append(step(mask))

    toks.append(SPECIAL["EOS"])
    sqs = tokens_to_squares(toks)
    return canonicalize_squares(sqs)[:n]

# ========= recombination & lifting =========
def recombine_seeds(elites: List[Tuple[float, Squares]],
                    k: int,
                    rng: random.Random,
                    n: int,
                    side: int,
                    grid: int) -> List[Squares]:
    """
    Mix squares from two elite parents (A,B). Keep n squares. Jitter duplicates.
    """
    if not elites: return []
    out=[]
    for _ in range(k):
        _, A = rng.choice(elites)
        _, B = rng.choice(elites)
        pickA = rng.sample(range(len(A)), k=min(len(A), max(1, n//2)))
        S = [A[i] for i in pickA]
        # fill remainder from B
        bidx = list(range(len(B)))
        rng.shuffle(bidx)
        for j in bidx:
            if len(S) >= n: break
            S.append(B[j])
        # dedupe by small jitters if needed
        used=set()
        S2=[]
        for (x,y) in S[:n]:
            if (x,y) in used:
                x,y = clip_square(x + rng.choice([-1,0,1]),
                                  y + rng.choice([-1,0,1]), side, grid)
            used.add((x,y)); S2.append((x,y))
        out.append(canonicalize_squares(S2))
    return out

def lift_instance(sqs: Squares, n_new: int, rng: random.Random,
                  side: int, grid: int) -> Squares:
    """
    Add (n_new - n) new squares near existing ones with a bias to dense areas.
    """
    S = list(sqs)
    need = n_new - len(S)
    if need <= 0: return canonicalize_squares(S[:n_new])
    # build a simple density map: count occurrences in 2x2 neighborhoods
    H = np.zeros((grid, grid), dtype=np.int32)
    for (x,y) in S:
        H[x, y] += 1
    # sample new squares with prob ∝ (1 + local density)
    for _ in range(need):
        # draw a source anchor
        if len(S)>0 and rng.random()<0.8:
            x0, y0 = rng.choice(S)
        else:
            x0, y0 = random_square(rng, side, grid)
        # jitter around anchor
        dx = rng.randint(-max(1,side), max(1,side))
        dy = rng.randint(-max(1,side), max(1,side))
        x, y = clip_square(x0+dx, y0+dy, side, grid)
        S.append((x,y))
    return canonicalize_squares(S[:n_new])

# ========= globals for neighbor grid access (simple pass-through) =========
SIDEGRID = {"side": 4, "grid": 64}  # will be updated per-n in run

# ========= main driver (PatternBoost loop) =========
def run_patternboost(
    seed: int = 123,
    n_start: int = 12,
    n_target: int = 48,
    rounds_per_n: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    elites_to_train: int = 96,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    # search shaping
    alpha_ilp: float = 0.15,
    beta_lp: float = 0.10,
    gurobi_threads: int = 0,
    lift_step: int = 4,
    # grid / geometry
    base_grid: int = 64,
    base_side: int = 4,
    grid_growth: float = 1.05,  # grid scales gently as n grows
    motifs_on: bool = True,
    # I/O
    out_root: str = "runs",
    # optional seed pickle [(ratio, squares)] or [squares]
    seed_pkl: str = "",
    seed_top_k: int = 8,
    candidate_lines: str = "right_top",  # "right_top" or "ends" (wired in solver)
):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    n_list = ns_sequence(n_start, n_target, lift_step)
    run_dir = make_run_dirs(out_root, seed, n_list)
    print(f"[run_dir] {run_dir}")
    print(f"[device] {DEVICE}")

    # persist args
    args_path = os.path.join(run_dir, "run_args.json")
    with open(args_path, "w") as f:
        json.dump({
            "seed": seed, "n_start": n_start, "n_target": n_target,
            "rounds_per_n": rounds_per_n, "seeds_per_round": seeds_per_round,
            "local_time_per_seed": local_time_per_seed,
            "elites_to_train": elites_to_train, "batch_size": batch_size,
            "train_steps_per_round": train_steps_per_round,
            "temperature": temperature, "top_p": top_p,
            "alpha_ilp": alpha_ilp, "beta_lp": beta_lp,
            "gurobi_threads": gurobi_threads, "lift_step": lift_step,
            "base_grid": base_grid, "base_side": base_side, "grid_growth": grid_growth,
            "motifs_on": motifs_on, "candidate_lines": candidate_lines
        }, f, indent=2)

    # model/opt
    model = TinyGPT(d=192, nhead=6, nlayers=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    elites: List[Tuple[float, Squares]] = []
    def push_elite(score, S):
        elites.append((score, list(S)))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    # helpers to choose grid/side per n
    def grid_for_n(n: int) -> int:
        # gentle growth with n
        g = int(round(base_grid * (grid_growth ** max(0, (n - n_start)//max(1,lift_step)))))
        return min(MAX_COORD-1, max(base_side+2, g))

    def side_for_n(n: int) -> int:
        # keep side relatively small wrt grid
        return max(2, base_side)

    n = n_start
    best_overall = 0.0

    # initial geometry
    grid = grid_for_n(n); side = side_for_n(n)
    SIDEGRID['grid'] = grid; SIDEGRID['side'] = side

    # seeds
    seeds = seeded_pool(n, rng, seeds_per_round, side=side, grid=grid, motifs_on=motifs_on)
    if seed_pkl and os.path.exists(seed_pkl):
        try:
            with open(seed_pkl, "rb") as f:
                data = pickle.load(f)
            bag=[]
            for item in data:
                if isinstance(item, (list,tuple)) and len(item)>=2 and isinstance(item[0], (list,tuple)):
                    # (squares) or (ratio, squares)
                    if isinstance(item[0][0], (list,tuple)):  # could be nested
                        continue
                    if len(item)>=2 and isinstance(item[1], (list,tuple)) and isinstance(item[1][0], (tuple,list)):
                        ratio, S = item[:2]
                        if len(S)==n: bag.append((float(ratio), canonicalize_squares(list(map(tuple,S)))))
                    elif all(isinstance(p, (tuple,list)) for p in item):
                        S = list(map(tuple,item))
                        if len(S)==n: bag.append((0.0, canonicalize_squares(S)))
            bag.sort(key=lambda t: -t[0])
            injected = [S for _,S in bag[:seed_top_k]]
            if injected:
                take = min(len(injected), seeds_per_round)
                seeds = injected[:take] + seeds[:max(0, seeds_per_round - take)]
                print(f"[seed_inject] loaded {len(injected)} seeds from {seed_pkl} for n={n}")
        except Exception as e:
            print(f"[seed_inject] failed to read {seed_pkl}: {e}")

    while n <= n_target:
        # refresh geometry for this n
        grid = grid_for_n(n); side = side_for_n(n)
        SIDEGRID['grid'] = grid; SIDEGRID['side'] = side

        n_dir = os.path.join(run_dir, f"n{n:02d}")
        print(f"\n=== SIZE n={n} (grid={grid}, side={side}) : {len(seeds)} seeds ===")
        for r in range(rounds_per_n):
            # 1) Local search over seeds
            for S in seeds:
                es, best = local_search(
                    S,
                    time_budget_s=local_time_per_seed,
                    rng=rng,
                    side=side,
                    alpha_ilp=alpha_ilp,
                    beta_lp=beta_lp,
                    gurobi_threads=gurobi_threads,
                    elite_size=64,
                    neighbor_k=96
                )
                for (score, SS) in es:
                    push_elite(score, SS)
                if best is not None:
                    best_overall = max(best_overall, best)
            print(f"[round {r+1}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.3f}")

            # 2) Train TinyGPT on top elites
            topk = elites[:max(elites_to_train, min(32, len(elites)))]
            if topk:
                last_loss = None
                for _ in range(train_steps_per_round):
                    batch = make_batch(topk, min(batch_size, len(topk)), rng, grid_bins=grid)
                    last_loss = train_one_step(model, opt, batch)
                print(f"   trained {train_steps_per_round} steps, last loss ~ {last_loss:.3f}")

            # 3) New seeds: recombine + model sample + motif refresh + jitter
            new_seeds: List[Squares] = []
            new_seeds.extend(recombine_seeds(elites, k=max(1, seeds_per_round//4),
                                             rng=rng, n=n, side=side, grid=grid))
            while len(new_seeds) < seeds_per_round:
                elite_mut = (rng.random() < 0.25) and (len(elites)>0)
                if elite_mut:
                    _, S = rng.choice(elites[:min(64, len(elites))])
                    S = list(S)
                    # jitter a few squares
                    for _m in range(rng.randint(1, max(1, len(S)//4))):
                        i = rng.randrange(len(S))
                        x,y = S[i]
                        dx = rng.randint(-2,2); dy = rng.randint(-2,2)
                        S[i] = clip_square(x+dx,y+dy,side,grid)
                    new_seeds.append(canonicalize_squares(S))
                else:
                    S = sample_model(model, n, grid_bins=grid, temperature=temperature, top_p=top_p)
                    if (not S) or len(S)!=n:
                        S = random_squares(n, rng, side, grid)
                    if rng.random()<0.35:
                        # reverse small block jitter (invariant to canonicalization but keeps diversity)
                        S2 = list(S)
                        a = rng.randrange(len(S2)); b = rng.randrange(len(S2))
                        a,b = min(a,b), max(a,b)
                        S2[a:b+1] = list(reversed(S2[a:b+1]))
                        S = canonicalize_squares(S2)
                    new_seeds.append(S)

            if motifs_on:
                mix = [motif_grid(n, side, grid),
                       motif_diagonal(n, side, grid),
                       motif_two_clusters(n, side, grid)]
                for i in range(min(len(mix), max(2, seeds_per_round//8))):
                    new_seeds[i] = mix[i]
            seeds = new_seeds

            # 4) SAVE elites (only this n) after every round
            elites_n_only = [(sc,S) for (sc,S) in elites if len(S)==n]
            ts_round = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_name = f"round{r+1:02d}_elites_n{n:02d}_{ts_round}.pkl"
            pkl_path = os.path.join(n_dir, pkl_name)
            with open(pkl_path, "wb") as f:
                pickle.dump(elites_n_only, f)
            with open(os.path.join(n_dir, "LATEST.txt"), "w") as f:
                f.write(pkl_name + "\n")

        # 5) lift n -> n+lift_step (curriculum)
        if elites:
            n_next = n + lift_step
            lifted=[]
            for _, S in elites[:min(96, len(elites))]:
                lifted.append(lift_instance(S, n_next, rng, side=side, grid=grid_for_n(n_next)))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
        else:
            # fallback
            seeds = seeded_pool(n, rng, seeds_per_round, side=side, grid=grid, motifs_on=motifs_on)

    # print best few and save final elites
    print("\n=== BEST ELITES ===")
    for i,(score,S) in enumerate(elites[:10]):
        print(f"#{i+1} ratio={score:.4f}  n={len(S)}")
    final_path = os.path.join(run_dir, "final_elites.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(elites[:256], f)
    print(f"Saved top elites to {final_path}")
    return elites, run_dir

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=12)
    ap.add_argument("--n_target", type=int, default=48)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--alpha_ilp", type=float, default=0.15)
    ap.add_argument("--beta_lp", type=float, default=0.10)
    ap.add_argument("--gurobi_threads", type=int, default=0)
    ap.add_argument("--lift_step", type=int, default=4)
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--base_grid", type=int, default=64)
    ap.add_argument("--base_side", type=int, default=4)
    ap.add_argument("--grid_growth", type=float, default=1.05)
    ap.add_argument("--motifs_on", action="store_true")
    ap.add_argument("--seed_pkl", type=str, default="")
    ap.add_argument("--seed_top_k", type=int, default=8)
    ap.add_argument("--candidate_lines", type=str, default="right_top",
                    choices=["right_top","ends"])
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    run_patternboost(
        seed=args.seed,
        n_start=args.n_start,
        n_target=args.n_target,
        rounds_per_n=args.rounds_per_n,
        seeds_per_round=args.seeds_per_round,
        local_time_per_seed=args.local_time_per_seed,
        elites_to_train=args.elites_to_train,
        batch_size=args.batch_size,
        train_steps_per_round=args.train_steps_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        alpha_ilp=args.alpha_ilp,
        beta_lp=args.beta_lp,
        gurobi_threads=args.gurobi_threads,
        lift_step=args.lift_step,
        out_root=args.out_root,
        base_grid=args.base_grid,
        base_side=args.base_side,
        grid_growth=args.grid_growth,
        motifs_on=args.motifs_on,
        seed_pkl=args.seed_pkl,
        seed_top_k=args.seed_top_k,
        candidate_lines=args.candidate_lines,
    )

if __name__ == "__main__":
    main()
