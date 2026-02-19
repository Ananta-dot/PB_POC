#!/usr/bin/env python3
# mistr_runner.py
# Minimal PatternBoost-style loop with local search + tiny GPT,
# plus optional Chuzhoy Appendix-C seed injection.

from __future__ import annotations
import argparse, hashlib, json, math, os, pickle, random, time, warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB

warnings.filterwarnings("ignore", message=r".*TF32 behavior.*", category=UserWarning)

# -----------------------
# Device
# -----------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -----------------------
# Types & constants
# -----------------------
Seq = List[int]
Instance = Tuple[Seq, Seq]

SPECIAL = {"BOS": 0, "SEP": 1, "EOS": 2}
BASE_VOCAB = 3
MAX_N = 128  # safety ceiling

# -----------------------
# Optional Chuzhoy seeds (inline to keep this file standalone)
# -----------------------
Rect = Tuple[float,float,float,float]  # (xl, xr, yb, yt)

def _rot90(rects: List[Rect]) -> List[Rect]:
    out=[]
    for (xl,xr,yb,yt) in rects:
        pts = [(xl,yb),(xl,yt),(xr,yb),(xr,yt)]
        rot = [(y, -x) for (x,y) in pts]
        xs = [p[0] for p in rot]
        ys = [p[1] for p in rot]
        out.append((min(xs), max(xs), min(ys), max(ys)))
    return _normalize_rects(out)

def _translate_rects(rects: List[Rect], dx: float, dy: float) -> List[Rect]:
    return [(xl+dx, xr+dx, yb+dy, yt+dy) for (xl,xr,yb,yt) in rects]

def _normalize_rects(rects: List[Rect]) -> List[Rect]:
    minx = min(r[0] for r in rects)
    miny = min(r[2] for r in rects)
    return _translate_rects(rects, -minx, -miny)

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
        out.append(((a-xl)*s, (b-xl)*s, (c-yb)*s, (d-yb)*s))
    return out

def _I1() -> List[Rect]:
    s=10.0
    A = (4*s, 6*s, 4*s, 6*s)         # R1
    B = (6*s, 8*s, 6*s, 8*s)         # R2 (touch A at corner)
    C = (4*s, 6*s, 8*s,10*s)         # R3 (touch B at corner)
    D = (2*s, 4*s, 6*s, 8*s)         # touch C at corner
    E = (2*s, 4*s, 2*s, 4*s)         # touch D and A at corners
    eps=1e-3
    A = (A[0], A[1], A[2], A[3])
    B = (B[0]+eps, B[1]+eps, B[2]+eps, B[3]+eps)
    C = (C[0], C[1], C[2]+2*eps, C[3]+2*eps)
    D = (D[0]-eps, D[1]-eps, D[2], D[3])
    E = (E[0], E[1], E[2], E[3])
    return _normalize_rects([A,B,C,D,E])

def build_chuzhoy_instance(depth: int) -> List[Rect]:
    core = _I1()
    idx_R1, idx_R2, idx_R3 = 0, 1, 2
    for _ in range(1, depth):
        core = _rot90(core)
        core = _scale_to(core, 100.0, 100.0)
        core = _normalize_rects(core)
        (L,R,B,T) = _bbox(core)
        W = R-L; H = T-B
        R1j = core[idx_R1]
        R2j = core[idx_R2]
        R3j = core[idx_R3]
        w = W*0.18; h = H*0.18; x_gap = W*0.04
        y_mid = (R1j[3] + R2j[2]) / 2.0
        R1_next = (L - x_gap - w, L - x_gap, y_mid - h/2.0, y_mid + h/2.0)
        y_mid2 = (R3j[2] + R3j[3]) / 2.0
        R2_next = (R + x_gap, R + x_gap + w, y_mid2 - h/2.0, y_mid2 + h/2.0)
        top_y = T + H*0.07
        mid_x = (((R1_next[0]+R1_next[1])*0.5) + ((R2_next[0]+R2_next[1])*0.5)) * 0.5
        R3_next = (mid_x - w*0.4, mid_x + w*0.4, top_y, top_y + h*0.8)
        core = core + [R1_next, R2_next, R3_next]
        idx_R1 = len(core)-3
        idx_R2 = len(core)-2
        idx_R3 = len(core)-1
    eps = 1e-5
    out=[]
    for (xl,xr,yb,yt) in core:
        out.append((xl+eps, xr-eps, yb+eps, yt-eps))
    return out

def rects_to_sequences(rects: List[Rect]) -> Tuple[List[int], List[int]]:
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

def seeds_for_n_from_chuzhoy(n: int, rng: random.Random, variants: int = 8) -> List[Instance]:
    depth=1
    while True:
        rects = build_chuzhoy_instance(depth)
        if len(rects) >= n: break
        depth += 1
        if depth > 10_000:
            raise RuntimeError("Depth search runaway for Chuzhoy seeds.")
    H, V = rects_to_sequences(rects)
    Hn = [x for x in H if x <= n]
    Vn = [x for x in V if x <= n]
    out=[(Hn, Vn)]
    for _ in range(variants-1):
        perm = list(range(1, n+1))
        rng.shuffle(perm)
        rel = {old: new for new, old in enumerate(perm, 1)}
        out.append(([rel[x] for x in Hn], [rel[x] for x in Vn]))
    return out

# -----------------------
# Utility & encoding
# -----------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seq_spans(seq: Seq) -> List[Tuple[int, int]]:
    first = {}; spans = {}
    for idx, lab in enumerate(seq):
        if lab not in first: first[lab] = idx
        else: spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def canonicalize(H: Seq, V: Seq) -> Instance:
    order = []; seen = set()
    for x in H:
        if x not in seen:
            order.append(x); seen.add(x)
    rel = {old: new for new, old in enumerate(order, 1)}
    return [rel[x] for x in H], [rel[x] for x in V]

def instance_key(H: Seq, V: Seq) -> str:
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def random_valid_seq(n: int, rng: random.Random) -> Seq:
    seq = [i for i in range(1, n + 1) for _ in range(2)]
    rng.shuffle(seq); return seq

def motif_rainbow(n: int) -> Seq:
    return list(range(1, n+1)) + list(range(n, 0, -1))

def motif_doubled(n: int) -> Seq:
    out=[];  [out.extend([i,i]) for i in range(1, n+1)]; return out

def motif_interleave(n: int) -> Seq:
    out=[]
    for i in range(1, n+1, 2):
        j = i+1 if i+1<=n else i
        out += [i, j, i, j]
    cnt = {i:0 for i in range(1,n+1)}; fixed=[]
    for x in out:
        if cnt[x] < 2:
            fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2:
            fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_zipper(n: int) -> Seq:
    out=[]
    for i in range(1, (n//2)+1):
        j = n - i + 1
        out += [i, j, i, j]
    if n % 2 == 1:
        k = (n//2)+1; out += [k,k]
    return out[:2*n]

def motif_ladder(n: int) -> Seq:
    out=[]; a, b = 1, 2
    while len(out) < 2*n:
        out += [a, b if b<=n else a]
        a += 1; b += 1
        if a > n: a = n
        if b > n: b = n
    cnt = {i:0 for i in range(1,n+1)}; fixed=[]
    for x in out:
        if cnt[x] < 2: fixed.append(x); cnt[x]+=1
    for i in range(1,n+1):
        while cnt[i] < 2: fixed.append(i); cnt[i]+=1
    return fixed[:2*n]

def motif_corner_combo(n: int) -> Tuple[Seq, Seq]:
    return motif_rainbow(n), motif_doubled(n)

def motif_seeds(n: int) -> List[Seq]:
    S = [
        motif_rainbow(n), motif_doubled(n), motif_interleave(n),
        motif_zipper(n),  motif_ladder(n),
        list(range(1, n+1)) + list(range(1, n+1)),
        [x for pair in zip(range(1,n+1), range(1,n+1)) for x in pair],
    ]
    out=[]
    for s in S:
        if len(s)==2*n and all(s.count(i)==2 for i in range(1,n+1)):
            out.append(s)
    return out

def seeded_pool(n: int, rng: random.Random, base_count: int) -> List[Instance]:
    seeds = []; motifs = motif_seeds(n)
    for m in motifs:
        seeds.append((m[:], random_valid_seq(n, rng)))
        seeds.append((random_valid_seq(n, rng), m[:]))
    RH, RV = motif_corner_combo(n)
    seeds.append((RH[:], RV[:])); seeds.append((RV[:], RH[:]))
    for i in range(min(len(motifs) - 1, 3)):
        seeds.append((motifs[i][:], motifs[i + 1][:]))
    while len(seeds) < base_count:
        seeds.append((random_valid_seq(n, rng), random_valid_seq(n, rng)))
    return seeds[:base_count]

def lift_instance(H: Seq, V: Seq, n_new: int, rng: random.Random) -> Instance:
    assert n_new >= max(H)
    H2, V2 = H[:], V[:]
    def depth(seq: Seq) -> List[int]:
        sp = seq_spans(seq); line = [0]*(len(seq)+1)
        for (l,r) in sp:
            if l < r:
                line[l] += 1
                if r+1 < len(line): line[r+1] -= 1
        out=[]; cur=0
        for i in range(len(seq)): cur += line[i]; out.append(cur)
        return out
    def weighted_idx(weights: List[int]) -> int:
        tot = sum(w+1 for w in weights); x = rng.randrange(tot); s=0
        for i,w in enumerate(weights):
            s += (w+1)
            if x < s: return i
        return len(weights)-1
    for lab in range(max(H)+1, n_new+1):
        for seq in (H2, V2):
            d = depth(seq)
            i = weighted_idx(d); j = min(i+1, len(seq))
            seq.insert(i, lab); seq.insert(j, lab)
    return canonicalize(H2, V2)

# -----------------------
# MISR scoring (LP & ILP)
# -----------------------
def build_rects(H: Seq, V: Seq) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    X = seq_spans(H); Y = seq_spans(V); rects=[]
    for (x1,x2),(y1,y2) in zip(X,Y):
        if x1>x2: x1,x2=x2,x1
        if y1>y2: y1,y2=y2,y1
        rects.append(((x1,x2),(y1,y2)))
    return rects

def grid_points(rects: List[Tuple[Tuple[int,int], Tuple[int,int]]]) -> List[Tuple[int,int]]:
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

def solve_lp_ilp(rects, grb_threads: int = 0) -> Tuple[float, float]:
    pts = grid_points(rects); covers = covers_grid_closed(rects, pts)
    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_lp.setParam('Threads', grb_threads)
    n = len(rects)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0
    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', 0)
    if grb_threads > 0: m_ilp.setParam('Threads', grb_threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    return lp, ilp

def score_ratio(H: Seq, V: Seq, alpha_lp: float = 0.0, beta_ilp: float = 0.0, grb_threads: int = 0
               ) -> Tuple[float,float,float,float]:
    rects = build_rects(H,V)
    lp, ilp = solve_lp_ilp(rects, grb_threads=grb_threads)
    ratio = (lp/ilp) if ilp > 0 else 0.0
    n = max(H) if H else 1
    blended = ratio + alpha_lp * (lp / n) - beta_ilp * (ilp / n)
    return lp, ilp, ratio, blended

# -----------------------
# Neighborhood & local search
# -----------------------
def neighbors(H: Seq, V: Seq, rng: random.Random, k: int = 96) -> List[Instance]:
    out=[]; L = len(H)
    moves = ['swapH','swapV','moveH','moveV','blockH','blockV','revH','revV','pairH','pairV','pairHV']
    for _ in range(k):
        which = rng.choice(moves)
        A = H[:] if 'H' in which else V[:]
        i = rng.randrange(L); j = rng.randrange(L)
        if which.startswith('swap'):
            A[i], A[j] = A[j], A[i]
        elif which.startswith('move'):
            if i!=j:
                x = A.pop(i); A.insert(j, x)
        elif which.startswith('block'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b:
                blk = A[a:b+1]; del A[a:b+1]
                t = rng.randrange(len(A)+1); A[t:t]=blk
        elif which == 'pairHV':
            labs = list(set(H) & set(V))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                AH, AV = H[:], V[:]
                for S in (AH,):
                    pa = [idx for idx,x in enumerate(S) if x==a_lab]
                    pb = [idx for idx,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                for S in (AV,):
                    pa = [idx for idx,x in enumerate(S) if x==a_lab]
                    pb = [idx for idx,x in enumerate(S) if x==b_lab]
                    for ia, ib in zip(pa, pb): S[ia], S[ib] = S[ib], S[ia]
                out.append(canonicalize(AH, AV)); continue
        elif which.startswith('rev'):
            a,b = (i,j) if i<j else (j,i)
            if a!=b: A[a:b+1] = list(reversed(A[a:b+1]))
        else:
            labs = list(set(A))
            if len(labs) >= 2:
                a_lab, b_lab = rng.sample(labs, 2)
                pa = [idx for idx,x in enumerate(A) if x==a_lab]
                pb = [idx for idx,x in enumerate(A) if x==b_lab]
                if len(pa)==2 and len(pb)==2:
                    for ia, ib in zip(pa, pb): A[ia], A[ib] = A[ib], A[ia]
        out.append(canonicalize(A, V) if 'H' in which else canonicalize(H, A))
    return out

def local_search(seed: Instance,
                 time_budget_s: float,
                 rng: random.Random,
                 alpha_lp: float,
                 beta_ilp: float,
                 grb_threads: int = 0,
                 tabu_seconds: float = 20.0,
                 elite_size: int = 64,
                 neighbor_k: int = 96):
    start = time.time()
    H, V = canonicalize(*seed)
    seen: Dict[str, Tuple[float,float,float,float]] = {}
    elites: List[Tuple[float, Seq, Seq]] = []
    tabu: Dict[str, float] = {}
    best = -1.0

    def push(score: float, h: Seq, v: Seq):
        elites.append((score, h[:], v[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > elite_size: elites.pop()

    while time.time() - start < time_budget_s:
        key = instance_key(H,V); now = time.time()
        if key in tabu and (now - tabu[key] < tabu_seconds):
            if elites: _, H, V = random.choice(elites)
            else: H, V = H[::-1], V[::-1]
            continue

        if key not in seen:
            lp, ilp, ratio, blended = score_ratio(H,V, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
            seen[key] = (lp, ilp, ratio, blended)
            push(ratio, H, V)
            best = max(best, ratio)
        else:
            _, _, _, blended = seen[key]

        cand = neighbors(H,V,rng,neighbor_k)
        best_nb=None; best_sc=-1e9
        for (h2,v2) in cand:
            k2 = instance_key(h2,v2)
            if k2 in seen:
                lp2, ilp2, r2, b2 = seen[k2]
            else:
                lp2, ilp2, r2, b2 = score_ratio(h2,v2, alpha_lp=alpha_lp, beta_ilp=beta_ilp, grb_threads=grb_threads)
                seen[k2] = (lp2, ilp2, r2, b2)
                push(r2, h2, v2)
            if b2 > best_sc:
                best_sc = b2; best_nb=(h2,v2,lp2,ilp2,r2,b2)

        if best_nb:
            _,_,lp2,ilp2,r2,b2 = best_nb
            # greedy + mild SA
            if b2 >= seen[key][3]:
                H, V = best_nb[0], best_nb[1]
            else:
                delta = b2 - seen[key][3]
                T = 0.03
                if math.exp(delta/max(T,1e-6)) > random.random():
                    H, V = best_nb[0], best_nb[1]
                else:
                    tabu[key] = now
                    if elites: _, H, V = random.choice(elites)
                    else: H, V = H[::-1], V[::-1]
            best = max(best, r2)

    elites_sorted = sorted(elites, key=lambda x: -x[0])
    return elites_sorted, best

# -----------------------
# Tiny GPT
# -----------------------
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
    """Decoder-only (TransformerEncoder used causally) with n-conditioning."""
    def __init__(self, d=192, nhead=6, nlayers=3, dropout=0.1):
        super().__init__()
        self.label_embed = nn.Embedding(BASE_VOCAB + MAX_N, d)
        self.n_embed     = nn.Embedding(MAX_N + 1, d)
        self.pos         = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4*d, dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d, BASE_VOCAB + MAX_N)
    def forward(self, tokens, n_scalar):
        tok_emb = self.label_embed(tokens)
        n_emb  = self.n_embed(n_scalar).unsqueeze(1).expand(-1, tok_emb.size(1), -1)
        x = self.pos(tok_emb + n_emb)
        L = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.enc(x, mask=causal)
        return self.out(h)

def seq_to_tokens(seq: Seq) -> List[int]:
    return [BASE_VOCAB + (i-1) for i in seq]

def tokens_to_seq(tokens: List[int]) -> Seq:
    return [t - BASE_VOCAB + 1 for t in tokens]

@dataclass
class Batch:
    tokens: torch.Tensor    # [B, L] long
    n_scalar: torch.Tensor  # [B] long
    targets: torch.Tensor   # [B, L] long

def make_batch(elites: List[Tuple[float, Seq, Seq]], B: int, rng: random.Random) -> Batch:
    tlist=[]; tglist=[]; ns=[]
    for _ in range(B):
        _, H, V = rng.choice(elites)
        n = max(H)
        tok = [SPECIAL["BOS"]] + seq_to_tokens(H) + [SPECIAL["SEP"]] + seq_to_tokens(V) + [SPECIAL["EOS"]]
        tgt = tok[1:] + [SPECIAL["EOS"]]
        tlist.append(torch.tensor(tok, dtype=torch.long))
        tglist.append(torch.tensor(tgt, dtype=torch.long))
        ns.append(n)
    L = max(len(t) for t in tlist)
    pad = SPECIAL["EOS"]
    tokens = torch.full((B, L), pad, dtype=torch.long)
    targets = torch.full((B, L), pad, dtype=torch.long)
    for i,(t,tt) in enumerate(zip(tlist,tglist)):
        tokens[i,:len(t)] = t
        targets[i,:len(tt)] = tt
    return Batch(tokens=tokens.to(DEVICE),
                 n_scalar=torch.tensor(ns, dtype=torch.long, device=DEVICE),
                 targets=targets.to(DEVICE))

@torch.no_grad()
def sample_model(model: TinyGPT, n: int, temperature: float = 1.0, top_p: float = 0.9, max_len: int = 4096
                ) -> Instance:
    model.eval()
    toks = [SPECIAL["BOS"]]
    def step(mask_valid: List[bool]) -> int:
        inp = torch.tensor(toks, dtype=torch.long, device=DEVICE).unsqueeze(0)
        nvec = torch.tensor([n], dtype=torch.long, device=DEVICE)
        logits = model(inp, nvec)[0, -1]  # [V]
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
    # build H
    counts = [0]*(n+1)
    while sum(counts) < 2*n:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1
    toks.append(SPECIAL["SEP"])
    # build V
    counts = [0]*(n+1)
    while sum(counts) < 2*n and len(toks) < max_len:
        mask = [False]*(BASE_VOCAB + MAX_N)
        for i in range(1, n+1):
            if counts[i] < 2:
                mask[BASE_VOCAB + (i-1)] = True
        toks.append(step(mask))
        lab = toks[-1] - BASE_VOCAB + 1
        counts[lab] += 1
    sep_idx = toks.index(SPECIAL["SEP"])
    H_tok = toks[1:sep_idx]
    V_tok = toks[sep_idx+1:]
    H = tokens_to_seq(H_tok); V = tokens_to_seq(V_tok)
    return canonicalize(H,V)

def train_one_step(model: nn.Module, opt: torch.optim.Optimizer, batch: Batch) -> float:
    model.train()
    logits = model(batch.tokens, batch.n_scalar)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.targets.reshape(-1),
        ignore_index=SPECIAL["EOS"]
    )
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())

# -----------------------
# Utility for elites
# -----------------------
def elites_for_n(elites: List[Tuple[float, Seq, Seq]], n: int) -> List[Tuple[float, Seq, Seq]]:
    return [e for e in elites if e[1] and max(e[1]) == n]

def recombine_seeds(elites: List[Tuple[float, Seq, Seq]], k: int, rng: random.Random, n: int) -> List[Instance]:
    pool = elites_for_n(elites, n)
    if not pool: return []
    out=[]
    for _ in range(k):
        _, H1, _ = rng.choice(pool)
        _, _, V2 = rng.choice(pool)
        h, v = canonicalize(H1, V2); out.append((h,v))
    return out

def ns_sequence(n_start: int, n_target: int, step: int) -> List[int]:
    out=[]; n=n_start
    while n<=n_target: out.append(n); n+=step
    return out

def make_run_dirs(out_root: str, seed: int, n_list: List[int]) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(out_root, f"misr_run-{ts}-seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    for n in n_list:
        os.makedirs(os.path.join(run_dir, f"n{n:02d}"), exist_ok=True)
    return run_dir

# -----------------------
# Main runner
# -----------------------
def run_patternboost(
    seed: int = 123,
    n_start: int = 8,
    n_target: int = 32,
    rounds_per_n: int = 10,
    seeds_per_round: int = 32,
    local_time_per_seed: float = 3.0,
    elites_to_train: int = 96,
    batch_size: int = 32,
    train_steps_per_round: int = 60,
    temperature: float = 1.0,
    top_p: float = 0.9,
    alpha_lp: float = 0.15,
    beta_ilp: float = 0.10,
    grb_threads: int = 0,
    lift_step: int = 3,
    out_root: str = "runs",
    seed_pkl: str = "",
    seed_top_k: int = 4,
    chuzhoy_seeds: bool = False,
    chuzhoy_variants: int = 8,
):
    rng = random.Random(seed)
    seed_everything(seed)

    n_list = ns_sequence(n_start, n_target, lift_step)
    run_dir = make_run_dirs(out_root, seed, n_list)
    print(f"[run_dir] {run_dir}")

    with open(os.path.join(run_dir, "run_args.json"), "w") as f:
        json.dump({
            "seed": seed, "n_start": n_start, "n_target": n_target,
            "rounds_per_n": rounds_per_n, "seeds_per_round": seeds_per_round,
            "local_time_per_seed": local_time_per_seed,
            "elites_to_train": elites_to_train, "batch_size": batch_size,
            "train_steps_per_round": train_steps_per_round,
            "temperature": temperature, "top_p": top_p,
            "alpha_lp": alpha_lp, "beta_ilp": beta_ilp,
            "grb_threads": grb_threads, "lift_step": lift_step,
            "chuzhoy_seeds": chuzhoy_seeds, "chuzhoy_variants": chuzhoy_variants
        }, f, indent=2)

    model = TinyGPT(d=192, nhead=6, nlayers=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    print(f"Device: {DEVICE}")
    nparams = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {nparams/1e6:.2f}M")

    elites: List[Tuple[float, Seq, Seq]] = []
    def push_elite(score, H, V):
        elites.append((score, H[:], V[:]))
        elites.sort(key=lambda x: -x[0])
        if len(elites) > 4096: elites[:] = elites[:4096]

    n = n_start
    best_overall = 0.0
    seeds = seeded_pool(n, rng, seeds_per_round)

    # optional: inject Chuzhoy bundle at start of each n
    if chuzhoy_seeds:
        chu = seeds_for_n_from_chuzhoy(n, rng, variants=chuzhoy_variants)
        take = min(len(chu), max(1, seeds_per_round//4))
        seeds = chu[:take] + seeds[:max(0, seeds_per_round - take)]
        print(f"[chuzhoy] injected {take} seeds at n={n}")

    # optional: inject from pkl
    if seed_pkl:
        try:
            with open(seed_pkl, "rb") as f:
                data = pickle.load(f)
            injected=[]
            for item in data:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 3 and isinstance(item[1], list) and isinstance(item[2], list):
                        _, H, V = item[:3]
                        if H and max(H) == n: injected.append((H,V))
                    elif len(item) == 2 and isinstance(item[0], list) and isinstance(item[1], list):
                        H, V = item
                        if H and max(H) == n: injected.append((H,V))
            take = min(len(injected), seed_top_k, seeds_per_round)
            if take > 0:
                seeds = injected[:take] + seeds[:max(0, seeds_per_round - take)]
                print(f"[seed_inject] loaded {take} seeds from {seed_pkl} for n={n}")
        except Exception as e:
            print(f"[seed_inject] failed to load {seed_pkl}: {e}")

    while n <= n_target:
        n_dir = os.path.join(run_dir, f"n{n:02d}")
        print(f"\n=== SIZE n={n} ({len(seeds)} seeds) ===")

        for r in range(rounds_per_n):
            # 1) Local search for each seed
            for (H,V) in seeds:
                es, best = local_search(
                    (H,V),
                    time_budget_s=local_time_per_seed,
                    rng=rng,
                    alpha_lp=alpha_lp,
                    beta_ilp=beta_ilp,
                    grb_threads=grb_threads,
                    elite_size=64,
                    neighbor_k=96
                )
                for (score, h, v) in es:
                    push_elite(score, h, v)
                if best is not None:
                    best_overall = max(best_overall, best)
            print(f"[round {r+1}/{rounds_per_n}] elites={len(elites)} best_so_far={best_overall:.4f}")

            # 2) Train transformer on elites
            topk = elites[:max(elites_to_train, min(32, len(elites)))]
            if topk:
                last_loss = None
                for _ in range(train_steps_per_round):
                    batch = make_batch(topk, min(batch_size, len(topk)), rng)
                    last_loss = train_one_step(model, opt, batch)
                print(f"   trained {train_steps_per_round} steps, last loss ~ {last_loss:.3f}")

            # 3) New seeds for next round at same n
            new_seeds: List[Instance] = []
            new_seeds.extend(recombine_seeds(elites, k=max(1, seeds_per_round//4), rng=rng, n=n))
            while len(new_seeds) < seeds_per_round:
                elite_mut = (rng.random() < 0.25)
                pool_n = elites_for_n(elites, n)
                if elite_mut and pool_n:
                    _, h, v = rng.choice(pool_n[:min(64, len(pool_n))])
                    h = h[:]; v = v[:]
                    for S in (h, v):
                        if rng.random() < 0.6:
                            i, j = rng.randrange(len(S)), rng.randrange(len(S))
                            S[i], S[j] = S[j], S[i]
                    new_seeds.append((h, v))
                else:
                    h, v = sample_model(model, n, temperature=temperature, top_p=top_p)
                    if (not h) or (not v) or len(h)!=2*n or len(v)!=2*n:
                        h, v = random_valid_seq(n, rng), random_valid_seq(n, rng)
                    if rng.random() < 0.35:
                        for S in (h, v):
                            a = rng.randrange(len(S)); b = rng.randrange(len(S))
                            a, b = min(a,b), max(a,b)
                            if a != b: S[a:b+1] = list(reversed(S[a:b+1]))
                    new_seeds.append((h, v))

            # motif refresh up front
            mix = []
            RH, RV = motif_corner_combo(n)
            mix.append((RH[:], RV[:])); mix.append((RV[:], RH[:]))
            for m in motif_seeds(n)[:2]:
                mix.append((m[:], motif_rainbow(n)))
            for i in range(min(len(mix), max(2, seeds_per_round//8))):
                new_seeds[i] = mix[i]
            seeds = new_seeds

            # 4) SAVE elites for this n after every round
            elites_n_only = elites_for_n(elites, n)
            ts_round = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_name = f"round{r+1:02d}_elites_n{n:02d}_{ts_round}.pkl"
            pkl_path = os.path.join(n_dir, pkl_name)
            with open(pkl_path, "wb") as f:
                pickle.dump(elites_n_only, f)
            with open(os.path.join(n_dir, "LATEST.txt"), "w") as f:
                f.write(pkl_name + "\n")

        # curriculum lift to n_next
        if elites:
            n_next = n + lift_step
            lifted=[]
            for _, h, v in elites[:min(96, len(elites))]:
                h2, v2 = lift_instance(h, v, n_next, rng)
                lifted.append((h2,v2))
            seeds = lifted + seeds[:max(0, seeds_per_round - len(lifted))]
            n = n_next
            if n <= n_target and chuzhoy_seeds:
                chu = seeds_for_n_from_chuzhoy(n, rng, variants=chuzhoy_variants)
                take = min(len(chu), max(1, seeds_per_round//5))
                seeds = chu[:take] + seeds[:max(0, seeds_per_round - take)]
                print(f"[chuzhoy] injected {take} seeds at n={n} (after lift)")
        else:
            seeds = seeded_pool(n, rng, seeds_per_round)

    print("\n=== BEST ELITES ===")
    for i,(score,h,v) in enumerate(elites[:10]):
        print(f"#{i+1} ratio={score:.4f}  n={max(h)}")
    final_path = os.path.join(run_dir, "final_elites.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(elites[:256], f)
    print(f"Saved top elites to {final_path}")
    return elites, run_dir

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_start", type=int, default=8)
    ap.add_argument("--n_target", type=int, default=32)
    ap.add_argument("--rounds_per_n", type=int, default=10)
    ap.add_argument("--seeds_per_round", type=int, default=32)
    ap.add_argument("--local_time_per_seed", type=float, default=3.0)
    ap.add_argument("--elites_to_train", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--train_steps_per_round", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--alpha_lp", type=float, default=0.15)
    ap.add_argument("--beta_ilp", type=float, default=0.10)
    ap.add_argument("--grb_threads", type=int, default=8)
    ap.add_argument("--lift_step", type=int, default=3)
    ap.add_argument("--out_root", type=str, default="runs",
                    help="root directory for run outputs")
    ap.add_argument("--seed_pkl", type=str, default="",
                    help="Optional pickle with seeds; each item is (ratio,H,V) or (H,V)")
    ap.add_argument("--seed_top_k", type=int, default=4,
                    help="How many seeds to inject from seed_pkl at n_start")
    ap.add_argument("--chuzhoy_seeds", action="store_true",
                    help="Inject Chuzhoy Appendix-C-style seeds at each n")
    ap.add_argument("--chuzhoy_variants", type=int, default=8,
                    help="How many Chuzhoy variants to generate per n")
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
        alpha_lp=args.alpha_lp,
        beta_ilp=args.beta_ilp,
        grb_threads=args.grb_threads,
        lift_step=args.lift_step,
        out_root=args.out_root,
        seed_pkl=args.seed_pkl,
        seed_top_k=args.seed_top_k,
        chuzhoy_seeds=args.chuzhoy_seeds,
        chuzhoy_variants=args.chuzhoy_variants,
    )

if __name__ == "__main__":
    main()
