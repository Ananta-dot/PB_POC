#!/usr/bin/env python3
# transformer.py
# Train a simple causal Transformer LM over (H', V') sequences and then generate new instances.
# Also verify candidates by solving MIS and LP relaxation on the rectangle intersection graph.

import os, sys, math, json, time, random, argparse, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Globals / Types
# -------------------------
Seq = List[int]
Rect = Tuple[Tuple[int,int], Tuple[int,int]]  # ((x1,x2),(y1,y2)) in index coords
RNG = random.Random(12345)

PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_OFFSET = 10  # first numeric token is NUM_OFFSET + 0

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# -------------------------
# Utilities
# -------------------------
def parse_seq(val) -> Seq:
    """Parse a semicolon-separated integer sequence; accept NaN/None/float gracefully."""
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    if isinstance(val, float):
        s = str(val)
        if s.lower() == 'nan':
            return []
        return []
    if not isinstance(val, str):
        val = str(val)
    s = val.strip()
    if not s:
        return []
    out = []
    for x in s.split(';'):
        x = x.strip()
        if x == "" or x.lower() == "nan":
            continue
        try:
            out.append(int(x))
        except Exception:
            continue
    return out

def _positions(seq: Seq, lab: int) -> List[int]:
    return [i for i, v in enumerate(seq) if v == lab]

def build_rects_from_seqs(H: Seq, V: Seq, strict: bool = False) -> Tuple[List[int], List[Rect], List[str]]:
    """Return (labels, rects, issues). 'labels' sorted ascending; rects[i] corresponds to labels[i]."""
    issues = []
    cntH, cntV = {}, {}
    for x in H:
        if x != 0:
            cntH[x] = cntH.get(x, 0) + 1
    for x in V:
        if x != 0:
            cntV[x] = cntV.get(x, 0) + 1

    badH = sorted([l for l, c in cntH.items() if c != 2])
    badV = sorted([l for l, c in cntV.items() if c != 2])
    onlyH = sorted(set(cntH) - set(cntV))
    onlyV = sorted(set(cntV) - set(cntH))

    if badH: issues.append(f"H': labels not occurring exactly twice: {badH}")
    if badV: issues.append(f"V': labels not occurring exactly twice: {badV}")
    if onlyH: issues.append(f"labels only in H': {onlyH}")
    if onlyV: issues.append(f"labels only in V': {onlyV}")

    label_set = sorted(l for l in (set(cntH) & set(cntV)) if cntH[l] == 2 and cntV[l] == 2)

    if strict and (badH or badV or onlyH or onlyV):
        return [], [], issues

    rects: List[Rect] = []
    labels: List[int] = []
    for l in label_set:
        hx = sorted(_positions(H, l))
        vy = sorted(_positions(V, l))
        if len(hx) == 2 and len(vy) == 2:
            x1, x2 = hx
            y1, y2 = vy
            if x1 == x2 or y1 == y2:
                issues.append(f"degenerate span for label {l}: x={hx}, y={vy}")
                continue
            rects.append(((x1, x2), (y1, y2)))
            labels.append(l)
        else:
            issues.append(f"unexpected multiplicity for label {l}: H={hx}, V={vy}")
    return labels, rects, issues

def rects_intersect(r1: Rect, r2: Rect) -> bool:
    # Open-interval intersection (touching edges = non-intersecting)
    (x1, x2), (y1, y2) = r1
    (u1, u2), (v1, v2) = r2
    return (x1 < u2) and (u1 < x2) and (y1 < v2) and (v1 < y2)

def intersection_adj(rects: List[Rect]):
    n = len(rects)
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        xi = rects[i]
        for j in range(i+1, n):
            xj = rects[j]
            if rects_intersect(xi, xj):
                adj[i][j] = adj[j][i] = True
    return adj

def mis_exact_and_lp(adj, use_gurobi: bool = True):
    """Return (mis_val, mis_set, lp_val). Falls back to greedy if Gurobi unavailable or disabled."""
    n = len(adj)
    if n == 0:
        return 0, [], 0.0
    try:
        if not use_gurobi:
            raise RuntimeError("Gurobi disabled by flag")
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model()
        m.Params.OutputFlag = 0
        x = m.addVars(n, vtype=GRB.BINARY, name="x")
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    m.addConstr(x[i] + x[j] <= 1)
        m.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
        m.optimize()
        mis_val = int(round(m.ObjVal))
        mis_set = [i for i in range(n) if x[i].X > 0.5]

        mlp = gp.Model()
        mlp.Params.OutputFlag = 0
        y = mlp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    mlp.addConstr(y[i] + y[j] <= 1.0)
        mlp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
        mlp.optimize()
        lp_val = float(mlp.ObjVal)
        return mis_val, mis_set, lp_val
    except Exception:
        degrees = [sum(row) for row in adj]
        order = sorted(range(n), key=lambda i: degrees[i])
        chosen = []
        banned = [False]*n
        for i in order:
            if not banned[i]:
                chosen.append(i)
                for j in range(n):
                    if adj[i][j]:
                        banned[j] = True
        mis_val = len(chosen)
        lp_val = float(mis_val)
        return mis_val, chosen, lp_val

# -------------------------
# Codec
# -------------------------
@dataclass
class Codec:
    max_label: int  # highest positive label allowed in data/generation

    @property
    def vocab_size(self) -> int:
        return NUM_OFFSET + (self.max_label + 1)  # include token for 0

    def tok_for_num(self, t: int) -> int:
        t = max(0, min(self.max_label, int(t)))
        return NUM_OFFSET + t

    def num_for_tok(self, tok: int) -> int:
        return max(0, tok - NUM_OFFSET)

    def encode_pair(self, H: Seq, V: Seq) -> List[int]:
        out = [BOS]
        out.extend(self.tok_for_num(t) for t in H)
        out.append(SEP)
        out.extend(self.tok_for_num(t) for t in V)
        out.append(EOS)
        return out

    def decode_pair(self, toks: List[int]) -> Tuple[Seq, Seq]:
        H, V = [], []
        mode = "H"
        for tok in toks:
            if tok == BOS:
                continue
            if tok == SEP:
                mode = "V"; continue
            if tok == EOS:
                break
            if tok < NUM_OFFSET:
                continue
            n = self.num_for_tok(tok)
            if mode == "H": H.append(n)
            else:           V.append(n)
        return H, V

# -------------------------
# Model (GPT-lite)
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int=256, nhead: int=8, num_layers: int=6, dim_feedforward: int=1024, dropout: float=0.1, max_len: int=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x, attn_mask=None):
        h = self.tok_emb(x)
        h = self.pos(h)
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.tr(h, mask=causal)
        logits = self.lm_head(h)
        return logits

# -------------------------
# Data loading with ceiling & augmentation
# -------------------------
def load_csv_as_action_pairs(csv_path: str, label_ceiling: int = 0, augment_copies: int = 0):
    df = pd.read_csv(csv_path)
    raw_pairs = []
    max_lab_seen = 0
    for _, row in df.iterrows():
        H = parse_seq(row.get("H_prime"))
        V = parse_seq(row.get("V_prime"))
        if len(H) == 0 or len(V) == 0:
            continue
        max_lab_seen = max(max_lab_seen, max([0]+H+[0]+V))
        raw_pairs.append((H, V))
    L = max(max_lab_seen, label_ceiling) if label_ceiling > 0 else max_lab_seen
    L = max(L, 1)
    codec = Codec(max_label=L)

    def relabel_once(H, V, L):
        labs = sorted({t for t in H+V if t > 0})
        if len(labs) > L:
            return None
        slots = list(range(1, L+1))
        RNG.shuffle(slots)
        mapping = {lab: slots[i] for i, lab in enumerate(labs)}
        def f(seq):
            out = []
            for t in seq:
                if t <= 0: out.append(0)
                else:      out.append(mapping[t])
            return out
        return f(H), f(V)

    rows = []
    for H, V in raw_pairs:
        Hc = [min(max(t,0), L) for t in H]
        Vc = [min(max(t,0), L) for t in V]
        rows.append(codec.encode_pair(Hc, Vc))
        for _ in range(max(0, augment_copies)):
            rv = relabel_once(H, V, L)
            if rv is None: break
            Ha, Va = rv
            rows.append(codec.encode_pair(Ha, Va))

    max_len = max((len(r) for r in rows), default=0)
    print(f"Loaded {len(rows)} rows; skipped {len(df)-len(raw_pairs)}. Using label_ceiling={L}, max seq len={max_len}")
    return rows, codec

# -------------------------
# Dataset / Collate
# -------------------------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[List[int]]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return torch.tensor(self.rows[idx], dtype=torch.long)

def collate_pad(batch: List[torch.Tensor]):
    T = max(x.size(0) for x in batch)
    out = torch.full((len(batch), T), PAD, dtype=torch.long)
    for i, x in enumerate(batch):
        out[i, :x.size(0)] = x
    y = out.clone()
    y[:, :-1] = out[:, 1:]
    y[:, -1] = EOS
    return out, y

# -------------------------
# Training
# -------------------------
def split_train_val(rows, val_ratio=0.10):
    idx = list(range(len(rows)))
    RNG.shuffle(idx)
    n_val = max(1, int(round(len(rows)*val_ratio)))
    val_idx = set(idx[:n_val])
    train = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    return train, val

def loss_on_batch(model, batch_inputs, batch_targets):
    logits = model(batch_inputs)
    B, T, V = logits.size()
    loss = F.cross_entropy(logits.view(B*T, V), batch_targets.view(B*T), ignore_index=PAD)
    return loss

# -------------------------
# Sampling
# -------------------------
@torch.no_grad()
def sample_tokens(model, codec: Codec, max_len: int, temperature: float=1.0, topk: int=0, topp: float=0.0, device="cpu"):
    model.eval()
    x = torch.tensor([[BOS]], dtype=torch.long, device=device)
    for _ in range(max_len-1):
        logits = model(x)[:, -1, :]
        logits[:, PAD] = -1e9
        logits[:, BOS] = -1e9
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        if topk and topk > 0:
            kth = torch.topk(logits, min(topk, logits.size(-1)), dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
        if topp and topp > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cdf = torch.cumsum(probs, dim=-1)
            mask = cdf > topp
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, -1e9)
            unsorted = torch.full_like(logits, -1e9)
            unsorted.scatter_(1, sorted_idx, sorted_logits)
            logits = unsorted
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_tok], dim=1)
        if next_tok.item() == EOS:
            break
    toks = x[0].tolist()
    H, V = codec.decode_pair(toks)
    return H, V, toks

# -------------------------
# Generation filters
# -------------------------
def valid_pair(H: Seq, V: Seq, n: int, max_zeros_h: int, max_zeros_v: int) -> bool:
    if H.count(0) > max_zeros_h: return False
    if V.count(0) > max_zeros_v: return False
    cntH, cntV = {}, {}
    for t in H:
        if t>0: cntH[t] = cntH.get(t,0)+1
    for t in V:
        if t>0: cntV[t] = cntV.get(t,0)+1
    labs = sorted(set(cntH) & set(cntV))
    if len(labs) != n: return False
    if any(cntH[l] != 2 for l in labs): return False
    if any(cntV[l] != 2 for l in labs): return False
    return True

def to_semicolon(seq: Seq) -> str:
    return ";".join(str(int(x)) for x in seq)

def inst_key_from_HV(H: Seq, V: Seq) -> str:
    s = f"H={to_semicolon(H)}|V={to_semicolon(V)}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# -------------------------
# Verify writer
# -------------------------
def write_candidates_csv(cands: List[Dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(cands)
    df.to_csv(out_csv, index=False)
    kept = len(df)
    tries = sum(c.get("_tries", 1) for c in cands) if cands else 0
    print(f"Wrote candidates → {out_csv} (kept {kept}, tries {tries})")

def write_verified_csv(rows: List[Dict], out_csv: str, min_gap: float, max_gap_seen: Optional[float]):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    columns = [
        "idx","instance_key","n","issues_count",
        "mis_exact","mis_lp_relax","lp_over_int_gap",
        "source","H_prime","V_prime"
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values(["lp_over_int_gap", "n"], ascending=[False, True])
        df.to_csv(out_csv, index=False)
        print(f"Wrote verified → {out_csv} (verified {len(df)} with gap ≥ {min_gap})")
    else:
        pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
        mg = "NaN" if max_gap_seen is None else f"{max_gap_seen:.6f}"
        print(f"Wrote verified → {out_csv} (0 candidates ≥ min_gap={min_gap}). Max gap observed = {mg}")

# -------------------------
# Commands
# -------------------------
def add_model_args(p):
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=1024)  # CLI arg kept as dim_ff
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p

def cmd_train(args):
    set_seed(args.seed)
    rows, codec = load_csv_as_action_pairs(
        args.csv,
        label_ceiling=args.label_ceiling,
        augment_copies=args.augment_copies
    )
    if len(rows) < 2:
        print("Not enough rows to train.", file=sys.stderr)
        sys.exit(1)
    train_rows, val_rows = split_train_val(rows, val_ratio=0.10)
    print(f"Train={len(train_rows)}  Val={len(val_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerLM(
        vocab_size=codec.vocab_size, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        dropout=args.dropout, max_len=args.max_len
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = torch.utils.data.DataLoader(SeqDataset(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader   = torch.utils.data.DataLoader(SeqDataset(val_rows),   batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = loss_on_batch(model, xb, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses)) if losses else float("nan")

        model.eval()
        vlosses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                vlosses.append(loss_on_batch(model, xb, yb).item())
        val_loss = float(np.mean(vlosses)) if vlosses else float("nan")
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "codec": {"max_label": codec.max_label},
                "model_cfg": {
                    # store using canonical key 'dim_feedforward'
                    "d_model": args.d_model, "nhead": args.nhead, "num_layers": args.num_layers,
                    "dim_feedforward": args.dim_ff, "dropout": args.dropout, "max_len": args.max_len
                }
            }, args.ckpt)
            print(f"  ↳ saved best → {args.ckpt} (val {best_val:.4f})")

def load_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    max_label = ckpt["codec"]["max_label"]
    codec = Codec(max_label=max_label)
    # default config
    cfg = ckpt.get("model_cfg", {"d_model":256,"nhead":8,"num_layers":6,"dim_feedforward":1024,"dropout":0.1,"max_len":256})
    # backward-compat: older ckpts saved 'dim_ff'
    if "dim_ff" in cfg and "dim_feedforward" not in cfg:
        cfg["dim_feedforward"] = cfg.pop("dim_ff")
    model = CausalTransformerLM(vocab_size=codec.vocab_size, **cfg)
    model.load_state_dict(ckpt["model"])
    return model, codec, cfg

def cmd_generate(args):
    set_seed(args.seed)
    model, codec, cfg = load_ckpt(args.ckpt)
    if args.n > codec.max_label:
        print(f"Requested n={args.n} but ckpt supports only labels up to {codec.max_label}. "
              f"Retrain with --label_ceiling ≥ {args.n}, then try again.")
        write_verified_csv([], args.verify_out, args.min_gap, max_gap_seen=None)
        write_candidates_csv([], args.out_csv)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate candidates
    cands = []
    tries = 0
    target = args.count
    max_tries = max(args.count * 10, args.count)
    while len(cands) < target and tries < max_tries:
        tries += 1
        H, V, toks = sample_tokens(
            model, codec, max_len=cfg.get("max_len", 256),
            temperature=args.temperature, topk=args.topk, topp=args.topp, device=device
        )
        if valid_pair(H, V, args.n, args.max_zeros_h, args.max_zeros_v):
            key = inst_key_from_HV(H, V)
            cands.append({
                "idx": len(cands),
                "instance_key": key,
                "H_prime": to_semicolon(H),
                "V_prime": to_semicolon(V),
                "source": "gpt_gen",
                "_tries": 1
            })

    write_candidates_csv(cands, args.out_csv)

    # Verify & score
    ver_rows = []
    max_gap_seen = None
    for i, c in enumerate(cands):
        H = parse_seq(c["H_prime"]); V = parse_seq(c["V_prime"])
        labels, rects, issues = build_rects_from_seqs(H, V, strict=False)
        adj = intersection_adj(rects)
        mis_val, mis_set, lp_val = mis_exact_and_lp(adj, use_gurobi=(not args.no_gurobi))
        gap = (lp_val / mis_val) if mis_val > 0 else float("nan")
        if not math.isnan(gap):
            max_gap_seen = gap if (max_gap_seen is None or gap > max_gap_seen) else max_gap_seen
        row = {
            "idx": i,
            "instance_key": c["instance_key"],
            "n": len(labels),
            "issues_count": len(issues),
            "mis_exact": int(mis_val),
            "mis_lp_relax": float(lp_val),
            "lp_over_int_gap": (float(gap) if not math.isnan(gap) else None),
            "source": c.get("source",""),
            "H_prime": c["H_prime"],
            "V_prime": c["V_prime"]
        }
        if row["lp_over_int_gap"] is not None and row["lp_over_int_gap"] >= args.min_gap:
            ver_rows.append(row)

    write_verified_csv(ver_rows, args.verify_out, args.min_gap, max_gap_seen=max_gap_seen)

# -------------------------
# Main / CLI
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Train and generate MISR rectangle instances via a causal Transformer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="train on csv of H'/V'")
    pt.add_argument("--csv", type=str, required=True)
    pt.add_argument("--ckpt", type=str, required=True)
    pt.add_argument("--epochs", type=int, default=25)
    pt.add_argument("--batch_size", type=int, default=128)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--label_ceiling", type=int, default=0, help="force vocab labels up to this ID (>= n you want)")
    pt.add_argument("--augment_copies", type=int, default=3, help="random relabel copies per row")
    add_model_args(pt)

    # generate
    pg = sub.add_parser("generate", help="generate & verify candidates")
    pg.add_argument("--ckpt", type=str, required=True)
    pg.add_argument("--n", type=int, required=True)
    pg.add_argument("--count", type=int, default=300)
    pg.add_argument("--out_csv", type=str, required=True)
    pg.add_argument("--verify_out", type=str, required=True)
    pg.add_argument("--min_gap", type=float, default=1.10)
    pg.add_argument("--topk", type=int, default=50)
    pg.add_argument("--topp", type=float, default=0.95)
    pg.add_argument("--temperature", type=float, default=1.0)
    pg.add_argument("--max_zeros_h", type=int, default=10)
    pg.add_argument("--max_zeros_v", type=int, default=10)
    pg.add_argument("--no-gurobi", action="store_true")
    add_model_args(pg)

    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
