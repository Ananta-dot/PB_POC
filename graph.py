# plot_and_graph_batch_info_preserving.py
import os, json, pickle, hashlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

# ========== config ==========
PICKLE_PATH    = 'misr_elites_1.pkl'
PLOTS_OUTDIR   = 'plots/n12'
GRAPHS_OUTDIR  = 'graphs/n12'
LIMIT          = 256
COUNT_TOUCHING = True
GRB_THREADS    = 0
OUTPUT_FLAG    = 0
EPS            = 1e-12

EXPORT_FULL_GRAPH     = True    # export graph for ALL rectangles (no filtering)
EXPORT_FILTERED_GRAPH = True    # export graph for kept rectangles (x_lp>0 or y_ilp=1)
EXPORT_GRID_HYPER     = True    # export rectangle↔grid-point hypergraph (covers sets)

### NEW ###
EXPORT_FILTERED_LISTS = True    # write filtered lists per-instance + a summary bundle

# ---------- core helpers ----------
def instance_key(H, V):
    s = ','.join(map(str, H)) + '|' + ','.join(map(str, V))
    return hashlib.blake2b(s.encode(), digest_size=16).hexdigest()

def seq_spans(seq):
    first, spans = {}, {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    return [spans[i] for i in range(1, n + 1)]

def build_rects(H, V):
    X, Y = seq_spans(H), seq_spans(V)
    rects = []
    for (x1, x2), (y1, y2) in zip(X, Y):
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        rects.append(((x1, x2), (y1, y2)))
    return rects  # index order = label 1..n

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

# ---------- optimization (LP + ILP) ----------
def solve_lp_ilp_with_vars(rects, threads=0, output_flag=0):
    import gurobipy as gp
    from gurobipy import GRB

    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    n = len(rects)

    # LP
    m_lp = gp.Model("misr_lp"); m_lp.setParam('OutputFlag', output_flag)
    if threads > 0: m_lp.setParam('Threads', threads)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0
    x_vals = [float(x[i].X) for i in range(n)] if m_lp.status == GRB.OPTIMAL else [0.0]*n

    # ILP
    m_ilp = gp.Model("misr_ilp"); m_ilp.setParam('OutputFlag', output_flag)
    if threads > 0: m_ilp.setParam('Threads', threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S: m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    y_vals = [int(round(y[i].X)) for i in range(n)] if m_ilp.status == GRB.OPTIMAL else [0]*n

    return lp, ilp, x_vals, y_vals, pts, covers

# ---------- filtering ----------
def filter_by_solution(rects, x_lp, y_ilp, eps=1e-12):
    kept_rects, kept_x, kept_y, kept_ids = [], [], [], []
    for i, r in enumerate(rects, start=1):
        xi = x_lp[i-1]; yi = y_ilp[i-1]
        if (abs(xi) > eps) or (yi == 1):
            kept_rects.append(r); kept_x.append(xi); kept_y.append(yi); kept_ids.append(i)
    return kept_rects, kept_x, kept_y, kept_ids

# ---------- plotting ----------
def plot_instance(H, V, title=None, save_path=None,
                  highlight_ilp=True, threads=0, output_flag=0,
                  show=True, filter_zero=True, eps=EPS):
    rects_all = build_rects(H, V)
    lp, ilp, x_lp, y_ilp, _, _ = solve_lp_ilp_with_vars(rects_all, threads=threads, output_flag=output_flag)

    if filter_zero:
        rects, x_f, y_f, ids = filter_by_solution(rects_all, x_lp, y_ilp, eps=eps)
    else:
        rects, x_f, y_f, ids = rects_all, x_lp, y_ilp, list(range(1, len(rects_all)+1))

    kept = len(rects); removed = len(rects_all) - kept
    if kept == 0:
        print("[INFO] Nothing to plot after filtering.")
        return lp, ilp, x_lp, y_ilp, ids

    n_orig = max(H) if H else 0
    fig, ax = plt.subplots(figsize=(7, 7))
    for k, ((x1, x2), (y1, y2)) in enumerate(rects):
        w = (x2 - x1); h = (y2 - y1)
        selected = (y_f[k] == 1) if highlight_ilp else False
        edgecolor = 'tab:red' if selected else '0.4'
        lw = 2.8 if selected else 1.2
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=lw, edgecolor=edgecolor))
        ax.text(x1 + w/2, y1 + h/2, f"{ids[k]}", ha="center", va="center", fontsize=10)

    ax.set_xlim(-1, 2*n_orig); ax.set_ylim(-1, 2*n_orig)
    ax.set_xlabel("H positions"); ax.set_ylabel("V positions")
    ax.set_aspect("equal", "box"); ax.grid(True, linestyle=":", linewidth=0.6); ax.invert_yaxis()
    if title is None:
        title = f"Rectangles | LP={lp:.2f}, ILP={ilp:.2f}, kept={kept}, removed={removed}"
    ax.set_title(title)

    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=220); plt.close(fig)
    else:
        plt.tight_layout(); plt.show()
    return lp, ilp, x_lp, y_ilp, ids

# ---------- richer relations for edges ----------
def overlap_kind(r1, r2):
    (xa1, xa2), (ya1, ya2) = r1
    (xb1, xb2), (yb1, yb2) = r2
    ox = max(0, min(xa2, xb2) - max(xa1, xb1))
    oy = max(0, min(ya2, yb2) - max(ya1, yb1))
    touch_x = (min(xa2, xb2) - max(xa1, xb1) == 0)
    touch_y = (min(ya2, yb2) - max(ya1, yb1) == 0)
    if ox > 0 and oy > 0: kind = 'proper'
    elif (ox > 0 and touch_y) or (oy > 0 and touch_x): kind = 'edge'
    elif touch_x and touch_y: kind = 'corner'
    else: kind = 'none'
    return kind, float(ox), float(oy)

def rel1d(a1, a2, b1, b2):
    # coarse Allen-ish relation for 1D intervals
    if a2 < b1:  return 'before'
    if b2 < a1:  return 'after'
    if a1 == b1 and a2 == b2: return 'equal'
    if a1 == b1 and a2 <  b2: return 'starts'
    if a2 == b2 and a1 >  b1: return 'finishes'
    if a1 <= b1 and a2 >= b2: return 'contains'
    if b1 <= a1 and b2 >= a2: return 'inside'
    if a2 == b1 or b2 == a1: return 'meet'
    return 'overlap'

# ---------- graph builders ----------
def build_graph_with_attrs(rects, node_ids, x_lp=None, y_ilp=None, pts=None, covers=None, title=None):
    """
    Information-preserving: node attrs include exact (x1,x2,y1,y2).
    Edge attrs include overlap kind and 1-D relations.
    """
    G = nx.Graph()
    n = len(rects)
    # nodes
    for k, nid in enumerate(node_ids):
        (x1,x2),(y1,y2) = rects[k]
        width = x2 - x1; height = y2 - y1
        G.add_node(nid,
                   original_index=int(nid),
                   x1=int(x1), x2=int(x2), y1=int(y1), y2=int(y2),
                   width=int(width), height=int(height), area=int(width*height),
                   lp=float(x_lp[k]) if x_lp else 0.0,
                   ilp=int(y_ilp[k]) if y_ilp else 0)
    # edges
    for i in range(n):
        for j in range(i+1, n):
            kind, ox, oy = overlap_kind(rects[i], rects[j])
            if kind != 'none':
                (xa1,xa2),(ya1,ya2) = rects[i]
                (xb1,xb2),(yb1,yb2) = rects[j]
                G.add_edge(node_ids[i], node_ids[j],
                           kind=kind, ox=ox, oy=oy,
                           x_rel=rel1d(xa1,xa2,xb1,xb2),
                           y_rel=rel1d(ya1,ya2,yb1,yb2))
    if title: G.graph['title'] = title
    return G

def save_graph_bundle(G, basepath_no_ext):
    # CSV edgelist with attributes
    csv_path = basepath_no_ext + ".csv"
    with open(csv_path, "w") as f:
        # header
        f.write("u,v,kind,ox,oy,x_rel,y_rel\n")
        for u, v, d in G.edges(data=True):
            f.write(f"{u},{v},{d.get('kind','')},{d.get('ox',0)},{d.get('oy',0)},{d.get('x_rel','')},{d.get('y_rel','')}\n")
    # GraphML (preserves node/edge attrs)
    nx.write_graphml(G, basepath_no_ext + ".graphml")
    # quick PNG
    try:
        plt.figure(figsize=(6,6))
        pos = nx.kamada_kawai_layout(G) if len(G) > 0 else {}
        nx.draw(G, pos, with_labels=True, node_size=520, font_size=8)
        if 'title' in G.graph: plt.title(G.graph['title'])
        plt.tight_layout(); plt.savefig(basepath_no_ext + ".png", dpi=220); plt.close()
    except Exception as e:
        print(f"[WARN] PNG render failed for {basepath_no_ext}: {e}")

# ---------- reconstruct / sanity ----------
def rects_to_HV(rects):
    # Given ((x1,x2),(y1,y2)) for labels 1..n in order, rebuild H,V by placing each label at its endpoints
    max_x = max(max(r[0]) for r in rects) if rects else -1
    max_y = max(max(r[1]) for r in rects) if rects else -1
    H = [0]*(max_x+1) if max_x >= 0 else []
    V = [0]*(max_y+1) if max_y >= 0 else []
    for i, ((x1,x2),(y1,y2)) in enumerate(rects, start=1):
        H[x1] = i; H[x2] = i; V[y1] = i; V[y2] = i
    return H, V

### NEW ###
def relabel_rects(kept_rects, kept_ids):
    """
    Map original ids -> 1..k in kept order.
    Returns: rects_relabelled (same geometry), id_map dict (old->new), H', V' for relabelled rects.
    """
    id_map = {old_id: new_id for new_id, old_id in enumerate(kept_ids, start=1)}
    rects_relab = list(kept_rects)  # geometry unchanged; label meaning changes via id_map
    Hp, Vp = rects_to_HV(rects_relab)  # labels 1..k in kept order
    return rects_relab, id_map, Hp, Vp

# ---------- hypergraph export (rectangles ↔ grid points) ----------
def save_grid_hypergraph(pts, covers, basepath_no_ext):
    # CSV: gx,gy,rect_ids (semicolon-separated)
    path = basepath_no_ext + "_gridcliques.csv"
    with open(path, "w") as f:
        f.write("gx,gy,rect_ids\n")
        for (gx,gy), S in zip(pts, covers):
            if S:  # keep non-empty cliques only
                ids = [s+1 for s in S]  # 1-based rect IDs
                f.write(f"{gx},{gy},{';'.join(map(str,ids))}\n")

# ---------- main ----------
def main():
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(PLOTS_OUTDIR, exist_ok=True)
    os.makedirs(GRAPHS_OUTDIR, exist_ok=True)

    ### NEW ### aggregate summary we’ll write at the end
    filtered_summary = []

    N = min(LIMIT, len(data))
    print(f"Found {len(data)} instances, exporting {N} of them.")
    for idx in range(N):
        try:
            # expecting records like (ratio, H, V) or (score, H, V, ...)
            H, V = data[idx][1], data[idx][2]
            key = instance_key(H, V)
            rects_all = build_rects(H, V)
            lp, ilp, x_lp, y_ilp, pts, covers = solve_lp_ilp_with_vars(rects_all, threads=GRB_THREADS, output_flag=OUTPUT_FLAG)
            ratio = (lp/ilp) if ilp>0 else 0.0

            # plot (filtered view for legibility)
            plot_path  = os.path.join(PLOTS_OUTDIR, f"misr_{idx:03d}.png")
            plot_title = f"MISR {idx}  LP={lp:.2f} ILP={ilp:.2f} ratio={ratio:.3f}"
            _, _, _, _, kept_ids_plot = plot_instance(
                H, V, title=plot_title, save_path=plot_path,
                highlight_ilp=True, threads=GRB_THREADS, output_flag=OUTPUT_FLAG,
                show=False, filter_zero=True, eps=EPS)

            # filtered set for graph (keeps: x_lp>0 OR y_ilp=1)
            kept_rects, kept_x, kept_y, kept_ids = filter_by_solution(rects_all, x_lp, y_ilp, eps=EPS)

            # ----- FULL graph (no filtering) -----
            if EXPORT_FULL_GRAPH:
                node_ids_full = list(range(1, len(rects_all)+1))
                Gfull = build_graph_with_attrs(rects_all, node_ids_full, x_lp=x_lp, y_ilp=y_ilp)
                base_full = os.path.join(GRAPHS_OUTDIR, f"misr_graph_{idx:03d}_full")
                Gfull.graph['title'] = f"Intersection Graph (full) — {idx} (|V|={Gfull.number_of_nodes()}, |E|={Gfull.number_of_edges()})"
                save_graph_bundle(Gfull, base_full)
                if EXPORT_GRID_HYPER:
                    save_grid_hypergraph(pts, covers, base_full)
            else:
                base_full = ""

            # ----- FILTERED graph -----
            if EXPORT_FILTERED_GRAPH and len(kept_rects)>0:
                Gf = build_graph_with_attrs(kept_rects, kept_ids, x_lp=kept_x, y_ilp=kept_y)
                base_filt = os.path.join(GRAPHS_OUTDIR, f"misr_graph_{idx:03d}_filtered")
                Gf.graph['title'] = f"Intersection Graph (filtered) — {idx} (|V|={Gf.number_of_nodes()}, |E|={Gf.number_of_edges()})"
                save_graph_bundle(Gf, base_filt)
            else:
                base_filt = ""

            # ----- metadata (makes it bijective) -----
            meta = {
                "instance_index": idx,
                "instance_key": key,
                "H": H, "V": V,
                "rects": [((int(a),int(b)),(int(c),int(d))) for ((a,b),(c,d)) in rects_all],
                "lp": lp, "ilp": ilp, "ratio": ratio,
                "x_lp": x_lp, "y_ilp": y_ilp,
                "kept_ids": kept_ids,
                "removed_ids": [i for i in range(1,len(rects_all)+1) if i not in kept_ids],
                "paths": {
                    "plot_png": plot_path,
                    "graph_full_prefix": EXPORT_FULL_GRAPH and base_full or "",
                    "graph_filtered_prefix": (EXPORT_FILTERED_GRAPH and len(kept_rects)>0) and base_filt or ""
                }
            }
            meta_path = os.path.join(GRAPHS_OUTDIR, f"misr_meta_{idx:03d}.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # ----- reconstruct sanity check (from rect attributes) -----
            H2, V2 = rects_to_HV(rects_all)
            if H2 != H or V2 != V:
                print(f"[WARN] Reconstruct mismatch at {idx} (should not happen)")

            ### NEW ### write filtered lists (per-instance) + add to summary
            if EXPORT_FILTERED_LISTS:
                filt_path = os.path.join(GRAPHS_OUTDIR, f"misr_filtered_{idx:03d}.json")
                if kept_rects:
                    rects_relab, id_map, Hp, Vp = relabel_rects(kept_rects, kept_ids)
                else:
                    rects_relab, id_map, Hp, Vp = [], {}, [], []
                filtered_payload = {
                    "instance_index": idx,
                    "instance_key": key,
                    "kept_ids_original": kept_ids,                 # original labels kept
                    "id_map_old_to_new": id_map,                   # {old_id: new_id}
                    "kept_rects_original": [((int(a),int(b)),(int(c),int(d))) for ((a,b),(c,d)) in kept_rects],
                    "kept_rects_relabelled": [((int(a),int(b)),(int(c),int(d))) for ((a,b),(c,d)) in rects_relab],
                    "kept_x_lp": [float(v) for v in kept_x],
                    "kept_y_ilp": [int(v) for v in kept_y],
                    "H_prime": Hp,                                  # H' for relabelled kept set
                    "V_prime": Vp,                                  # V' for relabelled kept set
                    "counts": {"kept": len(kept_ids), "removed": len(meta["removed_ids"]), "total": len(rects_all)},
                    "lp": lp, "ilp": ilp, "ratio": ratio,
                    "paths": {
                        "plot_png": plot_path,
                        "graph_full_prefix": base_full,
                        "graph_filtered_prefix": base_filt
                    }
                }
                with open(filt_path, "w") as f:
                    json.dump(filtered_payload, f, indent=2)

                filtered_summary.append({
                    "idx": idx,
                    "instance_key": key,
                    "kept_ids": kept_ids,
                    "removed_ids": meta["removed_ids"],
                    "kept_count": len(kept_ids),
                    "total": len(rects_all),
                    "lp": lp,
                    "ilp": ilp,
                    "ratio": ratio,
                    "filtered_json": filt_path
                })

            print(f"[OK] {idx:03d} | LP={lp:.2f} ILP={ilp:.2f} ratio={ratio:.3f} | plot:{plot_path}")

        except Exception as e:
            print(f"[WARN] Skipped instance {idx} due to error: {e}")

    ### NEW ### write a compact bundle for all filtered sets
    if EXPORT_FILTERED_LISTS:
        bundle_json = os.path.join(GRAPHS_OUTDIR, "misr_filtered_summary.json")
        with open(bundle_json, "w") as f:
            json.dump({"items": filtered_summary}, f, indent=2)

        # light CSV for quick grep/sheets
        bundle_csv = os.path.join(GRAPHS_OUTDIR, "misr_filtered_ids.csv")
        with open(bundle_csv, "w") as f:
            f.write("idx,instance_key,kept_count,total,kept_ids,removed_ids,lp,ilp,ratio\n")
            for row in filtered_summary:
                kept_str = ";".join(map(str, row["kept_ids"]))
                rem_str  = ";".join(map(str, row["removed_ids"]))
                f.write(f"{row['idx']},{row['instance_key']},{row['kept_count']},{row['total']},{kept_str},{rem_str},{row['lp']:.6f},{row['ilp']:.6f},{row['ratio']:.6f}\n")

if __name__ == "__main__":
    main()
