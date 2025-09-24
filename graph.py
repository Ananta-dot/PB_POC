# plot_and_graph_batch_filtered.py
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ========== config ==========
PICKLE_PATH    = 'runs/misr_run-2025-09-24_01-03-03-seed123/n16/round15_elites_n16_2025-09-24_03-33-08.pkl'     # input pickle file
PLOTS_OUTDIR   = 'plots/n16'   # where rectangle plots go
GRAPHS_OUTDIR  = 'graphs/n16'  # where graph CSV/PNG/GraphML go
LIMIT          = 256                      # cap on number of instances to export
COUNT_TOUCHING = True                     # True: count edge/corner touching as intersection
GRB_THREADS    = 0                        # 0 -> let Gurobi decide; else pin threads
OUTPUT_FLAG    = 0                        # 1 to show Gurobi solver output
EPS            = 1e-12                    # numeric tolerance for "LP==0"

# ========== helpers (geometry/rectangles) ==========
def seq_spans(seq):
    """Return [(l_i, r_i)] for labels i=1..n (each label must appear exactly twice)."""
    first, spans = {}, {}
    for idx, lab in enumerate(seq):
        if lab not in first:
            first[lab] = idx
        else:
            spans[lab] = (first[lab], idx)
    n = max(seq) if seq else 0
    # assumes labels are 1..n
    return [spans[i] for i in range(1, n + 1)]

def build_rects(H, V):
    X = seq_spans(H)
    Y = seq_spans(V)
    rects = []
    for (x1, x2), (y1, y2) in zip(X, Y):
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        rects.append(((x1, x2), (y1, y2)))
    return rects  # list of ((x1,x2),(y1,y2)), in rectangle index order 1..n

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

# ========== optimization (LP + ILP) ==========
def solve_lp_ilp_with_vars(rects, threads=0, output_flag=0):
    """Solve LP+ILP with grid clique constraints; return (lp, ilp, x_lp, y_ilp)."""
    import gurobipy as gp
    from gurobipy import GRB

    pts = grid_points(rects)
    covers = covers_grid_closed(rects, pts)
    n = len(rects)

    # LP
    m_lp = gp.Model("misr_lp")
    m_lp.setParam('OutputFlag', output_flag)
    if threads > 0:
        m_lp.setParam('Threads', threads)
    x = m_lp.addVars(n, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x')
    m_lp.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_lp.addConstr(gp.quicksum(x[i] for i in S) <= 1)
    m_lp.optimize()
    lp = float(m_lp.objVal) if m_lp.status == GRB.OPTIMAL else 0.0
    x_vals = [float(x[i].X) for i in range(n)] if m_lp.status == GRB.OPTIMAL else [0.0]*n

    # ILP
    m_ilp = gp.Model("misr_ilp")
    m_ilp.setParam('OutputFlag', output_flag)
    if threads > 0:
        m_ilp.setParam('Threads', threads)
    y = m_ilp.addVars(n, vtype=GRB.BINARY, name='y')
    m_ilp.setObjective(gp.quicksum(y[i] for i in range(n)), GRB.MAXIMIZE)
    for S in covers:
        if S:
            m_ilp.addConstr(gp.quicksum(y[i] for i in S) <= 1)
    m_ilp.optimize()
    ilp = float(m_ilp.objVal) if m_ilp.status == GRB.OPTIMAL else 0.0
    y_vals = [int(round(y[i].X)) for i in range(n)] if m_ilp.status == GRB.OPTIMAL else [0]*n

    return lp, ilp, x_vals, y_vals

# ========== filtering based on LP/ILP solution ==========
def filter_by_solution(rects, x_lp, y_ilp, eps=1e-12):
    """
    Keep rectangles where x_lp > eps OR y_ilp == 1.
    Returns:
      filtered_rects, filtered_x_lp, filtered_y_ilp, kept_ids (original 1-based indices)
    """
    kept_rects, kept_x, kept_y, kept_ids = [], [], [], []
    for i, r in enumerate(rects, start=1):
        xi = x_lp[i-1]
        yi = y_ilp[i-1]
        if (abs(xi) > eps) or (yi == 1):
            kept_rects.append(r)
            kept_x.append(xi)
            kept_y.append(yi)
            kept_ids.append(i)
    return kept_rects, kept_x, kept_y, kept_ids

# ========== plotting ==========
def plot_instance(H, V, title=None, save_path=None,
                  highlight_ilp=True, threads=0, output_flag=0,
                  show=True, filter_zero=True, eps=EPS):
    """
    If filter_zero=True, compute LP/ILP on the full set, then drop rectangles with x_lp≈0 and y_ilp=0.
    Plot the remaining rectangles; labels show ORIGINAL indices.
    """
    rects_all = build_rects(H, V)
    lp, ilp, x_lp, y_ilp = solve_lp_ilp_with_vars(rects_all, threads=threads, output_flag=output_flag)

    if filter_zero:
        rects, x_f, y_f, ids = filter_by_solution(rects_all, x_lp, y_ilp, eps=eps)
    else:
        rects, x_f, y_f, ids = rects_all, x_lp, y_ilp, list(range(1, len(rects_all)+1))

    kept = len(rects)
    removed = len(rects_all) - kept
    if kept == 0:
        print("[INFO] Nothing to plot after filtering (all rectangles had LP≈0 and ILP=0).")
        return lp, ilp, x_lp, y_ilp, ids  # ids is empty here

    n_orig = max(H) if H else 0
    fig, ax = plt.subplots(figsize=(7, 7))

    for k, ((x1, x2), (y1, y2)) in enumerate(rects):
        w = (x2 - x1)
        h = (y2 - y1)
        selected = (y_f[k] == 1) if highlight_ilp else False
        edgecolor = 'tab:red' if selected else '0.4'
        lw = 2.8 if selected else 1.2
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=lw, edgecolor=edgecolor))
        # Label with original rectangle ID
        ax.text(x1 + w/2, y1 + h/2, f"{ids[k]}", ha="center", va="center", fontsize=10)

    ax.set_xlim(-1, 2*n_orig)
    ax.set_ylim(-1, 2*n_orig)
    ax.set_xlabel("H positions (indices)")
    ax.set_ylabel("V positions (indices)")
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.invert_yaxis()

    if title is None:
        title = f"Rectangles (filtered) | LP={lp:.2f}, ILP={ilp:.2f}, kept={kept}, removed={removed}"
    else:
        title = f"{title} | LP={lp:.2f}, ILP={ilp:.2f}, kept={kept}, removed={removed}"
    ax.set_title(title)

    import matplotlib.lines as mlines
    leg_items = [
        mlines.Line2D([], [], color='tab:red', linewidth=2.8, label='ILP selected (y=1)'),
        mlines.Line2D([], [], color='0.4', linewidth=1.2, label='Not selected (y=0)'),
    ]
    ax.legend(handles=leg_items, loc='upper left', frameon=True)

    # Console summary like your example
    print(f"n={len(rects_all)} | LP={lp:.2f} ILP={ilp:.2f} ratio={(lp/ilp if ilp>0 else 0):.3f}")
    print("rect vars (i: x_lp, y_ilp):")
    for i in range(len(rects_all)):
        print(f"{i+1:5d}: {x_lp[i]:6.3f}, {y_ilp[i]}")

    if removed > 0:
        removed_ids = [i+1 for i,(xi,yi) in enumerate(zip(x_lp,y_ilp)) if (abs(xi) <= eps and yi == 0)]
        print(f"Removed rectangles (LP≈0 & ILP=0): {removed_ids}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return lp, ilp, x_lp, y_ilp, ids  # ids = kept original IDs (in render order)

# ========== graph construction ==========
import networkx as nx

def rects_intersect(r1, r2, touch=True):
    """
    Return True iff rectangles r1 and r2 intersect.
    touch=True  -> count touching edges/corners as intersection (closed intervals).
    touch=False -> require positive-area overlap (strict).
    r = ((x1,x2),(y1,y2)) with x1<x2, y1<y2
    """
    (xa1, xa2), (ya1, ya2) = r1
    (xb1, xb2), (yb1, yb2) = r2
    if touch:
        # Closed-interval overlap in both x and y
        return not (xa2 < xb1 or xb2 < xa1 or ya2 < yb1 or yb2 < ya1)
    else:
        # Strict overlap (positive area)
        return not (xa2 <= xb1 or xb2 <= xa1 or ya2 <= yb1 or yb2 <= ya1)

def build_intersection_graph(rects, node_ids, touch=True):
    """
    rects:    [ ((x1,x2),(y1,y2)), ... ] (filtered order)
    node_ids: [ original_id1, original_id2, ... ] same length/order as rects
    Returns a networkx.Graph with nodes labeled by ORIGINAL rectangle indices.
    """
    n = len(rects)
    G = nx.Graph()
    # add nodes with an attribute for clarity
    for nid in node_ids:
        G.add_node(nid, original_index=nid)
    for i in range(n):
        for j in range(i+1, n):
            if rects_intersect(rects[i], rects[j], touch=touch):
                G.add_edge(node_ids[i], node_ids[j])
    return G

def save_graph_artifacts(G, basepath_no_ext, title=None):
    """
    Save CSV edgelist, GraphML, and a PNG visualization.
    basepath_no_ext: path without extension (we'll add .csv, .graphml, .png)
    """
    # CSV edgelist
    csv_path = basepath_no_ext + ".csv"
    with open(csv_path, "w") as f:
        f.write("u,v\n")
        for u, v in G.edges():
            f.write(f"{u},{v}\n")

    # GraphML
    graphml_path = basepath_no_ext + ".graphml"
    nx.write_graphml(G, graphml_path)

    # PNG (quick layout)
    try:
        plt.figure(figsize=(6.0, 6.0))
        pos = nx.kamada_kawai_layout(G) if len(G) > 0 else {}
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
        if title:
            plt.title(title)
        png_path = basepath_no_ext + ".png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=220)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to render graph PNG for {basepath_no_ext}: {e}")

# ========== main batch ==========
def main():
    # load pickle
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)

    os.makedirs(PLOTS_OUTDIR, exist_ok=True)
    os.makedirs(GRAPHS_OUTDIR, exist_ok=True)

    N = min(LIMIT, len(data))
    print(f"Found {len(data)} instances, exporting {N} of them.")
    print(f"Filtering criterion: keep if (x_lp > {EPS}) or (y_ilp == 1).")

    for idx in range(N):
        try:
            # Expecting each record like (something, H, V, ...)
            H = data[idx][1]
            V = data[idx][2]

            # per-instance checks
            assert len(H) == len(V), f"[{idx}] H and V must have same length (2n)."
            labels = set(range(1, max(H + V) + 1))
            for L in labels:
                assert H.count(L) == 2 and V.count(L) == 2, f"[{idx}] Label {L} must appear exactly twice in both H and V."

            # ===== rectangle plot with filtering =====
            plot_path  = os.path.join(PLOTS_OUTDIR, f"misr_{idx:03d}.png")
            plot_title = f"MISR instance {idx}"
            lp, ilp, x_lp, y_ilp, kept_ids_plot = plot_instance(
                H, V,
                title=plot_title,
                save_path=plot_path,
                highlight_ilp=True,
                threads=GRB_THREADS,
                output_flag=OUTPUT_FLAG,
                show=False,
                filter_zero=True,
                eps=EPS,
            )

            # If no rectangles survived filtering, skip graph
            # (plot_instance already printed a message)
            # Recompute the kept rects here to build graph
            rects_all = build_rects(H, V)
            kept_rects, kept_x, kept_y, kept_ids = filter_by_solution(rects_all, x_lp, y_ilp, eps=EPS)

            if len(kept_rects) == 0:
                print(f"[SKIP] Instance {idx}: no rectangles after filtering; graph not generated.")
                continue

            # ===== intersection graph on filtered set (nodes labeled by ORIGINAL indices) =====
            G = build_intersection_graph(kept_rects, kept_ids, touch=COUNT_TOUCHING)
            base_graph_path = os.path.join(GRAPHS_OUTDIR, f"misr_graph_{idx:03d}")
            title = f"Intersection Graph — instance {idx} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})"
            save_graph_artifacts(G, basepath_no_ext=base_graph_path, title=title)

            removed_ct = len(rects_all) - len(kept_rects)
            print(f"[OK] Instance {idx}: kept {len(kept_rects)}, removed {removed_ct} -> "
                  f"plot: {plot_path}; graph: {base_graph_path}.[csv|graphml|png]")

        except Exception as e:
            print(f"[WARN] Skipped instance {idx} due to error: {e}")

if __name__ == "__main__":
    main()
