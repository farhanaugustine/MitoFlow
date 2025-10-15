#!/usr/bin/env python3
"""
mito_lineage_viewer_plotly.py - Interactive HTML viewer for lineage graphs (Plotly)

- Avoids pyvis/Jinja/encoding issues; writes UTF-8 HTML
- Time flows left->right on x; tracks occupy horizontal lanes on y
- Edge colors: continuation = light gray, fission = magenta, fusion = orange
- Node size ~ volume_vox (if provided via --objects)

Usage:
  python mito_lineage_viewer_plotly.py --edges edges_all.csv --objects objects.csv --out lineage.html
  # or
  python mito_lineage_viewer_plotly.py --graph lineage.graphml --objects objects.csv --out lineage.html

Options:
  --time_scale 160     # pixels per frame on x-axis
  --track_spacing 80   # pixels per track lane on y-axis
  --min_node_size 6 --max_node_size 18
  --palette viridis    # matplotlib colormap for node colors by track_id (fallback: label_id)
  --embed_js inline|cdn  # default inline for firewall-safe HTML
  --open               # open HTML after writing
  --debug              # print a few parsed nodes

Requires:
  pip install networkx pandas plotly matplotlib
"""

import argparse, os, sys, math, colorsys, re
import pandas as pd
import networkx as nx

def colormap_hex(val, vmin, vmax, cmap_name="viridis"):
    try:
        import matplotlib
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgba = cmap(norm(val))
        rgb = tuple(int(255*x) for x in rgba[:3])
        return "#%02x%02x%02x" % rgb
    except Exception:
        h = (hash((val, cmap_name)) % 360) / 360.0
        r,g,b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
        return "#%02x%02x%02x" % (int(255*r), int(255*g), int(255*b))

def _infer_t_label_from_node(n, data):
    """
    Try to get integer t and label_id from:
    - explicit attributes: 't', 'label_id' or 'label'
    - parse node id string for two ints (e.g., "(12, 34)", "12:34", "t12_l34")
    - tuple node ids like (t,label)
    """
    # 1) direct attributes
    t = None
    for key in ("t", "time", "frame"):
        if key in data:
            try:
                t = int(data[key]); break
            except Exception:
                pass
    lab = None
    for key in ("label_id", "label", "id"):
        if key in data:
            try:
                lab = int(data[key]); break
            except Exception:
                pass
    # 2) parse from node id string
    if (t is None or lab is None) and not isinstance(n, (tuple,list)):
        s = str(n)
        nums = re.findall(r"\d+", s)
        if len(nums) >= 2:
            if t is None:
                try: t = int(nums[0])
                except Exception: pass
            if lab is None:
                try: lab = int(nums[1])
                except Exception: pass
    # 3) tuple form
    if (t is None or lab is None) and isinstance(n, (tuple,list)) and len(n) >= 2:
        try:
            if t is None: t = int(n[0])
            if lab is None: lab = int(n[1])
        except Exception:
            pass
    if t is None: t = 0
    if lab is None: lab = -1
    return t, lab

def load_graph(graph_path=None, edges_csv=None):
    if edges_csv and os.path.exists(edges_csv):
        df = pd.read_csv(edges_csv)
        G = nx.DiGraph()
        for _, r in df.iterrows():
            t0 = int(r["t_from"]); t1 = int(r["t_to"])
            p  = int(r["parent_label"]); c = int(r["child_label"])
            et = str(r.get("edge_type", "continuation"))
            G.add_node((t0,p), t=t0, label_id=p)
            G.add_node((t1,c), t=t1, label_id=c)
            G.add_edge((t0,p), (t1,c), edge_type=et)
        G.graph["source"] = "edges_csv"
        return G
    elif graph_path and os.path.exists(graph_path):
        G = nx.read_graphml(graph_path)
        for u,v in G.edges():
            if "edge_type" not in G[u][v]:
                G[u][v]["edge_type"] = "continuation"
        G.graph["source"] = "graphml"
        return G
    else:
        print("Provide either --edges or --graph path", file=sys.stderr)
        sys.exit(2)

def attach_objects(G, objects_csv=None):
    if not objects_csv or not os.path.exists(objects_csv):
        return
    df = pd.read_csv(objects_csv, sep=None, engine="python")
    # harmonize column names
    if 'label_id' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label':'label_id'})
    if 'track_id' not in df.columns and 'track' in df.columns:
        df = df.rename(columns={'track':'track_id'})
    # coerce types
    df['t'] = df['t'].astype(int)
    df['label_id'] = df['label_id'].astype(int)
    df = df.drop_duplicates(subset=['t','label_id'])
    lut = df.set_index(['t','label_id']).to_dict(orient='index')
    tids = []
    for n in G.nodes:
        d = G.nodes[n]
        t, lab = _infer_t_label_from_node(n, d)
        meta = lut.get((t, lab), {})
        d.update(meta)
        if "track_id" in meta:
            try: tids.append(int(meta["track_id"]))
            except Exception: pass
    if tids:
        G.graph["track_id_min"] = int(min(tids))
        G.graph["track_id_max"] = int(max(tids))

def build_positions(G, time_scale=160, track_spacing=80):
    # ensure nodes have t,label_id fields
    for n in G.nodes:
        t, lab = _infer_t_label_from_node(n, G.nodes[n])
        G.nodes[n]["t"] = t
        G.nodes[n]["label_id"] = lab

    tracks = sorted({int(G.nodes[n].get('track_id', -1)) for n in G.nodes})
    if tracks == [-1]:
        # fallback: spread by label modulo a base
        labels = [int(G.nodes[n]['label_id']) for n in G.nodes]
        base = max(20, min(200, len(set(labels))//10 + 20))
        lanes = sorted({lab % base for lab in labels})
        track_to_lane = {t:i for i,t in enumerate(lanes)}
        def lane_of(n):
            lab = int(G.nodes[n]['label_id'])
            return track_to_lane.get(lab % base, 0)
    else:
        track_to_lane = {tid:i for i,tid in enumerate(sorted([t for t in tracks if t!=-1]))}
        def lane_of(n):
            return track_to_lane.get(int(G.nodes[n].get('track_id', -1)), len(track_to_lane))

    pos = {}
    for n in G.nodes:
        t = int(G.nodes[n]['t'])
        x = t * time_scale
        y = lane_of(n) * track_spacing
        pos[n] = (x,y)
    return pos

def build_plotly(G, positions, min_node_size=6, max_node_size=18, palette="viridis"):
    import plotly.graph_objects as go

    vols = [float(G.nodes[n].get("volume_vox", 0.0)) for n in G.nodes]
    vmin, vmax = (min(vols), max(vols)) if vols else (0.0, 1.0)
    tmin = int(G.graph.get("track_id_min", 0))
    tmax = int(G.graph.get("track_id_max", 1))

    # edges grouped by type
    edge_types = {"continuation": "#d0d0d0", "fission": "#ff00ff", "fusion": "#ff8800"}
    edge_traces = {k: {"x": [], "y": []} for k in edge_types}
    for u,v,data in G.edges(data=True):
        et = str(data.get("edge_type", "continuation")).lower()
        et = "fission" if et.startswith("fiss") else ("fusion" if et.startswith("fus") else "continuation")
        x0,y0 = positions[u]; x1,y1 = positions[v]
        edge_traces[et]["x"] += [x0, x1, None]
        edge_traces[et]["y"] += [y0, y1, None]

    edge_fig_traces = []
    for et, color in edge_types.items():
        edge_fig_traces.append(
            go.Scattergl(
                x=edge_traces[et]["x"], y=edge_traces[et]["y"],
                mode="lines", line=dict(color=color, width=2 if et=='continuation' else 3),
                hoverinfo="skip", name=et.capitalize()
            )
        )

    xs, ys, sizes, colors, texts = [], [], [], [], []
    for n in G.nodes:
        x,y = positions[n]
        d = G.nodes[n]
        t   = int(d.get("t", 0))
        lab = int(d.get("label_id", -1))
        tid = d.get("track_id", None)
        try: tid = int(tid) if tid is not None and str(tid)!='nan' else None
        except Exception: tid = None
        vol = float(d.get("volume_vox", 0.0))
        size = min_node_size
        if vmax > vmin and vol > 0:
            size = min_node_size + (max_node_size - min_node_size) * (vol - vmin) / (vmax - vmin)
        val_for_color = tid if tid is not None else lab
        color = colormap_hex(val_for_color, tmin, tmax if tmax>tmin else tmin+1, palette)
        xs.append(x); ys.append(y); sizes.append(size); colors.append(color)
        texts.append(f"t={t}, label={lab}, track_id={tid}, vol={vol:.1f}")

    node_trace = None
    try:
        import plotly.graph_objects as go
        node_trace = go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#333")),
            text=texts, hoverinfo="text", name="Nodes"
        )
    except Exception:
        node_trace = go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#333")),
            text=texts, hoverinfo="text", name="Nodes"
        )

    fig = None
    import plotly.graph_objects as go
    fig = go.Figure(edge_fig_traces + [node_trace])
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="time (frames)", showgrid=False, zeroline=False),
        yaxis=dict(title="track lanes", showgrid=True, zeroline=False),
        template="plotly_white",
        height=800
    )
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=str, help="Path to lineage.graphml")
    ap.add_argument("--edges", type=str, help="Path to edges_all.csv")
    ap.add_argument("--objects", type=str, help="Path to objects.csv (optional, for node sizes/track colors)")
    ap.add_argument("--out", type=str, required=True, help="Output HTML path")
    ap.add_argument("--time_scale", type=int, default=160)
    ap.add_argument("--track_spacing", type=int, default=80)
    ap.add_argument("--min_node_size", type=int, default=6)
    ap.add_argument("--max_node_size", type=int, default=18)
    ap.add_argument("--palette", type=str, default="viridis")
    ap.add_argument("--embed_js", type=str, default="inline", choices=["inline","cdn"], help="Embed Plotly JS inline (default) or load from CDN")
    ap.add_argument("--open", action="store_true", help="Open the HTML in the default browser after writing")
    ap.add_argument("--debug", action="store_true", help="Print a few sample nodes after parsing")
    args = ap.parse_args()

    # auto-prefer edges CSV if both exist
    edges_path = args.edges if args.edges else None
    if not edges_path and args.graph:
        # if out dir has a sibling edges_all.csv, prefer it
        cand = os.path.join(os.path.dirname(os.path.abspath(args.graph)), "edges_all.csv")
        if os.path.exists(cand):
            edges_path = cand

    G = load_graph(graph_path=args.graph, edges_csv=edges_path)
    attach_objects(G, args.objects)
    pos = build_positions(G, time_scale=args.time_scale, track_spacing=args.track_spacing)

    if args.debug:
        # print 5 sample nodes
        print("Parsed sample nodes:")
        for i, n in enumerate(list(G.nodes)[:5]):
            d = G.nodes[n]
            print("  node:", n, "=> t=", d.get("t"), "label_id=", d.get("label_id"), "track_id=", d.get("track_id"))
            if i>=4: break

    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (source={G.graph.get('source')})")

    fig = build_plotly(G, pos, min_node_size=args.min_node_size, max_node_size=args.max_node_size, palette=args.palette)
    out_html = os.path.abspath(args.out)
    if not out_html.lower().endswith(".html"): out_html += ".html"

    # Write as self-contained HTML
    html = fig.to_html(full_html=True, include_plotlyjs=("cdn" if args.embed_js=="cdn" else "inline"))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote:", out_html)
    if args.open:
        import webbrowser
        webbrowser.open('file://' + out_html)

if __name__ == "__main__":
    print("LINEAGE VIEWER (Plotly) - UTF-8 safe, no pyvis")
    main()
