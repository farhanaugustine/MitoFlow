#!/usr/bin/env python3
"""
mito_report_html.py - No-code HTML report for mitochondrial tracking

Inputs
------
--input: path to the same 4D TIFF used for tracking
--axis:  axes order (permutation of tzyx) of the TIFF
--voxel_size: Z Y X in microns
--outdir: directory that contains objects.csv, events_*_edges.csv, edges_all.csv (from tracker)
--crop_um_radius:  Z Y X radius in microns for local crops (default: 1.0 2.0 2.0)
--frames_before / --frames_after: how many frames around the event to include (default: 1 1)
--make_gifs: save animated GIFs for each event (default off)
--max_events: limit number of events rendered (default: 100)
--render_lineage: render a whole-graph lineage PNG if lineage.graphml is present
--dpi: image DPI for saved PNGs (default 110)

Install
-------
pip install numpy pandas tifffile imageio matplotlib networkx

Usage
-----
python mito_report_html.py ^
  --input stack.tif --axis xyzt ^
  --voxel_size 0.25 0.25 0.5 ^
  --outdir results_plus ^
  --crop_um_radius 1.0 2.0 2.0 --frames_before 1 --frames_after 2 --make_gifs --max_events 50
"""

from __future__ import annotations
import argparse, os, sys, io, base64
from typing import Tuple, List
import numpy as np
import pandas as pd
from tifffile import imread
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError

def parse_axis(axis: str) -> Tuple[int,int,int,int]:
    axis = axis.lower()
    if set(axis) != set("tzyx"):
        raise ValueError("--axis must be a permutation of 'tzyx'")
    return axis.index('t'), axis.index('z'), axis.index('y'), axis.index('x')

def to_tzyx(arr, axis):
    t,z,y,x = parse_axis(axis)
    return np.transpose(arr, (t,z,y,x))

def mip_z(vol_zyx: np.ndarray) -> np.ndarray:
    # Z-MIP
    return vol_zyx.max(axis=0)

def crop_bounds(center_zyx: Tuple[int,int,int], radius_zyx: Tuple[int,int,int], shape_zyx: Tuple[int,int,int]):
    cz, cy, cx = center_zyx
    rz, ry, rx = radius_zyx
    z0, z1 = max(0, cz - rz), min(shape_zyx[0], cz + rz + 1)
    y0, y1 = max(0, cy - ry), min(shape_zyx[1], cy + ry + 1)
    x0, x1 = max(0, cx - rx), min(shape_zyx[2], cx + rx + 1)
    return (z0, z1, y0, y1, x0, x1)

def save_png(arr2d: np.ndarray, path: str, dpi: int):
    fig = plt.figure(figsize=(4,4), dpi=dpi)
    ax = plt.gca()
    ax.imshow(arr2d, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def ndarray_to_base64_png(arr2d: np.ndarray, dpi: int) -> str:
    fig = plt.figure(figsize=(4,4), dpi=dpi)
    ax = plt.gca()
    ax.imshow(arr2d, interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def read_csv_safe(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns or [])
    if columns:
        missing = [col for col in columns if col not in df.columns]
        for col in missing:
            df[col] = np.nan
        df = df.reindex(columns=columns)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--axis", default="tzyx")
    ap.add_argument("--voxel_size", nargs=3, type=float, required=True, metavar=("Z_UM","Y_UM","X_UM"))
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--crop_um_radius", nargs=3, type=float, default=[1.0, 2.0, 2.0], metavar=("ZR","YR","XR"))
    ap.add_argument("--frames_before", type=int, default=1)
    ap.add_argument("--frames_after", type=int, default=1)
    ap.add_argument("--make_gifs", action="store_true")
    ap.add_argument("--max_events", type=int, default=100)
    ap.add_argument("--render_lineage", action="store_true")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--crop-pad", type=float, default=2.0, help="Multiplier to expand the crop radius for more context (default 2.0)")

    args = ap.parse_args()
    vz, vy, vx = args.voxel_size

    # Load image and CSVs
    arr = imread(args.input)
    if arr.ndim != 4:
        print(f"Expected 4D TIFF, got {arr.shape}", file=sys.stderr)
        sys.exit(2)
    arr = to_tzyx(arr, args.axis)  # (T,Z,Y,X)
    T, Z, Y, X = arr.shape

    objects_path = os.path.join(args.outdir, "objects.csv")
    fiss_path = os.path.join(args.outdir, "events_fission_edges.csv")
    fus_path  = os.path.join(args.outdir, "events_fusion_edges.csv")
    edges_all_path = os.path.join(args.outdir, "edges_all.csv")
    if not (os.path.exists(objects_path) and (os.path.exists(fiss_path) or os.path.exists(fus_path))):
        print("Missing objects.csv or events edges in --outdir", file=sys.stderr)
        sys.exit(3)

    object_columns = ["t", "label_id", "track_id", "centroid_z_um", "centroid_y_um", "centroid_x_um",
                      "centroid_z_vox", "centroid_y_vox", "centroid_x_vox"]
    event_columns = [
        "event", "t_from", "t_to", "parent_label", "child_label",
        "parent_track_id", "child_track_id",
        "parent_volume_vox", "child_volume_vox",
        "parent_centroid_z_um", "parent_centroid_z_vox", "parent_slice_index",
        "child_centroid_z_um", "child_centroid_z_vox", "child_slice_index"
    ]

    objects = read_csv_safe(objects_path, object_columns)
    fission = read_csv_safe(fiss_path, event_columns)
    fusion = read_csv_safe(fus_path, event_columns)
    edges_all = read_csv_safe(edges_all_path)

    # Basic stats
    n_frames = T
    n_obj = len(objects)
    n_tracks = int(objects["track_id"].nunique()) if "track_id" in objects else 0
    n_fis = len(fission)
    n_fus = len(fusion)

    # Map for quick lookups
    obj_lookup = objects.set_index(["t","label_id"])
    # Convert um to vox radius
    pad = float(getattr(args, "crop_pad", 2.0))
    rz = max(1, int(round((args.crop_um_radius[0] / vz) * pad)))
    ry = max(1, int(round((args.crop_um_radius[1] / vy) * pad)))
    rx = max(1, int(round((args.crop_um_radius[2] / vx) * pad)))

    # Prepare output dirs
    report_dir = ensure_dir(os.path.join(args.outdir, "report"))
    asset_dir  = ensure_dir(os.path.join(report_dir, "assets"))
    crops_dir  = ensure_dir(os.path.join(asset_dir, "crops"))
    gifs_dir   = ensure_dir(os.path.join(asset_dir, "gifs"))

    # Helper to crop around child node at t_to
    def event_crop_images(ev_row) -> Tuple[str, str]:
        """Return (png_rel_path, gif_rel_path or '') for this event."""
        t_to = int(ev_row["t_to"])
        c_lab = int(ev_row["child_label"])
        # Get centroid in voxel coords at t_to
        try:
            cz_um = float(obj_lookup.loc[(t_to, c_lab), "centroid_z_um"])
            cy_um = float(obj_lookup.loc[(t_to, c_lab), "centroid_y_um"])
            cx_um = float(obj_lookup.loc[(t_to, c_lab), "centroid_x_um"])
        except Exception:
            return "", ""
        cz = int(round(cz_um / vz))
        cy = int(round(cy_um / vy))
        cx = int(round(cx_um / vx))

        # time window
        t0 = max(0, t_to - args.frames_before)
        t1 = min(T-1, t_to + args.frames_after)

        png_entries = []
        stack_for_gif = []
        for tt in range(t0, t1+1):
            z0,z1,y0,y1,x0,x1 = crop_bounds((cz,cy,cx), (rz,ry,rx), (Z,Y,X))
            vol = arr[tt, z0:z1, y0:y1, x0:x1]
            if vol.size == 0:
                continue
            mip = mip_z(vol)
            png_name = f"event_t{t_to}_c{c_lab}_f{tt}.png"
            png_path = os.path.join(crops_dir, png_name)
            save_png(mip, png_path, dpi=args.dpi)
            png_entries.append((tt, os.path.relpath(png_path, report_dir)))
            stack_for_gif.append((mip * 255.0 / max(1e-6, mip.max())).astype(np.uint8))

        gif_rel = ""
        if args.make_gifs and stack_for_gif:
            gif_name = f"event_t{t_to}_c{c_lab}.gif"
            gif_path = os.path.join(gifs_dir, gif_name)
            imageio.mimsave(gif_path, stack_for_gif, duration=0.5, loop=0)
            gif_rel = os.path.relpath(gif_path, report_dir)

        # Return first PNG (or ""), and GIF (or "")
        thumb_rel = ""
        if png_entries:
            match = [path for frame, path in png_entries if frame == t_to]
            if match:
                thumb_rel = match[0]
            else:
                thumb_rel = png_entries[len(png_entries)//2][1]
        return (thumb_rel, gif_rel)

    # Render lineage graph if present (optional)
    lineage_png_rel = ""
    if args.render_lineage:
        graphml_path = os.path.join(args.outdir, "lineage.graphml")
        if os.path.exists(graphml_path):
            try:
                import networkx as nx
                G = nx.read_graphml(graphml_path)
                # Draw a simple layout
                pos = nx.spring_layout(G, seed=0, k=None, iterations=50)
                plt.figure(figsize=(10,8), dpi=args.dpi)
                nx.draw_networkx_nodes(G, pos, node_size=10)
                nx.draw_networkx_edges(G, pos, arrows=False, width=0.5)
                plt.axis("off")
                lineage_png = os.path.join(asset_dir, "lineage.png")
                plt.savefig(lineage_png, bbox_inches="tight", pad_inches=0.0)
                plt.close()
                lineage_png_rel = os.path.relpath(lineage_png, report_dir)
            except Exception as e:
                print("Lineage rendering failed:", e, file=sys.stderr)

    # Compose HTML
    html_parts = []
    html_parts.append(f"<h1>Mitochondria Tracking Report</h1>")
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Frames (T): <b>{n_frames}</b></li>")
    html_parts.append(f"<li>Total objects: <b>{n_obj}</b></li>")
    html_parts.append(f"<li>Tracks: <b>{n_tracks}</b></li>")
    html_parts.append(f"<li>Fission edges: <b>{n_fis}</b></li>")
    html_parts.append(f"<li>Fusion edges: <b>{n_fus}</b></li>")
    html_parts.append("</ul>")

    if lineage_png_rel:
        html_parts.append("<h2>Lineage Graph (overview)</h2>")
        html_parts.append(f'<img src="{lineage_png_rel}" style="max-width:100%;height:auto;"/>')

    # Tables: show top rows inline; save full CSV links
    def df_to_html_snippet(df: pd.DataFrame, title: str, link_name: str):
        html_parts.append(f"<h2>{title}</h2>")
        if df.empty:
            html_parts.append("<p><i>No entries</i></p>")
        else:
            preview = df.head(20).to_html(index=False, escape=False)
            html_parts.append(preview)
        html_parts.append(f'<p>See full table: <code>{link_name}.csv</code></p>')

    df_to_html_snippet(objects, "Objects (first 20 rows)", "objects")
    df_to_html_snippet(fission, "Fission edges (first 20 rows)", "events_fission_edges")
    df_to_html_snippet(fusion, "Fusion edges (first 20 rows)", "events_fusion_edges")

    # Event gallery (crops)
    # Merge fission+fusion with an event_type column
    events = []
    if not fission.empty:
        tmp = fission.copy(); tmp["event_type"] = "fission"; events.append(tmp)
    if not fusion.empty:
        tmp = fusion.copy(); tmp["event_type"] = "fusion"; events.append(tmp)
    gallery_rows = []
    if events:
        E = pd.concat(events, ignore_index=True)
        E = E.sort_values(["t_to", "event_type"]).head(args.max_events)
        html_parts.append("<h2>Event Gallery</h2>")
        html_parts.append("<p>Each tile shows a Z-MIP crop around the <b>child</b> at t_to. Hover filenames show t/label.</p>")
        html_parts.append('<div style="display:flex;flex-wrap:wrap;gap:12px;">')
        for _, ev in E.iterrows():
            png_rel, gif_rel = event_crop_images(ev)
            info = f'{ev["event_type"]} | t_from={int(ev["t_from"])}, t_to={int(ev["t_to"])}, parent_label={int(ev.get("parent_label",-1))}, child_label={int(ev.get("child_label",-1))}'
            if png_rel:
                html_parts.append(f'<div style="border:1px solid #ddd;padding:6px;border-radius:8px;max-width:256px;">')
                html_parts.append(f'<div style="font-size:12px;margin-bottom:4px;">{info}</div>')
                html_parts.append(f'<img src="{png_rel}" title="{info}" style="width:240px;height:auto;display:block;"/>')
                if gif_rel:
                    html_parts.append(f'<div style="margin-top:6px;"><img src="{gif_rel}" style="width:240px;height:auto;"/></div>')
                html_parts.append('</div>')
        html_parts.append('</div>')

    # Write HTML
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Mitochondria Tracking Report</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial,sans-serif; margin:20px;}
h1,h2{margin:0.4em 0;}
table{border-collapse:collapse;font-size:13px;}
td,th{border:1px solid #ccc;padding:4px 6px;}
code{background:#f6f6f6;padding:2px 4px;border-radius:4px;}
</style>
</head>
<body>
""" + "\n".join(html_parts) + """
</body>
</html>"""
    report_path = os.path.join(report_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Report written to:", report_path)

if __name__ == "__main__":
    main()
