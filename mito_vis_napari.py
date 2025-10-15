#!/usr/bin/env python3
"""
mito_vis_napari.py - Rich Napari visualization for 4D mitochondrial tracking

Purpose
-------
- Color each object's centroid by its track_id
- Draw inter-frame edges:
    - continuation edges (assigned 1:1) in white
    - fission edges in magenta
    - fusion edges in orange
- Mark event nodes:
    - fission children at t+1 in magenta points
    - fusion child at t+1 in orange points
- Optionally show Z-MIP label previews if available (or just the binary image)

Inputs
-------
--input:   path to the source 4D TIFF (same used for tracking)
--axis:    axis order of the TIFF (permutation of tzyx)
--voxel_size: Z Y X physical voxel size (microns) used to convert um back to voxel coords
--outdir:  folder containing objects.csv and edges_all.csv (outputs of tracking script)

Usage
-----
python mito_vis_napari.py \
  --input stack.tif --axis xyzt \
  --voxel_size 0.25 0.25 0.5 \
  --outdir results_plus
"""

import argparse, os, sys
import numpy as np
import pandas as pd
from tifffile import imread

def parse_axis(axis):
    axis = axis.lower()
    if set(axis) != set("tzyx"):
        raise ValueError("--axis must be a permutation of 'tzyx'")
    return axis.index('t'), axis.index('z'), axis.index('y'), axis.index('x')

def to_tzyx(arr, axis):
    t,z,y,x = parse_axis(axis)
    return np.transpose(arr, (t,z,y,x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--axis", default="tzyx")
    ap.add_argument("--voxel_size", nargs=3, type=float, required=True, metavar=("Z_UM","Y_UM","X_UM"))
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    vz, vy, vx = args.voxel_size

    # Load image (binary or grayscale ok), reshape to (T,Z,Y,X)
    arr = imread(args.input)
    if arr.ndim != 4:
        print(f"ERROR: expected 4D arr, got {arr.shape}", file=sys.stderr); sys.exit(2)
    arr = to_tzyx(arr, args.axis)

    # Load CSVs
    obj_path = os.path.join(args.outdir, "objects.csv")
    edges_path = os.path.join(args.outdir, "edges_all.csv")
    if not (os.path.exists(obj_path) and os.path.exists(edges_path)):
        print("ERROR: objects.csv and edges_all.csv are required in --outdir", file=sys.stderr)
        sys.exit(3)

    objects = pd.read_csv(obj_path)
    edges_all = pd.read_csv(edges_path)

    # Build centroid points in voxel coords for Napari (t,z,y,x)
    # objects.csv stores centroid_*_um in the fixed tracker
    if not set(["centroid_z_um","centroid_y_um","centroid_x_um"]).issubset(objects.columns):
        print("objects.csv missing centroid_*_um columns; cannot convert to voxel coords.", file=sys.stderr)
        sys.exit(4)
    objects = objects.copy()

    def safe_div_series(series_um: pd.Series, alt_series: pd.Series | None, scale: float) -> np.ndarray:
        valid_scale = np.isfinite(scale) and scale > 0
        if valid_scale:
            return (series_um / scale).to_numpy(float)
        if alt_series is not None:
            return alt_series.astype(float).to_numpy()
        return np.full(len(series_um), np.nan, dtype=float)

    if "centroid_z_vox" not in objects.columns:
        objects["centroid_z_vox"] = np.nan
    if "centroid_y_vox" not in objects.columns:
        objects["centroid_y_vox"] = np.nan
    if "centroid_x_vox" not in objects.columns:
        objects["centroid_x_vox"] = np.nan

    objects["cz"] = safe_div_series(objects["centroid_z_um"], objects["centroid_z_vox"], vz)
    objects["cy"] = safe_div_series(objects["centroid_y_um"], objects["centroid_y_vox"], vy)
    objects["cx"] = safe_div_series(objects["centroid_x_um"], objects["centroid_x_vox"], vx)
    objects["track_id"] = objects["track_id"].fillna(-1).astype(int)

    # Prepare continuation edges (assigned 1:1 links)
    cont_edges = []
    # and split/fusion edges from events
    split_edges = []
    fusion_edges = []

    # Helper: quickly map (t,label)-> centroid vox
    obj_key = {}
    dropped = 0
    for _, r in objects.iterrows():
        coords = (r.cz, r.cy, r.cx)
        if not np.all(np.isfinite(coords)):
            dropped += 1
            continue
        obj_key[(int(r.t), int(r.label_id))] = (float(r.cz), float(r.cy), float(r.cx), int(r.track_id))
    if dropped:
        print(f"Warning: dropped {dropped} detections with invalid voxel coordinates.", file=sys.stderr)

    fission_nodes_coords = {}
    fusion_nodes_coords = {}

    skip_cont = skip_fiss = skip_fus = 0
    for _, edge in edges_all.iterrows():
        try:
            t0 = int(edge["t_from"])
            t1 = int(edge["t_to"])
            parent = int(edge["parent_label"])
            child = int(edge["child_label"])
        except Exception:
            continue
        start = obj_key.get((t0, parent))
        end = obj_key.get((t1, child))
        if start is None or end is None:
            et = str(edge.get("edge_type", "continuation")).lower()
            if et == "fission":
                skip_fiss += 1
            elif et == "fusion":
                skip_fus += 1
            else:
                skip_cont += 1
            continue
        z0,y0,x0,_ = start
        z1,y1,x1,_ = end
        coords = np.array([z0, y0, x0, z1, y1, x1], dtype=float)
        if not np.all(np.isfinite(coords)):
            et = str(edge.get("edge_type", "continuation")).lower()
            if et == "fission":
                skip_fiss += 1
            elif et == "fusion":
                skip_fus += 1
            else:
                skip_cont += 1
            continue
        seg = np.array([[t0, coords[0], coords[1], coords[2]], [t1, coords[3], coords[4], coords[5]]], dtype=float)
        edge_type = str(edge.get("edge_type", "continuation")).lower()
        if edge_type == "fission":
            split_edges.append(seg)
            fission_nodes_coords[(t1, child)] = [t1, z1, y1, x1]
        elif edge_type == "fusion":
            fusion_edges.append(seg)
            fusion_nodes_coords[(t1, child)] = [t1, z1, y1, x1]
        else:
            cont_edges.append(seg)

    # Launch Napari
    import napari
    v = napari.Viewer(ndisplay=3)
    v.dims.ndisplay = 3

    # Image
    v.add_image(arr, name="image", contrast_limits=[float(arr.min()), float(arr.max())])

    # Centroids colored by track_id
    pts_all = objects[["t","cz","cy","cx"]].to_numpy(float)
    valid_mask = np.all(np.isfinite(pts_all), axis=1)
    if not np.all(valid_mask):
        print(f"Dropping {int((~valid_mask).sum())} centroid rows with invalid coordinates.", file=sys.stderr)
    pts = pts_all[valid_mask]
    feats = {"track_id": objects.loc[valid_mask, "track_id"].astype(int).to_numpy()}
    cent_layer = v.add_points(pts, name="centroids", features=feats, ndim=4, size=2.0, face_color="track_id", blending="translucent")
    try:
        texts = [str(tid) if tid > 0 else "" for tid in feats["track_id"]]
        cent_layer.text = {"string": texts, "size": 12, "color": "white", "anchor": "upper_left", "translation": (0, 0, 0, 0)}
    except Exception:
        pass

    # Event markers (children of fission in magenta, fusion child in orange)
    if fission_nodes_coords:
        v.add_points(np.array(list(fission_nodes_coords.values()), float), name="fission nodes", ndim=4, size=3.0, face_color="magenta")
    if fusion_nodes_coords:
        v.add_points(np.array(list(fusion_nodes_coords.values()), float), name="fusion nodes", ndim=4, size=3.0, face_color="orange")

    # Edge overlays as line segments in a Shapes layer
    shapes = []
    edge_colors = []
    for seg in cont_edges:
        shapes.append(seg)
        edge_colors.append("white")
    for seg in split_edges:
        shapes.append(seg)
        edge_colors.append("magenta")
    for seg in fusion_edges:
        shapes.append(seg)
        edge_colors.append("orange")
    if shapes:
        v.add_shapes(shapes, shape_type="path", name="links", edge_color=edge_colors, edge_width=2.0)

    if skip_cont or skip_fiss or skip_fus:
        print(f"Napari overlays skipped edges (continuation={skip_cont}, fission={skip_fiss}, fusion={skip_fus}) due to missing coordinates.", file=sys.stderr)

    napari.run()

if __name__ == "__main__":
    main()
