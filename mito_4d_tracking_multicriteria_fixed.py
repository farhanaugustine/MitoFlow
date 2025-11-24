#!/usr/bin/env python3
"""
mito_4d_tracking_multicriteria_fixed.py

4D (t,z,y,x) mitochondrial tracking with multi-criteria linking, robust fission/fusion,
richer Napari visualization (colored edges & event nodes), and safer data exports.

Key features
------------
- Multi-criteria linking with Hungarian assignment:
  cost = w_iou*(1 - IoU) + w_dist*(dist_um / (r_i + r_j)) + w_vol*|deltaV|/max(Vi,Vj)
- Split/Merge (fission/fusion) require overlap evidence + approximate volume conservation
- Persistence filter (min frames before/after) to reduce 1-frame artifacts
- Napari overlay (if --napari): 
  * Centroids colored by track_id
  * Continuation edges (white), Fission edges (magenta), Fusion edges (orange)
  * Toggle overlays with --napari_show_fission / --napari_show_fusion
- Safer data saving:
  * Per-frame objects in objects.csv
  * Candidate links in links.csv
  * Assigned 1-to-1 matches in matches.csv
  * Normalized event edges in events_fission_edges.csv / events_fusion_edges.csv (numeric columns; no "t:id" strings)
  * Combined edge list (continuations + events) in edges_all.csv
  * Optional Parquet mirrors (use --save_parquet)
- Summary CSVs per frame (`events_per_t.csv`), per time/slice (`events_per_slice.csv`), and per slice across time (`events_per_z.csv`)
- Graph exports (use --export_graph):
  * Whole-graph GraphML and DOT
  * Optional per-track subgraphs

Install
-------
pip install numpy scipy scikit-image tifffile pandas pillow networkx

Example
-------
python mito_4d_tracking_multicriteria_fixed.py ^
  --input stack.tif --axis xyzt ^
  --voxel_size 0.25 0.25 0.5 ^
  --min_size 50 ^
  --iou_thr_event 0.06 ^
  --max_disp_um 2.5 ^
  --w_iou 0.6 --w_dist 0.3 --w_vol 0.1 ^
  --max_cost 1.2 --vol_tol 0.35 --min_event_persistence 1 ^
  --outdir results_plus ^
  --save_pngs --napari --napari_show_fission --napari_show_fusion ^
  --export_graph --graph_per_track

Updates
-------
- Exports Z-slice indices in event tables for easier manual validation.
"""
from __future__ import annotations
import argparse, os, sys, json
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import remove_small_objects, binary_closing, ball, disk
from skimage.segmentation import relabel_sequential
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from PIL import Image

# --------------------- helpers ---------------------

def parse_axis(axis: str) -> str:
    """
    Validate axis string allowing optional channel (c) and optional time (t).
    Requires z,y,x to be present. Returns lowercase normalized axis.
    """
    axis = axis.lower()
    allowed = set("tzyxc")
    if not set(axis) <= allowed:
        raise ValueError(f"--axis may contain only t,z,y,x,c characters, got '{axis}'")
    if axis.count("t") > 1 or axis.count("c") > 1:
        raise ValueError(f"--axis may contain at most one 't' and one 'c', got '{axis}'")
    for ch in "zyx":
        if ch not in axis:
            raise ValueError(f"--axis must include z,y,x; got '{axis}'")
    return axis

def to_tzyx(arr: np.ndarray, axis: str) -> np.ndarray:
    """
    Reorder array to (t,z,y,x).
    - Drops a channel axis (c) if present, taking channel 0.
    - Adds a singleton time axis if missing.
    """
    axis = parse_axis(axis)
    data = arr
    if "c" in axis:
        c_pos = axis.index("c")
        if c_pos >= data.ndim:
            raise ValueError(f"--axis='{axis}' expects a channel axis, but data has shape {data.shape}")
        data = np.take(data, indices=0, axis=c_pos)
        axis = axis.replace("c", "")
    if "t" not in axis:
        data = np.expand_dims(data, axis=0)
        axis = "t" + axis
    if len(axis) != data.ndim:
        raise ValueError(f"--axis='{axis}' expects {len(axis)} dims, but data has shape {data.shape}")
    order = [axis.index(ch) for ch in "tzyx"]
    return np.transpose(data, order)

def ensure_binary(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.uint8)

def preprocess_binary_volume(vol: np.ndarray,
                             smooth_sigma: float,
                             closing_radius: float,
                             closing_xy: bool) -> np.ndarray:
    """
    Optional pre-processing to mitigate segmentation flicker:
    - Gaussian smoothing + threshold
    - Morphological closing (3D ball or per-slice disk)
    """
    out = vol.astype(np.float32, copy=False)
    if smooth_sigma and smooth_sigma > 0:
        out = gaussian_filter(out, float(smooth_sigma))
        out = (out >= 0.5).astype(np.uint8)
    if closing_radius and closing_radius > 0:
        radius = max(1, int(np.ceil(float(closing_radius))))
        if closing_xy:
            selem = disk(radius)
            closed = np.stack([binary_closing(sl.astype(bool), selem) for sl in out], axis=0)
        else:
            selem = ball(radius)
            closed = binary_closing(out.astype(bool), selem)
        out = closed.astype(np.uint8)
    return out

def label_3d(vol: np.ndarray, connectivity: int) -> np.ndarray:
    lab = sk_label(vol, connectivity=connectivity)
    lab, _, _ = relabel_sequential(lab)
    return lab

def apply_track_mapping(mapping: Dict[int, int], values: pd.Series) -> pd.Series:
    """Remap track ids using a canonical parent->root map."""
    if not mapping:
        return values
    cache: Dict[int, int] = {}
    def _canon(val: int) -> int:
        if val in cache:
            return cache[val]
        orig = val
        seen = set()
        while val in mapping and val not in seen:
            seen.add(val)
            nxt = mapping[val]
            if nxt == val:
                break
            val = nxt
        cache[orig] = val
        return val
    return values.apply(lambda v: _canon(int(v)) if pd.notna(v) and int(v) >= 0 else v)

def gap_close_tracks(objects_df: pd.DataFrame,
                     gap_frames: int,
                     max_disp_um: float,
                     gap_max_disp_factor: float,
                     w_dist: float,
                     w_vol: float,
                     frame_penalty: float) -> Tuple[Dict[int, int], List[Tuple[dict, dict]]]:
    """
    Bridge short gaps by merging track IDs when a new track starts within a
    small temporal/spatial window of a recently ended track.
    Returns (mapping {new_track_id -> existing_track_id}, gap_edge_rows metadata).
    """
    if gap_frames <= 0 or objects_df.empty:
        return {}, []
    tracks = {}
    for tid, grp in objects_df.groupby("track_id"):
        if tid <= 0:
            continue
        g = grp.sort_values("t")
        tracks[int(tid)] = {
            "start_t": int(g.iloc[0]["t"]),
            "end_t": int(g.iloc[-1]["t"]),
            "start_row": g.iloc[0],
            "end_row": g.iloc[-1],
        }
    if not tracks:
        return {}, []
    candidates: List[Tuple[float, int, int]] = []
    for tid_from, info_from in tracks.items():
        end_t = info_from["end_t"]
        end_row = info_from["end_row"]
        for tid_to, info_to in tracks.items():
            if tid_from == tid_to:
                continue
            start_t = info_to["start_t"]
            gap = start_t - end_t
            if gap <= 0 or gap > gap_frames + 1:
                continue
            cz0, cy0, cx0 = float(end_row["centroid_z_um"]), float(end_row["centroid_y_um"]), float(end_row["centroid_x_um"])
            cz1, cy1, cx1 = float(info_to["start_row"]["centroid_z_um"]), float(info_to["start_row"]["centroid_y_um"]), float(info_to["start_row"]["centroid_x_um"])
            dist_um = float(np.sqrt((cz0-cz1)**2 + (cy0-cy1)**2 + (cx0-cx1)**2))
            if max_disp_um is not None and np.isfinite(max_disp_um):
                max_jump = float(max_disp_um) * float(gap) * float(gap_max_disp_factor)
                if dist_um > max_jump:
                    continue
            rnorm = float(end_row["r_eq_um"] + info_to["start_row"]["r_eq_um"] + 1e-6)
            denom = max(float(end_row.get("volume_um3", end_row["volume_vox"])), float(info_to["start_row"].get("volume_um3", info_to["start_row"]["volume_vox"])))
            dvol = abs(float(end_row.get("volume_um3", end_row["volume_vox"])) - float(info_to["start_row"].get("volume_um3", info_to["start_row"]["volume_vox"]))) / denom if denom > 0 else 0.0
            cost = w_dist * (dist_um / rnorm) + w_vol * dvol + frame_penalty * max(0, gap - 1)
            candidates.append((cost, tid_from, tid_to))

    if not candidates:
        return {}, []
    candidates.sort(key=lambda v: v[0])
    mapping: Dict[int, int] = {}
    gap_edges: List[Tuple[dict, dict]] = []
    used_from: set[int] = set()
    used_to: set[int] = set()
    for cost, tid_from, tid_to in candidates:
        if tid_from in used_from or tid_to in used_to:
            continue
        mapping[tid_to] = tid_from
        gap_edges.append((tracks[tid_from]["end_row"].to_dict(), tracks[tid_to]["start_row"].to_dict()))
        used_from.add(tid_from)
        used_to.add(tid_to)
    return mapping, gap_edges

def props_table(lab: np.ndarray, vz, vy, vx) -> pd.DataFrame:
    rows = []
    voxel_volume_um3 = float(vz * vy * vx)
    for r in regionprops(lab):
        z0,y0,x0,z1,y1,x1 = (*r.bbox[:3], *r.bbox[3:6])
        cz,cy,cx = r.centroid
        V = float(r.area)
        volume_um3 = float(V * voxel_volume_um3)
        r_eq_um = ((3.0*volume_um3)/(4.0*np.pi))**(1/3) if volume_um3>0 else 0.0
        rows.append({
            "label_id": int(r.label),
            "volume_vox": V,
            "volume_um3": volume_um3,
            "centroid_z": float(cz), "centroid_y": float(cy), "centroid_x": float(cx),
            "centroid_z_um": float(cz*vz), "centroid_y_um": float(cy*vy), "centroid_x_um": float(cx*vx),
            "r_eq_um": float(r_eq_um),
            "bbox_zyx0": f"{z0},{y0},{x0}",
            "bbox_zyx1": f"{z1},{y1},{x1}",
        })
    return pd.DataFrame(rows)

def overlap_counts(a: np.ndarray, b: np.ndarray) -> Tuple[Dict[int, Dict[int, int]], np.ndarray, np.ndarray]:
    A, B = a.ravel().astype(np.int64, copy=False), b.ravel().astype(np.int64, copy=False)
    max_a = int(A.max()) if A.size else 0
    max_b = int(B.max()) if B.size else 0
    fg = (A > 0) & (B > 0)
    overlaps: Dict[int, Dict[int, int]] = defaultdict(dict)
    if fg.any():
        pairs = np.column_stack((A[fg], B[fg]))
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        for (i, j), cnt in zip(uniq, counts):
            overlaps[int(i)][int(j)] = int(cnt)
    sizes_a = np.bincount(A, minlength=max_a + 1)
    sizes_b = np.bincount(B, minlength=max_b + 1)
    return overlaps, sizes_a, sizes_b

def build_candidates(props_t: pd.DataFrame,
                     props_tp1: pd.DataFrame,
                     overlaps: Dict[int, Dict[int, int]],
                     sizes_a: np.ndarray,
                     sizes_b: np.ndarray,
                     iou_thr_event: float,
                     event_proximity_um: float,
                     max_disp_um: float,
                     w_iou: float, w_dist: float, w_vol: float,
                     max_cost: float) -> Tuple[pd.DataFrame, Dict[int,List[int]], Dict[int,List[int]]]:
    i_list, j_list = [], []
    iou_list, dist_list, dvol_list, cost_list, assignable_list = [], [], [], [], []
    parents_of = {}
    children_of = {}
    P = props_t.set_index("label_id")
    C = props_tp1.set_index("label_id")

    child_coords = None
    tree = None
    max_search = 0.0
    for val in (max_disp_um, event_proximity_um):
        if val is not None and np.isfinite(val):
            max_search = max(max_search, float(val))
    if len(C):
        child_coords = C[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].to_numpy(float)
        if max_search > 0:
            tree = cKDTree(child_coords)

    for i in P.index:
        pi = P.loc[i]
        candidate_children: set[int] = set(overlaps.get(int(i), {}).keys())
        if tree is not None and max_search > 0:
            center = [pi["centroid_z_um"], pi["centroid_y_um"], pi["centroid_x_um"]]
            nearby_idx = tree.query_ball_point(center, r=float(max_search))
            candidate_children.update(int(C.index[j]) for j in nearby_idx)
        else:
            candidate_children.update(int(idx) for idx in C.index)

        for j in candidate_children:
            if j not in C.index:
                continue
            cj = C.loc[j]
            ov = overlaps.get(int(i), {}).get(int(j), 0)
            union = sizes_a[i] + sizes_b[j] - ov
            iou = float(ov / union) if union > 0 else 0.0
            dz = pi["centroid_z_um"] - cj["centroid_z_um"]
            dy = pi["centroid_y_um"] - cj["centroid_y_um"]
            dx = pi["centroid_x_um"] - cj["centroid_x_um"]
            dist_um = float(np.sqrt(dz*dz + dy*dy + dx*dx))
            event_ev = (iou >= iou_thr_event) or (event_proximity_um is not None and event_proximity_um > 0 and dist_um <= float(event_proximity_um))
            # prune assignment search but allow event evidence beyond max_disp_um if flagged
            if (iou <= 0.0) and (max_disp_um is not None and np.isfinite(max_disp_um) and dist_um > max_disp_um) and not event_ev:
                continue
            rnorm = float(pi["r_eq_um"] + cj["r_eq_um"] + 1e-6)
            f_dist = dist_um / rnorm
            Vi = float(pi["volume_um3"]) if "volume_um3" in P.columns else float(pi["volume_vox"])
            Vj = float(cj["volume_um3"]) if "volume_um3" in C.columns else float(cj["volume_vox"])
            denom = max(Vi, Vj)
            dvol = abs(Vj - Vi) / denom if denom > 0 else 0.0
            cost = w_iou*(1.0 - iou) + w_dist*(f_dist) + w_vol*(dvol)
            assignable = cost <= max_cost
            if (not assignable) and (not event_ev):
                continue
            i_list.append(int(i)); j_list.append(int(j))
            iou_list.append(iou); dist_list.append(dist_um); dvol_list.append(dvol); cost_list.append(cost)
            assignable_list.append(bool(assignable))
            parents_of.setdefault(int(j), []).append(int(i))
            children_of.setdefault(int(i), []).append(int(j))

    df = pd.DataFrame({
        "parent": i_list, "child": j_list,
        "iou": iou_list, "dist_um": dist_list, "dvol": dvol_list, "cost": cost_list,
        "assignable": assignable_list
    })
    df["overlap_event_evidence"] = df["iou"] >= iou_thr_event
    if event_proximity_um is not None and event_proximity_um > 0:
        df["proximity_event_evidence"] = df["dist_um"] <= float(event_proximity_um)
    else:
        df["proximity_event_evidence"] = False
    df["event_evidence"] = df["overlap_event_evidence"] | df["proximity_event_evidence"]
    return df, parents_of, children_of

def hungarian_1to1(df_cand: pd.DataFrame) -> pd.DataFrame:
    if df_cand.empty:
        return df_cand.assign(assigned=False)
    df_assignable = df_cand[df_cand.get("assignable", True)]
    parents = sorted(df_assignable["parent"].unique())
    children = sorted(df_assignable["child"].unique())
    p_idx = {p:i for i,p in enumerate(parents)}
    c_idx = {c:j for j,c in enumerate(children)}
    Cmat = np.full((len(parents), len(children)), 1e6, dtype=float)
    for _, row in df_assignable.iterrows():
        Cmat[p_idx[row["parent"]], c_idx[row["child"]]] = row["cost"]
    r_idx, c_jdx = linear_sum_assignment(Cmat)
    valid = {(parents[ri], children[cj]) for ri, cj in zip(r_idx, c_jdx) if Cmat[ri, cj] < 1e6}
    assigned = [(row["parent"], row["child"]) in valid for _, row in df_cand.iterrows()]
    return df_cand.assign(assigned=assigned)

def detect_events(df_cand: pd.DataFrame,
                  props_t: pd.DataFrame,
                  props_tp1: pd.DataFrame,
                  iou_thr_event: float,
                  vol_tol: float) -> List[dict]:
    events = []
    P = props_t.set_index("label_id")
    C = props_tp1.set_index("label_id")
    ev_col = "event_evidence" if "event_evidence" in df_cand.columns else "overlap_event_evidence"
    evidence = df_cand[df_cand[ev_col]]

    def _vol(df, idx, column):
        if column in df.columns:
            return float(df.loc[idx, column])
        return float(df.loc[idx, "volume_vox"])

    # fission
    for p, sub in evidence.groupby("parent"):
        kids = list(sub["child"].unique())
        if len(kids) >= 2:
            Vi = _vol(P, p, "volume_um3")
            Vsum = float(C.loc[kids, "volume_um3"].sum()) if "volume_um3" in C.columns else float(C.loc[kids, "volume_vox"].sum())
            denom = max(Vsum, Vi)
            if denom == 0:
                continue
            if abs(Vsum - Vi)/denom <= vol_tol:
                events.append({"event":"fission","parent":int(p),"children":[int(c) for c in kids]})
    # fusion
    for c, sub in evidence.groupby("child"):
        pars = list(sub["parent"].unique())
        if len(pars) >= 2:
            Vj = _vol(C, c, "volume_um3")
            Vsum = float(P.loc[pars, "volume_um3"].sum()) if "volume_um3" in P.columns else float(P.loc[pars, "volume_vox"].sum())
            denom = max(Vsum, Vj)
            if denom == 0:
                continue
            if abs(Vsum - Vj)/denom <= vol_tol:
                events.append({"event":"fusion","child":int(c),"parents":[int(p) for p in pars]})
    return events

def save_label_preview_pngs(sequence_labels: List[np.ndarray], outdir: str):
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(outdir, "quicklook_pngs"), exist_ok=True)
    for t, lab in enumerate(sequence_labels):
        lab, _, _ = relabel_sequential(lab)
        maxlab = int(lab.max())
        if maxlab == 0:
            img = np.zeros((lab.shape[1], lab.shape[2], 3), dtype=np.uint8)
        else:
            colors = np.zeros((maxlab+1, 3), dtype=np.uint8)
            colors[1:] = rng.integers(0,255,size=(maxlab,3), dtype=np.uint8)
            mip = lab.max(axis=0)
            img = colors[mip]
        Image.fromarray(img).save(os.path.join(outdir, "quicklook_pngs", f"t{t:03d}.png"))

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--axis", default="tzyx")
    ap.add_argument("--voxel_size", nargs=3, type=float, default=[1.0,1.0,1.0], metavar=("Z_UM","Y_UM","X_UM"))
    ap.add_argument("--min_size", type=int, default=0)
    ap.add_argument("--connectivity", type=int, default=1, choices=[1,2])
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian sigma (voxels) applied to the binary mask before labeling; 0 disables.")
    ap.add_argument("--closing_radius", type=float, default=0.0, help="Binary closing radius (voxels) to bridge small gaps and reduce flicker; 0 disables.")
    ap.add_argument("--closing_xy", action="store_true", help="Apply closing per Z-slice (disk) instead of full 3D ball.")

    # multi-criteria params
    ap.add_argument("--iou_thr_event", type=float, default=0.06)
    ap.add_argument("--max_disp_um", type=float, default=2.0)
    ap.add_argument("--w_iou", type=float, default=0.6)
    ap.add_argument("--w_dist", type=float, default=0.3)
    ap.add_argument("--w_vol", type=float, default=0.1)
    ap.add_argument("--max_cost", type=float, default=1.2)
    ap.add_argument("--vol_tol", type=float, default=0.35)
    ap.add_argument("--min_event_persistence", type=int, default=1)
    ap.add_argument("--event_proximity_um", type=float, default=0.0, help="If >0, also accept event evidence when centroids are within this distance even if IoU is low.")
    ap.add_argument("--gap_frames", type=int, default=1, help="Max number of missing frames to bridge when reconnecting tracks (0 disables).")
    ap.add_argument("--gap_max_disp_factor", type=float, default=1.5, help="Multiplier on max_disp_um per skipped frame during gap closing.")
    ap.add_argument("--gap_frame_penalty", type=float, default=0.1, help="Additive cost penalty per skipped frame during gap closing.")

    # I/O and visualization
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save_pngs", action="store_true")
    ap.add_argument("--save_labels", action="store_true", help="Save labels stack as labels.tif in outdir")
    ap.add_argument("--napari", action="store_true")
    ap.add_argument("--napari_show_fission", action="store_true", help="If set, show magenta fission edges/nodes.")
    ap.add_argument("--napari_show_fusion", action="store_true", help="If set, show orange fusion edges/nodes.")
    ap.add_argument("--save_parquet", action="store_true", help="Also save .parquet versions of all tables.")
    ap.add_argument("--export_graph", action="store_true", help="Export graph as GraphML and DOT.")
    ap.add_argument("--graph_per_track", action="store_true", help="When exporting graph, also save per-track subgraphs.")

    args = ap.parse_args()

    vz, vy, vx = args.voxel_size

    arr = imread(args.input)
    if arr.ndim < 3 or arr.ndim > 5:
        print(f"ERROR: expected 3D/4D stack (optionally with channel), got shape {arr.shape}", file=sys.stderr); sys.exit(2)
    try:
        arr = to_tzyx(arr, args.axis)  # (T,Z,Y,X) with singleton time added when needed
    except ValueError as exc:
        print(f"ERROR interpreting --axis with data shape {arr.shape}: {exc}", file=sys.stderr); sys.exit(2)
    T,Z,Y,X = arr.shape
    arr = ensure_binary(arr)

    labels = []
    for t in range(T):
        vol = arr[t]
        if args.smooth_sigma > 0 or args.closing_radius > 0:
            vol = preprocess_binary_volume(vol, args.smooth_sigma, args.closing_radius, args.closing_xy)
            arr[t] = vol
        if args.min_size>0:
            vol = remove_small_objects(vol.astype(bool), args.min_size).astype(np.uint8)
            arr[t] = vol  # keep filtered volume for downstream previews/Napari
        lab = label_3d(vol, connectivity=args.connectivity)
        labels.append(lab)

    props = []
    for t in range(T):
        df = props_table(labels[t], vz, vy, vx)
        df["t"] = t
        props.append(df)

    track_of = {}
    next_tid = 1
    objects_rows, links_rows, matches_rows, events_rows_raw = [], [], [], []

    # init tracks at t=0
    for _, row in props[0].iterrows():
        track_of[(0, int(row["label_id"]))] = next_tid; next_tid += 1

    # iterate frames
    for t in range(T-1):
        A, B = labels[t], labels[t+1]
        overlaps, sizes_a, sizes_b = overlap_counts(A, B)
        df_cand, parents_of, children_of = build_candidates(
            props[t], props[t+1], overlaps, sizes_a, sizes_b,
            iou_thr_event=args.iou_thr_event,
            event_proximity_um=args.event_proximity_um,
            max_disp_um=args.max_disp_um,
            w_iou=args.w_iou, w_dist=args.w_dist, w_vol=args.w_vol,
            max_cost=args.max_cost
        )
        df_assign = hungarian_1to1(df_cand)
        assigned_pairs = df_assign[df_assign["assigned"]][["parent","child","cost","iou","dist_um","dvol"]].values.tolist()
        assigned_set = {(int(p),int(c)) for p,c,_,_,_,_ in assigned_pairs}

        # links table (all candidates)
        for _, r in df_cand.iterrows():
            links_rows.append({
                "t_from": t,
                "t_to": t+1,
                "parent_label": int(r["parent"]),
                "child_label": int(r["child"]),
                "assigned": bool((int(r["parent"]), int(r["child"])) in assigned_set),
                "assignable": bool(r.get("assignable", True)),
                "cost": float(r["cost"]), "iou": float(r["iou"]),
                "dist_um": float(r["dist_um"]), "dvol": float(r["dvol"]),
                "overlap_event_evidence": bool(r["overlap_event_evidence"]),
                "proximity_event_evidence": bool(r.get("proximity_event_evidence", False)),
                "event_evidence": bool(r.get("event_evidence", False)),
            })

        # matches table (only assigned 1:1)
        for p,c, cost,iou,dist,dvol in assigned_pairs:
            ptid = track_of.get((t, int(p)))
            if ptid is None: ptid = next_tid; next_tid += 1
            track_of[(t+1, int(c))] = ptid
            matches_rows.append({
                "t_from": t, "t_to": t+1,
                "parent_label": int(p), "child_label": int(c),
                "parent_track_id": int(ptid), "child_track_id": int(ptid),
                "cost": float(cost), "iou": float(iou), "dist_um": float(dist), "dvol": float(dvol)
            })

        # any child not assigned -> new track
        assigned_c = {int(c) for _,c,_,_,_,_ in assigned_pairs}
        B_labels = set(props[t+1]["label_id"].astype(int).tolist())
        for c in B_labels - assigned_c:
            track_of[(t+1, int(c))] = next_tid
            matches_rows.append({
                "t_from": t, "t_to": t+1,
                "parent_label": -1, "child_label": int(c),
                "parent_track_id": -1, "child_track_id": int(next_tid),
                "cost": np.nan, "iou": 0.0, "dist_um": np.nan, "dvol": np.nan
            })
            next_tid += 1

        # event detection
        events = detect_events(df_cand, props[t], props[t+1],
                               iou_thr_event=args.iou_thr_event,
                               vol_tol=args.vol_tol)
        for ev in events:
            if ev["event"] == "fission":
                for ch in ev["children"]:
                    events_rows_raw.append({
                        "event": "fission",
                        "t_from": t, "t_to": t+1,
                        "parent_label": int(ev["parent"]),
                        "child_label": int(ch)
                    })
            else:
                for pr in ev["parents"]:
                    events_rows_raw.append({
                        "event": "fusion",
                        "t_from": t, "t_to": t+1,
                        "parent_label": int(pr),
                        "child_label": int(ev["child"])
                    })

    # objects table
    for t in range(T):
        for _, r in props[t].iterrows():
            objects_rows.append({
                "t": t,
                "label_id": int(r["label_id"]),
                "track_id": int(track_of.get((t, int(r['label_id'])), -1)),
                "centroid_z_um": float(r["centroid_z_um"]), "centroid_y_um": float(r["centroid_y_um"]), "centroid_x_um": float(r["centroid_x_um"]),
                "centroid_z_vox": float(r["centroid_z"]), "centroid_y_vox": float(r["centroid_y"]), "centroid_x_vox": float(r["centroid_x"]),
                "volume_vox": float(r["volume_vox"]),
                "volume_um3": float(r.get("volume_um3", r["volume_vox"])),
                "r_eq_um": float(r["r_eq_um"]),
                "bbox_zyx0": r["bbox_zyx0"], "bbox_zyx1": r["bbox_zyx1"],
            })

    objects_columns = [
        "t",
        "label_id",
        "track_id",
        "centroid_z_um",
        "centroid_y_um",
        "centroid_x_um",
        "centroid_z_vox",
        "centroid_y_vox",
        "centroid_x_vox",
        "volume_vox",
        "volume_um3",
        "r_eq_um",
        "bbox_zyx0",
        "bbox_zyx1",
    ]
    objects_df = pd.DataFrame(objects_rows, columns=objects_columns)
    if not objects_df.empty:
        objects_df = objects_df.sort_values(["t", "label_id"]).reset_index(drop=True)

    links_columns = [
        "t_from",
        "t_to",
        "parent_label",
        "child_label",
        "assigned",
        "assignable",
        "cost",
        "iou",
        "dist_um",
        "dvol",
        "overlap_event_evidence",
        "proximity_event_evidence",
        "event_evidence",
    ]
    links_df = pd.DataFrame(links_rows, columns=links_columns)
    if not links_df.empty:
        links_df = links_df.sort_values(["t_from", "parent_label", "child_label"]).reset_index(drop=True)

    matches_columns = [
        "t_from",
        "t_to",
        "parent_label",
        "child_label",
        "parent_track_id",
        "child_track_id",
        "cost",
        "iou",
        "dist_um",
        "dvol",
    ]
    matches_df = pd.DataFrame(matches_rows, columns=matches_columns)
    if not matches_df.empty:
        matches_df = matches_df.sort_values(["t_from", "parent_label", "child_label"]).reset_index(drop=True)

    # optional gap-closing to reconnect tracks across short segmentation dropouts
    track_remap, gap_edge_meta = gap_close_tracks(
        objects_df=objects_df,
        gap_frames=int(args.gap_frames),
        max_disp_um=args.max_disp_um,
        gap_max_disp_factor=float(args.gap_max_disp_factor),
        w_dist=float(args.w_dist),
        w_vol=float(args.w_vol),
        frame_penalty=float(args.gap_frame_penalty),
    )
    def canonical_tid(tid: int) -> int:
        try:
            tid_int = int(tid)
        except Exception:
            return tid
        seen = set()
        while tid_int in track_remap and tid_int not in seen:
            seen.add(tid_int)
            nxt = track_remap[tid_int]
            if nxt == tid_int:
                break
            tid_int = nxt
        return tid_int

    if track_remap:
        objects_df["track_id"] = apply_track_mapping(track_remap, objects_df["track_id"])
        if "parent_track_id" in matches_df.columns:
            matches_df["parent_track_id"] = apply_track_mapping(track_remap, matches_df["parent_track_id"])
        if "child_track_id" in matches_df.columns:
            matches_df["child_track_id"] = apply_track_mapping(track_remap, matches_df["child_track_id"])
        # keep track_of in sync for downstream label exports
        track_of = {(t, l): canonical_tid(v) for (t, l), v in track_of.items()}
        objects_df = objects_df.sort_values(["t", "label_id"]).reset_index(drop=True)
        matches_df = matches_df.sort_values(["t_from", "parent_label", "child_label"]).reset_index(drop=True)
        merged = {k: v for k, v in track_remap.items()}
        if merged:
            print(f"Gap closing merged {len(merged)} track(s): " + ", ".join(f"{dst}->{src}" for dst, src in merged.items()))

    # persistence filtering for events -> create normalized edges tables (no "t:id" strings)
    # map (t,label) -> (track_id, centroids, volume)
    tl2tr = {(int(r["t"]), int(r["label_id"])): int(r["track_id"]) for _, r in objects_df.iterrows()}
    obj_lookup = objects_df.set_index(["t","label_id"])
    times_by_track = objects_df.groupby("track_id")["t"].apply(list).to_dict()
    def persists(track_id: int, start_t: int, forward: bool, N: int) -> bool:
        if N <= 1:
            return True
        ts = times_by_track.get(track_id, [])
        if not ts:
            return False
        ts_set = set(int(v) for v in ts)
        count = 0
        t = int(start_t)
        step = 1 if forward else -1
        while count < N and t in ts_set:
            count += 1
            t += step
        return count >= N

    def centroid_z_info(frame: int, label: int):
        key = (frame, label)
        if key not in obj_lookup.index:
            return (np.nan, np.nan, -1)
        z_um = float(obj_lookup.loc[key, "centroid_z_um"])
        if (vz is None) or (not np.isfinite(vz)) or (vz <= 0):
            z_vox = float(obj_lookup.loc[key, "centroid_z_vox"]) if "centroid_z_vox" in obj_lookup.columns else np.nan
        else:
            z_vox = z_um / vz
        slice_idx = int(round(z_vox)) if np.isfinite(z_vox) else -1
        return (z_um, z_vox, slice_idx)

    # Build event edge rows with track IDs & volumes
    fission_rows = []
    fusion_rows  = []
    for e in events_rows_raw:
        t_from, t_to = int(e["t_from"]), int(e["t_to"])
        p, c = int(e["parent_label"]), int(e["child_label"])
        # look up track ids (may be -1 if missing)
        try: ptid = tl2tr[(t_from, p)]
        except: ptid = -1
        try: ctid = tl2tr[(t_to, c)]
        except: ctid = -1
        # persistence filter
        ok = True
        if args.min_event_persistence > 1:
            ok = (ptid != -1 and persists(ptid, t_from, forward=False, N=args.min_event_persistence)) and \
                 (ctid != -1 and persists(ctid, t_to,   forward=True,  N=args.min_event_persistence))
        if not ok:
            continue

        vol_parent = float(obj_lookup.loc[(t_from,p), "volume_vox"]) if (t_from,p) in obj_lookup.index else np.nan
        vol_child  = float(obj_lookup.loc[(t_to,c), "volume_vox"])   if (t_to,c)   in obj_lookup.index else np.nan

        parent_cz_um, parent_cz_vox, parent_slice_idx = centroid_z_info(t_from, p)
        child_cz_um, child_cz_vox, child_slice_idx = centroid_z_info(t_to, c)

        row = {
            "event": e["event"],
            "t_from": t_from, "t_to": t_to,
            "parent_label": p, "child_label": c,
            "parent_track_id": ptid, "child_track_id": ctid,
            "parent_volume_vox": vol_parent, "child_volume_vox": vol_child,
            "parent_centroid_z_um": parent_cz_um, "parent_centroid_z_vox": parent_cz_vox,
            "parent_slice_index": int(parent_slice_idx),
            "child_centroid_z_um": child_cz_um, "child_centroid_z_vox": child_cz_vox,
            "child_slice_index": int(child_slice_idx)
        }
        if e["event"] == "fission":
            fission_rows.append(row)
        else:
            fusion_rows.append(row)

    event_columns = [
        "event",
        "t_from",
        "t_to",
        "parent_label",
        "child_label",
        "parent_track_id",
        "child_track_id",
        "parent_volume_vox",
        "child_volume_vox",
        "parent_centroid_z_um",
        "parent_centroid_z_vox",
        "parent_slice_index",
        "child_centroid_z_um",
        "child_centroid_z_vox",
        "child_slice_index",
    ]
    fission_df = pd.DataFrame(fission_rows, columns=event_columns)
    if not fission_df.empty:
        fission_df = fission_df.sort_values(["t_from", "parent_label", "child_label"]).reset_index(drop=True)
    else:
        fission_df = pd.DataFrame(columns=event_columns)

    fusion_df = pd.DataFrame(fusion_rows, columns=event_columns)
    if not fusion_df.empty:
        fusion_df = fusion_df.sort_values(["t_from", "parent_label", "child_label"]).reset_index(drop=True)
    else:
        fusion_df = pd.DataFrame(columns=event_columns)

    # combined edge list for graph building (continuations + events)
    # continuation edges from matches_df where parent_label != -1 (1:1 or appearance)
    cont_edges = matches_df[matches_df["parent_label"] != -1][[
        "t_from","t_to","parent_label","child_label","parent_track_id","child_track_id"
    ]].copy()
    cont_edges["edge_type"] = "continuation"

    gap_edges_rows = []
    for end_row, start_row in gap_edge_meta:
        try:
            t_from = int(end_row["t"]); t_to = int(start_row["t"])
            p_lbl = int(end_row["label_id"]); c_lbl = int(start_row["label_id"])
            p_tid = canonical_tid(end_row.get("track_id", -1))
            c_tid = canonical_tid(start_row.get("track_id", -1))
            gap_edges_rows.append({
                "t_from": t_from,
                "t_to": t_to,
                "parent_label": p_lbl,
                "child_label": c_lbl,
                "parent_track_id": p_tid,
                "child_track_id": c_tid,
                "edge_type": "gap"
            })
        except Exception:
            continue
    gap_edges_df = pd.DataFrame(gap_edges_rows, columns=["t_from","t_to","parent_label","child_label","parent_track_id","child_track_id","edge_type"])

    e_fis = fission_df.copy(); e_fis["edge_type"] = "fission"
    e_fus = fusion_df.copy();  e_fus["edge_type"] = "fusion"

    edges_all = pd.concat([cont_edges, gap_edges_df, e_fis, e_fus], ignore_index=True)

    # aggregate event stats (per frame and per slice)
    if not fission_df.empty or not fusion_df.empty:
        events_all = pd.concat([fission_df, fusion_df], ignore_index=True)
    else:
        events_all = pd.DataFrame(columns=["event", "t_from", "t_to"])

    if not events_all.empty:
        per_t_counts = events_all.groupby(["t_to", "event"]).size().unstack(fill_value=0)
    else:
        per_t_counts = pd.DataFrame()
    for col in ("fission", "fusion"):
        if col not in per_t_counts:
            per_t_counts[col] = 0
    per_t_df = per_t_counts.reset_index().rename(columns={"t_to": "t", "fission": "fission_count", "fusion": "fusion_count"})
    if per_t_df.empty:
        per_t_df = pd.DataFrame({"t": [], "fission_count": [], "fusion_count": [], "total_events": []})
    else:
        per_t_df["total_events"] = per_t_df["fission_count"] + per_t_df["fusion_count"]
    all_frames = pd.DataFrame({"t": np.arange(T, dtype=int)})
    per_t_df = all_frames.merge(per_t_df, on="t", how="left").fillna(0)
    for col in ["fission_count", "fusion_count", "total_events"]:
        per_t_df[col] = per_t_df[col].astype(int)

    if not events_all.empty and "child_slice_index" in events_all.columns:
        per_slice_events = events_all.dropna(subset=["child_slice_index"]).copy()
        per_slice_events = per_slice_events[per_slice_events["child_slice_index"] >= 0]
        if per_slice_events.empty:
            per_slice_df = pd.DataFrame({"t": [], "slice_index": [], "slice_z_um": [], "fission_count": [], "fusion_count": [], "total_events": []})
        else:
            per_slice_events["child_slice_index"] = per_slice_events["child_slice_index"].astype(int)
            per_slice_counts = per_slice_events.groupby(["t_to", "child_slice_index", "event"]).size().unstack(fill_value=0)
            for col in ("fission", "fusion"):
                if col not in per_slice_counts:
                    per_slice_counts[col] = 0
            per_slice_df = per_slice_counts.reset_index().rename(columns={
                "t_to": "t",
                "child_slice_index": "slice_index",
                "fission": "fission_count",
                "fusion": "fusion_count"
            })
            per_slice_df["slice_z_um"] = per_slice_df["slice_index"] * vz
            per_slice_df["total_events"] = per_slice_df["fission_count"] + per_slice_df["fusion_count"]
            per_slice_df = per_slice_df.sort_values(["t", "slice_index"]).reset_index(drop=True)
            for col in ["fission_count", "fusion_count", "total_events"]:
                per_slice_df[col] = per_slice_df[col].astype(int)
    else:
        per_slice_df = pd.DataFrame({"t": [], "slice_index": [], "slice_z_um": [], "fission_count": [], "fusion_count": [], "total_events": []})

    if not events_all.empty and "child_slice_index" in events_all.columns:
        per_z_events = events_all.dropna(subset=["child_slice_index"]).copy()
        per_z_events = per_z_events[per_z_events["child_slice_index"] >= 0]
        if per_z_events.empty:
            per_z_df = pd.DataFrame({"slice_index": [], "slice_z_um": [], "fission_count": [], "fusion_count": [], "total_events": []})
        else:
            per_z_events["child_slice_index"] = per_z_events["child_slice_index"].astype(int)
            per_z_counts = per_z_events.groupby(["child_slice_index", "event"]).size().unstack(fill_value=0)
            for col in ("fission", "fusion"):
                if col not in per_z_counts:
                    per_z_counts[col] = 0
            per_z_df = per_z_counts.reset_index().rename(columns={
                "child_slice_index": "slice_index",
                "fission": "fission_count",
                "fusion": "fusion_count"
            })
            per_z_df["slice_z_um"] = per_z_df["slice_index"] * vz
            per_z_df["total_events"] = per_z_df["fission_count"] + per_z_df["fusion_count"]
            per_z_df = per_z_df.sort_values(["slice_index"]).reset_index(drop=True)
            for col in ["fission_count", "fusion_count", "total_events"]:
                per_z_df[col] = per_z_df[col].astype(int)
    else:
        per_z_df = pd.DataFrame({"slice_index": [], "slice_z_um": [], "fission_count": [], "fusion_count": [], "total_events": []})

    # --------------------- save tables ---------------------
    os.makedirs(args.outdir, exist_ok=True)
    def save(df: pd.DataFrame, name: str):
        path_csv = os.path.join(args.outdir, f"{name}.csv")
        df.to_csv(path_csv, index=False)
        if args.save_parquet:
            df.to_parquet(os.path.join(args.outdir, f"{name}.parquet"), index=False)
        return path_csv

    p_objects = save(objects_df, "objects")
    p_links   = save(links_df,   "links")
    p_matches = save(matches_df, "matches")
    p_fiss    = save(fission_df, "events_fission_edges")
    p_fus     = save(fusion_df,  "events_fusion_edges")
    p_edges   = save(edges_all,  "edges_all")
    p_t       = save(per_t_df, "events_per_t")
    p_slice   = save(per_slice_df, "events_per_slice")
    p_z       = save(per_z_df, "events_per_z")

    print("Saved:")
    for p in [p_objects, p_links, p_matches, p_fiss, p_fus, p_edges, p_t, p_slice, p_z]:
        print("  ", p)

    if args.save_labels:
        try:
            lab_stack = np.stack(labels, axis=0).astype("uint32")
            lab_hyper = np.expand_dims(lab_stack, axis=1)  # (T,1,Z,Y,X)
            labels_path = os.path.join(args.outdir, "labels.tif")
            imwrite(labels_path, lab_hyper, ome=True, metadata={"axes": "TCZYX", "Channel": [{"Name": "label_id"}]})
            print("Saved labels stack to", labels_path)

            track_volumes: List[np.ndarray] = []
            max_track_id = 0
            for t_idx, lab in enumerate(labels):
                if lab.size == 0:
                    track_volumes.append(np.zeros_like(lab, dtype=np.uint32))
                    continue
                max_label = int(lab.max())
                if max_label == 0:
                    track_volumes.append(np.zeros_like(lab, dtype=np.uint32))
                    continue
                lut = np.zeros(max_label + 1, dtype=np.uint32)
                for label_id in range(1, max_label + 1):
                    tid = track_of.get((t_idx, label_id), -1)
                    lut[label_id] = tid if tid > 0 else 0
                track_volume = lut[lab]
                max_track_id = max(max_track_id, int(track_volume.max()))
                track_volumes.append(track_volume)

            track_stack = np.stack(track_volumes, axis=0).astype("uint32")
            track_hyper = np.expand_dims(track_stack, axis=1)
            tracks_path = os.path.join(args.outdir, "tracks.tif")
            imwrite(tracks_path, track_hyper, ome=True, metadata={"axes": "TCZYX", "Channel": [{"Name": "track_id"}]})
            print("Saved track-labeled stack to", tracks_path)

            if max_track_id > 0:
                rng = np.random.default_rng(42)
                colors = rng.integers(0, 256, size=(max_track_id + 1, 3), dtype=np.uint8)
                colors[0] = np.array([0, 0, 0], dtype=np.uint8)
                color_volumes = [colors[vol] for vol in track_volumes]
                color_stack = np.stack(color_volumes, axis=0)  # (T,Z,Y,X,3)
                color_stack = np.moveaxis(color_stack, -1, 1)  # (T,3,Z,Y,X)
                tracks_rgb_path = os.path.join(args.outdir, "tracks_rgb.tif")
                imwrite(
                    tracks_rgb_path,
                    color_stack,
                    ome=True,
                    metadata={
                        "axes": "TCZYX",
                        "Channel": [
                            {"Name": "track_R"},
                            {"Name": "track_G"},
                            {"Name": "track_B"},
                        ],
                    },
                )
                print("Saved track RGB stack to", tracks_rgb_path)
        except Exception as e:
            print("Failed to save labels/track stacks:", e, file=sys.stderr)

    if args.save_pngs:
        save_label_preview_pngs(labels, args.outdir)

    # --------------------- graph exports ---------------------
    if args.export_graph:
        try:
            import networkx as nx
            G = nx.DiGraph()
            # nodes are (t,label) with attributes
            for _, r in objects_df.iterrows():
                G.add_node((int(r["t"]), int(r["label_id"])), 
                           track_id=int(r["track_id"]),
                           volume_vox=float(r["volume_vox"]),
                           volume_um3=float(r.get("volume_um3", np.nan)),
                           cz_um=float(r["centroid_z_um"]), cy_um=float(r["centroid_y_um"]), cx_um=float(r["centroid_x_um"]))
            # edges: from edges_all
            for _, r in edges_all.iterrows():
                u = (int(r["t_from"]), int(r["parent_label"]))
                v = (int(r["t_to"]), int(r["child_label"]))
                G.add_edge(u, v, edge_type=r["edge_type"],
                           parent_track_id=int(r["parent_track_id"]), child_track_id=int(r["child_track_id"]))
            # whole graph
            nx.write_graphml(G, os.path.join(args.outdir, "lineage.graphml"))
            try:
                nx.drawing.nx_pydot.write_dot(G, os.path.join(args.outdir, "lineage.dot"))
            except Exception:
                pass
            print("Saved graph: lineage.graphml (and DOT if pydot available)")

            if args.graph_per_track:
                for tid, sub in objects_df.groupby("track_id"):
                    if tid == -1: continue
                    nodes = [(int(r["t"]), int(r["label_id"])) for _, r in sub.iterrows()]
                    # expand neighborhood by including edges touching these nodes
                    H = nx.DiGraph()
                    for n in nodes:
                        if n in G:
                            H.add_node(n, **G.nodes[n])
                    for u,v,data in G.edges(data=True):
                        if u in H and v in H:
                            H.add_edge(u,v, **data)
                    if H.number_of_edges() == 0 and H.number_of_nodes() <= 1:
                        continue
                    nx.write_graphml(H, os.path.join(args.outdir, f"track_{int(tid)}.graphml"))
                print("Saved per-track subgraphs (graphml).")
        except Exception as e:
            print("Graph export failed:", e, file=sys.stderr)

    # --------------------- Napari overlay ---------------------
    if args.napari:
        try:
            import napari
            v = napari.Viewer(ndisplay=3)
            v.dims.ndisplay = 3

            v.add_image(arr, name="binary", contrast_limits=[0,1])

            def safe_div_series(series_um: pd.Series, series_vox: pd.Series | None, scale: float | None) -> np.ndarray:
                valid_scale = scale is not None and np.isfinite(scale) and scale > 0
                if valid_scale:
                    return (series_um / scale).to_numpy(float)
                if series_vox is not None:
                    return series_vox.astype(float).to_numpy()
                return np.full(len(series_um), np.nan, dtype=float)

            def safe_div_value(um_val: float, vox_val: float | None, scale: float | None) -> float:
                valid_scale = scale is not None and np.isfinite(scale) and scale > 0
                if valid_scale:
                    return float(um_val) / float(scale)
                if vox_val is not None:
                    try:
                        return float(vox_val)
                    except Exception:
                        return float("nan")
                return float("nan")

            # centroids in voxel coords
            objects_df["cz"] = safe_div_series(objects_df["centroid_z_um"], objects_df.get("centroid_z_vox"), vz)
            objects_df["cy"] = safe_div_series(objects_df["centroid_y_um"], objects_df.get("centroid_y_vox"), vy)
            objects_df["cx"] = safe_div_series(objects_df["centroid_x_um"], objects_df.get("centroid_x_vox"), vx)
            pts_all = objects_df[["t","cz","cy","cx"]].to_numpy(float)
            valid_mask = np.all(np.isfinite(pts_all), axis=1)
            if not np.all(valid_mask):
                missing_pts = int((~valid_mask).sum())
                print(f"Napari overlay dropping {missing_pts} centroid rows with invalid coordinates.", file=sys.stderr)
            pts = pts_all[valid_mask]
            feats = {"track_id": objects_df.loc[valid_mask, "track_id"].astype(int).to_numpy()}
            cent_layer = v.add_points(pts, name="centroids", features=feats, ndim=4, size=2.0, face_color="track_id", blending="translucent")
            try:
                labels_text = [str(tid) if tid > 0 else "" for tid in feats["track_id"]]
                cent_layer.text = {"string": labels_text, "size": 12, "color": "white", "anchor": "upper_left", "translation": (0, 0, 0, 0)}
            except Exception:
                pass

            # Build edges
            shapes = []
            colors = []
            skip_cont = skip_gap = skip_fiss = skip_fus = 0

            # continuation edges (white)
            for _, r in matches_df[matches_df["parent_label"] != -1].iterrows():
                t0, t1 = int(r["t_from"]), int(r["t_to"])
                p, c = int(r["parent_label"]), int(r["child_label"])
                try:
                    z0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_z_um"], obj_lookup.loc[(t0,p), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                    y0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_y_um"], obj_lookup.loc[(t0,p), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                    x0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_x_um"], obj_lookup.loc[(t0,p), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                    z1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_z_um"], obj_lookup.loc[(t1,c), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                    y1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_y_um"], obj_lookup.loc[(t1,c), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                    x1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_x_um"], obj_lookup.loc[(t1,c), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                    coords = np.array([z0, y0, x0, z1, y1, x1], dtype=float)
                    if not np.all(np.isfinite(coords)):
                        skip_cont += 1
                        continue
                    seg = np.array([[t0, coords[0], coords[1], coords[2]],[t1, coords[3], coords[4], coords[5]]], dtype=float)
                    shapes.append(seg); colors.append("white")
                except Exception:
                    skip_cont += 1

            # gap edges (cyan) from gap closing
            if not gap_edges_df.empty:
                for _, r in gap_edges_df.iterrows():
                    t0, t1 = int(r["t_from"]), int(r["t_to"])
                    p, c = int(r["parent_label"]), int(r["child_label"])
                    try:
                        z0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_z_um"], obj_lookup.loc[(t0,p), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_y_um"], obj_lookup.loc[(t0,p), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_x_um"], obj_lookup.loc[(t0,p), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        z1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_z_um"], obj_lookup.loc[(t1,c), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_y_um"], obj_lookup.loc[(t1,c), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_x_um"], obj_lookup.loc[(t1,c), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        coords = np.array([z0, y0, x0, z1, y1, x1], dtype=float)
                        if not np.all(np.isfinite(coords)):
                            skip_gap += 1
                            continue
                        seg = np.array([[t0, coords[0], coords[1], coords[2]],[t1, coords[3], coords[4], coords[5]]], dtype=float)
                        shapes.append(seg); colors.append("cyan")
                    except Exception:
                        skip_gap += 1

            # fission edges (magenta)
            if args.napari_show_fission and not fission_df.empty:
                for _, r in fission_df.iterrows():
                    t0, t1 = int(r["t_from"]), int(r["t_to"])
                    p, c = int(r["parent_label"]), int(r["child_label"])
                    try:
                        z0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_z_um"], obj_lookup.loc[(t0,p), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_y_um"], obj_lookup.loc[(t0,p), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_x_um"], obj_lookup.loc[(t0,p), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        z1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_z_um"], obj_lookup.loc[(t1,c), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_y_um"], obj_lookup.loc[(t1,c), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_x_um"], obj_lookup.loc[(t1,c), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        coords = np.array([z0, y0, x0, z1, y1, x1], dtype=float)
                        if not np.all(np.isfinite(coords)):
                            skip_fiss += 1
                            continue
                        seg = np.array([[t0, coords[0], coords[1], coords[2]],[t1, coords[3], coords[4], coords[5]]], dtype=float)
                        shapes.append(seg); colors.append("magenta")
                    except Exception:
                        skip_fiss += 1

            # fusion edges (orange)
            if args.napari_show_fusion and not fusion_df.empty:
                for _, r in fusion_df.iterrows():
                    t0, t1 = int(r["t_from"]), int(r["t_to"])
                    p, c = int(r["parent_label"]), int(r["child_label"])
                    try:
                        z0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_z_um"], obj_lookup.loc[(t0,p), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_y_um"], obj_lookup.loc[(t0,p), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x0 = safe_div_value(obj_lookup.loc[(t0,p), "centroid_x_um"], obj_lookup.loc[(t0,p), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        z1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_z_um"], obj_lookup.loc[(t1,c), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz)
                        y1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_y_um"], obj_lookup.loc[(t1,c), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy)
                        x1 = safe_div_value(obj_lookup.loc[(t1,c), "centroid_x_um"], obj_lookup.loc[(t1,c), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx)
                        coords = np.array([z0, y0, x0, z1, y1, x1], dtype=float)
                        if not np.all(np.isfinite(coords)):
                            skip_fus += 1
                            continue
                        seg = np.array([[t0, coords[0], coords[1], coords[2]],[t1, coords[3], coords[4], coords[5]]], dtype=float)
                        shapes.append(seg); colors.append("orange")
                    except Exception:
                        skip_fus += 1

            if shapes:
                v.add_shapes(shapes, shape_type="path", name="links", edge_color=colors, edge_width=2.0, blending="translucent")

            # event nodes for child objects
            def child_nodes(df):
                L = []
                for _, r in df.iterrows():
                    t1 = int(r["t_to"]); c = int(r["child_label"])
                    if (t1, c) in obj_lookup.index:
                        try:
                            candidate = [
                                t1,
                                safe_div_value(obj_lookup.loc[(t1,c), "centroid_z_um"], obj_lookup.loc[(t1,c), "centroid_z_vox"] if "centroid_z_vox" in obj_lookup.columns else None, vz),
                                safe_div_value(obj_lookup.loc[(t1,c), "centroid_y_um"], obj_lookup.loc[(t1,c), "centroid_y_vox"] if "centroid_y_vox" in obj_lookup.columns else None, vy),
                                safe_div_value(obj_lookup.loc[(t1,c), "centroid_x_um"], obj_lookup.loc[(t1,c), "centroid_x_vox"] if "centroid_x_vox" in obj_lookup.columns else None, vx),
                            ]
                            if all(np.isfinite(candidate[1:])):
                                L.append(candidate)
                        except Exception:
                            continue
                return np.array(L, float) if L else None

            if args.napari_show_fission:
                pts = child_nodes(fission_df)
                if pts is not None:
                    layer = v.add_points(pts, name="fission nodes", ndim=4, size=3.0, face_color="magenta", blending="additive")
                    try:
                        v.layers.move(v.layers.index(layer), len(v.layers) - 1)
                    except Exception:
                        pass
            if args.napari_show_fusion:
                pts = child_nodes(fusion_df)
                if pts is not None:
                    layer = v.add_points(pts, name="fusion nodes", ndim=4, size=3.0, face_color="orange", blending="additive")
                    try:
                        v.layers.move(v.layers.index(layer), len(v.layers) - 1)
                    except Exception:
                        pass

            if skip_cont or skip_gap or skip_fiss or skip_fus:
                print(f"Napari overlay skipped edges (continuation={skip_cont}, gap={skip_gap}, fission={skip_fiss}, fusion={skip_fus}) due to missing coordinates.", file=sys.stderr)

            napari.run()
        except Exception as e:
            print("Napari overlay failed:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
