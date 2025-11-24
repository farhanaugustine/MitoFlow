#!/usr/bin/env python3
"""
napari_id_viewer.py - Interactive Napari viewer for mitochondrial tracking outputs.

Features
--------
- Loads raw image stacks, segmentation labels, and objects/edges tables.
- Colors centroids by track or label id and overlays text IDs.
- Draws continuation/fission/fusion edges plus event nodes when edges_all.csv is present.
- Provides in-view controls (point size, text size, edge width, toggles) via magicgui.
- Auto-discovers standard mito_toolkit outputs when given an output folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops_table

import napari
from magicgui.widgets import ComboBox, CheckBox, Container, FileEdit, FloatSpinBox, Label, PushButton


@dataclass
class _State:
    viewer: napari.Viewer
    points_layer: Optional[napari.layers.Points] = None
    labels_layer: Optional[napari.layers.Labels] = None
    edge_layers: Dict[str, Optional[napari.layers.Shapes]] = field(default_factory=dict)
    node_layers: Dict[str, Optional[napari.layers.Points]] = field(default_factory=dict)
    props: Dict[str, List] = field(default_factory=dict)
    lookup: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    color_key: str = "track_id"
    cmap: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    objects_path: Optional[str] = None
    edges_path: Optional[str] = None
    edges_df: Optional[pd.DataFrame] = None
    text_size: float = 14.0
    event_size: float = 6.0
    edge_width: float = 2.0
    show_cont_edges: bool = True
    show_fission_edges: bool = True
    show_fusion_edges: bool = True
    show_fission_nodes: bool = True
    show_fusion_nodes: bool = True
    local_filter_enabled: bool = False
    local_filter_radius_um: float = 0.0
    local_filter_time_window: int = 0
    edge_time_window: int = 0
    filter_focus_edges: bool = False
    verified_path: Optional[str] = None
    events_df: Optional[pd.DataFrame] = None
    event_rows: Optional[pd.DataFrame] = None
    current_event_idx: Optional[int] = None
    show_only_event: bool = False
    col_t: str = "t"
    col_id: str = "label_id"
    col_track: str = "track_id"
    col_y: str = "centroid_y_um"
    col_x: str = "centroid_x_um"
    col_z: str = "centroid_z_um"
    voxel_size_um: Optional[Tuple[float, float, float]] = None
    show_ids_default: bool = True
    objects_df: Optional[pd.DataFrame] = None
    points_table: Optional[pd.DataFrame] = None
    focus_points_layer: Optional[napari.layers.Points] = None
    focus_edges_layer: Optional[napari.layers.Shapes] = None
    track_ids: List[int] = field(default_factory=list)
    label_ids: List[int] = field(default_factory=list)
    audit_history: List[int] = field(default_factory=list)


def to_tzyx(arr: np.ndarray, axis: str) -> np.ndarray:
    axis = axis.lower()
    allowed = set("tzyxc")
    if not set(axis) <= allowed:
        raise ValueError(f"Invalid axis spec '{axis}'. Use characters t, z, y, x, c.")
    if axis.count("t") > 1 or axis.count("c") > 1:
        raise ValueError(f"Axis spec '{axis}' may contain at most one 't' and one 'c'.")
    for ch in "zyx":
        if ch not in axis:
            raise ValueError(f"Axis spec '{axis}' must include z, y, and x.")
    data = arr
    if "c" in axis:
        c_pos = axis.index("c")
        if c_pos >= data.ndim:
            raise ValueError(f"Axis spec '{axis}' expects a channel axis, data shape is {data.shape}.")
        data = np.take(data, indices=0, axis=c_pos)
        axis = axis.replace("c", "")
    if "t" not in axis:
        data = np.expand_dims(data, axis=0)
        axis = "t" + axis
    if len(axis) != data.ndim:
        raise ValueError(f"Axis spec '{axis}' expects {len(axis)} dims, data shape is {data.shape}.")
    order = [axis.index(ch) for ch in "tzyx"]
    return np.transpose(data, order)


def discover_paths(root: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    image = labels = objects = edges = None
    try:
        entries = list(os.scandir(root))
    except FileNotFoundError:
        return image, labels, objects, edges
    for entry in entries:
        name = entry.name.lower()
        if entry.is_file():
            if name.endswith((".tif", ".tiff")):
                if "label" in name:
                    labels = labels or entry.path
                else:
                    image = image or entry.path
            elif name.endswith(".csv"):
                if "objects" in name or "detections" in name:
                    objects = objects or entry.path
                elif "edges_all" in name:
                    edges = entry.path
                elif "edges" in name and "events" not in name and edges is None:
                    edges = entry.path
    return image, labels, objects, edges


def distinct_color_map(ids: Iterable[int]) -> Dict[int, Tuple[float, float, float, float]]:
    ids = [int(i) for i in ids if i is not None and not (isinstance(i, float) and np.isnan(i))]
    unique_ids = sorted(set(ids))
    rng = np.random.default_rng(42)
    rng.shuffle(unique_ids)
    colors: Dict[int, Tuple[float, float, float, float]] = {-1: (0.7, 0.7, 0.7, 1.0)}
    golden = 0.61803398875
    for k, value in enumerate(unique_ids):
        h = (k * golden) % 1.0
        s = 0.65
        v = 0.95
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors[int(value)] = (r, g, b, 1.0)
    return colors


def _get_layer(viewer: napari.Viewer, name: str) -> Optional[napari.layers.Layer]:
    try:
        for layer in viewer.layers:
            if layer.name == name:
                return layer
    except Exception:
        pass
    return None


def _maybe_to_pixels(
    value: Optional[float],
    axis_index: int,
    column_name: Optional[str],
    voxel_size_um: Optional[Tuple[float, float, float]],
) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    val = float(value)
    if voxel_size_um is None:
        return val
    name = (column_name or "").lower()
    if "um" in name or "micron" in name:
        scale = voxel_size_um[axis_index]
        if (not np.isfinite(scale)) or (scale <= 0):
            return float("nan")
        return val / scale
    return val


def build_points_from_objects(
    df: pd.DataFrame,
    col_t: str,
    col_id: str,
    col_track: str,
    col_y: str,
    col_x: str,
    col_z: Optional[str],
    voxel_size_um: Optional[Tuple[float, float, float]],
) -> Tuple[np.ndarray, Dict[str, List], pd.DataFrame]:
    required_cols = [col_t, col_y, col_x]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in objects table.")
    props: Dict[str, List] = {"label_id": [], "track_id": []}
    coords: List[List[float]] = []
    records: List[Dict[str, float]] = []
    dropped = 0
    z_vox_col = "centroid_z_vox" if "centroid_z_vox" in df.columns else None
    y_vox_col = "centroid_y_vox" if "centroid_y_vox" in df.columns else None
    x_vox_col = "centroid_x_vox" if "centroid_x_vox" in df.columns else None
    for _, row in df.iterrows():
        t = int(row[col_t])
        y = _maybe_to_pixels(row[col_y], 1, col_y, voxel_size_um)
        if not np.isfinite(y) and y_vox_col:
            alt = row.get(y_vox_col)
            y = float(alt) if pd.notna(alt) else float("nan")
        x = _maybe_to_pixels(row[col_x], 2, col_x, voxel_size_um)
        if not np.isfinite(x) and x_vox_col:
            alt = row.get(x_vox_col)
            x = float(alt) if pd.notna(alt) else float("nan")
        z = float("nan")
        if col_z and col_z in df.columns and not pd.isna(row[col_z]):
            z = _maybe_to_pixels(row[col_z], 0, col_z, voxel_size_um)
        if (not np.isfinite(z)) and z_vox_col:
            alt = row.get(z_vox_col)
            if pd.notna(alt):
                z = float(alt)
        if not np.isfinite(z):
            z = 0.0
        if not np.all(np.isfinite([z, y, x])):
            dropped += 1
            continue
        coords.append([t, z, y, x])
        label_val = None
        if col_id in df.columns and not pd.isna(row[col_id]):
            try:
                label_val = int(row[col_id])
            except Exception:
                label_val = int(float(row[col_id]))
        track_val = None
        if col_track in df.columns and not pd.isna(row[col_track]):
            try:
                track_val = int(row[col_track])
            except Exception:
                track_val = int(float(row[col_track]))
        props["label_id"].append(label_val)
        props["track_id"].append(track_val)
        records.append({"t": t, "label_id": label_val, "track_id": track_val, "z": z, "y": y, "x": x})
    if dropped:
        print(f"Note: skipped {dropped} detections with invalid coordinates in objects table.", file=sys.stderr)
    table = pd.DataFrame(records)
    points = np.asarray(coords, dtype=float) if coords else np.empty((0, 4), dtype=float)
    return points, props, table


def build_points_from_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, List], pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    coords: List[List[float]] = []
    props: Dict[str, List] = {"label_id": [], "track_id": []}
    t_frames = labels.shape[0]
    dropped = 0
    for t in range(t_frames):
        lbl = labels[t]
        if lbl.ndim == 3:
            if lbl.max() == 0:
                continue
            props_tbl = regionprops_table(lbl.astype(np.int32), properties=("label", "centroid"))
            labels_found = props_tbl.get("label", [])
            cz = props_tbl.get("centroid-0", [])
            cy = props_tbl.get("centroid-1", [])
            cx = props_tbl.get("centroid-2", [])
            for lab_id, z, y, x in zip(labels_found, cz, cy, cx):
                if not np.all(np.isfinite([z, y, x])):
                    dropped += 1
                    continue
                coords.append([t, float(z), float(y), float(x)])
                props["label_id"].append(int(lab_id))
                props["track_id"].append(None)
                rows.append({"t": t, "label_id": int(lab_id), "track_id": None, "z": float(z), "y": float(y), "x": float(x)})
        else:
            if lbl.max() == 0:
                continue
            props_tbl = regionprops_table(lbl.astype(np.int32), properties=("label", "centroid"))
            labels_found = props_tbl.get("label", [])
            cy = props_tbl.get("centroid-0", [])
            cx = props_tbl.get("centroid-1", [])
            for lab_id, y, x in zip(labels_found, cy, cx):
                if not np.all(np.isfinite([y, x])):
                    dropped += 1
                    continue
                coords.append([t, 0.0, float(y), float(x)])
                props["label_id"].append(int(lab_id))
                props["track_id"].append(None)
                rows.append({"t": t, "label_id": int(lab_id), "track_id": None, "z": 0.0, "y": float(y), "x": float(x)})
    if dropped:
        print(f"Note: skipped {dropped} centroid(s) with invalid coordinates from labels stack.", file=sys.stderr)
    table = pd.DataFrame(rows)
    points = np.asarray(coords, dtype=float) if coords else np.empty((0, 4), dtype=float)
    return points, props, table


def build_edge_geometry(
    edges_df: Optional[pd.DataFrame], lookup: Dict[Tuple[int, int], np.ndarray], focus_track: Optional[int] = None
) -> Dict[str, List[np.ndarray]]:
    result = {
        "continuations": [],
        "gap": [],
        "fission": [],
        "fusion": [],
        "fission_nodes": [],
        "fusion_nodes": [],
        "fission_idx": [],
        "fusion_idx": [],
    }
    skipped = {"continuations": 0, "fission": 0, "fusion": 0}
    if edges_df is None or lookup is None:
        result["_skipped"] = skipped
        return result
    for row in edges_df.itertuples():
        try:
            t_from = int(getattr(row, "t_from"))
            t_to = int(getattr(row, "t_to"))
            parent = int(getattr(row, "parent_label"))
            child = int(getattr(row, "child_label"))
        except Exception:
            continue
        start = lookup.get((t_from, parent))
        end = lookup.get((t_to, child))
        if start is None or end is None:
            edge_type = str(getattr(row, "edge_type", "continuation")).lower()
            if edge_type == "fission":
                skipped["fission"] += 1
            elif edge_type == "fusion":
                skipped["fusion"] += 1
            else:
                skipped["continuations"] += 1
            continue
        segment = np.vstack([start, end]).astype(float)
        edge_type = str(getattr(row, "edge_type", "continuation")).lower()
        if focus_track is not None:
            ptid = getattr(row, "parent_track_id", None)
            ctid = getattr(row, "child_track_id", None)
            if ptid is None and ctid is None:
                pass
            else:
                keep = (ptid == focus_track) or (ctid == focus_track)
                if not keep:
                    continue
        idx_val = getattr(row, "Index", None)
        if edge_type == "fission":
            result["fission"].append(segment)
            result["fission_nodes"].append(end.copy())
            result["fission_idx"].append(idx_val if idx_val is not None else len(result["fission_idx"]))
        elif edge_type == "fusion":
            result["fusion"].append(segment)
            result["fusion_nodes"].append(end.copy())
            result["fusion_idx"].append(idx_val if idx_val is not None else len(result["fusion_idx"]))
        elif edge_type == "gap":
            result["gap"].append(segment)
        else:
            result["continuations"].append(segment)
    result["_skipped"] = skipped
    return result


def _apply_text(
    points_layer: Optional[napari.layers.Points],
    props: Dict[str, List],
    key: str,
    show: bool,
    text_size: float,
) -> None:
    if points_layer is None:
        return
    if not show:
        points_layer.text = None
        return
    labels = props.get("label_id", [])
    tracks = props.get("track_id", [])
    strings = []
    for lid, tid in zip(labels, tracks):
        if key == "track_id" and tid is not None:
            if lid is not None:
                strings.append(f"{int(tid)}:{int(lid)}")
            else:
                strings.append(f"{int(tid)}")
        else:
            strings.append(f"{int(lid)}" if lid is not None else "?")
    payload = {
        "string": list(strings),
        "anchor": "upper_left",
        "translation": (0.0, 0.0),
        "size": float(max(4.0, text_size)),
        "color": "white",
    }
    try:
        points_layer.text = payload
    except Exception:
        try:
            points_layer.text = list(strings)
        except Exception:
            points_layer.text = None


def _recolor_layers(state: _State) -> None:
    if state.points_layer is not None:
        colors = []
        default = state.cmap.get(-1, (0.7, 0.7, 0.7, 1.0))
        for lid, tid in zip(state.props.get("label_id", []), state.props.get("track_id", [])):
            if state.color_key == "track_id" and tid is not None:
                colors.append(state.cmap.get(int(tid), default))
            elif lid is not None:
                colors.append(state.cmap.get(int(lid), default))
            else:
                colors.append(default)
        if colors:
            state.points_layer.face_color = np.asarray(colors, dtype=float)


def _current_focus(state: _State) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        steps = state.viewer.dims.current_step
        if len(steps) >= 4:
            return np.array([float(steps[1]), float(steps[2]), float(steps[3])], dtype=float), int(steps[0])
    except Exception:
        pass
    return None, None

def _filter_geometry_local(geometry: Dict[str, List[np.ndarray]], state: _State) -> Dict[str, List[np.ndarray]]:
    focus_xyz, focus_t = _current_focus(state)
    spacing = np.asarray(state.voxel_size_um if state.voxel_size_um else (1.0, 1.0, 1.0), dtype=float)
    radius_um = float(state.local_filter_radius_um)
    dt_local = max(0, int(state.local_filter_time_window))
    dt_global = max(0, int(state.edge_time_window))
    use_space = bool(state.local_filter_enabled and radius_um > 0 and focus_xyz is not None)
    use_time = (dt_local > 0 or dt_global > 0) and focus_t is not None
    focus_um = focus_xyz * spacing if focus_xyz is not None else None

    def _segment_visible(seg: np.ndarray) -> bool:
        if seg.shape[0] < 2 or seg.shape[1] < 4:
            return False
        t0, t1 = seg[0, 0], seg[1, 0]
        if use_time:
            dt_eff = dt_local if state.local_filter_enabled else dt_global
            if min(abs(t0 - focus_t), abs(t1 - focus_t)) > dt_eff:
                return False
        if not use_space:
            return True
        pts = [seg[0][1:], seg[1][1:]]
        for pt in pts:
            dist = np.linalg.norm(np.asarray(pt, dtype=float) * spacing - focus_um)
            if dist <= radius_um:
                return True
        return False

    def _node_visible(pt: np.ndarray) -> bool:
        if pt.shape[0] < 4:
            return False
        if use_time:
            dt_eff = dt_local if state.local_filter_enabled else dt_global
            if abs(pt[0] - focus_t) > dt_eff:
                return False
        if not use_space:
            return True
        dist = np.linalg.norm(np.asarray(pt[1:], dtype=float) * spacing - focus_um)
        return dist <= radius_um

    filtered: Dict[str, List[np.ndarray]] = {k: [] for k in geometry.keys()}
    for key in ("continuations", "gap", "fission", "fusion"):
        for seg in geometry.get(key, []):
            if _segment_visible(seg):
                filtered[key].append(seg)
    for key in ("fission_nodes", "fusion_nodes"):
        for pt in geometry.get(key, []):
            arr = np.asarray(pt, dtype=float)
            if _node_visible(arr):
                filtered[key].append(arr)
    filtered["_skipped"] = geometry.get("_skipped", {})
    filtered["fission_idx"] = geometry.get("fission_idx", [])
    filtered["fusion_idx"] = geometry.get("fusion_idx", [])
    return filtered


def _refresh_edges(state: _State) -> None:
    if state.edges_df is None or state.edges_df.empty:
        return
    focus_tid = None
    if state.filter_focus_edges and state.audit_history:
        sel = state.audit_history[-1]
        focus_tid = sel
    df_use = state.edges_df
    if state.show_only_event and state.current_event_idx is not None and state.event_rows is not None and not state.event_rows.empty:
        if 0 <= state.current_event_idx < len(state.event_rows):
            ev = state.event_rows.iloc[state.current_event_idx]
            mask_event = (
                (df_use["edge_type"] == ev["edge_type"]) &
                (df_use["t_from"] == ev["t_from"]) &
                (df_use["t_to"] == ev["t_to"]) &
                (df_use["parent_label"] == ev["parent_label"]) &
                (df_use["child_label"] == ev["child_label"])
            )
            df_ev = df_use[mask_event]
            tracks_keep = set()
            try:
                tracks_keep.add(int(ev.get("parent_track_id", -1)))
                tracks_keep.add(int(ev.get("child_track_id", -1)))
            except Exception:
                pass
            df_cont = df_use[
                (df_use["edge_type"].isin(["continuation","gap"])) &
                (df_use["parent_track_id"].isin(tracks_keep) | df_use["child_track_id"].isin(tracks_keep))
            ]
            df_use = pd.concat([df_ev, df_cont], ignore_index=False)
    geometry = build_edge_geometry(df_use, state.lookup, focus_track=focus_tid)
    geometry = _filter_geometry_local(geometry, state)
    edge_specs = {
        "continuations": ("Continuation edges", "white", state.show_cont_edges),
        "gap": ("Gap edges", "cyan", state.show_cont_edges),
        "fission": ("Fission edges", "magenta", state.show_fission_edges),
        "fusion": ("Fusion edges", "orange", state.show_fusion_edges),
    }
    node_specs = {
        "fission_nodes": ("Fission nodes", "magenta", state.show_fission_nodes),
        "fusion_nodes": ("Fusion nodes", "orange", state.show_fusion_nodes),
    }

    def _node_in_labels(pt: np.ndarray) -> bool:
        if state.labels_layer is None:
            return True
        data = getattr(state.labels_layer, "data", None)
        if data is None:
            return True
        try:
            t = int(round(pt[0]))
            z = int(round(pt[1]))
            y = int(round(pt[2]))
            x = int(round(pt[3]))
        except Exception:
            return True
        try:
            if data.ndim == 5:  # (T, C, Z, Y, X)
                if t < 0 or t >= data.shape[0]:
                    return False
                z_idx = np.clip(z, 0, data.shape[2]-1)
                y_idx = np.clip(y, 0, data.shape[3]-1)
                x_idx = np.clip(x, 0, data.shape[4]-1)
                return data[t, 0, z_idx, y_idx, x_idx] != 0
            elif data.ndim == 4:  # (T, Z, Y, X)
                if t < 0 or t >= data.shape[0]:
                    return False
                z_idx = np.clip(z, 0, data.shape[1]-1)
                y_idx = np.clip(y, 0, data.shape[2]-1)
                x_idx = np.clip(x, 0, data.shape[3]-1)
                return data[t, z_idx, y_idx, x_idx] != 0
            elif data.ndim == 3:  # (Z, Y, X) no time axis
                z_idx = np.clip(z, 0, data.shape[0]-1)
                y_idx = np.clip(y, 0, data.shape[1]-1)
                x_idx = np.clip(x, 0, data.shape[2]-1)
                return data[z_idx, y_idx, x_idx] != 0
        except Exception:
            return True
        return True

    for key, (title, color, visible) in edge_specs.items():
        data = geometry.get(key, [])
        layer = state.edge_layers.get(key)
        if data:
            if layer is None:
                layer = state.viewer.add_shapes(
                    data,
                    shape_type="path",
                    name=title,
                    edge_color=color,
                    edge_width=max(state.edge_width, 1.5 if key in ("fission","fusion") else state.edge_width),
                    blending="additive",
                )
                state.edge_layers[key] = layer
            else:
                layer.data = data
                layer.edge_color = color
            layer.edge_width = max(state.edge_width, 1.5 if key in ("fission","fusion") else state.edge_width)
            layer.visible = bool(visible)
            try:
                state.viewer.layers.move(state.viewer.layers.index(layer), len(state.viewer.layers) - 1)
            except Exception:
                pass
        elif layer is not None:
            layer.data = []
            layer.visible = False
    state._geom_indices = {"fission_idx": geometry.get("fission_idx", []), "fusion_idx": geometry.get("fusion_idx", [])}
    for key, (title, color, visible) in node_specs.items():
        pts_raw = geometry.get(key, [])
        points = [pt for pt in pts_raw if _node_in_labels(np.asarray(pt, dtype=float))]
        layer = state.node_layers.get(key)
        if points:
            arr = np.asarray(points, dtype=float)
            texts = None
            idx_list = state._geom_indices.get("fission_idx" if key == "fission_nodes" else "fusion_idx", [])
            if idx_list and state.edges_df is not None:
                strings = []
                for idx_val in idx_list[:len(arr)]:
                    try:
                        row = state.edges_df.iloc[idx_val]
                        strings.append(f"{row['edge_type']} p{int(row['parent_label'])}->c{int(row['child_label'])}")
                    except Exception:
                        strings.append("")
                texts = {"string": strings, "size": 10, "color": "white", "anchor": "upper_left"}
            bright_color = "yellow" if key == "fission_nodes" else "cyan"
            if layer is None:
                layer = state.viewer.add_points(
                    arr,
                    name=title,
                    ndim=4,
                    size=max(state.event_size, 9.0),
                    face_color=bright_color,
                    edge_color="black",
                    edge_width=0.8,
                )
                state.node_layers[key] = layer
            else:
                layer.data = arr
            if texts:
                try:
                    layer.text = texts
                except Exception:
                    pass
            layer.face_color = bright_color
            layer.edge_color = "black"
            layer.edge_width = 0.8
            layer.size = max(state.event_size, 9.0)
            layer.visible = bool(visible)
            try:
                state.viewer.layers.move(state.viewer.layers.index(layer), len(state.viewer.layers) - 1)
            except Exception:
                pass
        elif layer is not None:
            layer.data = np.empty((0, 4), dtype=float)
            layer.visible = False

    skipped = geometry.get("_skipped", {})
    if isinstance(skipped, dict) and any(val > 0 for val in skipped.values()):
        print(
            f"Edges dropped (continuation={skipped.get('continuations', 0)}, fission={skipped.get('fission', 0)}, fusion={skipped.get('fusion', 0)}) due to missing coordinates.",
            file=sys.stderr,
        )


def _build_dock(state: _State) -> Container:
    viewer = state.viewer
    header = Label(value="<b>Mito controls</b>")
    objects_edit = FileEdit(mode="r", value=state.objects_path, filter="*.csv", label="Objects CSV")
    edges_edit = FileEdit(mode="r", value=state.edges_path, filter="*.csv", label="Edges CSV")
    reload_btn = PushButton(text="Reload files")

    show_ids = CheckBox(text="Show IDs", value=bool(state.show_ids_default and state.points_layer is not None))
    color_by = ComboBox(choices=["track", "label"], value="track" if state.color_key == "track_id" else "label")

    point_size_value = 6.0
    if state.points_layer is not None:
        size = state.points_layer.size
        if np.isscalar(size):
            point_size_value = float(size)
        elif len(size) > 0:
            point_size_value = float(np.mean(size))
    point_size = FloatSpinBox(value=point_size_value, step=0.5, min=1.0, max=40.0, label="Point size")
    text_size = FloatSpinBox(value=state.text_size, step=1.0, min=6.0, max=64.0, label="Text size")
    edge_width = FloatSpinBox(value=state.edge_width, step=0.5, min=0.5, max=10.0, label="Edge width")
    event_size = FloatSpinBox(value=state.event_size, step=0.5, min=1.0, max=20.0, label="Event size")
    edge_time_window = FloatSpinBox(value=float(state.edge_time_window), step=1.0, min=0.0, max=50.0, label="Edge time window (frames)")
    focus_edges_only = CheckBox(text="Show focused track edges", value=state.filter_focus_edges)
    btn_verify_mode = PushButton(text="Verify view")
    btn_context_mode = PushButton(text="Context view")
    btn_reset_filters = PushButton(text="Reset filters")

    show_cont = CheckBox(text="Show continuations", value=state.show_cont_edges)
    show_fission_edges = CheckBox(text="Show fission edges", value=state.show_fission_edges)
    show_fusion_edges = CheckBox(text="Show fusion edges", value=state.show_fusion_edges)
    show_fission_nodes = CheckBox(text="Show fission nodes", value=state.show_fission_nodes)
    show_fusion_nodes = CheckBox(text="Show fusion nodes", value=state.show_fusion_nodes)
    local_filter = CheckBox(text="Local edge filter", value=state.local_filter_enabled)
    local_radius = FloatSpinBox(
        value=state.local_filter_radius_um if state.local_filter_radius_um > 0 else 8.0,
        step=1.0,
        min=0.0,
        max=250.0,
        label="Local radius (um)",
    )
    local_dt = FloatSpinBox(
        value=state.local_filter_time_window,
        step=1.0,
        min=0.0,
        max=20.0,
        label="Local Δt (frames)",
    )
    state.local_filter_radius_um = float(local_radius.value)
    try:
        state.local_filter_time_window = int(float(local_dt.value))
    except Exception:
        state.local_filter_time_window = 0

    info = Label(value="IDs use labels.tif and objects.csv; edge overlays need edges_all.csv.")

    audit_header = Label(value="<b>Audit tracks</b>")
    if state.track_ids:
        track_choices = ["(all)"] + [str(t) for t in state.track_ids]
        track_default = "(all)"
    else:
        track_choices = ["(none)"]
        track_default = "(none)"
    track_combo = ComboBox(choices=track_choices, value=track_default, label="Track focus")
    focus_btn = PushButton(text="Highlight")
    clear_focus_btn = PushButton(text="Clear")
    prev_track_btn = PushButton(text="◀ Prev")
    next_track_btn = PushButton(text="Next ▶")
    audit_action_row = Container(widgets=[focus_btn, clear_focus_btn], layout="horizontal")
    audit_nav_row = Container(widgets=[prev_track_btn, next_track_btn], layout="horizontal")
    track_info = Label(value="Track info: –")

    review_header = Label(value="<b>Event review</b>")
    btn_verify = PushButton(text="Mark Verified")
    btn_question = PushButton(text="Mark Questionable")
    btn_reject = PushButton(text="Mark Reject")
    review_row = Container(widgets=[btn_verify, btn_question, btn_reject], layout="horizontal")
    event_info = Label(value="Event: –")
    btn_prev_event = PushButton(text="◀ Prev event")
    btn_next_event = PushButton(text="Next event ▶")
    toggle_only_event = CheckBox(text="Show only current event", value=False)
    goto_label = Label(value="Go to (t, label):")
    goto_t = FloatSpinBox(value=0, step=1, min=0, max=1e6)
    goto_id = FloatSpinBox(value=0, step=1, min=0, max=1e6)
    btn_goto = PushButton(text="Go")
    goto_row = Container(widgets=[goto_label, goto_t, goto_id, btn_goto], layout="horizontal")

    def _refresh_track_choices() -> None:
        ids = [str(t) for t in state.track_ids if t > 0]
        if ids:
            choices = ["(all)"] + ids
        else:
            choices = ["(none)"]
        track_combo.choices = choices
        if track_combo.value not in choices:
            track_combo.value = choices[0]

    def _get_selected_track() -> Optional[int]:
        val = track_combo.value
        if val in ("(all)", "(none)", None):
            return None
        try:
            return int(val)
        except Exception:
            return None

    def _clear_focus(reset_combo: bool = False) -> None:
        if state.focus_points_layer is not None and state.focus_points_layer in viewer.layers:
            state.focus_points_layer.data = np.empty((0, 4), dtype=float)
            state.focus_points_layer.visible = False
        if state.focus_edges_layer is not None and state.focus_edges_layer in viewer.layers:
            state.focus_edges_layer.data = []
            state.focus_edges_layer.visible = False
        track_info.value = "Track info: –"
        if reset_combo:
            if "(all)" in track_combo.choices:
                track_combo.value = "(all)"
            elif track_combo.choices:
                track_combo.value = track_combo.choices[0]

    def _set_focus(track_id: int) -> None:
        if state.points_table is None or state.points_table.empty:
            track_info.value = "Track info: –"
            return
        df_sel = state.points_table[state.points_table["track_id"] == track_id]
        if df_sel.empty:
            track_info.value = f"Track {track_id}: not found"
            return

        pts = df_sel[["t", "z", "y", "x"]].to_numpy(float)
        highlight_size = max(float(point_size.value) * 1.4, 2.0)

        if state.focus_points_layer is None or state.focus_points_layer not in viewer.layers:
            state.focus_points_layer = viewer.add_points(
                pts,
                name="track focus",
                ndim=4,
                size=highlight_size,
                face_color="yellow",
                edge_color="black",
                edge_width=0.2,
                opacity=0.9,
            )
        else:
            state.focus_points_layer.data = pts
            state.focus_points_layer.visible = True
            state.focus_points_layer.size = highlight_size
            state.focus_points_layer.face_color = "yellow"
        try:
            state.focus_points_layer.symbol = "ring"
        except Exception:
            pass

        segs: List[np.ndarray] = []
        if state.edges_df is not None and not state.edges_df.empty:
            parent_col = state.edges_df.get("parent_track_id")
            child_col = state.edges_df.get("child_track_id")
            if parent_col is not None and child_col is not None:
                mask = (parent_col == track_id) | (child_col == track_id)
            elif parent_col is not None:
                mask = parent_col == track_id
            else:
                mask = child_col == track_id if child_col is not None else None
            if mask is not None:
                subset = state.edges_df[mask]
                for _, row in subset.iterrows():
                    start = state.lookup.get((int(row["t_from"]), int(row["parent_label"])))
                    end = state.lookup.get((int(row["t_to"]), int(row["child_label"])))
                    if start is None or end is None:
                        continue
                    segs.append(np.vstack([start, end]).astype(float))
        if segs:
            edge_width_val = max(state.edge_width * 1.5, 1.0)
            if state.focus_edges_layer is None or state.focus_edges_layer not in viewer.layers:
                state.focus_edges_layer = viewer.add_shapes(
                    segs,
                    shape_type="path",
                    name="track edges",
                    edge_color="yellow",
                    edge_width=edge_width_val,
                    blending="translucent",
                )
            else:
                state.focus_edges_layer.data = segs
                state.focus_edges_layer.visible = True
                state.focus_edges_layer.edge_color = "yellow"
                state.focus_edges_layer.edge_width = edge_width_val
        elif state.focus_edges_layer is not None and state.focus_edges_layer in viewer.layers:
            state.focus_edges_layer.data = []
            state.focus_edges_layer.visible = False

        try:
            first = df_sel.iloc[0]
            viewer.dims.set_current_step(0, int(first["t"]))
            viewer.dims.set_current_step(1, int(round(first["z"])))
        except Exception:
            pass

        if state.objects_df is not None and state.col_track in state.objects_df.columns:
            obj_sel = state.objects_df[state.objects_df[state.col_track] == track_id]
            if not obj_sel.empty:
                frames = sorted(int(v) for v in obj_sel[state.col_t].unique())
                vol_col = "volume_um3" if "volume_um3" in obj_sel.columns else ("volume_vox" if "volume_vox" in obj_sel.columns else None)
                if vol_col:
                    volumes = obj_sel[vol_col].astype(float).to_numpy()
                    median_vol = float(np.median(volumes)) if volumes.size else float("nan")
                    track_info.value = f"Track {track_id}: detections {len(obj_sel)}, frames {frames[0]}–{frames[-1]}, median volume {median_vol:.1f}"
                else:
                    track_info.value = f"Track {track_id}: detections {len(obj_sel)}, frames {frames[0]}–{frames[-1]}"
            else:
                track_info.value = f"Track {track_id}: detections {len(df_sel)}"
        else:
            track_info.value = f"Track {track_id}: detections {len(df_sel)}"

        if track_id not in state.audit_history:
            state.audit_history.append(track_id)
        _refresh_edges(state)

    def _update_event_info():
        if state.event_rows is None or state.current_event_idx is None or state.current_event_idx < 0 or state.current_event_idx >= len(state.event_rows):
            event_info.value = "Event: –"
            return
        row = state.event_rows.iloc[state.current_event_idx]
        event_info.value = f"{row['edge_type']} t{int(row['t_from'])}->{int(row['t_to'])} p{int(row['parent_label'])}→c{int(row['child_label'])}"

    def _goto_event(delta: int):
        if state.event_rows is None or state.event_rows.empty:
            return
        idx = state.current_event_idx if state.current_event_idx is not None else 0
        idx = (idx + delta) % len(state.event_rows)
        state.current_event_idx = idx
        row = state.event_rows.iloc[idx]
        try:
            state._last_selected_event_idx = int(row["_edge_index"])
        except Exception:
            state._last_selected_event_idx = None
        _update_event_info()
        try:
            viewer.dims.set_current_step(0, int(row["t_to"]))
            if "centroid_z_um" in state.objects_df.columns and "centroid_z_vox" in state.objects_df.columns:
                pass
        except Exception:
            pass
        _refresh_edges(state)

    @btn_prev_event.changed.connect
    def _on_prev_event(event=None):
        _goto_event(-1)

    @btn_next_event.changed.connect
    def _on_next_event(event=None):
        _goto_event(1)

    @toggle_only_event.changed.connect
    def _on_toggle_only_event(event=None):
        state.show_only_event = bool(toggle_only_event.value)
        _refresh_edges(state)

    @btn_goto.changed.connect
    def _on_goto(event=None):
        try:
            t_val = int(goto_t.value)
            lbl_val = int(goto_id.value)
        except Exception:
            return
        coord = state.lookup.get((t_val, lbl_val))
        if coord is None:
            return
        try:
            viewer.dims.set_current_step(0, int(coord[0]))
            viewer.dims.set_current_step(1, int(round(coord[1])))
            viewer.dims.set_current_step(2, int(round(coord[2])))
            viewer.dims.set_current_step(3, int(round(coord[3])))
        except Exception:
            pass
        # highlight object if track_id known
        if state.points_layer is not None and state.points_table is not None:
            mask = (state.points_table["t"] == t_val) & (state.points_table["label_id"] == lbl_val)
            idxs = list(state.points_table[mask].index)
            if idxs:
                try:
                    state.points_layer.selected_data = {int(idxs[0])}
                except Exception:
                    pass

    def _highlight_current(event=None) -> None:
        tid = _get_selected_track()
        if tid is None:
            _clear_focus()
        else:
            _set_focus(tid)

    def _navigate_track(delta: int) -> None:
        if not state.track_ids:
            return
        current = _get_selected_track()
        if current is None or current not in state.track_ids:
            idx = 0 if delta >= 0 else len(state.track_ids) - 1
        else:
            idx = state.track_ids.index(current)
            idx = (idx + delta) % len(state.track_ids)
        track_combo.value = str(state.track_ids[idx])
        _set_focus(state.track_ids[idx])

    @focus_btn.changed.connect
    def _on_focus(event=None):
        _highlight_current()

    @clear_focus_btn.changed.connect
    def _on_clear(event=None):
        _clear_focus(reset_combo=True)

    @prev_track_btn.changed.connect
    def _on_prev(event=None):
        _navigate_track(-1)

    @next_track_btn.changed.connect
    def _on_next(event=None):
        _navigate_track(1)

    _refresh_track_choices()

    def _update_color_map(target: str) -> None:
        ids = state.props.get(target, [])
        state.cmap = distinct_color_map([i if i is not None else -1 for i in ids])
        _recolor_layers(state)
        _apply_text(state.points_layer, state.props, state.color_key, show_ids.value, text_size.value)

    @show_ids.changed.connect
    def _on_show_ids(event=None):
        _apply_text(state.points_layer, state.props, state.color_key, show_ids.value, text_size.value)

    @color_by.changed.connect
    def _on_color_by(event=None):
        state.color_key = "track_id" if color_by.value == "track" else "label_id"
        _update_color_map("track_id" if color_by.value == "track" else "label_id")
        _refresh_edges(state)

    @point_size.changed.connect
    def _on_point_size(event=None):
        if state.points_layer is not None:
            state.points_layer.size = float(point_size.value)
        _apply_text(state.points_layer, state.props, state.color_key, show_ids.value, text_size.value)

    @text_size.changed.connect
    def _on_text_size(event=None):
        state.text_size = float(text_size.value)
        _apply_text(state.points_layer, state.props, state.color_key, show_ids.value, text_size.value)

    @edge_width.changed.connect
    def _on_edge_width(event=None):
        state.edge_width = float(edge_width.value)
        for layer in state.edge_layers.values():
            if layer is not None:
                layer.edge_width = state.edge_width

    @event_size.changed.connect
    def _on_event_size(event=None):
        state.event_size = float(event_size.value)
        for layer in state.node_layers.values():
            if layer is not None:
                layer.size = state.event_size
        _refresh_edges(state)

    @show_cont.changed.connect
    def _toggle_cont(event=None):
        state.show_cont_edges = bool(show_cont.value)
        layer = state.edge_layers.get("continuations")
        if layer is not None:
            layer.visible = state.show_cont_edges
        layer = state.edge_layers.get("gap")
        if layer is not None:
            layer.visible = state.show_cont_edges

    @show_fission_edges.changed.connect
    def _toggle_fission_edges(event=None):
        state.show_fission_edges = bool(show_fission_edges.value)
        layer = state.edge_layers.get("fission")
        if layer is not None:
            layer.visible = state.show_fission_edges

    @show_fusion_edges.changed.connect
    def _toggle_fusion_edges(event=None):
        state.show_fusion_edges = bool(show_fusion_edges.value)
        layer = state.edge_layers.get("fusion")
        if layer is not None:
            layer.visible = state.show_fusion_edges

    @show_fission_nodes.changed.connect
    def _toggle_fission_nodes(event=None):
        state.show_fission_nodes = bool(show_fission_nodes.value)
        layer = state.node_layers.get("fission_nodes")
        if layer is not None:
            layer.visible = state.show_fission_nodes

    @show_fusion_nodes.changed.connect
    def _toggle_fusion_nodes(event=None):
        state.show_fusion_nodes = bool(show_fusion_nodes.value)
        layer = state.node_layers.get("fusion_nodes")
        if layer is not None:
            layer.visible = state.show_fusion_nodes

    @edge_time_window.changed.connect
    def _on_edge_time_window(event=None):
        state.edge_time_window = int(edge_time_window.value)
        _refresh_edges(state)

    @focus_edges_only.changed.connect
    def _on_focus_edges_only(event=None):
        state.filter_focus_edges = bool(focus_edges_only.value)
        _refresh_edges(state)

    def _select_from_points_layer(event=None):
        try:
            selection = state.points_layer.selected_data
            if not selection:
                return
            idx = list(selection)[0]
            tid = state.props.get("track_id", [None])[idx]
            if tid is not None and tid > 0:
                _set_focus(int(tid))
        except Exception:
            pass

    if state.points_layer is not None:
        try:
            state.points_layer.events.selected.connect(_select_from_points_layer)
        except Exception:
            pass

    def _set_view_mode(verify: bool):
        state.show_only_event = verify
        toggle_only_event.value = verify
        state.show_cont_edges = not verify
        show_cont.value = not verify
        state.edge_time_window = 0 if not verify else 0
        edge_time_window.value = state.edge_time_window
        focus_edges_only.value = False
        state.filter_focus_edges = False
        if state.labels_layer is not None:
            try:
                state.labels_layer.opacity = 0.3 if verify else 0.5
            except Exception:
                pass
        base_image = _get_layer(viewer, "image") or _get_layer(viewer, "binary")
        if base_image is not None:
            try:
                base_image.visible = not verify
            except Exception:
                pass
        _refresh_edges(state)

    @btn_verify_mode.changed.connect
    def _on_verify_mode(event=None):
        _set_view_mode(True)

    @btn_context_mode.changed.connect
    def _on_context_mode(event=None):
        _set_view_mode(False)

    @btn_reset_filters.changed.connect
    def _on_reset(event=None):
        state.local_filter_enabled = False
        local_filter.value = False
        state.local_filter_radius_um = float(local_radius.value)
        state.local_filter_time_window = 0
        local_dt.value = 0
        state.edge_time_window = 0
        edge_time_window.value = 0
        state.filter_focus_edges = False
        focus_edges_only.value = False
        state.show_only_event = False
        toggle_only_event.value = False
        _refresh_edges(state)

    def _current_event_row() -> Optional[pd.Series]:
        if state.event_rows is not None and state.current_event_idx is not None:
            if 0 <= state.current_event_idx < len(state.event_rows):
                return state.event_rows.iloc[state.current_event_idx]
        if state._last_selected_event_idx is None or state.edges_df is None or state.edges_df.empty:
            return None
        idx = state._last_selected_event_idx
        if idx < 0 or idx >= len(state.edges_df):
            return None
        row = state.edges_df.iloc[idx]
        if str(row.get("edge_type", "")).lower() not in ("fission", "fusion"):
            return None
        return row

    def _write_review(status: str):
        row = _current_event_row()
        if row is None or state.verified_path is None:
            return
        record = {
            "edge_type": row.get("edge_type"),
            "t_from": row.get("t_from"),
            "t_to": row.get("t_to"),
            "parent_label": row.get("parent_label"),
            "child_label": row.get("child_label"),
            "parent_track_id": row.get("parent_track_id"),
            "child_track_id": row.get("child_track_id"),
            "status": status,
        }
        try:
            if os.path.exists(state.verified_path):
                df = pd.read_csv(state.verified_path)
                # drop duplicates for same key
                df = df[~((df["edge_type"] == record["edge_type"]) &
                          (df["t_from"] == record["t_from"]) &
                          (df["t_to"] == record["t_to"]) &
                          (df["parent_label"] == record["parent_label"]) &
                          (df["child_label"] == record["child_label"]))]
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])
            df.to_csv(state.verified_path, index=False)
            print(f"Wrote review: {status} -> {state.verified_path}")
        except Exception as exc:
            print(f"Failed to write review: {exc}", file=sys.stderr)

    @btn_verify.changed.connect
    def _on_verify(event=None):
        _write_review("verified")
        _goto_event(1)

    @btn_question.changed.connect
    def _on_question(event=None):
        _write_review("questionable")
        _goto_event(1)

    @btn_reject.changed.connect
    def _on_reject(event=None):
        _write_review("reject")
        _goto_event(1)

    @local_filter.changed.connect
    def _toggle_local_filter(event=None):
        state.local_filter_enabled = bool(local_filter.value)
        _refresh_edges(state)

    @local_radius.changed.connect
    def _update_local_radius(event=None):
        state.local_filter_radius_um = float(local_radius.value)
        _refresh_edges(state)

    @local_dt.changed.connect
    def _update_local_dt(event=None):
        try:
            state.local_filter_time_window = int(float(local_dt.value))
        except Exception:
            state.local_filter_time_window = 0
        _refresh_edges(state)

    @reload_btn.changed.connect
    def _on_reload(event=None):
        obj_path = str(objects_edit.value) if objects_edit.value else None
        edge_path = str(edges_edit.value) if edges_edit.value else None
        previous_focus = _get_selected_track()
        try:
            if obj_path and os.path.exists(obj_path):
                df = pd.read_csv(obj_path)
                points, props, table = build_points_from_objects(
                    df,
                    state.col_t,
                    state.col_id,
                    state.col_track,
                    state.col_y,
                    state.col_x,
                    state.col_z,
                    state.voxel_size_um,
                )
                state.props = props
                state.lookup = {}
                for _, row in table.iterrows():
                    lid = row["label_id"]
                    if lid is None:
                        continue
                    state.lookup[(int(row["t"]), int(lid))] = np.array(
                        [row["t"], row["z"], row["y"], row["x"]], dtype=float
                    )
                if state.points_layer is None and len(points):
                    state.points_layer = state.viewer.add_points(
                        points,
                        name="centroids",
                        ndim=4,
                        size=float(point_size.value),
                    )
                    try:
                        state.points_layer.edge_color = "black"
                        state.points_layer.edge_width = 0.0
                    except AttributeError:
                        pass
                elif state.points_layer is not None:
                    state.points_layer.data = points
                    state.points_layer.size = float(point_size.value)
                state.objects_path = obj_path
                state.objects_df = df.copy()
                state.points_table = table
                state.track_ids = []
                state.label_ids = []
                if table is not None:
                    if "track_id" in table.columns:
                        track_vals = []
                        for val in table["track_id"].dropna():
                            try:
                                iv = int(val)
                            except Exception:
                                continue
                            if iv > 0:
                                track_vals.append(iv)
                        state.track_ids = sorted(set(track_vals))
                    if "label_id" in table.columns:
                        label_vals = []
                        for val in table["label_id"].dropna():
                            try:
                                iv = int(val)
                            except Exception:
                                continue
                            if iv >= 0:
                                label_vals.append(iv)
                        state.label_ids = sorted(set(label_vals))
                _refresh_track_choices()
                _update_color_map(state.color_key)
                _apply_text(state.points_layer, state.props, state.color_key, show_ids.value, text_size.value)
            if edge_path and os.path.exists(edge_path):
                state.edges_df = pd.read_csv(edge_path)
                state.edges_path = edge_path
                if not state.edges_df.empty:
                    ev = state.edges_df[state.edges_df["edge_type"].isin(["fission","fusion"])].copy()
                    ev["_edge_index"] = ev.index
                    ev = ev.reset_index(drop=True)
                    state.event_rows = ev
                    state.current_event_idx = 0 if len(ev) else None
                    state._last_selected_event_idx = int(ev["_edge_index"].iloc[0]) if len(ev) else None
                    _update_event_info()
                else:
                    state.event_rows = None
                    state.current_event_idx = None
                    state._last_selected_event_idx = None
            _refresh_edges(state)
            if previous_focus is not None and previous_focus in state.track_ids:
                _set_focus(previous_focus)
            elif previous_focus is not None:
                _clear_focus(reset_combo=True)
        except Exception as exc:
            print(f"Reload failed: {exc}", file=sys.stderr)

    _update_color_map(state.color_key)

    widgets = [
        header,
        info,
        objects_edit,
        edges_edit,
        reload_btn,
        show_ids,
        color_by,
        point_size,
        text_size,
        edge_width,
        event_size,
        edge_time_window,
        focus_edges_only,
        btn_reset_filters,
        btn_verify_mode,
        btn_context_mode,
        show_cont,
        show_fission_edges,
        show_fusion_edges,
        show_fission_nodes,
        show_fusion_nodes,
        local_filter,
        local_radius,
        local_dt,
        audit_header,
        track_combo,
        audit_action_row,
        audit_nav_row,
        track_info,
        review_header,
        review_row,
        event_info,
        btn_prev_event,
        btn_next_event,
        toggle_only_event,
        goto_row,
    ]
    container = Container(widgets=widgets, labels=False, layout="vertical")
    viewer.window.add_dock_widget(container, name="Mito controls", area="right")
    try:
        viewer.dims.events.current_step.connect(lambda event=None: _refresh_edges(state))
    except Exception:
        pass
    # optional event selection callbacks for node layers
    def _select_from_nodes():
        def _make_handler(idx_list_key: str, layer_key: str):
            layer = state.node_layers.get(layer_key)
            if layer is None:
                return None
            def _on_select(event=None):
                try:
                    sel = layer.selected_data
                    if not sel:
                        return
                    sel_idx = list(sel)[0]
                    indices = state._geom_indices.get(idx_list_key, [])
                    if sel_idx < len(indices):
                        idx_val = indices[sel_idx]
                        state._last_selected_event_idx = idx_val
                        if state.event_rows is not None and "_edge_index" in state.event_rows:
                            match = state.event_rows[state.event_rows["_edge_index"] == idx_val]
                            if not match.empty:
                                state.current_event_idx = int(match.index[0])
                                _update_event_info()
                                _refresh_edges(state)
                except Exception:
                    pass
            return _on_select
        for key, idx_key in (("fission_nodes","fission_idx"), ("fusion_nodes","fusion_idx")):
            handler = _make_handler(idx_key, key)
            if handler:
                try:
                    state.node_layers[key].events.selected.connect(handler)
                except Exception:
                    pass
    _select_from_nodes()
    _update_event_info()
    return container


def main() -> None:
    ap = argparse.ArgumentParser(description="Napari viewer for mito_toolkit outputs")
    ap.add_argument("--image", help="Path to raw or projection stack (optional)")
    ap.add_argument("--labels", help="Path to labels stack (optional but enables segmentation overlay)")
    ap.add_argument("--objects", help="Path to objects.csv (centroid and track table)")
    ap.add_argument("--edges", help="Path to edges_all.csv for continuation/fission/fusion overlays")
    ap.add_argument("--outdir", help="Output folder to auto-load standard files")
    ap.add_argument("--discover", nargs="?", const=".", help="Legacy alias for --outdir")
    ap.add_argument("--axis", default="tzyx", help="Axis order for image/labels (default: tzyx)")
    ap.add_argument("--color-by", choices=["track", "label"], default="track")
    ap.add_argument("--show-ids", action="store_true", help="Overlay IDs beside centroids")
    ap.add_argument("--col-t", default="t")
    ap.add_argument("--col-id", default="label_id")
    ap.add_argument("--col-track", default="track_id")
    ap.add_argument("--col-y", default="centroid_y_um")
    ap.add_argument("--col-x", default="centroid_x_um")
    ap.add_argument("--col-z", default="centroid_z_um")
    ap.add_argument(
        "--voxel-size-um",
        nargs=3,
        type=float,
        metavar=("Z", "Y", "X"),
        help="Voxel size in microns for (z,y,x) to convert centroids from um to pixels.",
    )
    ap.add_argument("--point-size", type=float, default=6.0)
    ap.add_argument("--text-size", type=float, default=16.0)
    ap.add_argument("--edge-width", type=float, default=2.0)
    ap.add_argument("--event-size", type=float, default=6.0)
    ap.add_argument("--local-edge-filter", action="store_true", help="Enable local density filter around the current cursor/time step for edges/nodes.")
    ap.add_argument("--local-edge-radius-um", type=float, default=0.0, help="Radius (microns) for the local edge filter.")
    ap.add_argument("--local-edge-dt", type=int, default=0, help="Half-window in frames for the local edge filter.")
    args = ap.parse_args()

    voxel_size = tuple(args.voxel_size_um) if args.voxel_size_um else None
    if voxel_size and any((not np.isfinite(v)) or (v <= 0) for v in voxel_size):
        print("Warning: voxel-size must be positive; falling back to voxel coordinates in overlays.", file=sys.stderr)
        voxel_size = None

    outdir = args.outdir or args.discover
    if outdir:
        if not args.objects:
            candidate = os.path.join(outdir, "objects.csv")
            if os.path.exists(candidate):
                args.objects = candidate
        if not args.edges:
            candidate = os.path.join(outdir, "edges_all.csv")
            if os.path.exists(candidate):
                args.edges = candidate
        if not args.labels:
            candidate = os.path.join(outdir, "labels.tif")
            if os.path.exists(candidate):
                args.labels = candidate
    if outdir and not args.image:
        img, lab, objs, edges = discover_paths(outdir)
        args.image = args.image or img
        args.labels = args.labels or lab
        args.objects = args.objects or objs
        args.edges = args.edges or edges

    if args.objects is None and args.labels is None:
        print("ERROR: need at least --objects or --labels to visualize.", file=sys.stderr)
        sys.exit(2)

    def _load_stack(path: str) -> np.ndarray:
        arr = imread(path)
        if arr.ndim < 3 or arr.ndim > 5:
            raise ValueError(f"Expected 3D/4D stack (optionally with channel), got shape {arr.shape} for {path}")
        return to_tzyx(arr, args.axis)

    img_arr = None
    lab_arr = None
    if args.image and os.path.exists(args.image):
        try:
            img_arr = _load_stack(args.image)
        except Exception as exc:
            print(f"ERROR loading image '{args.image}': {exc}", file=sys.stderr)
            sys.exit(2)
    if args.labels and os.path.exists(args.labels):
        try:
            lab_arr = _load_stack(args.labels)
        except Exception as exc:
            print(f"ERROR loading labels '{args.labels}': {exc}", file=sys.stderr)
            sys.exit(2)

    props = {"label_id": [], "track_id": []}
    lookup: Dict[Tuple[int, int], np.ndarray] = {}
    points = np.empty((0, 4), dtype=float)
    table: Optional[pd.DataFrame] = None
    objects_df: Optional[pd.DataFrame] = None

    if args.objects and os.path.exists(args.objects):
        df = pd.read_csv(args.objects)
        objects_df = df.copy()
        try:
            points, props, table = build_points_from_objects(
                df,
                args.col_t,
                args.col_id,
                args.col_track,
                args.col_y,
                args.col_x,
                args.col_z,
                voxel_size,
            )
        except Exception as exc:
            print(f"Failed to interpret objects table: {exc}", file=sys.stderr)
            sys.exit(3)
        for _, row in table.iterrows():
            lid = row["label_id"]
            if lid is None:
                continue
            lookup[(int(row["t"]), int(lid))] = np.array([row["t"], row["z"], row["y"], row["x"]], dtype=float)
    elif lab_arr is not None:
        points, props, table = build_points_from_labels(lab_arr)
        for _, row in table.iterrows():
            lookup[(int(row["t"]), int(row["label_id"]))] = np.array([row["t"], row["z"], row["y"], row["x"]], dtype=float)

    track_ids: List[int] = []
    label_ids: List[int] = []
    if table is not None:
        if "track_id" in table.columns:
            track_vals = []
            for val in table["track_id"].dropna():
                try:
                    iv = int(val)
                except Exception:
                    continue
                if iv > 0:
                    track_vals.append(iv)
            track_ids = sorted(set(track_vals))
        if "label_id" in table.columns:
            label_vals = []
            for val in table["label_id"].dropna():
                try:
                    iv = int(val)
                except Exception:
                    continue
                if iv >= 0:
                    label_vals.append(iv)
            label_ids = sorted(set(label_vals))

    cmap = distinct_color_map(props.get("track_id" if args.color_by == "track" else "label_id", []))

    viewer = napari.Viewer(title="Mito Napari - IDs & Tracks")
    try:
        viewer.dims.ndisplay = 3
    except Exception:
        pass
    if img_arr is not None:
        img_layer = viewer.add_image(img_arr, name="image", blending="additive", opacity=0.3)
        try:
            viewer.layers.move(viewer.layers.index(img_layer), 0)
        except Exception:
            pass
    if lab_arr is not None:
        lbl_layer = viewer.add_labels(lab_arr, name="labels", opacity=0.4)
        try:
            viewer.layers.move(viewer.layers.index(lbl_layer), 1)
        except Exception:
            pass

    points_layer = None
    if len(points):
        colors = []
        default = cmap.get(-1)
        for lid, tid in zip(props.get("label_id", []), props.get("track_id", [])):
            if args.color_by == "track" and tid is not None:
                colors.append(cmap.get(int(tid), default))
            elif lid is not None:
                colors.append(cmap.get(int(lid), default))
            else:
                colors.append(default)
        points_layer = viewer.add_points(
            points,
            name="centroids",
            ndim=4,
            size=float(args.point_size),
            face_color=np.asarray(colors, dtype=float) if colors else "white",
        )
        try:
            points_layer.edge_color = "black"
            points_layer.edge_width = 0.0
        except AttributeError:
            pass
        if args.show_ids:
            _apply_text(points_layer, props, "track_id" if args.color_by == "track" else "label_id", True, args.text_size)

    edges_df = pd.read_csv(args.edges) if args.edges and os.path.exists(args.edges) else None
    events_df = None
    if edges_df is not None and not edges_df.empty:
        events_df = edges_df[edges_df["edge_type"].isin(["fission","fusion"])].copy()
    verified_path = None
    if args.edges:
        verified_path = os.path.join(os.path.dirname(os.path.abspath(args.edges)), "events_verified.csv")

    state = _State(
        viewer=viewer,
        points_layer=points_layer,
        labels_layer=_get_layer(viewer, "labels"),
        edge_layers={},
        node_layers={},
        props=props,
        lookup=lookup,
        color_key="track_id" if args.color_by == "track" else "label_id",
        cmap=cmap,
        objects_path=args.objects,
        edges_path=args.edges,
        edges_df=edges_df,
        events_df=events_df,
        text_size=args.text_size,
        event_size=args.event_size,
        edge_width=args.edge_width,
        local_filter_enabled=bool(args.local_edge_filter),
        local_filter_radius_um=float(args.local_edge_radius_um),
        local_filter_time_window=int(args.local_edge_dt),
        edge_time_window=0,
        filter_focus_edges=False,
        verified_path=verified_path,
        show_cont_edges=True,
        show_fission_edges=True,
        show_fusion_edges=True,
        show_fission_nodes=True,
        show_fusion_nodes=True,
        col_t=args.col_t,
        col_id=args.col_id,
        col_track=args.col_track,
        col_y=args.col_y,
        col_x=args.col_x,
        col_z=args.col_z,
        voxel_size_um=voxel_size,
        show_ids_default=bool(args.show_ids),
        objects_df=objects_df,
        points_table=table,
        track_ids=track_ids,
        label_ids=label_ids,
    )
    state._last_selected_event_idx = None
    state._geom_indices = {}
    if events_df is not None and not events_df.empty:
        ev = events_df.copy()
        ev["_edge_index"] = ev.index
        ev = ev.reset_index(drop=True)
        state.event_rows = ev
        state.current_event_idx = 0 if len(ev) else None
    else:
        state.event_rows = None

    _refresh_edges(state)
    _build_dock(state)
    viewer.update_console({"state": state})
    napari.run()


if __name__ == "__main__":
    main()
