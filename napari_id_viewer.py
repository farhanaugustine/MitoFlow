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
    valid = set("tzyx")
    if any(ch not in valid for ch in axis):
        raise ValueError(f"Invalid axis spec '{axis}'. Use characters t, z, y, x.")
    need = ["t", "y", "x"]
    for ch in need:
        if ch not in axis:
            raise ValueError(f"Axis spec '{axis}' must include t, y, and x.")
    add_z = "z" not in axis
    data = arr
    order_axis = axis
    if add_z:
        data = np.expand_dims(data, axis=-1)
        order_axis = axis + "z"
    order = [order_axis.index(ch) for ch in "tzyx"]
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
        if not np.isfinite(scale) or scale == 0:
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
    edges_df: Optional[pd.DataFrame], lookup: Dict[Tuple[int, int], np.ndarray]
) -> Dict[str, List[np.ndarray]]:
    result = {
        "continuations": [],
        "fission": [],
        "fusion": [],
        "fission_nodes": [],
        "fusion_nodes": [],
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
        if edge_type == "fission":
            result["fission"].append(segment)
            result["fission_nodes"].append(end.copy())
        elif edge_type == "fusion":
            result["fusion"].append(segment)
            result["fusion_nodes"].append(end.copy())
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


def _refresh_edges(state: _State) -> None:
    geometry = build_edge_geometry(state.edges_df, state.lookup)
    edge_specs = {
        "continuations": ("Continuation edges", "white", state.show_cont_edges),
        "fission": ("Fission edges", "magenta", state.show_fission_edges),
        "fusion": ("Fusion edges", "orange", state.show_fusion_edges),
    }
    node_specs = {
        "fission_nodes": ("Fission nodes", "magenta", state.show_fission_nodes),
        "fusion_nodes": ("Fusion nodes", "orange", state.show_fusion_nodes),
    }
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
                    edge_width=state.edge_width,
                    blending="translucent",
                )
                state.edge_layers[key] = layer
            else:
                layer.data = data
                layer.edge_color = color
            layer.edge_width = state.edge_width
            layer.visible = bool(visible)
        elif layer is not None:
            layer.data = []
            layer.visible = False
    for key, (title, color, visible) in node_specs.items():
        points = geometry.get(key, [])
        layer = state.node_layers.get(key)
        if points:
            arr = np.asarray(points, dtype=float)
            if layer is None:
                layer = state.viewer.add_points(
                    arr,
                    name=title,
                    ndim=4,
                    size=state.event_size,
                    face_color=color,
                )
                state.node_layers[key] = layer
            else:
                layer.data = arr
            layer.size = state.event_size
            layer.visible = bool(visible)
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

    show_cont = CheckBox(text="Show continuations", value=state.show_cont_edges)
    show_fission_edges = CheckBox(text="Show fission edges", value=state.show_fission_edges)
    show_fusion_edges = CheckBox(text="Show fusion edges", value=state.show_fusion_edges)
    show_fission_nodes = CheckBox(text="Show fission nodes", value=state.show_fission_nodes)
    show_fusion_nodes = CheckBox(text="Show fusion nodes", value=state.show_fusion_nodes)

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

    @show_cont.changed.connect
    def _toggle_cont(event=None):
        state.show_cont_edges = bool(show_cont.value)
        layer = state.edge_layers.get("continuations")
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
        show_cont,
        show_fission_edges,
        show_fusion_edges,
        show_fission_nodes,
        show_fusion_nodes,
        audit_header,
        track_combo,
        audit_action_row,
        audit_nav_row,
        track_info,
    ]
    container = Container(widgets=widgets, labels=False, layout="vertical")
    viewer.window.add_dock_widget(container, name="Mito controls", area="right")
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
    args = ap.parse_args()

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

    img_arr = None
    lab_arr = None
    if args.image and os.path.exists(args.image):
        img_arr = to_tzyx(imread(args.image), args.axis)
    if args.labels and os.path.exists(args.labels):
        lab_arr = to_tzyx(imread(args.labels), args.axis)

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
                tuple(args.voxel_size_um) if args.voxel_size_um else None,
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
        viewer.add_image(img_arr, name="image", blending="additive")
    if lab_arr is not None:
        viewer.add_labels(lab_arr, name="labels", opacity=0.4)

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
        text_size=args.text_size,
        event_size=args.event_size,
        edge_width=args.edge_width,
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
        voxel_size_um=tuple(args.voxel_size_um) if args.voxel_size_um else None,
        show_ids_default=bool(args.show_ids),
        objects_df=objects_df,
        points_table=table,
        track_ids=track_ids,
        label_ids=label_ids,
    )

    _refresh_edges(state)
    _build_dock(state)
    viewer.update_console({"state": state})
    napari.run()


if __name__ == "__main__":
    main()
