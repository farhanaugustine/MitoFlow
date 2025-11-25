#!/usr/bin/env python3
"""
napari_event_focus_viewer.py

Interactive Napari viewer to browse events by time and slice:
- Load objects.csv and edges_all.csv (required), plus optional image/labels.
- Time selector (events_per_t.csv) and Z selector (events_per_z.csv) filter visible edges/events.
- Edge table updates with the filtered set; clicking a row focuses parent/child and jumps the view.
- Overlays show only the selected frame/slice, with labels derived from objects.csv.

Usage
-----
python napari_event_focus_viewer.py --objects out/objects.csv --edges out/edges_all.csv --events-t out/events_per_t.csv --events-z out/events_per_z.csv --axis tzyx --labels out/labels.tif --image raw.tif
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tifffile import imread
import napari
from qtpy import QtCore, QtWidgets


def parse_axis(axis: str) -> str:
    axis = axis.lower()
    allowed = set("tzyxc")
    if not set(axis) <= allowed:
        raise ValueError(f"--axis may contain only t,z,y,x,c; got '{axis}'")
    if axis.count("t") > 1 or axis.count("c") > 1:
        raise ValueError(f"--axis may contain at most one 't' and one 'c'; got '{axis}'")
    for ch in "zyx":
        if ch not in axis:
            raise ValueError(f"--axis must include z,y,x; got '{axis}'")
    return axis


def to_tzyx(arr: np.ndarray, axis: str) -> np.ndarray:
    axis = parse_axis(axis)
    data = arr
    if "c" in axis:
        c_pos = axis.index("c")
        if c_pos >= data.ndim:
            raise ValueError(f"--axis='{axis}' expects a channel axis, data shape {data.shape}")
        data = np.take(data, indices=0, axis=c_pos)
        axis = axis.replace("c", "")
    if "t" not in axis:
        data = np.expand_dims(data, axis=0)
        axis = "t" + axis
    if len(axis) != data.ndim:
        raise ValueError(f"--axis='{axis}' expects {len(axis)} dims, got {data.shape}")
    order = [axis.index(ch) for ch in "tzyx"]
    return np.transpose(data, order)


def load_stack(path: str, axis: str) -> np.ndarray:
    arr = imread(path)
    if arr.ndim < 3 or arr.ndim > 5:
        raise ValueError(f"expected 3D/4D (optional channel) stack, got shape {arr.shape}")
    return to_tzyx(arr, axis)


def safe_vox(um_val: float | None, vox_val: float | None, scale: float | None) -> float:
    if vox_val is not None and np.isfinite(vox_val):
        return float(vox_val)
    if um_val is not None and np.isfinite(um_val) and scale and np.isfinite(scale) and scale > 0:
        return float(um_val) / float(scale)
    return float("nan")


def build_points_and_lookup(df: pd.DataFrame, voxel_size: Optional[Tuple[float, float, float]]) -> Tuple[np.ndarray, Dict[str, List], Dict[Tuple[int, int], np.ndarray]]:
    props: Dict[str, List] = {"label_id": [], "track_id": []}
    coords: List[List[float]] = []
    lookup: Dict[Tuple[int, int], np.ndarray] = {}
    for _, r in df.iterrows():
        t = int(r["t"])
        z = safe_vox(r.get("centroid_z_um"), r.get("centroid_z_vox"), voxel_size[0] if voxel_size else None)
        y = safe_vox(r.get("centroid_y_um"), r.get("centroid_y_vox"), voxel_size[1] if voxel_size else None)
        x = safe_vox(r.get("centroid_x_um"), r.get("centroid_x_vox"), voxel_size[2] if voxel_size else None)
        if not np.all(np.isfinite([z, y, x])):
            continue
        lid = int(r["label_id"]) if pd.notna(r["label_id"]) else -1
        tid = int(r["track_id"]) if pd.notna(r["track_id"]) else -1
        coords.append([t, z, y, x])
        props["label_id"].append(lid)
        props["track_id"].append(tid)
        lookup[(t, lid)] = np.array([t, z, y, x], dtype=float)
    points = np.asarray(coords, dtype=float) if coords else np.empty((0, 4), dtype=float)
    return points, props, lookup


def filter_edges(edges_df: pd.DataFrame, t_sel: Optional[int], z_sel: Optional[int]) -> pd.DataFrame:
    df = edges_df
    if t_sel is not None:
        df = df[df["t_to"] == t_sel]
    if z_sel is not None and "child_slice_index" in df.columns:
        df = df[df["child_slice_index"] == z_sel]
    return df


class TimeSliceDock(QtWidgets.QWidget):
    time_changed = QtCore.Signal(object)  # None or int
    z_changed = QtCore.Signal(object)  # None or int

    def __init__(self, events_t: pd.DataFrame, events_z: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.events_t = events_t
        self.events_z = events_z
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # time selector
        layout.addWidget(QtWidgets.QLabel("Frames with events:"))
        self.lst_t = QtWidgets.QListWidget()
        for _, r in self.events_t.iterrows():
            t = int(r["t"])
            label = f"t={t}  (fiss={int(r.get('fission_count',0))}, fus={int(r.get('fusion_count',0))})"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, t)
            self.lst_t.addItem(item)
        self.lst_t.currentItemChanged.connect(self._on_t_changed)
        layout.addWidget(self.lst_t)

        layout.addWidget(QtWidgets.QLabel("Slices with events (select after time):"))
        self.cbo_z = QtWidgets.QComboBox()
        self.cbo_z.addItem("All slices", None)
        for _, r in self.events_z.iterrows():
            z = int(r["slice_index"])
            label = f"z={z}  (fiss={int(r.get('fission_count',0))}, fus={int(r.get('fusion_count',0))})"
            self.cbo_z.addItem(label, z)
        self.cbo_z.currentIndexChanged.connect(self._on_z_changed)
        layout.addWidget(self.cbo_z)

        btn_reset = QtWidgets.QPushButton("Reset filters")
        btn_reset.clicked.connect(self._reset)
        layout.addWidget(btn_reset)
        layout.addStretch()

    def _on_t_changed(self, curr, prev):
        if curr is None:
            return
        t = curr.data(QtCore.Qt.UserRole)
        self.time_changed.emit(int(t))

    def _on_z_changed(self, idx):
        z = self.cbo_z.itemData(idx)
        self.z_changed.emit(z)

    def _reset(self):
        self.lst_t.clearSelection()
        self.cbo_z.setCurrentIndex(0)
        self.time_changed.emit(None)
        self.z_changed.emit(None)


class EdgeTable(QtWidgets.QWidget):
    selected = QtCore.Signal(int)  # emits edges_df index

    def __init__(self, edges_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.edges_df = edges_df
        self.filtered_indices: List[int] = list(edges_df.index)
        self._build_ui()
        self._populate()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget()
        cols = ["edge_type", "t_from", "t_to", "parent_label", "child_label", "parent_track_id", "child_track_id", "child_slice_index"]
        self.cols = [c for c in cols if c in self.edges_df.columns]
        self.table.setColumnCount(len(self.cols))
        self.table.setHorizontalHeaderLabels(self.cols)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_sel)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def set_filtered(self, indices: List[int]):
        self.filtered_indices = indices
        self._populate()

    def _populate(self):
        self.table.setRowCount(len(self.filtered_indices))
        for row_idx, src_idx in enumerate(self.filtered_indices):
            r = self.edges_df.loc[src_idx, self.cols]
            for col_idx, col in enumerate(self.cols):
                val = r[col]
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()
        if len(self.filtered_indices):
            self.table.selectRow(0)

    def _on_sel(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        if 0 <= row < len(self.filtered_indices):
            self.selected.emit(self.filtered_indices[row])


def build_focus_layers(viewer: napari.Viewer):
    parent_pts = viewer.add_points(np.empty((0, 4)), name="focus_parent", ndim=4, size=6.0, face_color="yellow", blending="additive")
    child_pts = viewer.add_points(np.empty((0, 4)), name="focus_child", ndim=4, size=6.0, face_color="red", blending="additive")
    edge_layer = viewer.add_shapes(np.empty((0, 2, 4)), name="focus_edge", shape_type="path", edge_color="yellow", edge_width=3.0, blending="additive")
    label_layer = viewer.add_points(np.empty((0, 4)), name="focus_labels", ndim=4, size=1.0, face_color="white", text={"string": []}, blending="translucent")
    return parent_pts, child_pts, edge_layer, label_layer


def _valid_segment(start: np.ndarray, end: np.ndarray, eps: float = 1e-6) -> bool:
    if start is None or end is None:
        return False
    if start.shape != end.shape or start.shape[-1] != 4:
        return False
    if (not np.all(np.isfinite(start))) or (not np.all(np.isfinite(end))):
        return False
    # use spatial delta (z,y,x) to avoid zero-length in space
    delta = end[-3:] - start[-3:]
    return float(np.linalg.norm(delta)) > eps


def focus_on_edge(idx: int, edges: pd.DataFrame, lookup: Dict[Tuple[int, int], np.ndarray], parent_pts, child_pts, edge_layer, label_layer, viewer: napari.Viewer, objects_df: pd.DataFrame):
    if idx < 0 or idx >= len(edges):
        return
    row = edges.iloc[idx]
    t0, t1 = int(row["t_from"]), int(row["t_to"])
    p, c = int(row["parent_label"]), int(row["child_label"])
    et = str(row.get("edge_type", "continuation")).lower()
    start = lookup.get((t0, p))
    end = lookup.get((t1, c))
    pts_parent = np.empty((0, 4))
    pts_child = np.empty((0, 4))
    segs = np.empty((0, 2, 4))
    labels_pts = np.empty((0, 4))
    labels_txt: List[str] = []
    if start is not None:
        pts_parent = np.array([start], dtype=float)
    if end is not None:
        pts_child = np.array([end], dtype=float)
    if _valid_segment(start, end):
        segs = np.array([[start, end]], dtype=float)
    # labels from objects.csv for child
    try:
        meta = objects_df[(objects_df["t"] == t1) & (objects_df["label_id"] == c)].iloc[0]
        tid = int(meta.get("track_id", -1))
        labels_pts = np.array([end], dtype=float) if end is not None else np.empty((0, 4))
        labels_txt = [f"{tid}:{c}"]
    except Exception:
        labels_pts = np.empty((0, 4))
        labels_txt = []
    parent_pts.data = pts_parent
    child_pts.data = pts_child
    edge_layer.data = segs
    label_layer.data = labels_pts
    if labels_txt:
        label_layer.text = {"string": labels_txt, "color": "white", "size": 12}
    else:
        label_layer.text = None
    target = end if et in ("fission", "fusion") else (end if end is not None else start)
    if target is not None and len(target) == 4:
        try:
            viewer.dims.set_point(0, float(target[0]))
            viewer.dims.set_point(1, float(target[1]))
            viewer.dims.set_point(2, float(target[2]))
            viewer.dims.set_point(3, float(target[3]))
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Napari event focus viewer (time/slice filtering)")
    ap.add_argument("--objects", required=True, help="Path to objects.csv")
    ap.add_argument("--edges", required=True, help="Path to edges_all.csv")
    ap.add_argument("--events-t", help="Path to events_per_t.csv")
    ap.add_argument("--events-z", help="Path to events_per_z.csv")
    ap.add_argument("--image", help="Optional image stack")
    ap.add_argument("--labels", help="Optional labels stack")
    ap.add_argument("--axis", default="tzyx", help="Axis order for image/labels (default tzyx)")
    ap.add_argument("--voxel-size-um", nargs=3, type=float, metavar=("Z", "Y", "X"), help="Voxel size in microns for um->voxel when *_vox missing")
    args = ap.parse_args()

    voxel_size = tuple(args.voxel_size_um) if args.voxel_size_um else None
    if voxel_size and any((not np.isfinite(v)) or (v <= 0) for v in voxel_size):
        print("Warning: voxel-size must be positive; falling back to voxel columns only.", file=sys.stderr)
        voxel_size = None

    objects = pd.read_csv(args.objects)
    edges = pd.read_csv(args.edges)
    if edges.empty:
        print("edges_all.csv is empty; nothing to browse.", file=sys.stderr)
        sys.exit(2)

    events_t = pd.read_csv(args.events_t) if args.events_t and os.path.exists(args.events_t) else pd.DataFrame({"t": [], "fission_count": [], "fusion_count": []})
    events_z = pd.read_csv(args.events_z) if args.events_z and os.path.exists(args.events_z) else pd.DataFrame({"slice_index": [], "fission_count": [], "fusion_count": []})

    img_arr = None
    lab_arr = None
    if args.image and os.path.exists(args.image):
        img_arr = load_stack(args.image, args.axis)
    if args.labels and os.path.exists(args.labels):
        lab_arr = load_stack(args.labels, args.axis)

    points, props, lookup = build_points_and_lookup(objects, voxel_size)

    viewer = napari.Viewer(title="Mito - Time/Slice Event Focus")
    try:
        viewer.dims.ndisplay = 3
    except Exception:
        pass
    if img_arr is not None:
        viewer.add_image(img_arr, name="image", blending="additive", opacity=0.3)
    if lab_arr is not None:
        viewer.add_labels(lab_arr, name="labels", opacity=0.4)
    if len(points):
        viewer.add_points(points, name="centroids", ndim=4, size=3.0, face_color="track_id", features=props, blending="translucent")

    # Full edges layer (faded) for context
    all_shapes = []
    all_colors = []
    for et, color in (("continuation", "white"), ("gap", "cyan"), ("fission", "magenta"), ("fusion", "orange")):
        sub = edges[edges["edge_type"].str.lower() == et] if "edge_type" in edges else edges
        for _, r in sub.iterrows():
            start = lookup.get((int(r["t_from"]), int(r["parent_label"])))
            end = lookup.get((int(r["t_to"]), int(r["child_label"])))
            if not _valid_segment(start, end):
                continue
            all_shapes.append(np.vstack([start, end]))
            all_colors.append(color)
    if all_shapes:
        viewer.add_shapes(all_shapes, name="edges_all", shape_type="path", edge_color=all_colors, edge_width=1.0, blending="additive", opacity=0.2)

    # Focus layers
    focus_parent, focus_child, focus_edge, focus_label = build_focus_layers(viewer)

    # Table
    table = EdgeTable(edges)
    viewer.window.add_dock_widget(table, area="right")

    # Time/Z dock
    tz_dock = TimeSliceDock(events_t, events_z)
    viewer.window.add_dock_widget(tz_dock, area="left")

    # Current filtered indices
    current_filtered = list(edges.index)

    def apply_filters(t_sel: Optional[int], z_sel: Optional[int]):
        nonlocal current_filtered
        df = filter_edges(edges, t_sel, z_sel)
        current_filtered = list(df.index)
        table.set_filtered(current_filtered)
        if current_filtered:
            focus_on_edge(current_filtered[0], edges, lookup, focus_parent, focus_child, focus_edge, focus_label, viewer, objects)

    def on_time_changed(t_val):
        apply_filters(t_val, get_current_z())

    def on_z_changed(z_val):
        apply_filters(get_current_t(), z_val if z_val is not None else None)

    def get_current_t():
        sel = tz_dock.lst_t.currentItem()
        return sel.data(QtCore.Qt.UserRole) if sel else None

    def get_current_z():
        return tz_dock.cbo_z.currentData()

    tz_dock.time_changed.connect(on_time_changed)
    tz_dock.z_changed.connect(on_z_changed)

    def on_row_selected(idx: int):
        focus_on_edge(idx, edges, lookup, focus_parent, focus_child, focus_edge, focus_label, viewer, objects)

    table.selected.connect(on_row_selected)

    if current_filtered:
        focus_on_edge(current_filtered[0], edges, lookup, focus_parent, focus_child, focus_edge, focus_label, viewer, objects)

    napari.run()


if __name__ == "__main__":
    main()
