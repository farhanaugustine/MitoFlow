#!/usr/bin/env python3
"""
napari_event_table_viewer.py

Napari viewer with a docked, clickable table for edges_all.csv. Selecting a row
snaps the 4D view to the associated parent/child and highlights it for manual validation.

Inputs
------
--objects: path to objects.csv (required)
--edges:   path to edges_all.csv (required)
--image:   optional raw/projection stack (3D/4D, channel allowed; channel 0 used)
--labels:  optional labels stack (3D/4D, channel allowed; channel 0 used)
--axis:    axis order for image/labels, default tzyx (c ignored)
--voxel-size-um: optional z y x voxel size for um->voxel conversion when *_vox absent

Usage
-----
python napari_event_table_viewer.py --objects out/objects.csv --edges out/edges_all.csv --labels out/labels.tif --axis tzyx
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


def build_points_from_objects(df: pd.DataFrame, voxel_size: Optional[Tuple[float, float, float]]) -> Tuple[np.ndarray, Dict[str, List], Dict[Tuple[int, int], np.ndarray]]:
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
        coords.append([t, z, y, x])
        lid = int(r["label_id"]) if pd.notna(r["label_id"]) else -1
        tid = int(r["track_id"]) if pd.notna(r["track_id"]) else -1
        props["label_id"].append(lid)
        props["track_id"].append(tid)
        lookup[(t, lid)] = np.array([t, z, y, x], dtype=float)
    points = np.asarray(coords, dtype=float) if coords else np.empty((0, 4), dtype=float)
    return points, props, lookup


def edge_geometry(edges: pd.DataFrame, lookup: Dict[Tuple[int, int], np.ndarray]) -> Dict[str, List[np.ndarray]]:
    geom = {"continuation": [], "gap": [], "fission": [], "fusion": []}
    for row in edges.itertuples():
        try:
            t0 = int(getattr(row, "t_from"))
            t1 = int(getattr(row, "t_to"))
            p = int(getattr(row, "parent_label"))
            c = int(getattr(row, "child_label"))
            et = str(getattr(row, "edge_type", "continuation")).lower()
        except Exception:
            continue
        start = lookup.get((t0, p))
        end = lookup.get((t1, c))
        if start is None or end is None:
            continue
        geom.setdefault(et, []).append(np.vstack([start, end]))
    return geom


class EdgeTable(QtWidgets.QWidget):
    """Dock widget with a filter and table; emits selection changes."""

    selected = QtCore.Signal(int)  # emits original row index in edges_df

    def __init__(self, edges_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.edges_df = edges_df
        self.filtered_indices: List[int] = list(edges_df.index)
        self._build_ui()
        self._populate_table()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Edge type:"))
        self.cbo_type = QtWidgets.QComboBox()
        self.cbo_type.addItems(["all", "continuation", "gap", "fission", "fusion"])
        self.cbo_type.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self.cbo_type)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["edge_type", "t_from", "t_to", "parent_label", "child_label", "parent_track_id"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_sel_changed)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def _apply_filter(self):
        etype = self.cbo_type.currentText()
        if etype == "all":
            self.filtered_indices = list(self.edges_df.index)
        else:
            self.filtered_indices = [i for i, et in zip(self.edges_df.index, self.edges_df["edge_type"]) if str(et).lower() == etype]
        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(len(self.filtered_indices))
        cols = ["edge_type", "t_from", "t_to", "parent_label", "child_label", "parent_track_id"]
        for row_idx, src_idx in enumerate(self.filtered_indices):
            r = self.edges_df.loc[src_idx, cols]
            for col_idx, col in enumerate(cols):
                item = QtWidgets.QTableWidgetItem(str(r[col]))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()
        if len(self.filtered_indices):
            self.table.selectRow(0)

    def _on_sel_changed(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        row = sel[0].row()
        if 0 <= row < len(self.filtered_indices):
            self.selected.emit(self.filtered_indices[row])


def main():
    ap = argparse.ArgumentParser(description="Napari viewer with clickable edges table")
    ap.add_argument("--objects", required=True, help="Path to objects.csv")
    ap.add_argument("--edges", required=True, help="Path to edges_all.csv")
    ap.add_argument("--image", help="Optional raw/projection stack (channel 0 used if present)")
    ap.add_argument("--labels", help="Optional labels stack (channel 0 used if present)")
    ap.add_argument("--axis", default="tzyx", help="Axis order for image/labels (default tzyx)")
    ap.add_argument("--voxel-size-um", nargs=3, type=float, metavar=("Z", "Y", "X"), help="Voxel size in microns for um->voxel when *_vox columns missing")
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

    img_arr = None
    lab_arr = None
    if args.image and os.path.exists(args.image):
        img_arr = load_stack(args.image, args.axis)
    if args.labels and os.path.exists(args.labels):
        lab_arr = load_stack(args.labels, args.axis)

    points, props, lookup = build_points_from_objects(objects, voxel_size)
    geom = edge_geometry(edges, lookup)

    viewer = napari.Viewer(title="Mito - Table-driven viewer")
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
    # edges layer
    shapes = []
    colors = []
    for et, color in (("continuation", "white"), ("gap", "cyan"), ("fission", "magenta"), ("fusion", "orange")):
        for seg in geom.get(et, []):
            shapes.append(seg)
            colors.append(color)
    if shapes:
        viewer.add_shapes(shapes, name="edges", shape_type="path", edge_color=colors, edge_width=2.0, blending="translucent")

    # focus layers (updated on table selection)
    focus_parent = viewer.add_points(np.empty((0, 4)), name="focus_parent", ndim=4, size=6.0, face_color="yellow", blending="additive")
    focus_child = viewer.add_points(np.empty((0, 4)), name="focus_child", ndim=4, size=6.0, face_color="red", blending="additive")
    focus_edge = viewer.add_shapes(np.empty((0, 2, 4)), name="focus_edge", shape_type="path", edge_color="yellow", edge_width=3.0, blending="additive")

    def focus_row(idx: int):
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
        if start is not None:
            pts_parent = np.array([start], dtype=float)
        if end is not None:
            pts_child = np.array([end], dtype=float)
        if start is not None and end is not None:
            segs = np.array([[start, end]], dtype=float)
        focus_parent.data = pts_parent
        focus_child.data = pts_child
        focus_edge.data = segs
        # snap viewer dims
        target = end if et in ("fission", "fusion") else (end if end is not None else start)
        if target is not None and len(target) == 4:
            try:
                viewer.dims.set_point(0, float(target[0]))
                viewer.dims.set_point(1, float(target[1]))
                viewer.dims.set_point(2, float(target[2]))
                viewer.dims.set_point(3, float(target[3]))
            except Exception:
                pass

    dock = EdgeTable(edges)
    dock.selected.connect(focus_row)
    viewer.window.add_dock_widget(dock, area="right")
    # initial focus
    focus_row(0)

    napari.run()


if __name__ == "__main__":
    main()
