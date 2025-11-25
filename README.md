# Mitochondria 4D Tracking Toolkit

Point-and-click and command-line tools for tracking mitochondria in 4D (t,z,y,x) binary stacks, detecting **fission/fusion** events, visualizing in **Napari**, and generating a **no-code HTML report**.
Simply binarize your 4D hyperstacks in ImageJ/FIJI --> save as tiffs --> run MitoFlow to detect fission/fusion events.

Repo still under development. API may evolve as the project gains more advanced feature. 🚧🚩

## Contents
- `mito_gui_fixed.py` - Tkinter GUI wrapper (Windows-safe) for end users.
- `mito_4d_tracking_multicriteria_fixed.py` - tracker with multi-criteria linking, event detection, Napari overlays, graph export.
- `mito_report_html.py` - HTML report generator with Z-MIP crops & optional GIFs.
- `mito_vis_napari.py` - standalone visualization (reads CSVs) - optional.
- `requirements.txt` - Python dependencies.
- `PARAMETERS_AND_JUSTIFICATION.md` - parameter meanings and scientific rationale.
- `CHANGELOG.md` - changes and improvements.
- `LICENSE` - GNU General Public License.

## Quickstart (GUI)
1. Install Python 3.9+ (3.11 recommdended) and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the GUI:
   ```bash
   python mito_gui_fixed.py
   ```
3. In the GUI pick your 4D TIFF, axis order, voxel sizes (um), and output folder. Leave **Save labels.tif** checked so Napari/ImageJ can load the 4D label and track hyperstacks. Click **Run Tracking** to process the data, **Open Napari (view only)** to explore results, and **Generate HTML Report** for a shareable summary.

### Napari viewer tips
- Tracking writes `labels.tif`, `objects.csv`, and `edges_all.csv` into the output folder; the GUI passes these to the viewer automatically.  
- The dock on the right (inside Napari) lets you resize centroid markers, adjust text size, tweak edge thickness, and toggle continuations/fission/fusion overlays without touching code.  
- Use the **Audit tracks** controls (combo + highlight buttons) to step through track IDs, isolate their voxels, and review edges à la ROI manager.  
- IDs appear as `track:label` when track IDs are present; clear **Show IDs** if you prefer an uncluttered view.  
- Press `4` (Napari shortcut) or use the display dropdown to switch into a true 3D view; the viewer now defaults to 3D mode and respects anisotropic voxel spacing.  
- Every overlay layer reports its health: if centroids or edges are skipped because the source tables are incomplete, a summary is printed to the terminal so you can trace the cause instead of guessing.  
- Use the **Reload files** button to pick updated CSVs after manual edits; diagnostics will re-run on reload.

Launch the viewer manually from a terminal:
```bash
python napari_id_viewer.py --outdir path/to/results --show-ids
```

## Command-line
### Tracking
```bash
python mito_4d_tracking_multicriteria_fixed.py   --input stack.tif --axis tzyx   --voxel_size 0.25 0.25 0.5   --min_size 25 --connectivity 1   --iou_thr_event 0.06 --max_disp_um 2.0   --w_iou 0.6 --w_dist 0.3 --w_vol 0.1   --max_cost 1.2 --vol_tol 0.35 --min_event_persistence 1   --outdir results   --save_pngs --export_graph --graph_per_track   --napari --napari_show_fission --napari_show_fusion
```

### HTML report
```bash
python mito_report_html.py   --input stack.tif --axis tzyx   --voxel_size 0.25 0.25 0.5   --outdir results   --crop_um_radius 1.0 2.0 2.0   --frames_before 1 --frames_after 2   --make_gifs --max_events 100 --render_lineage
```

## Outputs at a glance
- `objects.csv` - per-frame objects (centroids reported in both µm and voxel units, voxel + µm³ volumes, `track_id`).
- `links.csv` - candidate parent->child links with costs & features.
- `matches.csv` - assigned 1->1 continuations (or new appearances).
- `events_fission_edges.csv` / `events_fusion_edges.csv` - normalized event edges (numeric IDs & times).
- `edges_all.csv` - union of continuations and events (with `edge_type`).
- `labels.tif` (optional) - OME-TIFF hyperstack (`TCZYX`) with connected-component IDs for every frame/slice.
- `tracks.tif` (optional) - OME-TIFF hyperstack (`TCZYX`) with per-voxel track IDs, preserving the lineage across T and Z.
- `tracks_rgb.tif` (optional) - 3-channel OME-TIFF (`TCZYX`) with stable pseudo-colors per track for quick auditing in ImageJ/FIJI.
- `lineage.graphml` (+ optional per-track `track_*.graphml`) - for Cytoscape/Gephi.
- `report/report.html` - no-code, shareable summary with event crops/GIFs.

## Axis order
Use `tifffile` to inspect axes & shape:
```python
from tifffile import TiffFile
with TiffFile("stack.tif") as tf:
    s = tf.series[0]
    print(s.axes, s.shape)
```
Pass the matching permutation (e.g., `tzyx`, `zyxt`, `xyzt`) via `--axis`.

## Built-in diagnostics
- **Centroid sanity checks:** Any object rows lacking usable coordinates (e.g., missing voxel scaling) are dropped with a printed count so you can patch the upstream CSV.  
- **Edge sanity checks:** Continuation/fission/fusion segments that cannot be reconstructed report their counts separately (`continuation=…`, `fission=…`, `fusion=…`).  
- **Event persistence:** Fission/fusion calls now require contiguous frame support on both sides of the event. This suppresses one-off glitches yet still allows biologically brief remodeling when it is sustained for the requested number of frames.  
- **3D defaults:** Napari viewers set `ndisplay=3`, add white track labels by default, and honour anisotropic voxel metadata or falls back to stored voxel centroids when spacing is zero/unknown.


## Quick Guide for Non-Coders (5 minutes)

**Step 1 - Install once**
1. Install Python 3.9+ (Anaconda is fine).
2. Open a terminal and run: `pip install -r requirements.txt`.

**Step 2 - Launch the app**
- Double-click `run_gui.bat` on Windows (or run `python mito_gui_fixed.py`).

**Step 3 - Pick files & defaults**
- **Input 4D TIFF**: your thresholded stack (t,z,y,x). If unsure about axis order, try `tzyx` first.
- **Output folder**: where results will be written.
- **Voxel size (um)**: enter Z, Y, X spacing from your microscope metadata (e.g., 0.3 0.1 0.1).

**Step 4 - Click "Run Tracking"**
- It creates CSVs, optional graphs, and quick-look PNGs.
- Leave **Save labels.tif** checked to export `labels.tif`, `tracks.tif`, and `tracks_rgb.tif` hyperstacks (TCZYX) for 4D inspection inside ImageJ, FIJI, or Napari.
- Use "Open Napari (view only)" to visualize edges and centroids, then use the right-hand control panel to tweak point size, text size, and edge visibility.

**Step 5 - Click "Generate HTML Report"**
- Opens `report/report.html` (no coding). It shows counts, example tables, and per-event crops/GIFs.

### Visual QA workflow
- Start Napari with **Show fission** only -> scan splits. Then **Show fusion** only -> scan merges.
- Adjust **point size**, **text size**, and **edge width** from the Napari control panel until edges stand out clearly—no restart needed.
- If continuations break (white edges missing), increase **Max displacement (um)** or weight **Dist** more.
- If too many false events, raise **IoU (event evidence)** and/or lower **Volume tolerance**.
- Re-generate the report and share `report.html` with colleagues for sign-off.

### Frequently Asked Questions
**Q: My events look like times (e.g., "12:14") in Excel.**  
A: Our event tables use *numeric columns only* to avoid this. If Excel still "guesses", import as *data* or open the Parquet files.

**Q: The image opens but I see no edges.**  
A: Edges appear only after tracking; ensure you ran "Run Tracking" first. Then check that **Show fission/fusion** are ticked.

**Q: How do I find the axis order?**  
A: In ImageJ: *Image -> Properties...* (Frames and Slices). Or use `tifffile` to print `axes`/`shape`. Try `tzyx` and see if `T` (frames) matches expectations.

**Q: Do I have to threshold?**  
A: Yes. This pipeline expects a binary stack (mitochondria = 1). You can segment upstream in FIJI/ImageJ/Napari/Cellpose/etc.

**Q: Can I review one mitochondrion's history?**  
A: Yes. Open `track_<id>.graphml` in Cytoscape/Gephi or filter `edges_all.csv` by that `track_id`.

### Troubleshooting checklist
- **No objects detected** -> lower `--min_size` or verify the stack is binary (0/1).
- **Over-merging thin tubules** -> set **Connectivity** = 1 (6-neigh).
- **Split/merge from touching tips** -> increase **IoU evidence** and **persistence**.
- **Fast transport** -> raise **Max displacement (um)** and maybe **Max cost** slightly.
- **Z-undersampling** -> increase **Volume tolerance** or acquire with thinner Z-steps.

### Glossary
- **Label**: per-frame connected component ID (what voxels belong to which object *now*).
- **Track ID**: object identity over time after linking.
- **Continuation**: the same object across frames (white edge).
- **Fission/Fusion**: split/merge events validated by overlap and volume conservation.

## Why this pipeline (in one slide)
- **Biology-aware**: split/merge requires overlap + volume conservation + persistence.
- **Stable tracks**: multi-cue costs (IoU + distance + volume) reduce breakage and identity swaps.
- **Metric-true**: linking and event tests operate in physical units, respecting anisotropic voxels across T–Z–Y–X.
- **Audit-ready**: hyperstack outputs plus the Napari audit panel make per-mitochondrion review as frictionless as an ROI manager.
- **Human-first QA**: Napari overlays with adjustable styles; no-code HTML report with event crops.
- **Open & portable**: CSV/Parquet, GraphML/DOT, and a simple GUI.

If you already use other 4D toolkits: keep them! Use this for **auditable mitochondrial remodeling** where conservation and persistence matter, and export GraphML/CSV back to your lab's ecosystem.

### Lineage graph (interactive HTML)
From the main window, click **Open Lineage (HTML)** to generate an interactive Plotly graph with time on the x-axis and track lanes on y.
- It **auto-uses `edges_all.csv`** (most reliable) when available, otherwise falls back to `lineage.graphml`.
- The HTML is self-contained (no internet required) and opens directly in your browser.
- Nodes are colored by `track_id` (from `objects.csv`) and sized by `volume_vox`. Edge colors: gray=continuation, magenta=fission, orange=fusion.

## Lineage Viewer (Interactive HTML, Plotly)

From the GUI, click **Open Lineage (HTML)** to generate a shareable, self-contained HTML:
- **Source preference:** uses `edges_all.csv` by default (most reliable for `t`/`label_id`); falls back to `lineage.graphml` if needed.
- **What you see:** time on the x-axis, track lanes on y; edges colored (gray=continuation, magenta=fission, orange=fusion); nodes colored by `track_id`, sized by `volume_vox`.
- **Where it goes:** writes `lineage.html` into your chosen output folder and opens it in your browser.

### CLI usage (advanced)
```bash
python mito_lineage_viewer_plotly.py       --edges results/edges_all.csv       --objects results/objects.csv       --out results/lineage.html --embed_js inline
```
Or with GraphML:
```bash
python mito_lineage_viewer_plotly.py       --graph results/lineage.graphml --objects results/objects.csv       --out results/lineage.html --embed_js inline
```

### Troubleshooting
- **Blank/empty page:** ensure `--embed_js inline` (GUI uses inline by default). Try another browser if your IT blocks inline scripts.
- **Everything in one row / single color:** your graph likely lacks `t`/`label_id` as node attributes. Prefer `--edges edges_all.csv`, or use the latest viewer which infers `(t,label)` from node IDs.
- **Huge graphs (slow):** the viewer uses WebGL (`Scattergl`). If still slow, reduce time range by filtering CSVs first.


