#!/usr/bin/env python3
"""
mito_gui_fixed.py - Windows-safe GUI for 4D mitochondrial tracking

Changes vs mito_gui.py:
- Build subprocess arguments as a list (no shell=True, no shlex.quote).
- Works on Windows paths with spaces.
"""

import os, sys, threading, subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def which_python():
    return sys.executable or "python"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mitochondria 4D Tracking - GUI (Fixed)")
        self.geometry("980x730")

        # Defaults
        self.var_input = tk.StringVar()
        self.var_axis  = tk.StringVar(value="tzyx")
        self.var_vz    = tk.StringVar(value="0.25")
        self.var_vy    = tk.StringVar(value="0.25")
        self.var_vx    = tk.StringVar(value="0.50")
        self.var_min_size = tk.StringVar(value="25")
        self.var_connectivity = tk.StringVar(value="1")

        self.var_iou_thr_event = tk.StringVar(value="0.06")
        self.var_max_disp_um   = tk.StringVar(value="2.0")
        self.var_w_iou         = tk.StringVar(value="0.6")
        self.var_w_dist        = tk.StringVar(value="0.3")
        self.var_w_vol         = tk.StringVar(value="0.1")
        self.var_max_cost      = tk.StringVar(value="1.2")
        self.var_vol_tol       = tk.StringVar(value="0.35")
        self.var_min_event_persistence = tk.StringVar(value="1")

        self.var_outdir = tk.StringVar()
        self.var_tracker_path = tk.StringVar(value=os.path.abspath("mito_4d_tracking_multicriteria_fixed.py"))
        self.var_reporter_path = tk.StringVar(value=os.path.abspath("mito_report_html.py"))
        self.var_viewer_path = tk.StringVar(value=os.path.abspath("mito_lineage_viewer_plotly.py"))
        self.flag_viewer_prefer_edges = tk.BooleanVar(value=True)
        # Napari opts
        self.flag_napari_show_ids = tk.BooleanVar(value=True)
        self.var_napari_color_by = tk.StringVar(value="track")
        self.var_napari_point_size = tk.StringVar(value="12")
        self.var_napari_text_size = tk.StringVar(value="16")
        self.var_napari_edge_width = tk.StringVar(value="2.0")
        self.var_napari_event_size = tk.StringVar(value="6.0")

        # Flags
        self.flag_save_pngs = tk.BooleanVar(value=False)
        self.flag_save_labels = tk.BooleanVar(value=True)
        self.flag_napari    = tk.BooleanVar(value=False)
        self.flag_show_fis  = tk.BooleanVar(value=True)
        self.flag_show_fus  = tk.BooleanVar(value=True)
        self.flag_save_parquet = tk.BooleanVar(value=False)
        self.flag_export_graph = tk.BooleanVar(value=False)
        self.flag_graph_per_track = tk.BooleanVar(value=False)

        # Report flags/options
        self.flag_make_gifs = tk.BooleanVar(value=False)
        self.flag_render_lineage = tk.BooleanVar(value=False)
        self.var_crop_rz = tk.StringVar(value="1.0")
        self.var_crop_ry = tk.StringVar(value="2.0")
        self.var_crop_rx = tk.StringVar(value="2.0")
        self.var_frames_before = tk.StringVar(value="1")
        self.var_frames_after  = tk.StringVar(value="2")
        self.var_max_events    = tk.StringVar(value="100")

        self._action_buttons = []
        self._build_ui()

    def _build_ui(self):
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=8, pady=8)

        frm_in = ttk.Frame(nb); nb.add(frm_in, text="Inputs & Parameters")
        row = 0
        ttk.Label(frm_in, text="Input 4D TIFF:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_in, textvariable=self.var_input, width=70).grid(row=row, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(frm_in, text="Browse...", command=self._pick_input).grid(row=row, column=2, padx=6); row += 1

        ttk.Label(frm_in, text="Output folder:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_in, textvariable=self.var_outdir, width=70).grid(row=row, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(frm_in, text="Choose...", command=self._pick_outdir).grid(row=row, column=2, padx=6); row += 1

        ttk.Label(frm_in, text="Axis order:").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Combobox(frm_in, textvariable=self.var_axis, values=["tzyx","zyxt","xyzt","tyzx","tzxy","ytzx"], width=10).grid(row=row, column=1, sticky="w", padx=6)
        ttk.Label(frm_in, text="Voxel size (um) Z Y X:").grid(row=row, column=1, sticky="e", padx=120)
        vs = ttk.Frame(frm_in); vs.grid(row=row, column=1, sticky="e", padx=6)
        ttk.Entry(vs, textvariable=self.var_vz, width=6).pack(side="left", padx=2)
        ttk.Entry(vs, textvariable=self.var_vy, width=6).pack(side="left", padx=2)
        ttk.Entry(vs, textvariable=self.var_vx, width=6).pack(side="left", padx=2); row += 1

        ttk.Label(frm_in, text="Min object size (vox):").grid(row=row, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_in, textvariable=self.var_min_size, width=8).grid(row=row, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(frm_in, text="Connectivity:").grid(row=row, column=1, sticky="e", padx=120)
        ttk.Combobox(frm_in, textvariable=self.var_connectivity, values=["1","2"], width=5).grid(row=row, column=1, sticky="e", padx=6); row += 1

        block = ttk.LabelFrame(frm_in, text="Linking / Event parameters"); block.grid(row=row, column=0, columnspan=3, sticky="we", padx=6, pady=6); r=0
        ttk.Label(block, text="IoU thr (event evidence):").grid(row=r, column=0, sticky="e", padx=6, pady=3)
        ttk.Entry(block, textvariable=self.var_iou_thr_event, width=8).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Label(block, text="Max displacement (um):").grid(row=r, column=2, sticky="e", padx=6)
        ttk.Entry(block, textvariable=self.var_max_disp_um, width=8).grid(row=r, column=3, sticky="w", padx=6); r+=1
        ttk.Label(block, text="Weights (IoU, Dist, Vol):").grid(row=r, column=0, sticky="e", padx=6)
        ttk.Entry(block, textvariable=self.var_w_iou, width=6).grid(row=r, column=1, sticky="w", padx=2)
        ttk.Entry(block, textvariable=self.var_w_dist, width=6).grid(row=r, column=1, sticky="w", padx=50)
        ttk.Entry(block, textvariable=self.var_w_vol, width=6).grid(row=r, column=1, sticky="w", padx=100); r+=1
        ttk.Label(block, text="Max cost:").grid(row=r, column=0, sticky="e", padx=6)
        ttk.Entry(block, textvariable=self.var_max_cost, width=8).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Label(block, text="Volume tolerance:").grid(row=r, column=2, sticky="e", padx=6)
        ttk.Entry(block, textvariable=self.var_vol_tol, width=8).grid(row=r, column=3, sticky="w", padx=6); r+=1
        ttk.Label(block, text="Min event persistence (frames):").grid(row=r, column=0, sticky="e", padx=6)
        ttk.Entry(block, textvariable=self.var_min_event_persistence, width=8).grid(row=r, column=1, sticky="w", padx=6); r+=1

        flags = ttk.LabelFrame(frm_in, text="Options"); flags.grid(row=row+1, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Checkbutton(flags, text="Save PNG Z-MIPs", variable=self.flag_save_pngs).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(flags, text="Napari view", variable=self.flag_napari).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(flags, text="Show fission", variable=self.flag_show_fis).grid(row=0, column=2, sticky="w", padx=6)
        ttk.Checkbutton(flags, text="Show fusion", variable=self.flag_show_fus).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Checkbutton(flags, text="Save labels.tif", variable=self.flag_save_labels).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Label(flags, text="Tip: labels.tif feeds Napari so you can see colored segments and IDs.", wraplength=360).grid(row=2, column=0, columnspan=4, sticky="w", padx=6, pady=(2,0))
        ttk.Checkbutton(flags, text="Save Parquet", variable=self.flag_save_parquet).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Checkbutton(flags, text="Export graph (GraphML/DOT)", variable=self.flag_export_graph).grid(row=1, column=2, sticky="w", padx=6)
        ttk.Checkbutton(flags, text="Per-track subgraphs", variable=self.flag_graph_per_track).grid(row=1, column=3, sticky="w", padx=6)

        frm_rep = ttk.Frame(nb); nb.add(frm_rep, text="HTML Report")
        ttk.Label(frm_rep, text="Reporter script:").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_rep, textvariable=self.var_reporter_path, width=70).grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(frm_rep, text="Browse...", command=self._pick_reporter).grid(row=0, column=2, padx=6)

        ttk.Label(frm_rep, text="Crop radius (um) Z Y X:").grid(row=1, column=0, sticky="e", padx=6)
        rfrm = ttk.Frame(frm_rep); rfrm.grid(row=1, column=1, sticky="w")
        ttk.Entry(rfrm, textvariable=self.var_crop_rz, width=6).pack(side="left", padx=2)
        ttk.Entry(rfrm, textvariable=self.var_crop_ry, width=6).pack(side="left", padx=2)
        ttk.Entry(rfrm, textvariable=self.var_crop_rx, width=6).pack(side="left", padx=2)

        ttk.Label(frm_rep, text="Frames before / after:").grid(row=2, column=0, sticky="e", padx=6)
        fafrm = ttk.Frame(frm_rep); fafrm.grid(row=2, column=1, sticky="w")
        ttk.Entry(fafrm, textvariable=self.var_frames_before, width=6).pack(side="left", padx=2)
        ttk.Entry(fafrm, textvariable=self.var_frames_after, width=6).pack(side="left", padx=2)

        ttk.Label(frm_rep, text="Max events:").grid(row=3, column=0, sticky="e", padx=6)
        ttk.Entry(frm_rep, textvariable=self.var_max_events, width=8).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Checkbutton(frm_rep, text="Make GIFs", variable=self.flag_make_gifs).grid(row=4, column=0, sticky="w", padx=6)
        ttk.Checkbutton(frm_rep, text="Render lineage PNG", variable=self.flag_render_lineage).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Checkbutton(frm_rep, text="Prefer edges_all.csv when viewing", variable=self.flag_viewer_prefer_edges).grid(row=5, column=0, columnspan=2, sticky="w", padx=6)

        frm_sc = ttk.Frame(nb); nb.add(frm_sc, text="Scripts")
        ttk.Label(frm_sc, text="Tracker script:").grid(row=0, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_sc, textvariable=self.var_tracker_path, width=70).grid(row=0, column=1, sticky="we", padx=6, pady=6)
        ttk.Button(frm_sc, text="Browse...", command=self._pick_tracker).grid(row=0, column=2, padx=6)

        ttk.Label(frm_sc, text="Viewer script:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        ttk.Entry(frm_sc, textvariable=self.var_viewer_path, width=70).grid(row=1, column=1, sticky="we", padx=6, pady=6)

        ttk.Button(frm_sc, text="Browse...", command=self._pick_viewer).grid(row=1, column=2, padx=6)

        napari_opts = ttk.LabelFrame(frm_sc, text="Napari viewer defaults")
        napari_opts.grid(row=2, column=0, columnspan=3, sticky="we", padx=6, pady=6)
        ttk.Checkbutton(napari_opts, text="Show IDs", variable=self.flag_napari_show_ids).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(napari_opts, text="Color by:").grid(row=0, column=1, sticky="e", padx=6)
        ttk.Combobox(napari_opts, textvariable=self.var_napari_color_by, values=["track","label"], width=10, state="readonly").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Label(napari_opts, text="Point size:").grid(row=1, column=0, sticky="e", padx=6, pady=2)
        ttk.Spinbox(napari_opts, from_=1.0, to=60.0, increment=1.0, textvariable=self.var_napari_point_size, width=6).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(napari_opts, text="Text size:").grid(row=1, column=2, sticky="e", padx=6, pady=2)
        ttk.Spinbox(napari_opts, from_=6.0, to=72.0, increment=1.0, textvariable=self.var_napari_text_size, width=6).grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(napari_opts, text="Edge width:").grid(row=2, column=0, sticky="e", padx=6, pady=2)
        ttk.Spinbox(napari_opts, from_=0.5, to=10.0, increment=0.5, textvariable=self.var_napari_edge_width, width=6).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(napari_opts, text="Event marker size:").grid(row=2, column=2, sticky="e", padx=6, pady=2)
        ttk.Spinbox(napari_opts, from_=1.0, to=20.0, increment=0.5, textvariable=self.var_napari_event_size, width=6).grid(row=2, column=3, sticky="w", padx=6)
        ttk.Label(napari_opts, text="Viewer auto-loads objects.csv, labels.tif, and edges_all.csv when present.", wraplength=360).grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(4,0))

        act = ttk.Frame(self); act.pack(fill="x", padx=8)
        btn = ttk.Button(act, text="Run Tracking", command=self.run_tracking); btn.pack(side="left", padx=6, pady=6)
        self._action_buttons.append(btn)
        btn = ttk.Button(act, text="Open Napari (view only)", command=self.view_napari_only); btn.pack(side="left", padx=6, pady=6)
        self._action_buttons.append(btn)
        btn = ttk.Button(act, text="Generate HTML Report", command=self.run_report); btn.pack(side="left", padx=6, pady=6)
        self._action_buttons.append(btn)
        btn = ttk.Button(act, text="Open Lineage (HTML)", command=self.open_lineage); btn.pack(side="left", padx=6, pady=6)
        self._action_buttons.append(btn)

        self.txt = tk.Text(self, height=16)
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self._log("Ready.\n")

    # pickers
    def _pick_input(self):
        p = filedialog.askopenfilename(title="Choose 4D TIFF", filetypes=[("TIFF","*.tif *.tiff"),("All","*.*")])
        if p: self.var_input.set(p)
    def _pick_outdir(self):
        d = filedialog.askdirectory(title="Choose output folder")
        if d: self.var_outdir.set(d)
    def _pick_tracker(self):
        p = filedialog.askopenfilename(title="Choose tracker script", filetypes=[("Python","*.py"),("All","*.*")])
        if p: self.var_tracker_path.set(p)
    def _pick_reporter(self):
        p = filedialog.askopenfilename(title="Choose reporter script", filetypes=[("Python","*.py"),("All","*.*")])
        if p: self.var_reporter_path.set(p)

    def _pick_viewer(self):
        p = filedialog.askopenfilename(title="Choose viewer script", filetypes=[("Python","*.py"),("All","*.*")])
        if p: self.var_viewer_path.set(p)

    def _log(self, msg):
        # Ensure UI updates run on the Tk main thread even when log calls originate from workers.
        def append():
            self.txt.insert("end", msg)
            self.txt.see("end")
        if threading.current_thread() is threading.main_thread():
            append()
        else:
            self.after(0, append)

    # Build args as list
    def _build_tracker_args(self, napari_override=False):
        args = [which_python(), self.var_tracker_path.get(),
                "--input", self.var_input.get(),
                "--axis", self.var_axis.get(),
                "--voxel_size", self.var_vz.get(), self.var_vy.get(), self.var_vx.get(),
                "--min_size", self.var_min_size.get(),
                "--connectivity", self.var_connectivity.get(),
                "--iou_thr_event", self.var_iou_thr_event.get(),
                "--max_disp_um", self.var_max_disp_um.get(),
                "--w_iou", self.var_w_iou.get(),
                "--w_dist", self.var_w_dist.get(),
                "--w_vol", self.var_w_vol.get(),
                "--max_cost", self.var_max_cost.get(),
                "--vol_tol", self.var_vol_tol.get(),
                "--min_event_persistence", self.var_min_event_persistence.get(),
                "--outdir", self.var_outdir.get()]
        if self.flag_save_pngs.get(): args.append("--save_pngs")
        save_labels = self.flag_save_labels.get()
        if (napari_override or self.flag_napari.get()) and not save_labels:
            save_labels = True
            self.flag_save_labels.set(True)
            self._log("Enabling --save_labels so Napari can load labels.tif.\n")
        if save_labels: args.append("--save_labels")
        if self.flag_save_parquet.get(): args.append("--save_parquet")
        if self.flag_export_graph.get(): args.append("--export_graph")
        if self.flag_graph_per_track.get(): args.append("--graph_per_track")
        if napari_override or self.flag_napari.get():
            args.append("--napari")
            if self.flag_show_fis.get(): args.append("--napari_show_fission")
            if self.flag_show_fus.get(): args.append("--napari_show_fusion")
        return args

    def _build_report_args(self):
        args = [which_python(), self.var_reporter_path.get(),
                "--input", self.var_input.get(),
                "--axis", self.var_axis.get(),
                "--voxel_size", self.var_vz.get(), self.var_vy.get(), self.var_vx.get(),
                "--outdir", self.var_outdir.get(),
                "--crop_um_radius", self.var_crop_rz.get(), self.var_crop_ry.get(), self.var_crop_rx.get(),
                "--frames_before", self.var_frames_before.get(),
                "--frames_after", self.var_frames_after.get(),
                "--max_events", self.var_max_events.get()]
        if self.flag_make_gifs.get(): args.append("--make_gifs")
        if self.flag_render_lineage.get(): args.append("--render_lineage")
        return args

    def _run_args_async(self, args):
        # Pretty print cmd preview
        self._log("\n$ " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n")
        btns = list(getattr(self, "_action_buttons", ()))
        saved_states = [b['state'] for b in btns]
        for b in btns: b['state'] = 'disabled'
        def target():
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self._log(line)
                ret = proc.wait()
                if ret != 0:
                    self._log(f"[ERROR] Process exited with code {ret}\n")
                    self.after(0, lambda: messagebox.showerror("Error", f"Command failed (exit {ret}). See log."))
                else:
                    self._log("[OK] Done.\n")
            except Exception as e:
                self._log(f"[EXCEPTION] {e}\n")
                self.after(0, lambda: messagebox.showerror("Exception", str(e)))
            finally:
                for b, st in zip(btns, saved_states): b['state'] = st
        threading.Thread(target=target, daemon=True).start()

    def run_tracking(self):
        if not self.var_input.get() or not self.var_outdir.get():
            messagebox.showwarning("Missing", "Please choose input TIFF and output folder.")
            return
        self._run_args_async(self._build_tracker_args(napari_override=False))

    def view_napari_only(self):
        if not self.var_outdir.get():
            messagebox.showwarning("Missing", "Please choose an output folder with tracker results.")
            return
        if not os.path.isdir(self.var_outdir.get()):
            messagebox.showwarning("Missing", "Output folder does not exist yet. Run tracking first.")
            return
        self._open_napari()

    def run_report(self):
        if not self.var_input.get() or not self.var_outdir.get():
            messagebox.showwarning("Missing", "Please choose input TIFF and output folder.")
            return
        self._run_args_async(self._build_report_args())


    
    def _build_napari_args(self):
        # use existing inputs to build args for napari_id_viewer
        py = which_python()
        viewer_path = os.path.abspath("napari_id_viewer.py")
        args = [py, viewer_path]
        # Ensure we have an output folder to discover results
        outdir = self.var_outdir.get()
        if not outdir:
            raise ValueError("Output folder is required to launch the Napari viewer.")
        args += ["--outdir", outdir]
        if os.path.isdir(outdir):
            labels_path = os.path.join(outdir, "labels.tif")
            if os.path.exists(labels_path):
                args += ["--labels", labels_path]
            objects_path = os.path.join(outdir, "objects.csv")
            if os.path.exists(objects_path):
                args += ["--objects", objects_path]
            edges_path = os.path.join(outdir, "edges_all.csv")
            if os.path.exists(edges_path):
                args += ["--edges", edges_path]
        # pass axis and voxel sizes
        args += ["--axis", self.var_axis.get()]
        if self.var_input.get():
            args += ["--image", self.var_input.get()]
        args += ["--voxel-size-um", self.var_vz.get(), self.var_vy.get(), self.var_vx.get()]
        # show ids + color by + point size
        if self.flag_napari_show_ids.get():
            args += ["--show-ids"]
        args += ["--color-by", self.var_napari_color_by.get()]
        args += ["--point-size", self.var_napari_point_size.get()]
        args += ["--text-size", self.var_napari_text_size.get()]
        args += ["--edge-width", self.var_napari_edge_width.get()]
        args += ["--event-size", self.var_napari_event_size.get()]
        return args

    def _build_viewer_args(self):
        args = [which_python(), self.var_viewer_path.get()]
        outdir = self.var_outdir.get()
        edges = os.path.join(outdir, "edges_all.csv")
        graphml = os.path.join(outdir, "lineage.graphml")
        objects = os.path.join(outdir, "objects.csv")
        if self.flag_viewer_prefer_edges.get() and os.path.exists(edges):
            args += ["--edges", edges]
        elif os.path.exists(graphml):
            args += ["--graph", graphml]
        else:
            raise FileNotFoundError("Could not find edges_all.csv or lineage.graphml in output folder.")
        if os.path.exists(objects):
            args += ["--objects", objects]
        out_html = os.path.join(outdir, "lineage.html")
        args += ["--out", out_html, "--embed_js", "inline", "--open"]
        return args

    def open_lineage(self):
        if not self.var_outdir.get():
            messagebox.showwarning("Missing", "Please choose an output folder where results were written.")
            return
        try:
            args = self._build_viewer_args()
        except FileNotFoundError as e:
            self._log(str(e) + "\nRun tracking first to generate outputs.")
            messagebox.showwarning("Missing outputs", str(e) + "\nRun tracking first to generate outputs.")
            return
        self._run_args_async(args)


    def _open_napari(self):
        try:
            outdir = self.var_outdir.get()
            if outdir and os.path.isdir(outdir):
                missing = []
                if not os.path.exists(os.path.join(outdir, "objects.csv")):
                    missing.append("objects.csv")
                if not os.path.exists(os.path.join(outdir, "labels.tif")):
                    missing.append("labels.tif")
                if not os.path.exists(os.path.join(outdir, "edges_all.csv")):
                    missing.append("edges_all.csv")
                if missing:
                    self._log("Note: missing {} in {}. Napari will open with reduced overlays.\n".format(", ".join(missing), outdir))
            args = self._build_napari_args()
        except Exception as e:
            self._log(str(e))
            messagebox.showwarning("Napari", str(e))
            return
        self._log("Launching Napari viewer...\n")
        self._run_args_async(args)

if __name__ == "__main__":
    app = App()
    app.mainloop()
