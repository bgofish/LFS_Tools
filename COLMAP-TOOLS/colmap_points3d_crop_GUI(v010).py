#!/usr/bin/env python3
"""
COLMAP Points3D Cropping Tool
==============================
View, crop, and export COLMAP Points3D files in BIN or TXT format.

Features:
  • Open  points3D.bin  or  points3D.txt  (auto-detected)
  • 3-D scatter-plot viewer with preset views and colour modes
  • Clip / crop panel with histogram-assisted bound picking
  • Export the cropped subset back to the *same format* that was read,
    or choose the other format explicitly
  • Save / load clip bounds as JSON
  • Full per-point track data preserved on read/write

Coordinate convention (COLMAP default):
  +Y up   +X left   −Z into scene
"""

import json
import struct
import threading
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ──────────────────────────────────────────────────────────────
#  COLMAP I/O  (reads / writes full point records including track)
# ──────────────────────────────────────────────────────────────

def read_points3d_txt(path):
    """Return list of (point3d_id, x, y, z, r, g, b, error, track[])."""
    points = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                pid   = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]),   int(parts[5]),   int(parts[6])
                err   = float(parts[7])
                track = []
                td = parts[8:]
                for i in range(0, len(td) - 1, 2):
                    try:
                        track.append((int(td[i]), int(td[i + 1])))
                    except ValueError:
                        break
                points.append((pid, x, y, z, r, g, b, err, track))
            except (ValueError, IndexError):
                continue
    return points


def read_points3d_bin(path):
    """Return list of (point3d_id, x, y, z, r, g, b, error, track[])."""
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            pid       = struct.unpack("<Q", f.read(8))[0]
            x, y, z   = struct.unpack("<ddd", f.read(24))
            r, g, b   = struct.unpack("<BBB", f.read(3))
            err       = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            raw_track = f.read(8 * track_len)
            track = []
            for i in range(track_len):
                img_id, pt2d = struct.unpack_from("<ii", raw_track, i * 8)
                track.append((img_id, pt2d))
            points.append((pid, x, y, z, r, g, b, err, track))
    return points


def write_points3d_txt(points, path):
    """Write list of point-tuples to points3D.txt."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for rec in points:
            pid, x, y, z, r, g, b, err = rec[:8]
            track = rec[8] if len(rec) > 8 else []
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            line = f"{pid} {x} {y} {z} {r} {g} {b} {err}"
            for img_id, pt2d in track:
                line += f" {img_id} {pt2d}"
            f.write(line + "\n")


def write_points3d_bin(points, path):
    """Write list of point-tuples to points3D.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for rec in points:
            pid, x, y, z, r, g, b, err = rec[:8]
            track = rec[8] if len(rec) > 8 else []
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            f.write(struct.pack("<Q",  pid))
            f.write(struct.pack("<ddd", x, y, z))
            f.write(struct.pack("<BBB", r, g, b))
            f.write(struct.pack("<d",  err))
            f.write(struct.pack("<Q",  len(track)))
            for img_id, pt2d in track:
                f.write(struct.pack("<ii", img_id, pt2d))


# ──────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────

def _entry_float(var, fallback):
    try:
        v = var.get().strip()
        return float(v) if v else fallback
    except ValueError:
        return fallback


# ──────────────────────────────────────────────────────────────
#  Main application
# ──────────────────────────────────────────────────────────────

class ColmapCropGUI(tk.Tk):

    _VIEWS = {
        "▲  Top":    ( 90,   0),
        "▼  Bottom": (-90,   0),
        "◄ Left":   (  0,  90),
        "►  Right":   (  0, -90),
        "▲  Front":   (  0, 180),
        "▼  Back":    (  0,   0),
        "☼  Free":    ( 20, -45),  # +Y up the screen
    }

    def __init__(self):
        super().__init__()
        self.title("COLMAP Points3D Crop Tool  (V01.0)")
        self.geometry("1460x900")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        # Data
        self._points   = []           # full list of tuples
        self._xyz      = None         # np float32 (N,3) for rendering
        self._rgb      = None         # np uint8   (N,3)
        self._source_path = None      # Path of loaded file
        self._source_fmt  = None      # "bin" | "txt"

        # Graph zoom (numpad +/-)
        self._graph_zoom = 1.0        # 1.0 = default size

        # Y-axis orientation: True = +Y up the screen, False = -Y up
        self._y_up = tk.BooleanVar(value=False)

        # ── UI colour palette ──────────────────────────────────
        # Each entry: (label, attribute_name, current_value)
        # attribute_name is the key used in _apply_colour()
        self.COLOURS = {
            # name              default value
            "bg_dark":          "#0d0d1a",   # main background / canvas
            "bg_panel":         "#16213e",   # sidebars / toolbar
            "bg_input":         "#0f3460",   # entry / button backgrounds
            "accent_red":       "#e94560",   # primary accent (headings, buttons)
            "accent_cyan":      "#00b4d8",   # Y-axis / secondary accent
            "accent_light":     "#90e0ef",   # Z-axis / crop info text
            "text_main":        "#e0e0e0",   # general widget text
            "text_muted":       "#8892a4",   # labels / stats bar
            "text_dim":         "#556677",   # tick marks / dim labels
            "separator":        "#1e3a5f",   # divider lines
            "btn_green":        "#1a7a4a",   # export / save buttons
            "plot_bg":          "#0d0d1a",   # matplotlib axes background
        }

        self._build_ui()
        self._bind_zoom_keys()

    # ──────────────────────────────────────────────────────────
    #  Numpad zoom
    # ──────────────────────────────────────────────────────────

    def _bind_zoom_keys(self):
        """Bind numpad + / - to grow / shrink the graph canvas area."""
        self.bind_all("<KP_Add>",      self._zoom_in)
        self.bind_all("<KP_Subtract>", self._zoom_out)
        # Also support regular keyboard + / - in case numlock is off
        self.bind_all("<plus>",        self._zoom_in)
        self.bind_all("<minus>",       self._zoom_out)

    def _zoom_in(self, _event=None):
        self._graph_zoom = min(self._graph_zoom + 0.15, 4.0)
        self._apply_zoom()

    def _zoom_out(self, _event=None):
        self._graph_zoom = max(self._graph_zoom - 0.15, 0.3)
        self._apply_zoom()

    def _apply_zoom(self):
        """Zoom: scale the matplotlib figure DPI so content grows/shrinks within the canvas."""
        base_dpi = 80
        dpi = max(40, int(base_dpi * self._graph_zoom))
        self._fig.set_dpi(dpi)
        self._canvas.draw_idle()

    # ──────────────────────────────────────────────────────────
    #  UI construction
    # ──────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_toolbar()
        self._build_stats_bar()

        content = tk.Frame(self, bg="#0d0d1a")
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_left_sidebar(content)
        self._build_right_sidebar(content)
        self._build_canvas(content)
        self._build_colour_panel(content)   # builds hidden; toggle button in toolbar shows it

    # ── toolbar ──────────────────────────────────────────────

    def _build_toolbar(self):
        tb = tk.Frame(self, bg="#16213e", pady=6, padx=10)
        tb.pack(side=tk.TOP, fill=tk.X)

        tk.Label(tb, text="⬡  COLMAP Points3D Crop",
                 font=("Courier New", 13, "bold"),
                 fg="#e94560", bg="#16213e").pack(side=tk.LEFT, padx=(0, 16))

        def _btn(text, cmd, bg="#e94560", fg="white"):
            tk.Button(tb, text=text, command=cmd,
                      font=("Courier New", 10, "bold"),
                      bg=bg, fg=fg,
                      activebackground="#c73652", activeforeground="white",
                      relief=tk.FLAT, padx=10, pady=3, cursor="hand2"
                      ).pack(side=tk.LEFT, padx=4)

        _btn("📂  Open File", self._open_file)
        _btn("X  Export Crop", self._export_crop, bg="#1a7a4a")
        # _btn("✂  Export Crop", self._export_crop, bg="#1a7a4a")

        self._file_label = tk.Label(tb, text="No file loaded",
                                    font=("Courier New", 8),
                                    fg="#8892a4", bg="#16213e")
        self._file_label.pack(side=tk.LEFT, padx=8)

        def _vsep():
            tk.Frame(tb, bg="#1e3a5f", width=1, height=22
                     ).pack(side=tk.LEFT, padx=10, pady=2)

        _vsep()

        # Point size
        tk.Label(tb, text="Pt size:", font=("Courier New", 8),
                 fg="#8892a4", bg="#16213e").pack(side=tk.LEFT, padx=(0, 3))
        self._pt_size = tk.DoubleVar(value=5)
        tk.Scale(tb, from_=0.5, to=10, resolution=0.5,
                 orient=tk.HORIZONTAL, variable=self._pt_size,
                 length=100, bg="#16213e", fg="#e0e0e0",
                 troughcolor="#0f3460", highlightthickness=0,
                 command=lambda _: self._redraw()
                 ).pack(side=tk.LEFT)

        _vsep()

        # Max pts
        tk.Label(tb, text="Max pts:", font=("Courier New", 8),
                 fg="#8892a4", bg="#16213e").pack(side=tk.LEFT, padx=(0, 3))
        self._max_pts = tk.IntVar(value=200_000)
        tk.Entry(tb, textvariable=self._max_pts, width=8,
                 font=("Courier New", 9),
                 bg="#0f3460", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(tb, text="↺", command=self._redraw,
                  font=("Courier New", 9),
                  bg="#0f3460", fg="#e0e0e0",
                  relief=tk.FLAT, padx=5, pady=3, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=4)

        _vsep()

        # Color mode
        tk.Label(tb, text="Color:", font=("Courier New", 8),
                 fg="#8892a4", bg="#16213e").pack(side=tk.LEFT, padx=(0, 3))
        self._color_mode = tk.StringVar(value="RGB")
        cm = ttk.Combobox(tb, textvariable=self._color_mode,
                          values=["RGB", "Depth (Z)", "Height (Y)", "Uniform"],
                          state="readonly", width=11, font=("Courier New", 9))
        cm.pack(side=tk.LEFT)
        cm.bind("<<ComboboxSelected>>", lambda _: self._redraw())

        # Export format (right-aligned)
        tk.Label(tb, text="Export as:", font=("Courier New", 8),
                 fg="#8892a4", bg="#16213e").pack(side=tk.RIGHT, padx=(0, 3))
        self._export_fmt = tk.StringVar(value="same")
        ef = ttk.Combobox(tb, textvariable=self._export_fmt,
                          values=["same", "bin", "txt"],
                          state="readonly", width=6, font=("Courier New", 9))
        ef.pack(side=tk.RIGHT, padx=(0, 8))

        # Convention badge
        tk.Label(tb, text="Convention:", font=("Courier New", 8),
                 fg="#8892a4", bg="#16213e").pack(side=tk.RIGHT, padx=(0, 2))
        tk.Label(tb, text=" +Y↑  +X←  −Z▶ ",
                 font=("Courier New", 8, "bold"),
                 fg="#0d0d1a", bg="#00b4d8", padx=4, pady=2
                 ).pack(side=tk.RIGHT, padx=(0, 8))

        # Colour panel toggle
        self._colour_panel_visible = tk.BooleanVar(value=False)
        self._colour_toggle_btn = tk.Button(
            tb, text="🎨 Colours",
            command=self._toggle_colour_panel,
            font=("Courier New", 9),
            bg="#0f3460", fg="#8892a4",
            relief=tk.FLAT, padx=8, pady=3, cursor="hand2"
        )
        self._colour_toggle_btn.pack(side=tk.RIGHT, padx=(0, 8))

    # ── stats bar ────────────────────────────────────────────

    def _build_stats_bar(self):
        self._stats_var = tk.StringVar(
            value="Points: —   X: —   Y: —   Z: —")
        tk.Label(self, textvariable=self._stats_var,
                 font=("Courier New", 9), fg="#8892a4",
                 bg="#0d0d1a", anchor="w", padx=10
                 ).pack(side=tk.BOTTOM, fill=tk.X)

    # ── left sidebar: view presets ────────────────────────────

    def _build_left_sidebar(self, parent):
        left = tk.Frame(parent, bg="#16213e", width=110, padx=6, pady=12)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self._section_hdr(left, "VIEW")

        self._view_var = tk.StringVar(value="⟳  Free")
        for label in self._VIEWS:
            tk.Radiobutton(left, text=label,
                           variable=self._view_var, value=label,
                           command=self._apply_view,
                           font=("Courier New", 9),
                           fg="#c8d0e0", bg="#16213e",
                           selectcolor="#0f3460",
                           activebackground="#16213e",
                           activeforeground="#e94560",
                           anchor="w", relief=tk.FLAT, cursor="hand2"
                           ).pack(fill=tk.X, pady=2)

        tk.Frame(left, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=8)
        tk.Button(left, text="⌂ Reset View",
                  command=self._reset_view,
                  font=("Courier New", 8),
                  bg="#0f3460", fg="#e0e0e0",
                  relief=tk.FLAT, pady=3, cursor="hand2"
                  ).pack(fill=tk.X)

        # ── Y orientation toggle ──────────────────────────────
        tk.Frame(left, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=8)
        tk.Label(left, text="Y orientation",
                 font=("Courier New", 8, "bold"),
                 fg="#e94560", bg="#16213e", anchor="w"
                 ).pack(fill=tk.X, pady=(0, 4))

        y_row = tk.Frame(left, bg="#16213e")
        y_row.pack(fill=tk.X)

        self._btn_yup = tk.Button(
            y_row, text="-Y  ↑",
            command=lambda: self._set_y_orientation(True),
            font=("Courier New", 8, "bold"),
            bg="#e94560", fg="white",          # active style (default)
            relief=tk.FLAT, pady=3, cursor="hand2"
        )
        self._btn_yup.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self._btn_ydown = tk.Button(
            y_row, text="+Y  ↑",
            command=lambda: self._set_y_orientation(False),
            font=("Courier New", 8),
            bg="#0f3460", fg="#8892a4",        # inactive style
            relief=tk.FLAT, pady=3, cursor="hand2"
        )
        self._btn_ydown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    # ── right sidebar: clip + export ──────────────────────────

    def _build_right_sidebar(self, parent):
        right = tk.Frame(parent, bg="#16213e", width=180, padx=8, pady=12)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self._section_hdr(right, "CLIP / CROP")

        self._clip_vars = {}
        axes_cfg = [("X", "#e94560"), ("Y", "#00b4d8"), ("Z", "#90e0ef")]

        for axis, color in axes_cfg:
            tk.Label(right, text=f"── {axis} ──",
                     font=("Courier New", 8, "bold"),
                     fg=color, bg="#16213e", anchor="w"
                     ).pack(fill=tk.X, pady=(6, 2))

            row = tk.Frame(right, bg="#16213e")
            row.pack(fill=tk.X)

            for suffix in ("min", "max"):
                key = f"{axis.lower()}{suffix}"
                var = tk.StringVar(value="")
                self._clip_vars[key] = var

                cell = tk.Frame(row, bg="#16213e")
                cell.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

                tk.Label(cell, text=suffix, font=("Courier New", 7),
                         fg="#556677", bg="#16213e").pack(anchor="w")
                e = tk.Entry(cell, textvariable=var, width=9,
                             font=("Courier New", 9),
                             bg="#0f3460", fg="white",
                             insertbackground="white", relief=tk.FLAT)
                e.pack(fill=tk.X)
                e.bind("<Return>",   lambda _: self._redraw())
                e.bind("<FocusOut>", lambda _: self._redraw())

        tk.Frame(right, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=10)

        def _btn(text, cmd, bg="#0f3460", fg="#e0e0e0", bold=False):
            font = ("Courier New", 9, "bold") if bold else ("Courier New", 8)
            tk.Button(right, text=text, command=cmd,
                      font=font, bg=bg, fg=fg,
                      activebackground="#c73652", activeforeground="white",
                      relief=tk.FLAT, pady=4, cursor="hand2"
                      ).pack(fill=tk.X, pady=2)

        _btn("X  Apply Clip",       self._redraw,              bg="#e94560", fg="white", bold=True)
        _btn("⬤  Fill from Data",   self._fill_clip_from_data)
        _btn("📊 Histogram Picker",  self._show_histogram_window)
        _btn("📐 Use 99% Centre",    self._use_99_centre)
        _btn("✕  Clear Clip",        self._clear_clip)

        tk.Frame(right, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=8)

        _btn("💾  Save Clip JSON",   self._save_clip_json,  bg="#1a7a4a", fg="white")
        _btn("📂  Load Clip JSON",   self._load_clip_json)

        # Crop summary label
        self._crop_info = tk.StringVar(value="")
        tk.Label(right, textvariable=self._crop_info,
                 font=("Courier New", 8), fg="#90e0ef",
                 bg="#16213e", wraplength=190, justify="left"
                 ).pack(fill=tk.X, pady=(8, 0))

    # ── colour panel toggle ───────────────────────────────────

    def _toggle_colour_panel(self):
        if self._colour_panel_visible.get():
            self._colour_panel_frame.pack_forget()
            self._colour_panel_visible.set(False)
            self._colour_toggle_btn.config(bg="#0f3460", fg="#8892a4")
        else:
            self._colour_panel_frame.pack(side=tk.RIGHT, fill=tk.Y)
            self._colour_panel_visible.set(True)
            self._colour_toggle_btn.config(bg="#e94560", fg="white")

    # ── colour palette panel ──────────────────────────────────

    def _build_colour_panel(self, parent):
        """Panel listing every named UI colour; changes are staged until Apply is clicked."""
        QUICK = [
            ("Black",     "#000000"), ("White",    "#ffffff"),
            ("Dark Navy", "#0d0d1a"), ("Navy",     "#16213e"),
            ("Deep Blue", "#0f3460"), ("Red",      "#e94560"),
            ("Cyan",      "#00b4d8"), ("Lt Cyan",  "#90e0ef"),
            ("Green",     "#1a7a4a"), ("Lt Grey",  "#e0e0e0"),
            ("Mid Grey",  "#8892a4"), ("Dim Grey", "#556677"),
            ("Divider",   "#1e3a5f"),
        ]

        # Pending colour edits (key → new hex).  Not applied until Apply is clicked.
        self._pending_colours = {}

        panel = tk.Frame(parent, bg="#16213e", width=178, padx=8, pady=10)
        # Start hidden — toggled via 🎨 button in toolbar
        self._colour_panel_frame = panel
        panel.pack_propagate(False)
        # don't pack yet

        self._section_hdr(panel, "COLOURS")

        # Zoom hint
        tk.Label(panel,
                 text="Graph zoom:  Numpad  +  /  −",
                 font=("Courier New", 7), fg="#556677",
                 bg="#16213e", wraplength=158, justify="left"
                 ).pack(fill=tk.X, pady=(0, 4))

        tk.Frame(panel, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=(0, 6))

        # ── scrollable list of colour rows ───────────────────
        list_frame = tk.Frame(panel, bg="#16213e")
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._colour_swatches = {}   # key → (live_swatch, pending_swatch)

        for key, default_hex in self.COLOURS.items():
            row = tk.Frame(list_frame, bg="#16213e")
            row.pack(fill=tk.X, pady=1)

            # Live swatch (current applied colour)
            live_sw = tk.Label(row, width=2, bg=default_hex, relief=tk.FLAT)
            live_sw.pack(side=tk.LEFT, padx=(0, 2))

            # Pending swatch (shows chosen-but-not-yet-applied colour)
            pend_sw = tk.Label(row, width=2, bg=default_hex, relief=tk.SUNKEN)
            pend_sw.pack(side=tk.LEFT, padx=(0, 4))

            lbl_text = key.replace("_", " ")
            tk.Label(row, text=lbl_text,
                     font=("Courier New", 7), fg="#8892a4",
                     bg="#16213e", anchor="w"
                     ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            def _make_pick(k, psw):
                def _pick():
                    from tkinter.colorchooser import askcolor
                    cur = self._pending_colours.get(k, self.COLOURS[k])
                    result = askcolor(color=cur,
                                      title=f"Choose colour — {k.replace('_',' ')}",
                                      parent=self)
                    if result and result[1]:
                        h = result[1].lower()
                        self._pending_colours[k] = h
                        psw.config(bg=h)
                        self._apply_btn.config(state=tk.NORMAL,
                                               bg="#e94560", fg="white")
                return _pick

            tk.Button(row, text="✎", command=_make_pick(key, pend_sw),
                      font=("Courier New", 7),
                      bg="#0f3460", fg="#e0e0e0",
                      relief=tk.FLAT, padx=3, pady=0, cursor="hand2"
                      ).pack(side=tk.RIGHT)

            self._colour_swatches[key] = (live_sw, pend_sw)

        # ── Apply / Discard ───────────────────────────────────
        tk.Frame(panel, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=(6, 4))

        btn_row = tk.Frame(panel, bg="#16213e")
        btn_row.pack(fill=tk.X)

        self._apply_btn = tk.Button(
            btn_row, text="✔  Apply",
            command=self._apply_pending_colours,
            font=("Courier New", 8, "bold"),
            bg="#333344", fg="#556677",          # greyed out until a change is pending
            relief=tk.FLAT, pady=3, cursor="hand2", state=tk.DISABLED
        )
        self._apply_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        tk.Button(btn_row, text="✕",
                  command=self._discard_pending_colours,
                  font=("Courier New", 8),
                  bg="#0f3460", fg="#8892a4",
                  relief=tk.FLAT, pady=3, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(2, 0))

        # ── Quick-pick swatches ───────────────────────────────
        tk.Frame(panel, bg="#1e3a5f", height=1).pack(fill=tk.X, pady=(6, 4))
        tk.Label(panel, text="Quick colours  (click = copy hex):",
                 font=("Courier New", 7), fg="#556677",
                 bg="#16213e").pack(anchor="w")

        grid = tk.Frame(panel, bg="#16213e")
        grid.pack(fill=tk.X, pady=2)
        for col_idx, (name, hex_val) in enumerate(QUICK):
            sw = tk.Label(grid, bg=hex_val, width=2,
                          relief=tk.RAISED, cursor="hand2")
            sw.grid(row=col_idx // 4, column=col_idx % 4, padx=2, pady=2)
            sw.bind("<Button-1>", lambda e, h=hex_val: self._copy_hex_to_clipboard(h))
            self._add_tooltip(sw, f"{name}  {hex_val}")

    def _apply_pending_colours(self):
        """Commit all pending colour choices and rebuild the UI."""
        if not self._pending_colours:
            return

        # Merge pending into live dict
        self.COLOURS.update(self._pending_colours)

        # Update live swatches
        for key, (live_sw, pend_sw) in self._colour_swatches.items():
            live_sw.config(bg=self.COLOURS[key])
            pend_sw.config(bg=self.COLOURS[key])

        # Apply to matplotlib plot
        c = self.COLOURS
        try:
            self._fig.set_facecolor(c["bg_dark"])
            self._ax.set_facecolor(c["plot_bg"])
            self._ax.xaxis.label.set_color(c["accent_red"])
            self._ax.yaxis.label.set_color(c["accent_cyan"])
            self._ax.zaxis.label.set_color(c["accent_light"])
            self._ax.tick_params(colors=c["text_dim"])
            for pane in (self._ax.xaxis.pane,
                         self._ax.yaxis.pane,
                         self._ax.zaxis.pane):
                pane.set_edgecolor(c["separator"])
            self._ax.grid(True, color=c["separator"],
                          linewidth=0.4, linestyle="--")
            self._canvas.draw()
        except Exception:
            pass

        # Apply to main window background
        try:
            self.configure(bg=c["bg_dark"])
        except Exception:
            pass

        self._pending_colours.clear()
        self._apply_btn.config(state=tk.DISABLED,
                               bg="#333344", fg="#556677")

    def _discard_pending_colours(self):
        """Throw away pending picks and reset pending swatches to live colours."""
        self._pending_colours.clear()
        for key, (live_sw, pend_sw) in self._colour_swatches.items():
            pend_sw.config(bg=self.COLOURS[key])
        self._apply_btn.config(state=tk.DISABLED,
                               bg="#333344", fg="#556677")

    def _copy_hex_to_clipboard(self, hex_val):
        self.clipboard_clear()
        self.clipboard_append(hex_val)

    def _add_tooltip(self, widget, text):
        tip = None
        def _show(e):
            nonlocal tip
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 14
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            tk.Label(tip, text=text, font=("Courier New", 7),
                     bg="#16213e", fg="#e0e0e0",
                     relief=tk.SOLID, borderwidth=1, padx=4, pady=2
                     ).pack()
        def _hide(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind("<Enter>", _show)
        widget.bind("<Leave>", _hide)

    # ── 3-D canvas ───────────────────────────────────────────

    def _build_canvas(self, parent):
        self._fig = plt.Figure(figsize=(8, 8), facecolor="#0d0d1a")
        # Squeeze out the top/bottom dead space around the 3D axes
        self._fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        try:
            self._ax = self._fig.add_subplot(111, projection="3d",
                                             computed_zorder=False)
            self._ax.set_proj_type('persp')
        except TypeError:
            self._ax = self._fig.add_subplot(111, projection="3d")
        try:
            self._ax.view_init(elev=20, azim=-45, vertical_axis='y')
        except TypeError:
            self._ax.view_init(elev=20, azim=-45)

        cf = tk.Frame(parent, bg="#0d0d1a")
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._canvas_frame = cf          # keep ref for zoom resizing

        self._canvas = FigureCanvasTkAgg(self._fig, master=cf)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self._canvas, cf)
        nav.config(bg="#16213e")
        nav.update()

        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._draw_placeholder()

    # ──────────────────────────────────────────────────────────
    #  Small helpers
    # ──────────────────────────────────────────────────────────

    def _section_hdr(self, parent, text):
        tk.Label(parent, text=text, font=("Courier New", 9, "bold"),
                 fg="#e94560", bg="#16213e", anchor="w"
                 ).pack(fill=tk.X, pady=(0, 2))
        tk.Frame(parent, bg="#e94560", height=1).pack(fill=tk.X, pady=(0, 8))

    def _style_axes(self):
        ax = self._ax
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#556677", labelsize=7)
        ax.xaxis.label.set_color("#e94560")
        ax.yaxis.label.set_color("#00b4d8")
        ax.zaxis.label.set_color("#90e0ef")
        ax.set_xlabel("X  (+left)",  labelpad=6)
        y_label = "Y  (+up)" if self._y_up.get() else "−Y  (+up)"
        ax.set_ylabel(y_label, labelpad=6)
        ax.set_zlabel("Z  (−front)", labelpad=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1e3a5f")
        ax.grid(True, color="#1e3a5f", linewidth=0.4, linestyle="--")

    def _draw_placeholder(self):
        self._ax.text(0, 0, 0,
                      "Open a points3D.txt or\npoints3D.bin file to begin",
                      color="#8892a4", fontsize=11,
                      ha="center", va="center", family="Courier New")
        self._canvas.draw()

    # ──────────────────────────────────────────────────────────
    #  View controls
    # ──────────────────────────────────────────────────────────

    def _set_y_orientation(self, y_up: bool):
        """Switch between +Y up and −Y up and redraw (data negation only, no camera change)."""
        self._y_up.set(y_up)
        if y_up:
            self._btn_yup.config(bg="#e94560",  fg="white",   font=("Courier New", 8, "bold"))
            self._btn_ydown.config(bg="#0f3460", fg="#8892a4", font=("Courier New", 8))
        else:
            self._btn_yup.config(bg="#0f3460",  fg="#8892a4", font=("Courier New", 8))
            self._btn_ydown.config(bg="#e94560", fg="white",  font=("Courier New", 8, "bold"))
        self._redraw()

    def _view_init_with_orientation(self, elev, azim):
        """Call view_init — Y orientation is handled by data negation, not camera."""
        try:
            self._ax.view_init(elev=elev, azim=azim, vertical_axis='y')
        except TypeError:
            self._ax.view_init(elev=elev, azim=azim)

    def _apply_view(self, *_):
        elev, azim = self._VIEWS[self._view_var.get()]
        self._view_init_with_orientation(elev, azim)
        self._canvas.draw()

    def _reset_view(self):
        self._view_var.set("⟳  Free")
        elev, azim = self._VIEWS["⟳  Free"]
        self._view_init_with_orientation(elev, azim)
        self._canvas.draw()

    def _on_mouse_release(self, event):
        if event.button == 1:
            self._view_var.set("⟳  Free")

    # ──────────────────────────────────────────────────────────
    #  Clip helpers
    # ──────────────────────────────────────────────────────────

    def _get_clip_bounds(self):
        INF = float("inf")
        v = self._clip_vars
        return (
            _entry_float(v["xmin"], -INF), _entry_float(v["xmax"],  INF),
            _entry_float(v["ymin"], -INF), _entry_float(v["ymax"],  INF),
            _entry_float(v["zmin"], -INF), _entry_float(v["zmax"],  INF),
        )

    def _clear_clip(self):
        for var in self._clip_vars.values():
            var.set("")
        self._crop_info.set("")
        self._redraw()

    def _fill_clip_from_data(self):
        if self._xyz is None:
            return
        x, y, z = self._xyz[:, 0], self._xyz[:, 1], self._xyz[:, 2]
        self._clip_vars["xmin"].set(f"{float(x.min()):.4f}")
        self._clip_vars["xmax"].set(f"{float(x.max()):.4f}")
        self._clip_vars["ymin"].set(f"{float(y.min()):.4f}")
        self._clip_vars["ymax"].set(f"{float(y.max()):.4f}")
        self._clip_vars["zmin"].set(f"{float(z.min()):.4f}")
        self._clip_vars["zmax"].set(f"{float(z.max()):.4f}")
        self._redraw()

    def _use_99_centre(self):
        """Fill clip bounds using the 0.5–99.5 percentile range."""
        if self._xyz is None:
            return
        x, y, z = self._xyz[:, 0], self._xyz[:, 1], self._xyz[:, 2]
        for coord, key_lo, key_hi in [
            (x, "xmin", "xmax"),
            (y, "ymin", "ymax"),
            (z, "zmin", "zmax"),
        ]:
            lo = float(np.percentile(coord, 0.5))
            hi = float(np.percentile(coord, 99.5))
            self._clip_vars[key_lo].set(f"{lo:.4f}")
            self._clip_vars[key_hi].set(f"{hi:.4f}")
        self._redraw()

    # ──────────────────────────────────────────────────────────
    #  Histogram picker window
    # ──────────────────────────────────────────────────────────

    def _show_histogram_window(self):
        if self._xyz is None:
            messagebox.showinfo("No Data", "Load a file first.")
            return

        win = tk.Toplevel(self)
        win.title("Histogram Bound Picker")
        win.configure(bg="#16213e")
        win.geometry("800x520")

        # Controls row
        ctrl = tk.Frame(win, bg="#16213e", pady=6, padx=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        axis_var = tk.StringVar(value="X")
        for ax_name in ("X", "Y", "Z"):
            tk.Radiobutton(ctrl, text=ax_name, variable=axis_var, value=ax_name,
                           font=("Courier New", 10, "bold"),
                           fg="#e94560", bg="#16213e", selectcolor="#0f3460",
                           activebackground="#16213e",
                           command=lambda: _draw_hist()
                           ).pack(side=tk.LEFT, padx=6)

        bins_var = tk.IntVar(value=100)
        tk.Label(ctrl, text="  Bins:", font=("Courier New", 9),
                 fg="#8892a4", bg="#16213e").pack(side=tk.LEFT, padx=(20, 4))
        tk.Entry(ctrl, textvariable=bins_var, width=6,
                 font=("Courier New", 9),
                 bg="#0f3460", fg="white", insertbackground="white",
                 relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(ctrl, text="↺ Refresh", command=lambda: _draw_hist(),
                  font=("Courier New", 8),
                  bg="#0f3460", fg="#e0e0e0",
                  relief=tk.FLAT, padx=6, pady=2, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=8)

        # Instructions
        tk.Label(ctrl,
                 text="Left-click = set min  |  Right-click = set max  (applies to main clip panel)",
                 font=("Courier New", 8), fg="#556677", bg="#16213e"
                 ).pack(side=tk.LEFT, padx=12)

        # Matplotlib figure
        fig = plt.Figure(figsize=(7, 4), facecolor="#0d0d1a")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#556677")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3a5f")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        def _draw_hist():
            ax.cla()
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="#556677", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#1e3a5f")

            a = axis_var.get()
            col = {"X": 0, "Y": 1, "Z": 2}[a]
            data = self._xyz[:, col]
            color = {"X": "#e94560", "Y": "#00b4d8", "Z": "#90e0ef"}[a]

            ax.hist(data, bins=bins_var.get(), color=color, alpha=0.7, edgecolor="none")

            # Draw existing clip lines
            lo_key = f"{a.lower()}min"
            hi_key = f"{a.lower()}max"
            lo = _entry_float(self._clip_vars[lo_key], None)
            hi = _entry_float(self._clip_vars[hi_key], None)
            if lo is not None:
                ax.axvline(lo, color="#ffffff", linewidth=1.2, linestyle="--", label="min")
            if hi is not None:
                ax.axvline(hi, color="#ffa500", linewidth=1.2, linestyle="--", label="max")
            if lo is not None or hi is not None:
                ax.legend(fontsize=7, facecolor="#16213e", labelcolor="#e0e0e0")

            ax.set_title(f"{a} distribution  ({len(data):,} pts)",
                         color="#8892a4", fontsize=9, fontfamily="Courier New")
            ax.set_ylabel("count", color="#556677", fontsize=8)
            ax.grid(True, color="#1e3a5f", linewidth=0.4, linestyle="--")
            canvas.draw()

        def _on_click(event):
            if event.inaxes != ax or event.xdata is None:
                return
            a  = axis_var.get().lower()
            val = f"{event.xdata:.4f}"
            if event.button == 1:
                self._clip_vars[f"{a}min"].set(val)
            elif event.button == 3:
                self._clip_vars[f"{a}max"].set(val)
            _draw_hist()
            self._redraw()

        canvas.mpl_connect("button_press_event", _on_click)
        _draw_hist()

    # ──────────────────────────────────────────────────────────
    #  JSON clip save / load
    # ──────────────────────────────────────────────────────────

    def _save_clip_json(self):
        INF = float("inf")
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_clip_bounds()

        def _v(val, is_min):
            ref = -INF if is_min else INF
            return None if val == ref else float(val)

        data = {
            "coordinate_convention": "+Y up, +X left, -Z front",
            "clip": {
                "xmin": _v(xmin, True),  "xmax": _v(xmax, False),
                "ymin": _v(ymin, True),  "ymax": _v(ymax, False),
                "zmin": _v(zmin, True),  "zmax": _v(zmax, False),
            }
        }
        path = filedialog.asksaveasfilename(
            title="Save clip bounds as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        messagebox.showinfo("Saved", f"Clip bounds saved to:\n{path}")

    def _load_clip_json(self):
        path = filedialog.askopenfilename(
            title="Load clip bounds JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            clip = data.get("clip", data)
            for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                val = clip.get(key)
                self._clip_vars[key].set("" if val is None else str(val))
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not parse JSON:\n{e}")

    # ──────────────────────────────────────────────────────────
    #  File I/O
    # ──────────────────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open COLMAP Points3D file",
            filetypes=[
                ("COLMAP Points3D", "*.txt *.bin"),
                ("Text files",      "*.txt"),
                ("Binary files",    "*.bin"),
                ("All files",       "*.*"),
            ]
        )
        if not path:
            return
        self._load_file(path)

    def _load_file(self, path):
        p = Path(path)
        fmt = "bin" if p.suffix.lower() == ".bin" else "txt"
        try:
            if fmt == "bin":
                pts = read_points3d_bin(path)
            else:
                pts = read_points3d_txt(path)
        except Exception as e:
            messagebox.showerror("Parse Error", f"Could not read file:\n{e}")
            return
        if not pts:
            messagebox.showwarning("Empty File", "No 3D points found.")
            return

        self._points     = pts
        self._source_path = p
        self._source_fmt  = fmt
        self._export_fmt.set("same")

        # Build numpy arrays for fast rendering
        self._xyz = np.array([[r[1], r[2], r[3]] for r in pts], dtype=np.float32)
        self._rgb = np.array([[r[4], r[5], r[6]] for r in pts], dtype=np.uint8)

        self._file_label.config(
            text=f"{p.name}  ({len(pts):,} pts)  [{fmt.upper()}]"
        )
        self._update_stats()
        self._redraw()

    # ──────────────────────────────────────────────────────────
    #  Export crop
    # ──────────────────────────────────────────────────────────

    def _export_crop(self):
        if not self._points:
            messagebox.showinfo("No Data", "Load a file first.")
            return

        # Determine format
        fmt_choice = self._export_fmt.get()
        if fmt_choice == "same":
            out_fmt = self._source_fmt
        else:
            out_fmt = fmt_choice   # "bin" or "txt"

        ext        = ".bin" if out_fmt == "bin" else ".txt"
        fmt_label  = out_fmt.upper()

        # Suggest a default output name
        if self._source_path:
            default_name = self._source_path.stem + "_cropped" + ext
            default_dir  = str(self._source_path.parent)
        else:
            default_name = "points3D_cropped" + ext
            default_dir  = ""

        path = filedialog.asksaveasfilename(
            title=f"Export cropped Points3D as {fmt_label}",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=ext,
            filetypes=[
                ("COLMAP Binary", "*.bin"),
                ("COLMAP Text",   "*.txt"),
                ("All files",     "*.*"),
            ]
        )
        if not path:
            return

        # Build crop mask
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_clip_bounds()
        cropped = [
            rec for rec in self._points
            if xmin <= rec[1] <= xmax
            and ymin <= rec[2] <= ymax
            and zmin <= rec[3] <= zmax
        ]

        if not cropped:
            messagebox.showwarning(
                "Empty Crop",
                "No points fall within the current clip bounds.\n"
                "Adjust the clip values and try again."
            )
            return

        try:
            if out_fmt == "bin":
                write_points3d_bin(cropped, path)
            else:
                write_points3d_txt(cropped, path)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not write file:\n{e}")
            return

        total = len(self._points)
        kept  = len(cropped)
        pct   = 100.0 * kept / total if total else 0
        messagebox.showinfo(
            "Export Complete",
            f"Saved {kept:,} / {total:,} points  ({pct:.1f}%)\n"
            f"Format : {fmt_label}\n"
            f"Output : {path}"
        )
        self._crop_info.set(
            f"Last export:\n{kept:,}/{total:,} pts ({pct:.1f}%)\n→ {Path(path).name}"
        )

    # ──────────────────────────────────────────────────────────
    #  Rendering
    # ──────────────────────────────────────────────────────────

    def _redraw(self, *_):
        if self._xyz is None:
            return

        xyz = self._xyz
        rgb = self._rgb

        # Apply clip mask
        xmin, xmax, ymin, ymax, zmin, zmax = self._get_clip_bounds()
        mask = (
            (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
        )
        xyz_c = xyz[mask].copy()
        rgb_c = rgb[mask]
        clipped_n = len(xyz_c)

        # Negate Y if −Y up is selected
        if not self._y_up.get():
            xyz_c[:, 1] = -xyz_c[:, 1]

        # Subsample
        max_n = max(1, self._max_pts.get())
        if clipped_n > max_n:
            idx   = np.random.choice(clipped_n, max_n, replace=False)
            xyz_c = xyz_c[idx]
            rgb_c = rgb_c[idx]

        # Colours
        mode = self._color_mode.get()
        if mode == "RGB":
            colors = rgb_c.astype(np.float32) / 255.0
        elif mode == "Depth (Z)":
            z = xyz_c[:, 2]
            z_norm = (z - z.min()) / (np.ptp(z) + 1e-9)
            colors = plt.cm.plasma(z_norm)[:, :3]
        elif mode == "Height (Y)":
            y = xyz_c[:, 1]
            y_norm = (y - y.min()) / (np.ptp(y) + 1e-9)
            colors = plt.cm.viridis(y_norm)[:, :3]
        else:
            colors = "#00b4d8"

        self._ax.cla()
        self._style_axes()

        if clipped_n == 0:
            self._ax.text(0, 0, 0, "No points in clip region",
                          color="#e94560", fontsize=10,
                          ha="center", va="center", family="Courier New")
        else:
            self._ax.scatter(
                xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2],
                c=colors, s=self._pt_size.get(),
                linewidths=0, alpha=0.85, depthshade=True
            )

        INF = float("inf")
        is_clipped = any(v != b for v, b in zip(
            [xmin, xmax, ymin, ymax, zmin, zmax],
            [-INF, INF, -INF, INF, -INF, INF]))
        clip_tag = "  ✂ clipped" if is_clipped else ""

        # Draw info text inside the axes (top-left corner) — no external title padding
        info = f"showing {len(xyz_c):,} / {clipped_n:,} pts  |  {mode}{clip_tag}"
        self._ax.text2D(0.01, 0.99, info,
                        transform=self._ax.transAxes,
                        color="#8892a4", fontsize=8,
                        fontfamily="Courier New",
                        va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#0d0d1a", edgecolor="#1e3a5f",
                                  alpha=0.7))
        self._canvas.draw()

    def _update_stats(self):
        if self._xyz is None:
            return
        x, y, z = self._xyz[:, 0], self._xyz[:, 1], self._xyz[:, 2]
        fmt_tag = f"  [{self._source_fmt.upper()}]" if self._source_fmt else ""
        self._stats_var.set(
            f"Total pts: {len(self._xyz):,}{fmt_tag}   "
            f"X: [{float(x.min()):.3f}, {float(x.max()):.3f}]   "
            f"Y: [{float(y.min()):.3f}, {float(y.max()):.3f}]   "
            f"Z: [{float(z.min()):.3f}, {float(z.max()):.3f}]"
        )


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ColmapCropGUI()
    app.mainloop()
