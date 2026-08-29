#!/usr/bin/env python3
"""
COLMAP → LichtFeld Studio Camera Path Converter
Drag-and-drop any of the following onto this window (mix and match):
  - the cameras/images .txt or .bin files themselves
  - the "sparse/0" folder that contains them
  - the top-level COLMAP project folder (searched recursively, bounded
    depth, so any layout — sparse/0, distorted/sparse/0, colmap/sparse/0,
    etc. — is found automatically)

Output: <same folder>/<stem>_lfs.json  ready to load into LFS sequencer.

COLMAP conventions handled:
  - cameras.txt / cameras.bin  (SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV …)
  - images.txt  / images.bin   (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME)

Coordinate conversion:
  COLMAP  →  LFS (SuperSplat)
  +X right    +X right
  +Y down      +Y up      ← flipped
  +Z forward   +Z forward (SuperSplat +Z = camera forward)

The COLMAP image quaternion (qw,qx,qy,qz) rotates world→camera.
LFS wants a quaternion that rotates the default camera orientation to the
camera's world pose, i.e. the camera-to-world rotation expressed in the
LFS (+Y-up) coordinate frame.

Steps per image:
  1. Invert COLMAP R_cw  →  R_wc  (world-to-camera → camera-to-world)
  2. Convert position:  t_world = -R_cw^T · t_colmap
  3. Flip Y axis: pos_lfs = (x, -y, z)
  4. Build R_lfs from R_wc with Y-flip: negate rows/cols touching Y
  5. Roll +90° about the camera's own forward (camera→target) axis, to
     match LFS's camera orientation convention (adjustable, default 90°)
  6. Convert R_lfs to quaternion  (SuperSplat +Z-forward convention)

Timeline: images are sorted by filename (natural sort), spaced by
user-supplied FPS or duration.
"""

import json
import math
import os
import re
import struct
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ── COLMAP binary readers ─────────────────────────────────────────────────────

def _read_cameras_bin(path: Path) -> dict:
    """Return {camera_id: {model, width, height, params[...]}}"""
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<IiQQ", f.read(24))
            # param counts per model_id
            nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 8, 7: 5}.get(model_id, 4)
            params = list(struct.unpack(f"<{nparams}d", f.read(8 * nparams)))
            cameras[cam_id] = {"model_id": model_id, "width": w, "height": h, "params": params}
    return cameras


def _read_cameras_txt(path: Path) -> dict:
    cameras = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            w, h   = int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {"model": model, "width": w, "height": h, "params": params}
    return cameras


def _read_images_bin(path: Path) -> list:
    """Return list of dicts sorted by name."""
    images = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz, tx, ty, tz = struct.unpack("<7d", f.read(56))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # skip 2D points
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images.append({"id": img_id, "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                           "tx": tx, "ty": ty, "tz": tz, "cam_id": cam_id, "name": name})
    return images


def _read_images_txt(path: Path) -> list:
    images = []
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz      = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name   = parts[9]
        images.append({"id": img_id, "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                       "tx": tx, "ty": ty, "tz": tz, "cam_id": cam_id, "name": name})
        i += 2  # skip the 2D-point line
    return images


# ── Quaternion / matrix helpers ───────────────────────────────────────────────

def _quat_to_matrix(qw, qx, qy, qz):
    """Unit quaternion → 3×3 rotation matrix (row-major list-of-lists)."""
    return [
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ]


def _mat_transpose(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def _mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def _mat_mul(a, b):
    """3×3 matrix multiply: a·b (row-major)."""
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rot_z(deg):
    """Rotation matrix about the LOCAL +Z axis (camera-to-target / forward axis), in degrees."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [[c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]]


def _matrix_to_quat(m):
    """3×3 rotation matrix → (qw, qx, qy, qz), SuperSplat +Z-forward convention."""
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2][1] - m[1][2]) * s
        qy = (m[0][2] - m[2][0]) * s
        qz = (m[1][0] - m[0][1]) * s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        qw = (m[2][1] - m[1][2]) / s; qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s; qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        qw = (m[0][2] - m[2][0]) / s; qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s;                 qz = (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        qw = (m[1][0] - m[0][1]) / s; qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s; qz = 0.25 * s
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    return qw/n, qx/n, qy/n, qz/n


def _focal_to_mm(focal_px, sensor_px, sensor_mm=36.0):
    """Convert focal length in pixels to mm given sensor pixel width."""
    return focal_px * sensor_mm / sensor_px


# ── Core conversion ───────────────────────────────────────────────────────────

def _natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def convert(images_path: Path, cameras_path: Path,
            fps: float, sensor_mm: float, focal_override_mm: float | None,
            scale: float, roll_deg: float = 90.0) -> dict:
    """
    Convert COLMAP sparse model to LFS camera path JSON.
    Returns the dict ready for json.dump.

    roll_deg: extra rotation (degrees) applied about the camera's own
              forward axis (the camera→target viewing direction) after
              the COLMAP→LFS conversion. LFS cameras come out rolled
              -90° relative to COLMAP's convention, so the default here
              is +90° to compensate. Set to 0 to disable.
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    if cameras_path.suffix == ".bin":
        cameras = _read_cameras_bin(cameras_path)
    else:
        cameras = _read_cameras_txt(cameras_path)

    if images_path.suffix == ".bin":
        images = _read_images_bin(images_path)
    else:
        images = _read_images_txt(images_path)

    if not images:
        raise ValueError("No images found in the images file.")

    # Sort by filename (natural sort so frame_1 < frame_2 < … < frame_10)
    images.sort(key=lambda im: _natural_key(im["name"]))

    keyframes = []
    for idx, im in enumerate(images):
        time_sec = round(idx / fps, 6)

        # ── Focal length ───────────────────────────────────────────────────────
        cam = cameras.get(im["cam_id"], cameras.get(list(cameras.keys())[0]))
        params = cam["params"]
        # params[0] is always fx (pixels)
        focal_px = params[0]
        w = cam["width"]
        if focal_override_mm is not None:
            fl_mm = focal_override_mm
        else:
            fl_mm = round(_focal_to_mm(focal_px, w, sensor_mm), 4)

        # ── COLMAP world-to-camera rotation & translation ──────────────────────
        qw, qx, qy, qz = im["qw"], im["qx"], im["qy"], im["qz"]
        tx, ty, tz      = im["tx"], im["ty"], im["tz"]

        # R_cw: world→camera
        R_cw = _quat_to_matrix(qw, qx, qy, qz)

        # Camera position in world coords: t_w = -R_cw^T · t_colmap
        R_wc = _mat_transpose(R_cw)
        t_colmap = [tx, ty, tz]
        t_world  = [-sum(R_wc[i][j] * t_colmap[j] for j in range(3)) for i in range(3)]

        # ── COLMAP (+Y-down) → LFS (+Y-up) ────────────────────────────────────
        # Position: flip Y
        px =  t_world[0] * scale
        py = -t_world[1] * scale
        pz = -t_world[2] * scale

        # Rotation conversion COLMAP → SuperSplat (+Z-forward, +Y-up):
        # R_wc columns are camera axes in COLMAP world space (+Y down).
        # Convert each column vector to LFS world (+Y up) by flipping its Y component.
        # COLMAP camera +Z = backward, so LFS forward col = -R_wc col2.
        # Per element: row_sign = +1 for rows 0,2; -1 for row 1 (Y-flip).
        #   col0 (right)   =  row_sign * R_wc[row][0]
        #   col1 (up)      =  row_sign * R_wc[row][1]
        #   col2 (forward) = -row_sign * R_wc[row][2]  (negate for COLMAP -Z→+Z)
        def _to_lfs_rot(m):
            r = [[0.0]*3 for _ in range(3)]
            for row in range(3):
                sy = -1 if row == 1 else 1
                r[row][0] =  sy * m[row][0]
                r[row][1] =  sy * m[row][1]
                r[row][2] = -sy * m[row][2]
            return r

        R_lfs = _to_lfs_rot(R_wc)

        # Roll the camera about its own forward axis (the camera→target
        # line). Post-multiplying applies the rotation in the camera's
        # LOCAL frame, i.e. it spins the view around its own look direction
        # rather than around a world axis.
        if roll_deg:
            R_lfs = _mat_mul(R_lfs, _rot_z(roll_deg))

        rqw, rqx, rqy, rqz = _matrix_to_quat(R_lfs)

        keyframes.append({
            "easing": 0,
            "focal_length_mm": fl_mm,
            "position": [round(px, 6), round(py, 6), round(pz, 6)],
            "rotation": [round(rqw, 6), round(rqx, 6), round(rqy, 6), round(rqz, 6)],
            "time": time_sec,
        })

    return {"keyframes": keyframes, "version": 3}


# ── File discovery ────────────────────────────────────────────────────────────

def _find_in_tree(d: Path, stem: str, max_depth: int = 5):
    """
    Search a directory tree (bounded depth) for '<stem>.bin' first, then
    '<stem>.txt', preferring the shallowest match. Handles any COLMAP
    layout — e.g. project/sparse/0/, project/distorted/sparse/0/,
    project/colmap/sparse/0/, or the files sitting right at the root.
    Returns a Path or None.
    """
    for ext in (".bin", ".txt"):
        target = f"{stem}{ext}"
        best = None  # (depth, path)
        for root, dirs, files in os.walk(d):
            depth = len(Path(root).relative_to(d).parts)
            if depth > max_depth:
                dirs[:] = []  # don't descend further (e.g. into huge image dirs)
                continue
            if target in files:
                candidate = Path(root) / target
                if best is None or depth < best[0]:
                    best = (depth, candidate)
        if best is not None:
            return best[1]
    return None


def _find_colmap_files(paths: list[str]):
    """
    Given a list of dropped paths (files or folders — the COLMAP project
    folder, the 'sparse/0' folder, or the cameras/images files themselves,
    in any combination), find:
      - images file  (.txt or .bin)
      - cameras file (.txt or .bin)
    Returns (images_path, cameras_path, search_dir) or raises FileNotFoundError.
    """
    search_dirs = []
    explicit_images  = None
    explicit_cameras = None

    for p in paths:
        p = Path(p)
        if p.is_dir():
            if p not in search_dirs:
                search_dirs.append(p)
        elif p.name in ("images.txt", "images.bin"):
            explicit_images = p
            if p.parent not in search_dirs:
                search_dirs.append(p.parent)
        elif p.name in ("cameras.txt", "cameras.bin"):
            explicit_cameras = p
            if p.parent not in search_dirs:
                search_dirs.append(p.parent)
        else:
            if p.parent not in search_dirs:
                search_dirs.append(p.parent)

    # Search each dropped/implied directory (bounded-depth recursive walk)
    for d in search_dirs:
        if explicit_images is None:
            explicit_images = _find_in_tree(d, "images")
        if explicit_cameras is None:
            explicit_cameras = _find_in_tree(d, "cameras")
        if explicit_images and explicit_cameras:
            break

    if explicit_images is None:
        raise FileNotFoundError("Could not find images.txt or images.bin")
    if explicit_cameras is None:
        raise FileNotFoundError("Could not find cameras.txt or cameras.bin")

    return explicit_images, explicit_cameras, explicit_images.parent


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("COLMAP → LichtFeld Studio Converter")
        self.resizable(False, False)
        self.configure(bg="#1e2130")

        # ── colour palette ────────────────────────────────────────────────────
        BG   = "#1e2130"
        BG2  = "#262a3d"
        ACC  = "#7fb8ff"
        FG   = "#d0d6e8"
        DIM  = "#6a7190"
        BTN  = "#2e3450"
        BTNH = "#3c4466"

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel",      background=BG,  foreground=FG,  font=("Segoe UI", 10))
        style.configure("Dim.TLabel",  background=BG,  foreground=DIM, font=("Segoe UI", 9))
        style.configure("Head.TLabel", background=BG,  foreground=ACC, font=("Segoe UI", 10, "bold"))
        style.configure("TFrame",      background=BG)
        style.configure("Card.TFrame", background=BG2)
        style.configure("TEntry",      fieldbackground=BG2, foreground=FG,
                        insertcolor=FG, bordercolor=DIM, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TButton",  background=BTN,  foreground=FG,  font=("Segoe UI", 10),
                        borderwidth=0, focusthickness=0, padding=6)
        style.map("TButton",
                  background=[("active", BTNH)],
                  foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background=ACC, foreground="#0d1120",
                        font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Accent.TButton",
                  background=[("active", "#a8d0ff")],
                  foreground=[("active", "#0d1120")])

        self.configure(bg=BG)

        root = ttk.Frame(self, padding=20); root.pack(fill="both", expand=True)

        # Header
        ttk.Label(root, text="COLMAP  →  LichtFeld Studio", style="Head.TLabel",
                  font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(root, text="Camera Path Converter", style="Dim.TLabel").pack(anchor="w")
        ttk.Separator(root).pack(fill="x", pady=10)

        # ── Drop zone ─────────────────────────────────────────────────────────
        self._drop_var = tk.StringVar(
            value="Drop the COLMAP project folder, or the cameras/images .txt / .bin files, here")
        drop_card = tk.Frame(root, bg=BG2, bd=0, highlightthickness=2,
                             highlightbackground="#3c4466", cursor="hand2")
        drop_card.pack(fill="x", pady=(0, 12))
        self._drop_label = tk.Label(
            drop_card, textvariable=self._drop_var,
            bg=BG2, fg=ACC, font=("Segoe UI", 10, "italic"),
            padx=20, pady=22, wraplength=460, justify="center"
        )
        self._drop_label.pack(fill="both")

        # Make drop zone work
        drop_card.drop_target_register = lambda *a: None  # fallback
        try:
            self._enable_dnd(drop_card)
        except Exception:
            pass

        browse_row = ttk.Frame(root); browse_row.pack(fill="x", pady=(0, 10))
        ttk.Button(browse_row, text="Browse images file…",
                   command=self._browse_images).pack(side="left", padx=(0, 6))
        ttk.Button(browse_row, text="Browse cameras file…",
                   command=self._browse_cameras).pack(side="left", padx=(0, 6))
        ttk.Button(browse_row, text="Browse folder…",
                   command=self._browse_folder).pack(side="left")

        self._images_path:  Path | None = None
        self._cameras_path: Path | None = None

        # ── Options ───────────────────────────────────────────────────────────
        opt = ttk.Frame(root); opt.pack(fill="x", pady=(4, 0))
        opt.columnconfigure(1, weight=1)
        opt.columnconfigure(3, weight=1)

        def row(parent, r, label, var, tooltip=None, col=0):
            ttk.Label(parent, text=label).grid(row=r, column=col,   sticky="w", pady=3, padx=(0,6))
            e = ttk.Entry(parent, textvariable=var, width=10)
            e.grid(row=r, column=col+1, sticky="w", pady=3, padx=(0,20))

        self._fps_var    = tk.StringVar(value="24")
        self._sensor_var = tk.StringVar(value="36")
        self._focal_var  = tk.StringVar(value="")   # blank = derive from COLMAP
        self._scale_var  = tk.StringVar(value="1.0")
        self._roll_var   = tk.StringVar(value="90")  # camera roll about its own forward axis

        row(opt, 0, "FPS:",           self._fps_var,    col=0)
        row(opt, 0, "Sensor (mm):",   self._sensor_var, col=2)
        row(opt, 1, "Scale:",         self._scale_var,  col=0)
        row(opt, 1, "Focal override (mm):", self._focal_var, col=2)
        row(opt, 2, "Roll (deg):",    self._roll_var,   col=0)

        ttk.Label(opt, text="Leave Focal blank to derive from COLMAP intrinsics",
                  style="Dim.TLabel").grid(row=3, column=0, columnspan=4, sticky="w", pady=(2,0))
        ttk.Label(opt, text="Roll rotates each camera about its own camera→target axis (LFS default needs +90°)",
                  style="Dim.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", pady=(2,0))

        ttk.Separator(root).pack(fill="x", pady=10)

        # ── Convert button & status ───────────────────────────────────────────
        ttk.Button(root, text="Convert →  LFS JSON", style="Accent.TButton",
                   command=self._convert).pack(fill="x")

        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(root, textvariable=self._status_var,
                                    bg=BG, fg="#4ecf7e", font=("Segoe UI", 9),
                                    wraplength=480, justify="left")
        self._status_lbl.pack(anchor="w", pady=(8, 0))

        self.geometry("520x460")
        self.update_idletasks()

    # ── DnD ───────────────────────────────────────────────────────────────────
    def _enable_dnd(self, widget):
        """Try tkinterdnd2 first, fall back to Tk built-in wm_protocol drag."""
        try:
            import tkinterdnd2 as dnd
            # If tkinterdnd2 is available use it
            self.drop_target_register = widget.drop_target_register
            widget.drop_target_register(dnd.DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)
        except ImportError:
            # Pure-Tk fallback: bind to the window's WM_DROP_FILES
            try:
                self.tk.call("package", "require", "tkdnd")
                widget.tk.call("::tkdnd::drag_target_register", widget, "*")
                widget.bind("<<Drop>>", self._on_drop_tk)
            except Exception:
                # No DnD available — user must use Browse buttons
                pass

    def _on_drop(self, event):
        raw = event.data
        # tkinterdnd2 wraps paths with spaces in braces
        paths = self.tk.splitlist(raw)
        self._load_paths(list(paths))

    def _on_drop_tk(self, event):
        self._load_paths(self.tk.splitlist(event.data))

    def _load_paths(self, paths: list[str]):
        try:
            img, cam, folder = _find_colmap_files(paths)
            self._images_path  = img
            self._cameras_path = cam
            self._drop_var.set(f"✓  {img.name}  +  {cam.name}\n{folder}")
            self._drop_label.config(fg="#4ecf7e", font=("Segoe UI", 10))
        except FileNotFoundError as e:
            self._drop_var.set(f"⚠  {e}")
            self._drop_label.config(fg="#ff6b6b", font=("Segoe UI", 10, "italic"))

    # ── Browse helpers ────────────────────────────────────────────────────────
    def _browse_images(self):
        p = filedialog.askopenfilename(
            title="Select images.txt or images.bin",
            filetypes=[("COLMAP images", "images.txt images.bin"), ("All", "*.*")]
        )
        if p:
            self._images_path = Path(p)
            self._try_auto_pair()

    def _browse_cameras(self):
        p = filedialog.askopenfilename(
            title="Select cameras.txt or cameras.bin",
            filetypes=[("COLMAP cameras", "cameras.txt cameras.bin"), ("All", "*.*")]
        )
        if p:
            self._cameras_path = Path(p)
            self._try_auto_pair()

    def _browse_folder(self):
        d = filedialog.askdirectory(title="Select COLMAP sparse folder")
        if d:
            self._load_paths([d])

    def _try_auto_pair(self):
        """After one file is chosen try to find its partner in the same dir."""
        for attr, other_attr, candidates in [
            ("_images_path",  "_cameras_path", ("cameras.bin", "cameras.txt")),
            ("_cameras_path", "_images_path",  ("images.bin",  "images.txt")),
        ]:
            p = getattr(self, attr)
            if p and not getattr(self, other_attr):
                for name in candidates:
                    candidate = p.parent / name
                    if candidate.exists():
                        setattr(self, other_attr, candidate)
                        break

        img = self._images_path
        cam = self._cameras_path
        if img and cam:
            self._drop_var.set(f"✓  {img.name}  +  {cam.name}\n{img.parent}")
            self._drop_label.config(fg="#4ecf7e", font=("Segoe UI", 10))
        elif img:
            self._drop_var.set(f"images: {img.name}  (cameras not found yet)")
            self._drop_label.config(fg=ACC, font=("Segoe UI", 10, "italic"))

    # ── Conversion ────────────────────────────────────────────────────────────
    def _status(self, msg, ok=True):
        self._status_var.set(msg)
        self._status_lbl.config(fg="#4ecf7e" if ok else "#ff6b6b")

    def _convert(self):
        if not self._images_path or not self._cameras_path:
            self._status("⚠  Please select images and cameras files first.", False)
            return
        try:
            fps    = float(self._fps_var.get())
            sensor = float(self._sensor_var.get())
            scale  = float(self._scale_var.get())
            roll   = float(self._roll_var.get())
            focal_str = self._focal_var.get().strip()
            focal  = float(focal_str) if focal_str else None
        except ValueError as e:
            self._status(f"⚠  Bad option value: {e}", False); return

        try:
            data = convert(self._images_path, self._cameras_path,
                           fps=fps, sensor_mm=sensor,
                           focal_override_mm=focal, scale=scale, roll_deg=roll)
        except Exception as e:
            self._status(f"⚠  Conversion failed: {e}", False); return

        # Write output next to images file
        out = self._images_path.parent / (
            self._images_path.stem.replace("images", "").strip("_. ") + "lfs_camera_path.json"
        )
        if out.name.startswith("lfs"):
            out = self._images_path.parent / "lfs_camera_path.json"

        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._status(f"⚠  Could not write output: {e}", False); return

        n = len(data["keyframes"])
        self._status(f"✓  {n} keyframes written to:\n{out}")

        # Ask to open in Finder/Explorer
        if messagebox.askyesno("Done", f"Saved {n} keyframes.\n\nOpen output folder?"):
            import subprocess
            folder = str(out.parent)
            if sys.platform == "win32":
                subprocess.Popen(["explorer", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])


def main():
    app = App()
    # If a path was dragged onto the .py file at launch, pre-load it
    if len(sys.argv) > 1:
        app._load_paths(sys.argv[1:])
    app.mainloop()


if __name__ == "__main__":
    main()