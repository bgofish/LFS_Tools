import sys
import os
import struct
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QWidget, QFileDialog, QSlider, QLabel, 
                             QComboBox, QGroupBox, QDoubleSpinBox, QRadioButton, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt
from scipy.spatial.transform import Rotation as Rot
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()
pv.global_theme.multi_samples = 0

# --- COLMAP CORE MATH ---
def qvec2rotmat(qvec): return Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
def rotmat2qvec(R_mat):
    q = Rot.from_matrix(R_mat).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

class COLMAPProject:
    def __init__(self): self.cameras = {}; self.images = {}; self.points3D = {}
    def load(self, folder):
        def find_file(base_name):
            for ext in ['.bin', '.txt']:
                for name in [base_name, base_name.lower(), base_name.upper()]:
                    p = os.path.join(folder, name + ext)
                    if os.path.exists(p): return p
            return None
        p_path = find_file("points3D")
        if not p_path: raise FileNotFoundError(f"Missing points3D in {folder}")
        with open(p_path, "rb") as f:
            num = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num):
                d = struct.unpack("<QdddBBB d", f.read(43))
                t_len = struct.unpack("<Q", f.read(8))[0]
                self.points3D[d[0]] = {'xyz': np.array(d[1:4], dtype=np.float32), 'rgb': np.array(d[4:7], dtype=np.uint8), 'tracks': f.read(t_len * 8)}
        i_path = find_file("images")
        if i_path:
            with open(i_path, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    h = struct.unpack("<IdddddddI", f.read(64))
                    name = ""
                    while True:
                        char = f.read(1)
                        if char == b"\x00" or not char: break
                        name += char.decode("utf-8")
                    num_p2d = struct.unpack("<Q", f.read(8))[0]
                    self.images[h[0]] = {'qvec': np.array(h[1:5]), 'tvec': np.array(h[5:8]), 'cam_id': h[8], 'name': name, 'p2d': f.read(num_p2d * 24)}
        c_path = find_file("cameras")
        if c_path:
            with open(c_path, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    header = f.read(24); cid = struct.unpack("<I", header[:4])[0]
                    self.cameras[cid] = {'header': header, 'params': f.read(32)}

class COLMAPExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LichtFeld Studio | Master Geometry & Alignment Tool v0.0.1")
        self.resize(1750, 1050)
        self.proj, self.cloud_poly = None, None
        self.bounds = [0.0]*6; self.current_crop = [0.0]*6
        self.picked_points, self.pick_mode = [], None

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- SIDEBAR ---
        sidebar = QVBoxLayout()
        
        # Project Controls
        p_grp = QGroupBox("Project")
        p_v = QVBoxLayout()
        btn_load = QPushButton("📂 Import Project"); btn_load.clicked.connect(self.open_file)
        self.btn_bake = QPushButton("✅ SRT Baked"); self.btn_bake.clicked.connect(self.bake_srt)
        btn_export = QPushButton("💾 [Export COLMAP]"); btn_export.clicked.connect(self.export_project)
        p_v.addWidget(btn_load); p_v.addWidget(self.btn_bake); p_v.addWidget(btn_export); p_grp.setLayout(p_v)
        sidebar.addWidget(p_grp)

        # 3x3 VIEW PRESETS
        v_grp = QGroupBox("View Presets")
        v_v = QVBoxLayout(); grid = QGridLayout()
        for i in range(9):
            btn = QPushButton(f"V0{i+1}")
            btn.clicked.connect(lambda checked, idx=i: self.apply_view_preset(idx))
            grid.addWidget(btn, i//3, i%3)
        v_v.addLayout(grid)
        v_v.addWidget(QLabel("Camera Distance:"))
        self.sld_dist = QSlider(Qt.Horizontal); self.sld_dist.setRange(10, 5000); self.sld_dist.setValue(200)
        self.sld_dist.valueChanged.connect(self.update_preview); v_v.addWidget(self.sld_dist)
        v_grp.setLayout(v_v); sidebar.addWidget(v_grp)

        # ALIGNMENT PICKING TOOLS
        pick_grp = QGroupBox("Alignment Tools")
        pick_v = QVBoxLayout(); self.pick_btn_map = {}
        pick_configs = [("[3PointPick] Level Floor", 'plane'), ("[New XYZ Origin]", 'xyz_origin'), 
                        ("[New XZ Origin]", 'xz_origin'), ("[ALIGN] Y-Axis", 'align')]
        for text, mode in pick_configs:
            btn = QPushButton(text); btn.setCheckable(True)
            btn.clicked.connect(lambda checked, m=mode: self.start_pick(m))
            pick_v.addWidget(btn); self.pick_btn_map[mode] = btn
        pick_grp.setLayout(pick_v); sidebar.addWidget(pick_grp)

        # GLOBAL SRT
        srt_grp = QGroupBox("Global SRT Transforms")
        srt_v = QVBoxLayout(); self.srt_ctrl = {}
        configs = [('Scale', 1.0, 0.01, 10.0), ('RotX',0.0,-180.0,180.0), ('RotY',0.0,-180.0,180.0), ('RotZ',0.0,-180.0,180.0),
                   ('TransX',0.0,-500.0,500.0), ('TransY',0.0,-500.0,500.0), ('TransZ',0.0,-500.0,500.0)]
        for l, s, r1, r2 in configs:
            srt_v.addWidget(QLabel(f"<b>{l}</b>"))
            h = QHBoxLayout()
            sp = QDoubleSpinBox(); sp.setRange(r1, r2); sp.setValue(s); sp.setDecimals(3)
            sl = QSlider(Qt.Horizontal); sl.setRange(int(r1*100), int(r2*100)); sl.setValue(int(s*100))
            sp.valueChanged.connect(lambda v, s_obj=sl: s_obj.blockSignals(True) or s_obj.setValue(int(v*100)) or s_obj.blockSignals(False))
            sl.valueChanged.connect(lambda v, p_obj=sp: p_obj.blockSignals(True) or p_obj.setValue(v/100.0) or p_obj.blockSignals(False))
            sp.valueChanged.connect(self.on_srt_change); sl.valueChanged.connect(self.on_srt_change)
            h.addWidget(sp); h.addWidget(sl); srt_v.addLayout(h); self.srt_ctrl[l] = sp

        srt_v.addWidget(QLabel("Step Size (Increment):"))
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.001, 10.0); self.spin_step.setValue(0.1)
        self.spin_step.valueChanged.connect(self.update_step_sizes); srt_v.addWidget(self.spin_step)
        srt_grp.setLayout(srt_v); sidebar.addWidget(srt_grp)

        # Visuals
        t_grp = QGroupBox("Visuals")
        t_v = QVBoxLayout()
        self.btn_origin_vis = QPushButton("[Origin Axis]"); self.btn_origin_vis.setCheckable(True); self.btn_origin_vis.setChecked(True)
        self.btn_origin_vis.toggled.connect(self.toggle_origin); t_v.addWidget(self.btn_origin_vis)
        self.sld_size = QSlider(Qt.Horizontal); self.sld_size.setRange(1, 15); self.sld_size.setValue(3)
        self.sld_size.valueChanged.connect(self.update_preview); t_v.addWidget(QLabel("Point Size:")); t_v.addWidget(self.sld_size)
        t_grp.setLayout(t_v); sidebar.addWidget(t_grp); sidebar.addStretch(); main_layout.addLayout(sidebar, 1)

        # --- MAIN CONTENT ---
        content = QVBoxLayout()
        self.plotter = QtInteractor(self); self.hist_plotter = QtInteractor(self); self.hist_plotter.setMaximumHeight(180)
        content.addWidget(self.plotter.interactor, 6); content.addWidget(self.hist_plotter.interactor, 2)
        
        crop_grp = QGroupBox("Data Limit Filtering")
        crop_v = QVBoxLayout(); row_top = QHBoxLayout()
        self.axis_sel = QComboBox(); self.axis_sel.addItems(["X Axis", "Y Axis", "Z Axis"]); self.axis_sel.setCurrentIndex(1)
        self.axis_sel.currentIndexChanged.connect(self.sync_crop_ui)
        self.rad_log = QRadioButton("Log Scale"); self.rad_log.toggled.connect(self.update_preview)
        row_top.addWidget(QLabel("Axis:")); row_top.addWidget(self.axis_sel); row_top.addStretch(); row_top.addWidget(self.rad_log)
        btn_res = QPushButton("Reset Crop"); btn_res.clicked.connect(self.reset_crop); row_top.addWidget(btn_res)
        crop_v.addLayout(row_top)

        self.crop_ui = {}
        for l in ["Min", "Max"]:
            row = QHBoxLayout(); sp = QDoubleSpinBox(); sp.setRange(-1000000, 1000000); sp.setDecimals(2)
            sl = QSlider(Qt.Horizontal); sl.setRange(0, 1000)
            sp.valueChanged.connect(self.on_ui_change); sl.valueChanged.connect(self.on_slider_move)
            row.addWidget(QLabel(f"{l}:")); row.addWidget(sp); row.addWidget(sl); crop_v.addLayout(row)
            self.crop_ui[l.lower()] = sp; self.crop_ui[f"{l.lower()}_sl"] = sl
        
        crop_grp.setLayout(crop_v); content.addWidget(crop_grp); main_layout.addLayout(content, 4)
        self.hist_plotter.iren.add_observer("LeftButtonPressEvent", lambda o,e: self.on_hist_pick(o,e,"min"))
        self.hist_plotter.iren.add_observer("RightButtonPressEvent", lambda o,e: self.on_hist_pick(o,e,"max"))
        self.plotter.set_background("black"); self.set_bake_state(False)

    # --- PICKING ROUTINES ---
    def start_pick(self, m):
        try: self.plotter.disable_picking()
        except: pass
        for btn in self.pick_btn_map.values(): btn.setChecked(False)
        self.pick_btn_map[m].setChecked(True); self.pick_mode = m; self.picked_points = []
        self.plotter.enable_point_picking(callback=self.pick_callback, show_message=True, color='yellow', point_size=12)

    def pick_callback(self, pt):
        if pt is None: return
        self.picked_points.append(pt)
        self.plotter.add_mesh(pv.Sphere(radius=0.15, center=pt), color="yellow", name=f"tp_{len(self.picked_points)}", reset_camera=False)
        
        if self.pick_mode in ['xyz_origin', 'xz_origin']:
            self.srt_ctrl['TransX'].setValue(self.srt_ctrl['TransX'].value() - pt[0])
            self.srt_ctrl['TransZ'].setValue(self.srt_ctrl['TransZ'].value() - pt[2])
            if self.pick_mode == 'xyz_origin': self.srt_ctrl['TransY'].setValue(self.srt_ctrl['TransY'].value() - pt[1])
            self.finish_pick()
        elif self.pick_mode == 'align' and len(self.picked_points) == 2:
            p1, p2 = self.picked_points; angle = np.degrees(np.arctan2(p2[0]-p1[0], p2[2]-p1[2]))
            self.srt_ctrl['RotY'].setValue(self.srt_ctrl['RotY'].value() - angle); self.finish_pick()
        elif self.pick_mode == 'plane' and len(self.picked_points) == 3:
            p1, p2, p3 = map(np.array, self.picked_points); v1, v2 = p2 - p1, p3 - p1
            norm = np.cross(v1, v2); norm /= np.linalg.norm(norm)
            if norm[1] < 0: norm *= -1
            rx = np.degrees(np.arctan2(norm[2], norm[1])); rz = np.degrees(np.arctan2(-norm[0], np.sqrt(norm[1]**2 + norm[2]**2)))
            self.srt_ctrl['RotX'].setValue(self.srt_ctrl['RotX'].value() - rx); self.srt_ctrl['RotZ'].setValue(self.srt_ctrl['RotZ'].value() - rz)
            self.finish_pick()

    def finish_pick(self):
        self.plotter.disable_picking()
        for b in self.pick_btn_map.values(): b.setChecked(False)
        for i in range(1, 5): self.plotter.remove_actor(f"tp_{i}")
        self.pick_mode = None; self.update_preview()

    # --- CORE FUNCTIONS ---
    def get_mat(self):
        S = self.srt_ctrl['Scale'].value()
        R = Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        T = [self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()]
        m = np.eye(4); m[:3,:3] = R * S; m[:3,3] = T
        return m

    def update_preview(self):
        if self.cloud_poly is None: return
        idx = self.axis_sel.currentIndex(); self.current_crop[2*idx], self.current_crop[2*idx+1] = self.crop_ui['min'].value(), self.crop_ui['max'].value()
        temp = self.cloud_poly.copy(); temp.transform(self.get_mat(), inplace=True); c = self.current_crop
        mask = (temp.points[:,0]>=c[0]) & (temp.points[:,0]<=c[1]) & (temp.points[:,1]>=c[2]) & (temp.points[:,1]<=c[3]) & (temp.points[:,2]>=c[4]) & (temp.points[:,2]<=c[5])
        active = temp.extract_points(mask) if np.any(mask) else None
        if active: self.plotter.add_mesh(active, name="cloud", scalars="rgb", rgb=True, point_size=self.sld_size.value(), reset_camera=False, show_scalar_bar=False)
        else: self.plotter.remove_actor("cloud")
        self.update_histogram(temp.points); self.plotter.render()
        if self.btn_origin_vis.isChecked(): self.toggle_origin(True)

    def apply_view_preset(self, idx):
        d = self.sld_dist.value()
        angles = [(0,0,-d), (0,0,d), (-d,0,0), (d,0,0), (0,d,0), (-d,d,-d), (d,d,-d), (-d,d,d), (d,d,d)]
        self.plotter.camera_position = [angles[idx], (0, 0, 0), (0, 1, 0)]; self.plotter.reset_camera()

    def open_file(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder")
        if f:
            self.proj = COLMAPProject(); self.proj.load(f)
            pts = np.array([p['xyz'] * [1, -1, 1] for p in self.proj.points3D.values()], dtype=np.float32)
            self.cloud_poly = pv.PolyData(pts); self.cloud_poly.point_data["rgb"] = np.array([p['rgb'] for p in self.proj.points3D.values()])
            self.bounds = [pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()]
            self.reset_crop(); self.plotter.camera_position = [(200, -100, 200), (0, 0, 0), (0, 1, 0)]; self.plotter.reset_camera()
            self.set_bake_state(False); self.update_preview()

    def toggle_origin(self, show):
        self.plotter.remove_actor("origin")
        if show and self.cloud_poly:
            axis_len = max(self.cloud_poly.bounds[1]-self.cloud_poly.bounds[0], self.cloud_poly.bounds[3]-self.cloud_poly.bounds[2]) * 0.3
            pts = np.array([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]], dtype=np.float32)
            lines = np.array([2, 0, 1, 2, 0, 2, 2, 0, 3])
            mesh = pv.PolyData(pts, lines=lines); mesh.cell_data["colors"] = np.array([0, 1, 2])
            self.plotter.add_mesh(mesh, name="origin", scalars="colors", cmap=["red","green","blue"], line_width=5, render_lines_as_tubes=True, show_scalar_bar=False)
        self.plotter.render()

    def bake_srt(self):
        if self.cloud_poly is None: return
        self.cloud_poly.transform(self.get_mat(), inplace=True)
        for k, sp in self.srt_ctrl.items(): sp.blockSignals(True); sp.setValue(1.0 if k=='Scale' else 0.0); sp.blockSignals(False)
        self.set_bake_state(False); self.update_preview()

    def export_project(self):
        f = QFileDialog.getExistingDirectory(self, "Export COLMAP Folder")
        if f and self.proj: 
            S = self.srt_ctrl['Scale'].value(); R_d = Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix(); T_d = np.array([self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()])
            final_mesh = self.cloud_poly.copy(); final_mesh.transform(self.get_mat(), inplace=True); pts = final_mesh.points * [1, -1, 1]
            with open(os.path.join(f, "points3D.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(self.proj.points3D)))
                for i, pid in enumerate(self.proj.points3D.keys()):
                    p = self.proj.points3D[pid]; file.write(struct.pack("<QdddBBB d", pid, *pts[i], *p['rgb'], 0.0) + struct.pack("<Q", len(p['tracks']) // 8) + p['tracks'])
            with open(os.path.join(f, "images.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(self.proj.images)))
                for iid, img in self.proj.images.items():
                    R_old = qvec2rotmat(img['qvec']); R_new = R_old @ R_d.T; T_new = img['tvec'] - (R_new @ T_d) / S
                    file.write(struct.pack("<I", iid) + struct.pack("<dddd", *rotmat2qvec(R_new)) + struct.unpack("<ddd", struct.pack("<ddd", *T_new)) + struct.pack("<I", img['cam_id']) + img['name'].encode() + b"\x00" + struct.pack("<Q", len(img['p2d']) // 24) + img['p2d'])
            with open(os.path.join(f, "cameras.bin"), "wb") as file:
                file.write(struct.pack("<Q", len(self.proj.cameras)))
                for c in self.proj.cameras.values(): file.write(c['header'] + c['params'])
            print(f"Exported to {f}")

    def set_bake_state(self, dirty):
        if dirty: self.btn_bake.setText("🔥 BAKE SRT (Pending)"); self.btn_bake.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold;")
        else: self.btn_bake.setText("✅ SRT Baked"); self.btn_bake.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
    def on_srt_change(self): self.on_srt_value_changed()
    def on_srt_value_changed(self):
        is_dirty = any([abs(self.srt_ctrl['Scale'].value()-1.0)>1e-4, any(abs(self.srt_ctrl[k].value())>1e-4 for k in ['RotX','RotY','RotZ','TransX','TransY','TransZ'])])
        self.set_bake_state(is_dirty); self.update_preview()
    def sync_crop_ui(self):
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']:
            sp, sl = self.crop_ui[l], self.crop_ui[f"{l}_sl"]; sp.blockSignals(True); sl.blockSignals(True); sp.setRange(d_min-100, d_max+100); sp.setValue(self.current_crop[2*idx + (0 if l=='min' else 1)])
            pct = (sp.value() - d_min) / (d_max - d_min) if d_max != d_min else 0.5
            sl.setValue(int(pct * 1000)); sp.blockSignals(False); sl.blockSignals(False)
        self.update_preview()
    def on_ui_change(self): self.update_preview()
    def on_slider_move(self):
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']:
            sp, sl = self.crop_ui[l], self.crop_ui[f"{l}_sl"]; val = d_min + (sl.value() / 1000.0) * (d_max - d_min)
            sp.blockSignals(True); sp.setValue(val); sp.blockSignals(False)
        self.update_preview()
    def update_histogram(self, pts):
        self.hist_plotter.clear(); idx = self.axis_sel.currentIndex(); data = pts[:, idx]
        if self.rad_log.isChecked(): data = np.sign(data) * np.log10(np.abs(data) + 1)
        counts, self.bins = np.histogram(data, bins=100); chart = pv.Chart2D(); chart.bar(self.bins[:-1], counts, color=["#FF6666", "#66FF66", "#6666FF"][idx])
        c_min, c_max = self.current_crop[2*idx], self.current_crop[2*idx+1]
        if self.rad_log.isChecked(): c_min = np.sign(c_min)*np.log10(np.abs(c_min)+1); c_max = np.sign(c_max)*np.log10(np.abs(c_max)+1)
        chart.line([c_min, c_min], [0, counts.max()], color="black", width=3); chart.line([c_max, c_max], [0, counts.max()], color="black", width=3, style="--"); self.hist_plotter.add_chart(chart); self.hist_plotter.render()
    def on_hist_pick(self, o, e, m):
        if self.bins is None: return
        x, _ = o.GetEventPosition(); pct = (x/self.hist_plotter.width() - 0.08)/0.84; val = self.bins + (max(0, min(1, pct)) * (self.bins[-1] - self.bins))
        if self.rad_log.isChecked(): val = np.sign(val) * (10**np.abs(val) - 1)
        self.crop_ui['min' if m == 'min' else 'max'].setValue(val)
    def reset_crop(self):
        if self.cloud_poly: b = self.cloud_poly.bounds; self.current_crop = [b[0]-1, b[1]+1, b[2]-1, b[3]+1, b[4]-1, b[5]+1]; self.sync_crop_ui()
    def update_step_sizes(self):
        s = self.spin_step.value()
        for c in self.srt_ctrl.values(): c.setSingleStep(s)
    def closeEvent(self, event):
        if hasattr(self, 'plotter'): self.plotter.close()
        if hasattr(self, 'hist_plotter'): self.hist_plotter.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); ex = COLMAPExplorer(); ex.show(); sys.exit(app.exec_())
