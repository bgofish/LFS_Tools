# self.setWindowTitle("LichtFeld Studio | COLMAP Point Editor v0.0.2")
#================================================================
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

# --- HELPERS ---
class CustomSlider(QSlider):
    def __init__(self, orientation, parent=None, default=0):
        super().__init__(orientation, parent); self.default_val = default
    def mouseDoubleClickEvent(self, event): self.setValue(self.default_val)

def qvec2rotmat(qvec): return Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
def rotmat2qvec(R):
    q = Rot.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]]) # w, x, y, z

class COLMAPProject:
    def __init__(self): self.cameras = {}; self.images = {}; self.points3D = {}
    def load(self, folder):
        def find_file(base_name):
            for ext in ['.bin', '.txt']:
                for name in [base_name, base_name.lower(), base_name.upper()]:
                    p = os.path.join(folder, name + ext)
                    if os.path.exists(p): return p, ext.lower()
            return None, None
        
        p_path, _ = find_file("points3D")
        if not p_path: raise FileNotFoundError(f"Missing points3D in {folder}")
        with open(p_path, "rb") as f:
            num = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num):
                d = struct.unpack("<QdddBBB d", f.read(43))
                t_len = struct.unpack("<Q", f.read(8))[0]
                self.points3D[d[0]] = {'xyz': np.array(d[1:4]), 'rgb': np.array(d[4:7]), 'tracks': f.read(t_len * 8)}
        
        i_path, _ = find_file("images")
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
        
        c_path, _ = find_file("cameras")
        if c_path:
            with open(c_path, "rb") as f:
                num = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num):
                    cid = struct.unpack("<I", f.read(4))[0]
                    model = struct.unpack("<i", f.read(4))[0]
                    hw = struct.unpack("<QQ", f.read(16))
                    params = f.read(32) 
                    self.cameras[cid] = {'model': model, 'hw': hw, 'params': params}

class COLMAPExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LichtFeld Studio | COLMAP Point Editor v0.0.2")
        self.resize(1750, 1050)
        self.proj, self.cloud_poly = None, None
        self.bounds = [0.0]*6; self.current_crop = [-50000.0, 50000.0]*3
        self.picked_points, self.pick_mode = [], None

        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- Sidebar ---
        sidebar = QVBoxLayout()
        p_grp = QGroupBox("Project"); p_v = QVBoxLayout()
        btn_load = QPushButton("📂 Import Project"); btn_load.clicked.connect(self.open_file)
        self.btn_bake = QPushButton("✅ SRT Baked"); self.btn_bake.clicked.connect(self.bake_srt)
        btn_export = QPushButton("💾 [Export COLMAP]"); btn_export.clicked.connect(self.export_project)
        p_v.addWidget(btn_load); p_v.addWidget(self.btn_bake); p_v.addWidget(btn_export); p_grp.setLayout(p_v); sidebar.addWidget(p_grp)

        v_grp = QGroupBox("View Control"); v_v = QVBoxLayout(); grid = QGridLayout()
        for i in range(9):
            btn = QPushButton(f"V0{i+1}"); btn.clicked.connect(lambda checked, idx=i: self.apply_view_preset(idx))
            grid.addWidget(btn, i//3, i%3)
        v_v.addLayout(grid)
        v_v.addWidget(QLabel("Camera FOV:"))
        self.spin_fov = QDoubleSpinBox(); self.spin_fov.setRange(10, 120); self.spin_fov.setValue(45)
        self.spin_fov.valueChanged.connect(self.update_fov); v_v.addWidget(self.spin_fov); v_grp.setLayout(v_v); sidebar.addWidget(v_grp)

        pick_grp = QGroupBox("Alignment Tools"); pick_v = QVBoxLayout(); self.pick_btn_map = {}
        for text, mode in [("3Point Floor", 'plane'), ("ALIGN Y-Axis", 'align'), ("New XYZ Origin", 'xyz_origin'), ("New XZ Origin", 'xz_origin')]:
            btn = QPushButton(text); btn.setCheckable(True); btn.clicked.connect(lambda checked, m=mode: self.start_pick(m))
            pick_v.addWidget(btn); self.pick_btn_map[mode] = btn
        pick_grp.setLayout(pick_v); sidebar.addWidget(pick_grp)

        srt_grp = QGroupBox("Global SRT"); srt_v = QVBoxLayout(); self.srt_ctrl = {}
        for l, s, rmin, rmax in [('Scale',1.0,0.01,20),('RotX',0.0,-180,180),('RotY',0.0,-180,180),('RotZ',0.0,-180,180),('TransX',0.0,-1000,1000),('TransY',0.0,-1000,1000),('TransZ',0.0,-1000,1000)]:
            srt_v.addWidget(QLabel(f"<b>{l}</b>"))
            h = QHBoxLayout(); sp = QDoubleSpinBox(); sp.setRange(rmin, rmax); sp.setValue(s); sp.setDecimals(3)
            sl = CustomSlider(Qt.Horizontal, default=int(s*100) if l!='Scale' else 100); sl.setRange(int(rmin*100), int(rmax*100)); sl.setValue(int(s*100))
            sp.valueChanged.connect(lambda v, s_obj=sl: s_obj.blockSignals(True) or s_obj.setValue(int(v*100)) or s_obj.blockSignals(False))
            sl.valueChanged.connect(lambda v, p_obj=sp: p_obj.blockSignals(True) or p_obj.setValue(v/100.0) or p_obj.blockSignals(False))
            sp.valueChanged.connect(self.on_srt_change); sl.valueChanged.connect(self.on_srt_change)
            h.addWidget(sp); h.addWidget(sl); srt_v.addLayout(h); self.srt_ctrl[l] = sp

        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0.001, 1.0); self.spin_step.setValue(0.02); self.spin_step.setDecimals(3)
        self.spin_step.valueChanged.connect(self.update_step_sizes)
        srt_v.addWidget(QLabel("Step Size:")); srt_v.addWidget(self.spin_step); srt_grp.setLayout(srt_v); sidebar.addWidget(srt_grp)

        t_grp = QGroupBox("Visuals"); t_v = QVBoxLayout()
        self.btn_origin_vis = QPushButton("[Origin Axis]"); self.btn_origin_vis.setCheckable(True); self.btn_origin_vis.setChecked(True); self.btn_origin_vis.toggled.connect(self.update_preview); t_v.addWidget(self.btn_origin_vis)
        self.sld_size = QSlider(Qt.Horizontal); self.sld_size.setRange(1, 15); self.sld_size.setValue(3); self.sld_size.valueChanged.connect(self.update_preview)
        t_v.addWidget(QLabel("Point Size:")); t_v.addWidget(self.sld_size)
        self.combo_color = QComboBox(); self.combo_color.addItems(["Source RGB", "Elevation (Y)", "Cyan Single"]); self.combo_color.currentIndexChanged.connect(self.update_preview); t_v.addWidget(self.combo_color)
        t_grp.setLayout(t_v); sidebar.addWidget(t_grp); sidebar.addStretch(); main_layout.addLayout(sidebar, 1)

        # --- Main Viewport ---
        content = QVBoxLayout()
        self.plotter = QtInteractor(self); self.hist_plotter = QtInteractor(self); self.hist_plotter.setMaximumHeight(180)
        content.addWidget(self.plotter.interactor, 6); content.addWidget(self.hist_plotter.interactor, 2)
        
        crop_grp = QGroupBox("Cropping Area"); crop_v = QVBoxLayout(); row_top = QHBoxLayout()
        self.axis_sel = QComboBox(); self.axis_sel.addItems(["X Axis", "Y Axis", "Z Axis"]); self.axis_sel.setCurrentIndex(1); self.axis_sel.currentIndexChanged.connect(self.sync_crop_ui)
        btn99 = QPushButton("Auto99"); btn99.clicked.connect(lambda: self.auto_crop_stat(0.99))
        btn95 = QPushButton("Auto95"); btn95.clicked.connect(lambda: self.auto_crop_stat(0.95))
        self.rad_log = QRadioButton("Log Scale"); self.rad_log.toggled.connect(self.update_preview)
        row_top.addWidget(QLabel("Axis:")); row_top.addWidget(self.axis_sel); row_top.addStretch(); row_top.addWidget(btn99); row_top.addWidget(btn95); row_top.addWidget(self.rad_log)
        btn_res = QPushButton("Reset"); btn_res.clicked.connect(self.reset_crop); row_top.addWidget(btn_res); crop_v.addLayout(row_top)

        self.crop_ui = {}
        for l in ["Min", "Max"]:
            row = QHBoxLayout(); sp = QDoubleSpinBox(); sp.setRange(-1e6, 1e6); sp.setDecimals(2)
            sl = CustomSlider(Qt.Horizontal); sl.setRange(0, 1000); sp.valueChanged.connect(self.on_ui_change); sl.valueChanged.connect(self.on_slider_move)
            row.addWidget(QLabel(f"{l}:")); row.addWidget(sp); row.addWidget(sl); crop_v.addLayout(row); self.crop_ui[l.lower()] = sp; self.crop_ui[f"{l.lower()}_sl"] = sl
        crop_grp.setLayout(crop_v); content.addWidget(crop_grp); main_layout.addLayout(content, 4)

        self.hist_plotter.iren.add_observer("LeftButtonPressEvent", lambda o,e: self.on_hist_click(o,e,"min"))
        self.hist_plotter.iren.add_observer("RightButtonPressEvent", lambda o,e: self.on_hist_click(o,e,"max"))
        self.plotter.set_background("black"); self.update_step_sizes()

    def update_fov(self, val):
        self.plotter.camera.view_angle = val; self.plotter.render()

    def update_preview(self):
        if not self.cloud_poly: return
        idx = self.axis_sel.currentIndex()
        self.current_crop[2*idx], self.current_crop[2*idx+1] = self.crop_ui['min'].value(), self.crop_ui['max'].value()
        
        temp = self.cloud_poly.copy(); temp.transform(self.get_mat(), inplace=True)
        c = self.current_crop
        # FIXED: Explicit element-by-element masking to avoid broadcast errors
        mask = (temp.points[:,0] >= c[0]) & (temp.points[:,0] <= c[1]) & \
               (temp.points[:,1] >= c[2]) & (temp.points[:,1] <= c[3]) & \
               (temp.points[:,2] >= c[4]) & (temp.points[:,2] <= c[5])
        
        active = temp.extract_points(mask) if np.any(mask) else None
        if active:
            cm = self.combo_color.currentIndex()
            if cm == 0: self.plotter.add_mesh(active, name="cloud", scalars="rgb", rgb=True, point_size=self.sld_size.value(), reset_camera=False)
            elif cm == 1: self.plotter.add_mesh(active, name="cloud", scalars=active.points[:,1], cmap="viridis", point_size=self.sld_size.value(), reset_camera=False)
            else: self.plotter.add_mesh(active, name="cloud", color="cyan", point_size=self.sld_size.value(), reset_camera=False)
        else: self.plotter.remove_actor("cloud")
        
        self.update_histogram(temp.points); self.toggle_origin(self.btn_origin_vis.isChecked()); self.plotter.render()

    def toggle_origin(self, show):
        self.plotter.remove_actor("origin")
        if show and self.cloud_poly:
            b = self.cloud_poly.bounds
            al = max(abs(b[1]-b[0]), abs(b[3]-b[2]), abs(b[5]-b[4])) * 0.3
            pts = np.array([[0,0,0], [al,0,0], [0,al,0], [0,0,al]], dtype=np.float32)
            lines = np.array([2,0,1, 2,0,2, 2,0,3])
            mesh = pv.PolyData(pts, lines=lines); mesh.cell_data["colors"] = np.array([0, 1, 2])
            self.plotter.add_mesh(mesh, name="origin", scalars="colors", cmap=["red","green","blue"], line_width=5, render_lines_as_tubes=True, show_scalar_bar=False)

    def open_file(self):
        f = QFileDialog.getExistingDirectory(self, "Select COLMAP Folder")
        if f:
            self.proj = COLMAPProject(); self.proj.load(f)
            pts = np.array([p['xyz'] * [1, -1, 1] for p in self.proj.points3D.values()], dtype=np.float32)
            self.cloud_poly = pv.PolyData(pts); self.cloud_poly.point_data["rgb"] = np.array([p['rgb'] for p in self.proj.points3D.values()])
            self.bounds = [pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), pts[:,2].min(), pts[:,2].max()]
            self.current_crop = list(self.bounds); self.apply_view_preset(5); self.sync_crop_ui()

    def export_project(self):
        f = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not (f and self.proj): return
        S = self.srt_ctrl['Scale'].value()
        R_delta = Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        T_delta = np.array([self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()])
        
        final_mesh = self.cloud_poly.copy(); final_mesh.transform(self.get_mat(), inplace=True); pts = final_mesh.points * [1, -1, 1]
        with open(os.path.join(f, "points3D.bin"), "wb") as file:
            file.write(struct.pack("<Q", len(self.proj.points3D)))
            for i, pid in enumerate(self.proj.points3D.keys()):
                p = self.proj.points3D[pid]
                file.write(struct.pack("<QdddBBB d", pid, *pts[i].tolist(), *p['rgb'].tolist(), 0.0))
                file.write(struct.pack("<Q", len(p['tracks']) // 8) + p['tracks'])
        with open(os.path.join(f, "images.bin"), "wb") as file:
            file.write(struct.pack("<Q", len(self.proj.images)))
            for iid, img in self.proj.images.items():
                R_old = qvec2rotmat(img['qvec']); R_new = R_old @ R_delta.T; T_new = img['tvec'] - (R_new @ T_delta) / S
                file.write(struct.pack("<I", iid) + struct.pack("<dddd", *rotmat2qvec(R_new).tolist()) + 
                           struct.pack("<ddd", *T_new.flatten().tolist()) + struct.pack("<I", img['cam_id']) + 
                           img['name'].encode() + b"\x00" + struct.pack("<Q", len(img['p2d']) // 24) + img['p2d'])
        with open(os.path.join(f, "cameras.bin"), "wb") as file:
            file.write(struct.pack("<Q", len(self.proj.cameras)))
            for cid, cam in self.proj.cameras.items():
                file.write(struct.pack("<IiQQ", cid, cam['model'], *cam['hw']) + cam['params'])
        print(f"Export Complete: {f}")

    def pick_callback(self, pt):
        if pt is None: return
        self.picked_points.append(pt)
        if self.pick_mode in ['xyz_origin', 'xz_origin']:
            self.srt_ctrl['TransX'].setValue(self.srt_ctrl['TransX'].value() - float(pt[0]))
            self.srt_ctrl['TransZ'].setValue(self.srt_ctrl['TransZ'].value() - float(pt[2]))
            if self.pick_mode == 'xyz_origin': self.srt_ctrl['TransY'].setValue(self.srt_ctrl['TransY'].value() - float(pt[1]))
            self.finish_pick()
        elif self.pick_mode == 'align' and len(self.picked_points) == 2:
            p1, p2 = self.picked_points; ang = np.degrees(np.arctan2(p2[0]-p1[0], p2[2]-p1[2]))
            self.srt_ctrl['RotY'].setValue(self.srt_ctrl['RotY'].value() - float(ang)); self.finish_pick()
        elif self.pick_mode == 'plane' and len(self.picked_points) == 3:
            p1,p2,p3 = map(np.array, self.picked_points); norm = np.cross(p2-p1, p3-p1); norm /= np.linalg.norm(norm)
            rx = np.degrees(np.arctan2(norm[1], norm[2])); rz = -np.degrees(np.arctan2(norm[0], np.sqrt(norm[1]**2+norm[2]**2)))
            self.srt_ctrl['RotX'].setValue(float(rx)); self.srt_ctrl['RotZ'].setValue(float(rz)); self.finish_pick()

    def start_pick(self, m):
        try: self.plotter.disable_picking()
        except: pass
        for b in self.pick_btn_map.values(): b.setChecked(False)
        self.pick_btn_map[m].setChecked(True); self.pick_mode = m; self.picked_points = []
        self.plotter.enable_point_picking(callback=self.pick_callback, show_message=True, color='yellow', point_size=12)

    def finish_pick(self): self.plotter.disable_picking(); self.pick_mode = None; [b.setChecked(False) for b in self.pick_btn_map.values()]; self.update_preview()
    def sync_crop_ui(self):
        if not self.cloud_poly: return
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']:
            sp, sl = self.crop_ui[l], self.crop_ui[f"{l}_sl"]; sp.blockSignals(True); sl.blockSignals(True)
            sp.setRange(d_min-5000, d_max+5000); sp.setValue(self.current_crop[2*idx + (0 if l=='min' else 1)])
            pct = (sp.value()-d_min)/(d_max-d_min) if d_max!=d_min else 0.5; sl.setValue(int(pct*1000)); sp.blockSignals(False); sl.blockSignals(False)
        self.update_preview()
    def on_srt_change(self):
        is_dirty = any([abs(self.srt_ctrl['Scale'].value()-1.0)>1e-4, any(abs(self.srt_ctrl[k].value())>1e-4 for k in ['RotX','RotY','RotZ','TransX','TransY','TransZ'])])
        self.btn_bake.setStyleSheet(f"background-color: {'#d32f2f' if is_dirty else '#2e7d32'}; color: white;"); self.update_preview()
    def on_slider_move(self):
        idx = self.axis_sel.currentIndex(); d_min, d_max = self.bounds[2*idx], self.bounds[2*idx+1]
        for l in ['min', 'max']:
            sp, sl = self.crop_ui[l], self.crop_ui[f"{l}_sl"]; sp.blockSignals(True); sp.setValue(d_min + (sl.value()/1000.0)*(d_max-d_min)); sp.blockSignals(False)
        self.update_preview()
    def update_histogram(self, pts):
        self.hist_plotter.clear(); idx = self.axis_sel.currentIndex(); data = pts[:, idx]
        counts, self.bins = np.histogram(data, bins=100); chart = pv.Chart2D(); chart.bar(self.bins[:-1], counts, color=["#FF6666","#66FF66","#6666FF"][idx])
        c_min, c_max = self.current_crop[2*idx], self.current_crop[2*idx+1]
        chart.line([c_min, c_min], [0, counts.max()], color="black", width=2); chart.line([c_max, c_max], [0, counts.max()], color="black", width=2, style="--"); self.hist_plotter.add_chart(chart); self.hist_plotter.render()
    def on_hist_click(self, o, e, mode):
        if not hasattr(self, 'bins'): return
        x, _ = o.GetEventPosition(); pct = (x/self.hist_plotter.width() - 0.08)/0.84
        val = self.bins[int(np.clip(pct, 0, 1) * (len(self.bins)-1))]; self.crop_ui[mode].setValue(val)
    def auto_crop_stat(self, pct):
        for i in range(3):
            d = self.cloud_poly.points[:,i]
            self.current_crop[2*i], self.current_crop[2*i+1] = np.percentile(d, (1-pct)/2*100), np.percentile(d, (1-(1-pct)/2)*100)
        self.sync_crop_ui()
    def reset_crop(self):
        if self.cloud_poly: self.current_crop = list(self.bounds); self.sync_crop_ui()
    def bake_srt(self):
        if self.cloud_poly: self.cloud_poly.transform(self.get_mat(), inplace=True)
        for k, sp in self.srt_ctrl.items(): sp.blockSignals(True); sp.setValue(1.0 if k=='Scale' else 0.0); sp.blockSignals(False)
        self.on_srt_change()
    def get_mat(self):
        S = self.srt_ctrl['Scale'].value(); R_m = Rot.from_euler('xyz', [self.srt_ctrl['RotX'].value(), self.srt_ctrl['RotY'].value(), self.srt_ctrl['RotZ'].value()], degrees=True).as_matrix()
        mat = np.eye(4); mat[:3,:3] = R_m * S; mat[:3,3] = [self.srt_ctrl['TransX'].value(), self.srt_ctrl['TransY'].value(), self.srt_ctrl['TransZ'].value()]
        return mat
    def apply_view_preset(self, idx):
        angles = [(0,0,-200),(0,0,200),(-200,0,0),(200,0,0),(0,200,0),(-200,200,-200),(200,200,-200),(-200,200,200),(200,200,200)]
        self.plotter.camera_position = [angles[idx], (0,0,0), (0,1,0)]; self.plotter.reset_camera()
    def update_step_sizes(self): [c.setSingleStep(self.spin_step.value()) for c in self.srt_ctrl.values()]
    def on_ui_change(self): self.update_preview()
    def closeEvent(self, event):
        if hasattr(self, 'plotter'): [p.close() for p in [self.plotter, self.hist_plotter]]
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); ex = COLMAPExplorer(); ex.show(); sys.exit(app.exec_())
