import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import laspy
import re
from plyfile import PlyData

class GS2LasTransformer:
    def __init__(self, root):
        self.root = root
        self.root.title("3DGS to LAS Transformer")
        self.root.geometry("500x350")

        self.ply_path = tk.StringVar()
        self.coord_path = tk.StringVar()
        self.output_path = tk.StringVar()

        # UI Layout
        tk.Label(root, text="Step 1: Select 3DGS PLY File").pack(pady=5)
        tk.Entry(root, textvariable=self.ply_path, width=60).pack()
        tk.Button(root, text="Browse PLY", command=self.browse_ply).pack()

        tk.Label(root, text="Step 2: Select COORD_00.TXT").pack(pady=5)
        tk.Entry(root, textvariable=self.coord_path, width=60).pack()
        tk.Button(root, text="Browse TXT", command=self.browse_txt).pack()

        tk.Label(root, text="Step 3: Save Result As").pack(pady=5)
        tk.Entry(root, textvariable=self.output_path, width=60).pack()
        tk.Button(root, text="Save As...", command=self.save_las).pack()

        tk.Button(root, text="RUN TRANSFORM", bg="#2ecc71", fg="white", 
                  font=('Arial', 12, 'bold'), command=self.process).pack(pady=20)

    def browse_ply(self):
        self.ply_path.set(filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")]))

    def browse_txt(self):
        self.coord_path.set(filedialog.askopenfilename(filetypes=[("Text files", "*.txt")]))

    def save_las(self):
        self.output_path.set(filedialog.asksaveasfilename(defaultextension=".las", filetypes=[("LAS files", "*.las")]))

    def parse_datum(self, path):
        with open(path, 'r') as f:
            content = f.read()
            # Extracts the specific numbers before 'm E', 'm N', and 'm RL'
            east = float(re.search(r'([\d\.]+)\s+m\s+E', content).group(1))
            north = float(re.search(r'([\d\.]+)\s+m\s+N', content).group(1))
            rl = float(re.search(r'([\d\.]+)\s+m\s+RL', content).group(1))
            return east, north, rl

    def process(self):
        try:
            off_e, off_n, off_z = self.parse_datum(self.coord_path.get())
            plydata = PlyData.read(self.ply_path.get())
            v = plydata['vertex']
            
            # --- 1. COORDINATE TRANSFORMATION (180 ROTATION) ---
            # East = -x, North = +z, Elevation = +y
            final_e = (-np.array(v['x'])) + off_e
            final_n = (-np.array(v['z'])) + off_n
            final_z = (-np.array(v['y'])) - off_z

            # --- 2. INTENSITY & COLORS ---
            opacity = 1 / (1 + np.exp(-np.array(v['opacity'])))
            intensity_vals = (opacity * 65535).astype(np.uint16)
            
            SH_C0 = 0.28209479177387814
            r = (np.clip(0.5 + (SH_C0 * np.array(v['f_dc_0'])), 0, 1) * 65535).astype(np.uint16)
            g = (np.clip(0.5 + (SH_C0 * np.array(v['f_dc_1'])), 0, 1) * 65535).astype(np.uint16)
            b = (np.clip(0.5 + (SH_C0 * np.array(v['f_dc_2'])), 0, 1) * 65535).astype(np.uint16)

            # --- 3. DYNAMIC OFFSETS (The "Stripe Killer") ---
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = [0.001, 0.001, 0.001] # 1mm is plenty if offsets are correct

            # This line tells laspy to look at your data and pick the 
            # best offset automatically so the numbers stay tiny.
            header.offsets = [np.min(final_e), np.min(final_n), np.min(final_z)]

            las = laspy.LasData(header)
            las.x = final_e
            las.y = final_n
            las.z = final_z
            las.red, las.green, las.blue = r, g, b
            las.intensity = intensity_vals

            las.write(self.output_path.get())
            messagebox.showinfo("Success", "Transformation complete. Stripe-free LAS saved!")

        except Exception as e:
            messagebox.showerror("Error", str(e))
  

if __name__ == "__main__":
    root = tk.Tk()
    app = GS2LasTransformer(root)
    root.mainloop()