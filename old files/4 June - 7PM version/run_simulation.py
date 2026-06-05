import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import numpy as np

from mesh_engine import TissueMesh
from mesh_plotter import plot_mesh
from compute_energy import compute_system_energy 

class TyssueFlowGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro Workbench")
        self.root.geometry("450x780")
        self.root.resizable(False, False)
        
        style = ttk.Style()
        if 'clam' in style.theme_names(): style.theme_use('clam')
            
        # Mesh Variables
        self.grid_type = tk.StringVar(value="hex")
        self.cols_var = tk.StringVar(value="5")
        self.rows_var = tk.StringVar(value="5")
        self.noise_var = tk.StringVar(value="25.0")
        
        # Physics Variables
        self.ka_var = tk.StringVar(value="1.0")
        self.gamma_var = tk.StringVar(value="0.1")
        self.p0_var = tk.StringVar(value="3.81")
        self.lambda_var = tk.StringVar(value="1.0")
        
        # Solver & T1 Variables
        self.dt_var = tk.StringVar(value="0.1")
        self.t1_thresh_var = tk.StringVar(value="0.05")
        self.t1_rest_var = tk.StringVar(value="0.08")
        
        self.is_running = False
        self.fig = None
        self.ax_mesh = None
        self.ax_energy = None
        
        self.energy_history = []
        
        self.show_faces = tk.BooleanVar(value=True)
        self.show_verts = tk.BooleanVar(value=True)
        self.show_f_total = tk.BooleanVar(value=False) 
        
        self.tissue = None 
        self._build_tabs()

    def _build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tab_mesh = ttk.Frame(self.notebook, padding=10)
        self.tab_physics = ttk.Frame(self.notebook, padding=10)
        self.tab_solver = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_mesh, text=" 1. Mesh Control ")
        self.notebook.add(self.tab_physics, text=" 2. Physics Lab ")
        self.notebook.add(self.tab_solver, text=" 3. Live Solver ")

        self._populate_mesh_tab()
        self._populate_physics_tab()
        self._populate_solver_tab()

    def _populate_mesh_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_mesh, text=" Lattice Configuration ", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Label(frame, text="Grid Geometry:").grid(row=0, column=0, sticky="w", pady=5)
        r_frame = ttk.Frame(frame)
        r_frame.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Radiobutton(r_frame, text="Square", variable=self.grid_type, value="square").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(r_frame, text="Hexagonal", variable=self.grid_type, value="hex").pack(side=tk.LEFT)
        ttk.Label(frame, text="Grid Columns (nx):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.cols_var, width=15).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Grid Rows (ny):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.rows_var, width=15).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Structural Noise (%):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.noise_var, width=15).grid(row=3, column=1, sticky="w", pady=5)
        ttk.Button(self.tab_mesh, text="⚡ 1. Initialize Noisy Mesh", command=self.execute_pipeline).pack(fill='x', pady=20)

    def _populate_physics_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_physics, text=" Energy Weights ", padding=15)
        frame.pack(fill=tk.BOTH, pady=5)
        ttk.Label(frame, text="Area Elasticity (K_A):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.ka_var, width=12).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(frame, text="Perimeter Cont. (Γ_P):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.gamma_var, width=12).grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(frame, text="Target Shape Index (p_0):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.p0_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(frame, text="Line Tension (Λ):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.lambda_var, width=12).grid(row=3, column=1, padx=10, pady=5)

    def _populate_solver_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_solver, text=" Overdamped Viscous Engine ", padding=15)
        frame.pack(fill='x', pady=5)
        ttk.Label(frame, text="Time Step (dt):").grid(row=0, column=0, sticky="w", pady=10)
        ttk.Entry(frame, textvariable=self.dt_var, width=12).grid(row=0, column=1, padx=10, pady=10)
        ttk.Checkbutton(frame, text="Show Live Force Vectors", variable=self.show_f_total).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=15, sticky="ew")
        self.step_btn = ttk.Button(btn_frame, text="⏭ 1 Step", command=self.step_simulation, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        self.run_btn = ttk.Button(btn_frame, text="▶ Run ASAP", command=self.toggle_continuous_run, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        t1_frame = ttk.LabelFrame(self.tab_solver, text=" Topological Surgery (T1) ", padding=15)
        t1_frame.pack(fill='x', pady=5)
        ttk.Label(t1_frame, text="T1 Threshold:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(t1_frame, textvariable=self.t1_thresh_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(t1_frame, text="Resolved Len:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(t1_frame, textvariable=self.t1_rest_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.t1_btn = ttk.Button(t1_frame, text="✂️ Identify & Resolve T1s", command=self.manual_resolve_t1, state=tk.DISABLED)
        self.t1_btn.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        self.energy_readout = tk.StringVar(value="System Energy: --")
        ttk.Label(self.tab_solver, textvariable=self.energy_readout, font=("Courier", 10), justify="center", background="#f1f5f9", padding=10).pack(fill='x', pady=5)
        self.view_btn = ttk.Button(self.tab_solver, text="👁 Open Dual-Dashboard", command=self.force_plot, state=tk.DISABLED)
        self.view_btn.pack(fill='x', pady=5)

    def execute_pipeline(self) -> None:
        try:
            nx, ny, noise = int(self.cols_var.get()), int(self.rows_var.get()), float(self.noise_var.get())
        except ValueError: return
        self.tissue = TissueMesh()
        self.tissue.generate_mesh(self.grid_type.get(), nx, ny)
        if noise > 0: self.tissue.apply_vertex_noise(noise)
        self.energy_history = []
        self.step_btn.config(state=tk.NORMAL)
        self.run_btn.config(state=tk.NORMAL)
        self.view_btn.config(state=tk.NORMAL)
        self.t1_btn.config(state=tk.NORMAL)
        self.force_plot()

    def force_plot(self) -> None:
        if not self.tissue: return
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            plt.ion() 
            self.fig, (self.ax_mesh, self.ax_energy) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2.5, 1]})
            self.fig.canvas.mpl_connect('close_event', self.on_close_plot)
            self.fig.patch.set_facecolor('#f8fafc')
            plt.show(block=False) 
        self._update_plot_windows("Initial Noisy State")

    def on_close_plot(self, event):
        self.is_running = False
        self.run_btn.config(text="▶ Run ASAP")
        self.fig = None

    def manual_resolve_t1(self) -> None:
        if not self.tissue: return
        try:
            thresh = float(self.t1_thresh_var.get())
            rest = float(self.t1_rest_var.get())
        except ValueError: return
        swaps = self.tissue.resolve_t1_transitions(threshold=thresh, rest_length=rest, max_iter=20)
        if swaps > 0:
            messagebox.showinfo("Surgery Complete", f"Resolved {swaps} T1 transitions!")
            self._update_plot_windows("Manual T1 Surgery")
        else:
            messagebox.showwarning("Scan Complete", "No edges found under threshold.")

    def step_simulation(self) -> None:
        if not self.tissue: return
        try:
            ka, gamma, lam, p0 = float(self.ka_var.get()), float(self.gamma_var.get()), float(self.lambda_var.get()), float(self.p0_var.get())
            dt, thresh, rest = float(self.dt_var.get()), float(self.t1_thresh_var.get()), float(self.t1_rest_var.get())
        except ValueError: return
        energy_data = compute_system_energy(self.tissue, K_A=ka, Gamma_P=gamma, p_0=p0, Lambda=lam)
        self.energy_history.append(energy_data['Total'])
        self.tissue.step_viscous(dt=dt, eta=1.0)
        self.tissue.resolve_t1_transitions(threshold=thresh, rest_length=rest)
        if self.fig: self._update_plot_windows(f"Active Relaxation | dt={dt}")

    def _update_plot_windows(self, title: str):
        plot_mesh(self.tissue, ax=self.ax_mesh, title=title)
        self.ax_energy.clear()
        self.ax_energy.plot(self.energy_history, color='#ef4444', lw=2)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def toggle_continuous_run(self) -> None:
        self.is_running = not self.is_running
        self.run_btn.config(text="⏸ Pause Loop" if self.is_running else "▶ Run ASAP")
        if self.is_running: self._continuous_loop()

    def _continuous_loop(self) -> None:
        if self.is_running:
            self.step_simulation()
            self.root.after(1, self._continuous_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = TyssueFlowGUI(root)
    root.mainloop()