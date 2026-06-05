import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
# Force Matplotlib to use TkAgg to prevent cross-platform event loop thread deadlocks
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
import traceback

from mesh_engine import TissueMesh
from mesh_plotter import plot_mesh
from compute_energy import compute_system_energy 

class TyssueFlowGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro Workbench")
        self.root.geometry("460x860")
        self.root.resizable(False, False)
        
        # --- DARK THEME AESTHETICS ---
        self.bg_main = "#1e1e1e"
        self.bg_panel = "#252526"
        self.fg_text = "#d4d4d4"
        self.accent_blue = "#4fc1ff"
        self.btn_bg = "#333333"
        self.btn_active = "#007acc"
        
        self.root.configure(bg=self.bg_main)
        plt.style.use('dark_background')
        
        style = ttk.Style()
        if 'clam' in style.theme_names(): 
            style.theme_use('clam')
            
        style.configure('.', background=self.bg_main, foreground=self.fg_text)
        style.configure('TFrame', background=self.bg_main)
        style.configure('TNotebook', background=self.bg_main, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.bg_panel, foreground=self.fg_text, padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', self.accent_blue)], foreground=[('selected', '#ffffff')])
        
        style.configure('TLabelframe', background=self.bg_panel, bordercolor=self.bg_main)
        style.configure('TLabelframe.Label', background=self.bg_panel, foreground=self.accent_blue, font=('Helvetica', 10, 'bold'))
        
        style.configure('TEntry', fieldbackground=self.btn_bg, foreground='#ffffff', borderwidth=1)
        style.configure('TCheckbutton', background=self.bg_panel, foreground=self.fg_text)
        style.map('TCheckbutton', background=[('active', self.bg_panel)])
        style.configure('TRadiobutton', background=self.bg_panel, foreground=self.fg_text)
        style.map('TRadiobutton', background=[('active', self.bg_panel)])

        # --- Variables ---
        self.grid_type = tk.StringVar(value="hex")
        self.cols_var = tk.StringVar(value="5")
        self.rows_var = tk.StringVar(value="5")
        self.noise_var = tk.StringVar(value="25.0")
        
        self.ka_var = tk.StringVar(value="1.0")
        self.gamma_var = tk.StringVar(value="0.1")
        self.p0_var = tk.StringVar(value="3.81")
        self.lambda_var = tk.StringVar(value="1.0")
        
        self.dt_var = tk.StringVar(value="0.05")
        self.use_brownian = tk.BooleanVar(value=False)
        self.brownian_mag_var = tk.StringVar(value="0.1")
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

    def _create_standard_button(self, parent, text, command, state=tk.NORMAL):
        """Helper to create highly responsive, cross-platform native buttons."""
        return tk.Button(parent, text=text, command=command, state=state,
                         bg=self.btn_bg, fg=self.fg_text, 
                         activebackground=self.btn_active, activeforeground="white", 
                         relief=tk.RAISED, bd=1, pady=5, cursor="hand2")

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
        # Use tk.X without expand=True to preserve native layout flow
        frame.pack(fill=tk.X, pady=5) 
        
        inner = ttk.Frame(frame, style='TLabelframe')
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="Grid Geometry:", background=self.bg_panel).grid(row=0, column=0, sticky="w", pady=5)
        r_frame = ttk.Frame(inner, style='TLabelframe')
        r_frame.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Radiobutton(r_frame, text="Square", variable=self.grid_type, value="square").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(r_frame, text="Hexagonal", variable=self.grid_type, value="hex").pack(side=tk.LEFT)
        ttk.Label(inner, text="Columns (nx):", background=self.bg_panel).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.cols_var, width=15).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(inner, text="Rows (ny):", background=self.bg_panel).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.rows_var, width=15).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(inner, text="Structural Noise (%):", background=self.bg_panel).grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.noise_var, width=15).grid(row=3, column=1, sticky="w", pady=5)
        
        self.init_btn = self._create_standard_button(self.tab_mesh, "⚡ 1. Initialize Noisy Mesh", self.execute_pipeline)
        self.init_btn.pack(fill='x', pady=20)

    def _populate_physics_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_physics, text=" Energy Weights ", padding=15)
        frame.pack(fill=tk.X, pady=5)
        inner = ttk.Frame(frame, style='TLabelframe')
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="Area Elasticity (K_A):", background=self.bg_panel).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.ka_var, width=12).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Perimeter Cont. (Γ_P):", background=self.bg_panel).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.gamma_var, width=12).grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Target Shape Index (p_0):", background=self.bg_panel).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.p0_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Line Tension (Λ):", background=self.bg_panel).grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.lambda_var, width=12).grid(row=3, column=1, padx=10, pady=5)

    def _populate_solver_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_solver, text=" Overdamped Viscous Engine ", padding=15)
        frame.pack(fill='x', pady=5)
        inner = ttk.Frame(frame, style='TLabelframe')
        inner.pack(fill=tk.BOTH)

        ttk.Label(inner, text="Time Step (dt):", background=self.bg_panel).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.dt_var, width=12).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Checkbutton(inner, text="Enable Brownian Noise", variable=self.use_brownian).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(inner, text="Noise Magnitude:", background=self.bg_panel).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.brownian_mag_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Checkbutton(inner, text="Show Live Force Vectors", variable=self.show_f_total).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        btn_frame = ttk.Frame(inner, style='TLabelframe')
        btn_frame.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")
        
        self.step_btn = self._create_standard_button(btn_frame, "⏭ 1 Step", self.step_simulation, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        self.run_btn = self._create_standard_button(btn_frame, "▶ Run ASAP", self.toggle_continuous_run, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        t1_frame = ttk.LabelFrame(self.tab_solver, text=" Topological Surgery (T1) ", padding=15)
        t1_frame.pack(fill='x', pady=5)
        t1_inner = ttk.Frame(t1_frame, style='TLabelframe')
        t1_inner.pack(fill=tk.BOTH)

        ttk.Label(t1_inner, text="T1 Threshold:", background=self.bg_panel).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(t1_inner, textvariable=self.t1_thresh_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(t1_inner, text="Resolved Len:", background=self.bg_panel).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(t1_inner, textvariable=self.t1_rest_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.t1_btn = self._create_standard_button(t1_inner, "✂️ Identify & Resolve T1s", self.manual_resolve_t1, state=tk.DISABLED)
        self.t1_btn.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        self.energy_readout = tk.StringVar(value="System Energy: --")
        
        s = ttk.Style()
        s.configure('TReadout.TLabel', background=self.btn_bg, foreground=self.accent_blue, font=("Courier", 10, "bold"))
        ttk.Label(self.tab_solver, textvariable=self.energy_readout, style='TReadout.TLabel', justify="center", padding=10).pack(fill='x', pady=10)
        
        self.view_btn = self._create_standard_button(self.tab_solver, "👁 Open Visualizer", self.force_plot, state=tk.DISABLED)
        self.view_btn.pack(fill='x', pady=5)

    def execute_pipeline(self) -> None:
        """Fully wrapped in Try/Except to prevent silent failures."""
        try:
            try:
                nx = int(self.cols_var.get())
                ny = int(self.rows_var.get())
                noise = float(self.noise_var.get())
            except ValueError: 
                messagebox.showerror("Input Error", "Please ensure nx, ny, and noise are valid numbers.")
                return

            self.is_running = False
            self.run_btn.config(text="▶ Run ASAP")

            # Initialize a fresh data instance 
            self.tissue = TissueMesh()
            self.tissue.generate_mesh(self.grid_type.get(), nx, ny)
            if noise > 0: 
                self.tissue.apply_vertex_noise(noise)
                
            self.tissue.compute_geometry()
            self.energy_history = []
            
            # Unlock core execution buttons
            self.step_btn.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)
            self.view_btn.config(state=tk.NORMAL)
            self.t1_btn.config(state=tk.NORMAL)
            
            self.force_plot()

        except Exception as e:
            messagebox.showerror("Initialization Error", f"A critical error occurred:\n\n{str(e)}\n\n{traceback.format_exc()}")

    def force_plot(self) -> None:
        if not self.tissue: return
        try:
            if self.fig is None or not plt.fignum_exists(self.fig.number):
                plt.ion() 
                self.fig, (self.ax_mesh, self.ax_energy) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.5, 1]})
                self.fig.patch.set_facecolor(self.bg_main)
                self.fig.canvas.mpl_connect('close_event', self.on_close_plot)
                plt.show(block=False) 

            self._update_plot_windows("Initial Noisy State")
        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred while plotting:\n\n{str(e)}\n\n{traceback.format_exc()}")

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
            b_mag = float(self.brownian_mag_var.get()) if self.use_brownian.get() else 0.0
        except ValueError: return
        
        try:
            energy_data = compute_system_energy(self.tissue, K_A=ka, Gamma_P=gamma, p_0=p0, Lambda=lam)
            self.energy_history.append(energy_data['Total'])
            self.energy_readout.set(f"Live Energy: {energy_data['Total']:,.2f}")
            
            self.tissue.step_viscous(dt=dt, eta=1.0, brownian_noise=b_mag)
            self.tissue.resolve_t1_transitions(threshold=thresh, rest_length=rest)
            
            if self.fig: self._update_plot_windows(f"Active Relaxation | dt={dt}")
        except Exception as e:
            self.is_running = False
            self.run_btn.config(text="▶ Run ASAP")
            messagebox.showerror("Physics Engine Error", f"The math engine crashed:\n\n{str(e)}\n\n{traceback.format_exc()}")

    def _update_plot_windows(self, title: str):
        plot_mesh(self.tissue, ax=self.ax_mesh, show_f_total=self.show_f_total.get(), title=title)
        
        self.ax_energy.clear()
        self.ax_energy.set_facecolor(self.bg_panel)
        if len(self.energy_history) > 0:
            self.ax_energy.plot(self.energy_history, color=self.accent_blue, lw=2)
            
        self.ax_energy.set_title("System Energy Drop", fontsize=10, fontweight='bold', color=self.fg_text, pad=10)
        self.ax_energy.set_xlabel("Simulation Steps", color=self.fg_text, fontsize=8)
        self.ax_energy.tick_params(colors=self.fg_text)
        
        for spine in self.ax_energy.spines.values():
            spine.set_color('#444444')
            
        self.ax_energy.grid(True, linestyle='--', color='#444444', alpha=0.6)
        
        # Ensures graph scales properly across re-initializations
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
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