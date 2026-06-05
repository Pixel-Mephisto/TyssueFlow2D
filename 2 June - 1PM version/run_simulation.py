import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

from mesh_engine import TissueMesh
from mesh_plotter import plot_mesh
from compute_energy import compute_system_energy 

class TyssueFlowGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro Workbench")
        self.root.geometry("450x650")
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
        
        # Solver Variables
        self.dt_var = tk.StringVar(value="0.1")
        self.fig = None
        self.ax = None
        
        # Display Toggles
        self.show_faces = tk.BooleanVar(value=True)
        self.show_verts = tk.BooleanVar(value=True)
        self.show_f_total = tk.BooleanVar(value=True) # Great for debugging steps
        
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
        self.notebook.add(self.tab_solver, text=" 3. Step Solver ")

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
        ttk.Label(frame, text="Perimeter Cont. (\u0393_P):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.gamma_var, width=12).grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(frame, text="Target Shape Index (p_0):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.p0_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(frame, text="Line Tension (\u039B):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.lambda_var, width=12).grid(row=3, column=1, padx=10, pady=5)

    def _populate_solver_tab(self) -> None:
        frame = ttk.LabelFrame(self.tab_solver, text=" Manual Viscous Stepper ", padding=15)
        frame.pack(fill='x', pady=5)
        
        ttk.Label(frame, text="Time Step (dt):").grid(row=0, column=0, sticky="w", pady=10)
        ttk.Entry(frame, textvariable=self.dt_var, width=12).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Checkbutton(frame, text="Show Live Force Vectors (Orange)", variable=self.show_f_total).grid(row=1, column=0, columnspan=2, sticky="w", pady=10)
        
        # New Manual Step Button
        self.step_btn = ttk.Button(self.tab_solver, text="⏭ Execute 1 Physics Step", command=self.step_simulation, state=tk.DISABLED)
        self.step_btn.pack(fill='x', pady=20)
        
        self.energy_readout = tk.StringVar(value="System Energy: --")
        ttk.Label(self.tab_solver, textvariable=self.energy_readout, font=("Courier", 10), justify="center", background="#f1f5f9", padding=10).pack(fill='x', pady=5)
        
        # Add a reset view button just in case the window gets closed
        self.view_btn = ttk.Button(self.tab_solver, text="👁 Re-open Plot Window", command=self.force_plot, state=tk.DISABLED)
        self.view_btn.pack(fill='x', pady=5)

    def execute_pipeline(self) -> None:
        try:
            nx, ny, noise = int(self.cols_var.get()), int(self.rows_var.get()), float(self.noise_var.get())
        except ValueError:
            return

        self.tissue = TissueMesh()
        self.tissue.generate_mesh(self.grid_type.get(), nx, ny)
        if noise > 0: self.tissue.apply_vertex_noise(noise)
            
        self.step_btn.config(state=tk.NORMAL)
        self.view_btn.config(state=tk.NORMAL)
        
        self.force_plot()

    def force_plot(self) -> None:
        """Forces the Matplotlib window to open and show the current static state."""
        if not self.tissue: return
        
        # If window doesn't exist, make a new one
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            # block=False allows Tkinter to keep running while the plot is open
            plt.show(block=False) 
            
        plot_mesh(self.tissue, ax=self.ax, show_labels=False, title="Current State")
        self.fig.canvas.draw()

    def step_simulation(self) -> None:
        """Executes exactly ONE step of overdamped physics and refreshes the plot."""
        if not self.tissue: return
        
        try:
            ka, gamma, lam, p0 = float(self.ka_var.get()), float(self.gamma_var.get()), float(self.lambda_var.get()), float(self.p0_var.get())
            dt = float(self.dt_var.get())
        except ValueError:
            messagebox.showerror("Error", "Check physics parameters.")
            return
            
        # 1. Physics: Compute Gradients & Energy
        energy_data = compute_system_energy(self.tissue, K_A=ka, Gamma_P=gamma, p_0=p0, Lambda=lam)
        self.energy_readout.set(f"Live Energy: {energy_data['Total']:,.2f}")
        
        # 2. Dynamics: Advance Coordinates by dt
        self.tissue.step_viscous(dt=dt, eta=1.0)
        
        # 3. Render: Ensure window is open, then draw
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.show(block=False)
            
        plot_mesh(self.tissue, ax=self.ax, show_faces=self.show_faces.get(), 
                  show_verts=self.show_verts.get(), show_labels=False, 
                  show_f_total=self.show_f_total.get(), title=f"Stepping Dynamics | dt={dt}")
                  
        # Safely force Matplotlib to update the canvas graphics
        self.fig.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TyssueFlowGUI(root)
    root.mainloop()