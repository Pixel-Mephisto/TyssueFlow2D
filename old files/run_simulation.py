import tkinter as tk
from tkinter import ttk, messagebox
from mesh_engine import TissueMesh
from mesh_plotter import plot_mesh

class TyssueFlowGUI:
    """
    Native Desktop Controller for the TissueMesh Simulation.
    """
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro Workbench")
        self.root.geometry("360x460")
        self.root.resizable(False, False)
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        # Application State Variables
        self.grid_type = tk.StringVar(value="hex")
        self.cols_var = tk.StringVar(value="3")
        self.rows_var = tk.StringVar(value="3")
        self.noise_var = tk.StringVar(value="15.0")
        
        self.show_faces = tk.BooleanVar(value=True)
        self.show_edges = tk.BooleanVar(value=True)
        self.show_verts = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="TyssueFlow Pro", font=("Helvetica", 14, "bold"), foreground="#0369a1").grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")

        # Grid Selection
        ttk.Label(frame, text="Lattice Geometry:", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        r_frame = ttk.Frame(frame)
        r_frame.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Radiobutton(r_frame, text="Square", variable=self.grid_type, value="square").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(r_frame, text="Hexagonal", variable=self.grid_type, value="hex").pack(side=tk.LEFT)

        # Dimensional Parameters
        ttk.Label(frame, text="Grid Columns (nx):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.cols_var, width=12).grid(row=2, column=1, sticky="w", pady=5)

        ttk.Label(frame, text="Grid Rows (ny):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.rows_var, width=12).grid(row=3, column=1, sticky="w", pady=5)

        ttk.Label(frame, text="Interior Noise (%):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.noise_var, width=12).grid(row=4, column=1, sticky="w", pady=5)

        ttk.Separator(frame, orient='horizontal').grid(row=5, column=0, columnspan=2, sticky='ew', pady=15)

        # Rendering Toggles
        ttk.Label(frame, text="Rendering Engine:", font=("Helvetica", 10, "bold")).grid(row=6, column=0, sticky="w", pady=(0, 5))
        ttk.Checkbutton(frame, text="Render Cell Faces", variable=self.show_faces).grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="Render Half-Edge Vectors", variable=self.show_edges).grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="Render Vertices", variable=self.show_verts).grid(row=9, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="Display Topological Labels", variable=self.show_labels).grid(row=10, column=0, columnspan=2, sticky="w", pady=2)

        # Execution
        ttk.Button(frame, text="⚡ Compile & Render Topology", command=self.execute_pipeline).grid(row=11, column=0, columnspan=2, pady=(20, 0), sticky="ew")

    def execute_pipeline(self) -> None:
        """Sanitizes inputs and orchestrates the backend engine and rendering system."""
        try:
            nx = int(self.cols_var.get())
            ny = int(self.rows_var.get())
            noise = float(self.noise_var.get())
            
            if nx <= 0 or ny <= 0: raise ValueError("Grid dimensions must be positive integers.")
            if not (0.0 <= noise <= 100.0): raise ValueError("Noise must be between 0 and 100.")
                
        except ValueError as err:
            messagebox.showerror("Configuration Error", str(err))
            return

        # Core Mathematical Engine
        tissue = TissueMesh()
        tissue.generate_mesh(self.grid_type.get(), nx, ny)
        
        if noise > 0:
            tissue.apply_vertex_noise(noise)
            
        # Graphical Display
        title = f"{self.grid_type.get().upper()} Topology ({nx}x{ny}) | Noise: {noise}%"
        plot_mesh(
            tissue,
            show_faces=self.show_faces.get(),
            show_edges=self.show_edges.get(),
            show_verts=self.show_verts.get(),
            show_labels=self.show_labels.get(),
            title=title
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = TyssueFlowGUI(root)
    root.mainloop()