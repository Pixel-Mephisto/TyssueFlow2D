import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import traceback
import imageio

from mesh_engine import TissueMesh
from mesh_plotter import plot_mesh
from compute_energy import compute_system_energy 

class TyssueFlowGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro Workbench - Presenter Mode")
        self.root.geometry("1600x900") 
        self.root.resizable(True, True)
        
        # --- DARK THEME AESTHETICS ---
        self.bg_main = "#1e1e1e"
        self.bg_panel = "#252526"
        self.fg_text = "#d4d4d4"
        self.accent_blue = "#4fc1ff"
        self.btn_bg = "#333333"
        self.btn_active = "#007acc"
        
        self.root.configure(bg=self.bg_main)
        plt.style.use('dark_background')
        
        # --- MASSIVELY ENLARGED GLOBAL FONTS ---
        self.font_base = ('Helvetica', 14)
        self.font_bold = ('Helvetica', 14, 'bold')
        self.font_title = ('Helvetica', 16, 'bold')
        
        style = ttk.Style()
        if 'clam' in style.theme_names(): 
            style.theme_use('clam')
            
        style.configure('.', background=self.bg_main, foreground=self.fg_text, font=self.font_base)
        style.configure('TFrame', background=self.bg_main)
        style.configure('TLabelframe', background=self.bg_panel, bordercolor=self.bg_main)
        style.configure('TLabelframe.Label', background=self.bg_panel, foreground=self.accent_blue, font=self.font_title)
        
        style.configure('TEntry', fieldbackground=self.btn_bg, foreground='#ffffff', borderwidth=1, font=self.font_base)
        style.configure('TCheckbutton', background=self.bg_panel, foreground=self.fg_text, font=self.font_base)
        style.map('TCheckbutton', background=[('active', self.bg_panel)])
        style.configure('TRadiobutton', background=self.bg_panel, foreground=self.fg_text, font=self.font_base)
        style.map('TRadiobutton', background=[('active', self.bg_panel)])

        # --- Variables ---
        self.grid_type = tk.StringVar(value="hex")
        self.cols_var = tk.StringVar(value="8") 
        self.rows_var = tk.StringVar(value="8")
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
        self.is_recording = False
        self.video_writer = None
        self.energy_history = []
        self.show_f_total = tk.BooleanVar(value=False) 
        self.tissue = None 
        
        self.fig = None
        self.ax_mesh = None
        self.ax_energy = None
        self.plot_canvas = None # FIXED: Separated Plot Canvas from Scroll Canvas
        
        # --- UI LAYOUT: COLLAPSIBLE PANED WINDOW ---
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.bg_main, sashwidth=8, sashrelief=tk.RAISED)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_panel_container = tk.Frame(self.paned_window, bg=self.bg_main)
        self.paned_window.add(self.left_panel_container, minsize=50, width=520) 

        self.right_panel = tk.Frame(self.paned_window, bg=self.bg_main)
        self.paned_window.add(self.right_panel, minsize=400)

        # --- SCROLLABLE CANVAS SETUP ---
        # FIXED: Variable renamed to scroll_canvas to prevent overwriting Matplotlib event bindings
        self.scroll_canvas = tk.Canvas(self.left_panel_container, bg=self.bg_main, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_panel_container, orient="vertical", command=self.scroll_canvas.yview)
        self.scrollable_frame = tk.Frame(self.scroll_canvas, bg=self.bg_main)

        self.scrollable_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        def _configure_canvas(event):
            self.scroll_canvas.itemconfig(self.canvas_window, width=event.width)
            
        self.scroll_canvas.bind("<Configure>", _configure_canvas)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scroll_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        self.scrollbar.pack(side="right", fill="y", pady=10)

        def _on_mousewheel(event):
            if event.num == 4 or getattr(event, 'delta', 0) > 0:
                self.scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or getattr(event, 'delta', 0) < 0:
                self.scroll_canvas.yview_scroll(1, "units")

        self.scroll_canvas.bind('<Enter>', lambda _: self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scroll_canvas.bind('<Enter>', lambda _: self.scroll_canvas.bind_all("<Button-4>", _on_mousewheel), add="+")
        self.scroll_canvas.bind('<Enter>', lambda _: self.scroll_canvas.bind_all("<Button-5>", _on_mousewheel), add="+")
        
        self.scroll_canvas.bind('<Leave>', lambda _: self.scroll_canvas.unbind_all("<MouseWheel>"))
        self.scroll_canvas.bind('<Leave>', lambda _: self.scroll_canvas.unbind_all("<Button-4>"))
        self.scroll_canvas.bind('<Leave>', lambda _: self.scroll_canvas.unbind_all("<Button-5>"))

        self._build_controls()
        self._initialize_canvas()

    def _create_standard_button(self, parent, text, command, state=tk.NORMAL, fg=None):
        fg_color = fg if fg else self.fg_text
        return tk.Button(parent, text=text, command=command, state=state,
                         bg=self.btn_bg, fg=fg_color, font=self.font_bold,
                         activebackground=self.btn_active, activeforeground="white", 
                         relief=tk.RAISED, bd=2, pady=8, cursor="hand2")

    def _build_controls(self) -> None:
        self._populate_mesh_controls()
        self._populate_physics_controls()
        self._populate_solver_controls()

    def _populate_mesh_controls(self) -> None:
        frame = ttk.LabelFrame(self.scrollable_frame, text=" 1. Lattice Configuration ", padding=15)
        frame.pack(fill=tk.X, pady=(0, 10)) 
        inner = ttk.Frame(frame, style='TLabelframe'); inner.pack(fill=tk.X)

        ttk.Label(inner, text="Grid Geometry:").grid(row=0, column=0, sticky="w", pady=5)
        r_frame = ttk.Frame(inner, style='TLabelframe'); r_frame.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Radiobutton(r_frame, text="Square", variable=self.grid_type, value="square").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(r_frame, text="Hexagonal", variable=self.grid_type, value="hex").pack(side=tk.LEFT)
        
        ttk.Label(inner, text="Columns (nx):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.cols_var, width=12).grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(inner, text="Rows (ny):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.rows_var, width=12).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(inner, text="Structural Noise (%):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.noise_var, width=12).grid(row=3, column=1, sticky="w", pady=5)
        
        self.init_btn = self._create_standard_button(frame, "⚡ Initialize New Mesh", self.execute_pipeline)
        self.init_btn.pack(fill='x', pady=(15, 0))

    def _populate_physics_controls(self) -> None:
        frame = ttk.LabelFrame(self.scrollable_frame, text=" 2. Physics & Energy Weights ", padding=15)
        frame.pack(fill=tk.X, pady=10)
        inner = ttk.Frame(frame, style='TLabelframe'); inner.pack(fill=tk.X)

        ttk.Label(inner, text="Area Elasticity (K_A):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.ka_var, width=12).grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Perimeter Cont. (Γ_P):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.gamma_var, width=12).grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Target Shape Index (p_0):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.p0_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(inner, text="Line Tension (Λ):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.lambda_var, width=12).grid(row=3, column=1, padx=10, pady=5)

    def _populate_solver_controls(self) -> None:
        frame = ttk.LabelFrame(self.scrollable_frame, text=" 3. Live Viscous Solver ", padding=15)
        frame.pack(fill=tk.BOTH, expand=True, pady=10)
        inner = ttk.Frame(frame, style='TLabelframe'); inner.pack(fill=tk.BOTH)

        ttk.Label(inner, text="Time Step (dt):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.dt_var, width=12).grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Checkbutton(inner, text="Enable Langevin Noise", variable=self.use_brownian).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(inner, text="Noise Magnitude:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(inner, textvariable=self.brownian_mag_var, width=12).grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Checkbutton(inner, text="Show Live Force Vectors", variable=self.show_f_total).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        btn_frame = ttk.Frame(inner, style='TLabelframe'); btn_frame.grid(row=4, column=0, columnspan=2, pady=(15, 10), sticky="ew")
        self.step_btn = self._create_standard_button(btn_frame, "⏭ Step", self.step_simulation, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        self.run_btn = self._create_standard_button(btn_frame, "▶ Run", self.toggle_continuous_run, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        self.record_btn = self._create_standard_button(btn_frame, "⏺ Record", self.toggle_recording, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, expand=True, fill="x", padx=2)
        
        t1_frame = ttk.LabelFrame(inner, text=" Topological Surgery (T1) ", padding=10)
        t1_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        
        ttk.Label(t1_frame, text="T1 Threshold:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(t1_frame, textvariable=self.t1_thresh_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(t1_frame, text="Resolved Len:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(t1_frame, textvariable=self.t1_rest_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        self.t1_btn = self._create_standard_button(t1_frame, "✂️ Resolve T1s", self.manual_resolve_t1, state=tk.DISABLED)
        self.t1_btn.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        self.energy_readout = tk.StringVar(value="System Energy: --")
        s = ttk.Style()
        s.configure('TReadout.TLabel', background=self.btn_bg, foreground=self.accent_blue, font=("Courier", 16, "bold"))
        ttk.Label(frame, textvariable=self.energy_readout, style='TReadout.TLabel', justify="center", padding=15).pack(fill='x', side=tk.BOTTOM, pady=(15, 0))

    def _initialize_canvas(self) -> None:
        self.fig, (self.ax_mesh, self.ax_energy) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2.5, 1]})
        self.fig.patch.set_facecolor(self.bg_main)
        self.ax_mesh.set_facecolor(self.bg_panel)
        self.ax_energy.set_facecolor(self.bg_panel)
        self.ax_mesh.axis('off')
        self.ax_energy.axis('off')
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas_widget = self.plot_canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # --- VIDEO EXPORT LOGIC ---
    def toggle_recording(self) -> None:
        if not self.is_recording:
            fps = simpledialog.askinteger("Export FPS", "Enter frame rate (e.g., 15 or 30):", initialvalue=15, minvalue=1, maxvalue=60)
            if not fps: return
            
            filepath = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if not filepath: return
            
            try:
                # FIXED: macro_block_size=2 ensures the dimensions are valid for the H264 codec
                self.video_writer = imageio.get_writer(filepath, fps=fps, macro_block_size=2)
                self.is_recording = True
                self.record_btn.config(text="⏹ Stop Rec", fg="#ff4444")
            except Exception as e:
                messagebox.showerror("Export Error", f"Ensure 'imageio[ffmpeg]' is installed.\n\n{e}")
        else:
            self.stop_recording()

    def stop_recording(self) -> None:
        if self.is_recording and self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            self.is_recording = False
            self.record_btn.config(text="⏺ Record", fg=self.fg_text)
            messagebox.showinfo("Export Complete", "Animation saved successfully!")

    def execute_pipeline(self) -> None:
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
            if self.is_recording: self.stop_recording()

            self.tissue = TissueMesh()
            self.tissue.generate_mesh(self.grid_type.get(), nx, ny)
            if noise > 0: 
                self.tissue.apply_vertex_noise(noise)
                
            try:
                ka, gamma, lam, p0 = float(self.ka_var.get()), float(self.gamma_var.get()), float(self.lambda_var.get()), float(self.p0_var.get())
            except ValueError:
                ka, gamma, p0, lam = 1.0, 0.1, 3.81, 1.0

            compute_system_energy(self.tissue, K_A=ka, Gamma_P=gamma, p_0=p0, Lambda=lam)
            self.energy_history = []
            
            self.step_btn.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)
            self.t1_btn.config(state=tk.NORMAL)
            self.record_btn.config(state=tk.NORMAL)
            
            self._update_plot_windows("Lattice Initialized")

        except Exception as e:
            messagebox.showerror("Initialization Error", f"Traceback:\n\n{str(e)}\n\n{traceback.format_exc()}")

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
            
            self._update_plot_windows(f"Relaxation Flow | dt={dt}")
        except Exception as e:
            self.is_running = False
            self.run_btn.config(text="▶ Run ASAP")
            if self.is_recording: self.stop_recording()
            messagebox.showerror("Physics Engine Error", f"The math engine crashed:\n\n{str(e)}\n\n{traceback.format_exc()}")

    def _update_plot_windows(self, title: str):
        self.ax_mesh.clear()
        plot_mesh(self.tissue, ax=self.ax_mesh, show_f_total=self.show_f_total.get(), title=title)
        
        self.ax_energy.clear()
        self.ax_energy.set_facecolor(self.bg_panel)
        if len(self.energy_history) > 0:
            self.ax_energy.plot(self.energy_history, color=self.accent_blue, lw=3)
            
        self.ax_energy.set_title("System Energy Drop", fontsize=14, fontweight='bold', color=self.fg_text, pad=10)
        self.ax_energy.set_xlabel("Simulation Steps", color=self.fg_text, fontsize=12)
        self.ax_energy.tick_params(colors=self.fg_text, labelsize=10)
        
        for spine in self.ax_energy.spines.values():
            spine.set_color('#444444')
            
        self.ax_energy.grid(True, linestyle='--', color='#444444', alpha=0.6)
        self.ax_energy.relim()
        self.ax_energy.autoscale_view()
        
        self.plot_canvas.draw()
        self.plot_canvas.flush_events()
        
        # FIXED: Forces the imageio buffer to extract properly sized, strictly even frames
        if self.is_recording and self.video_writer is not None:
            width, height = self.fig.canvas.get_width_height()
            buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape((height, width, 4))[:, :, :3]  # Drop the alpha channel for MP4
            
            # Crop 1 pixel off if the user resizes the window to an odd number
            if img.shape[0] % 2 != 0: img = img[:-1, :, :]
            if img.shape[1] % 2 != 0: img = img[:, :-1, :]
            
            self.video_writer.append_data(img)

    def toggle_continuous_run(self) -> None:
        self.is_running = not self.is_running
        self.run_btn.config(text="⏸ Pause" if self.is_running else "▶ Run")
        
        if not self.is_running and self.is_recording:
            self.stop_recording()
            
        if self.is_running: 
            self._continuous_loop()

    def _continuous_loop(self) -> None:
        if self.is_running:
            self.step_simulation()
            self.root.after(1, self._continuous_loop)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TyssueFlowGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"\nFATAL TERMINAL ERROR RECORDED:\n{str(e)}")
        traceback.print_exc()
        input("\n[CRASH LOGGED] Press Enter to exit terminal...")