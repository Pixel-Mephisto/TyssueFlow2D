import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import traceback
import cupy as cp

from model.config import MeshConfig, PhysicsConfig, T1Config, BiochemConfig
from core.mesh_factory import build_lattice
from core.io_exporter import TissueExporter
from view.control_panels import ScrollableControlPanel
from view.plot_renderer import render_mesh_to_axis
from controller.sim_manager import advance_simulation_frame

class MainDashboardApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TyssueFlow Pro - GPU Mechanochemical Engine")
        self.root.geometry("1600x900")
        
        self.styles = {
            'bg_main': "#1e1e1e", 'bg_panel': "#252526", 'fg_text': "#d4d4d4",
            'accent_blue': "#4fc1ff", 'btn_bg': "#333333", 'btn_active': "#007acc"
        }
        self.root.configure(bg=self.styles['bg_main'])
        plt.style.use('dark_background')
        
        style = ttk.Style()
        if 'clam' in style.theme_names(): style.theme_use('clam')
        style.configure('.', background=self.styles['bg_main'], foreground=self.styles['fg_text'], font=('Helvetica', 14))
        style.configure('TLabelframe', background=self.styles['bg_panel'], bordercolor=self.styles['bg_main'])
        style.configure('TLabelframe.Label', background=self.styles['bg_panel'], foreground=self.styles['accent_blue'], font=('Helvetica', 16, 'bold'))
        style.configure('TScale', background=self.styles['bg_panel'], troughcolor=self.styles['bg_main'], sliderthickness=20, sliderlength=20)
        style.map('TScale', slidercolor=[('active', self.styles['accent_blue']), ('!active', self.styles['accent_blue'])])
        style.configure('TCheckbutton', background=self.styles['bg_panel'], foreground=self.styles['fg_text'])
        style.map('TCheckbutton', background=[('active', self.styles['bg_panel'])])
        style.configure('TRadiobutton', background=self.styles['bg_panel'], foreground=self.styles['fg_text'])
        style.map('TRadiobutton', background=[('active', self.styles['bg_panel'])])

        self.m_cfg = MeshConfig()
        self.p_cfg = PhysicsConfig()
        self.t1_cfg = T1Config()
        self.bio_cfg = BiochemConfig()
        
        self._init_tk_variables()
        self.tissue = None
        self.is_running = False
        self.exporter = TissueExporter()
        self.current_time = 0.0

        self._assemble_layout()

    def _init_tk_variables(self):
        self.var_grid_type = tk.StringVar(value=self.m_cfg.grid_type)
        self.var_nx = tk.IntVar(value=self.m_cfg.nx)
        self.var_ny = tk.IntVar(value=self.m_cfg.ny)
        self.var_noise = tk.IntVar(value=int(self.m_cfg.noise_percent))

        self.var_ka = tk.DoubleVar(value=self.p_cfg.K_A)
        self.var_gamma = tk.DoubleVar(value=self.p_cfg.Gamma_P)
        self.var_p0 = tk.DoubleVar(value=self.p_cfg.p_0)
        
        init_p_ratio = self.p_cfg.Lambda / (2.0 * self.p_cfg.Gamma_P) if self.p_cfg.Gamma_P != 0 else 0.0
        self.var_p_ratio = tk.DoubleVar(value=init_p_ratio)

        self.var_dt = tk.DoubleVar(value=self.p_cfg.dt)
        self.var_b_mag = tk.DoubleVar(value=self.p_cfg.brownian_mag)
        self.use_brownian_var = tk.BooleanVar(value=self.p_cfg.use_brownian)
        self.show_f_total = tk.BooleanVar(value=False)
        
        self.var_t1_th = tk.DoubleVar(value=self.t1_cfg.threshold)
        self.var_t1_rst = tk.DoubleVar(value=self.t1_cfg.rest_length)

        self.var_kt_ui = tk.DoubleVar(value=self.bio_cfg.kT0 * 1e5) 
        self.var_chem_steps = tk.IntVar(value=self.bio_cfg.steps_per_mech)
        self.var_plot_interval = tk.DoubleVar(value=1.0) 

    def _assemble_layout(self):
        paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=self.styles['bg_main'], sashwidth=8, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ScrollableControlPanel(paned, self.styles)
        paned.add(self.sidebar, width=540)

        self.canvas_panel = tk.Frame(paned, bg=self.styles['bg_main'])
        paned.add(self.canvas_panel, minsize=400)
        
        self.plot_canvas = tk.Canvas(self.canvas_panel, bg=self.styles['bg_main'], highlightthickness=0)
        self.plot_scrollbar = ttk.Scrollbar(self.canvas_panel, orient="vertical", command=self.plot_canvas.yview)
        self.plot_inner = tk.Frame(self.plot_canvas, bg=self.styles['bg_main'])

        self.plot_inner.bind("<Configure>", lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))
        self.plot_window = self.plot_canvas.create_window((0, 0), window=self.plot_inner, anchor="nw")
        self.plot_canvas.bind("<Configure>", lambda e: self.plot_canvas.itemconfig(self.plot_window, width=e.width))
        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar.set)

        self.plot_canvas.pack(side="left", fill="both", expand=True)
        self.plot_scrollbar.pack(side="right", fill="y")
        
        def _scroll_plot(event):
            if event.num == 4 or getattr(event, 'delta', 0) > 0: self.plot_canvas.yview_scroll(-1, "units")
            else: self.plot_canvas.yview_scroll(1, "units")
        self.plot_canvas.bind('<Enter>', lambda _: self.plot_canvas.bind_all("<MouseWheel>", _scroll_plot))
        self.plot_canvas.bind('<Leave>', lambda _: self.plot_canvas.unbind_all("<MouseWheel>"))

        self._build_sidebar_widgets()
        self._embed_graph_canvas()

    def _create_btn(self, parent, text, cmd, bg=None):
        return tk.Button(parent, text=text, command=cmd, bg=bg if bg else self.styles['btn_bg'],
                         fg=self.styles['fg_text'], font=('Helvetica', 14, 'bold'), relief=tk.RAISED, bd=2, pady=10, cursor="hand2")

    def _build_sidebar_widgets(self):
        mech_accordion = self.sidebar.add_collapsible_section("Mechanical Parameters", expanded=False)

        m_box = self.sidebar.add_label_frame(mech_accordion, " 1. Lattice Configuration ")
        radio_frame = tk.Frame(m_box, bg=self.styles['bg_panel'])
        radio_frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(radio_frame, text="Geometry:", bg=self.styles['bg_panel'], fg=self.styles['fg_text'], font=('Helvetica', 13)).pack(side=tk.LEFT)
        ttk.Radiobutton(radio_frame, text="Square", variable=self.var_grid_type, value="square").pack(side=tk.RIGHT, padx=5)
        ttk.Radiobutton(radio_frame, text="Hexagonal", variable=self.var_grid_type, value="hex").pack(side=tk.RIGHT, padx=5)
        
        self.sidebar.add_slider(m_box, "Columns (nₓ):", self.var_nx, 1, 20, 1, is_int=True)
        self.sidebar.add_slider(m_box, "Rows (nᵧ):", self.var_ny, 1, 20, 1, is_int=True)
        self.sidebar.add_slider(m_box, "Noise (%):", self.var_noise, 0, 100, 1, is_int=True)
        self._create_btn(m_box, "⚡ Initialize Tissue Mesh", self.trigger_initialization).pack(fill=tk.X, pady=(15, 5))

        p_box = self.sidebar.add_label_frame(mech_accordion, " 2. Physics & Energy Weights ")
        self.sidebar.add_slider(p_box, "Area Elasticity (Kₐ):", self.var_ka, 0.0, 5.0, 0.01)
        self.sidebar.add_slider(p_box, "Perim Contractility (Γₚ):", self.var_gamma, 0.0, 1.0, 0.01)
        self.sidebar.add_slider(p_box, "Target Shape Index (p₀):", self.var_p0, 0.0, 5.0, 0.01)
        self.sidebar.add_slider(p_box, "Adhesion Ratio (p = Λ / 2Γₚ):", self.var_p_ratio, -5.0, 5.0, 0.01)

        t1_box = self.sidebar.add_label_frame(mech_accordion, " 3. Topological Surgery (T1) ")
        self.sidebar.add_slider(t1_box, "T1 Threshold:", self.var_t1_th, 0.01, 0.15, 0.01)
        self.sidebar.add_slider(t1_box, "Resolved Length:", self.var_t1_rst, 0.01, 0.20, 0.01)

        bio_accordion = self.sidebar.add_collapsible_section("Biochemical Parameters", expanded=True)
        
        b_box = self.sidebar.add_label_frame(bio_accordion, " Notch-Delta Kinetics ")
        self.sidebar.add_slider(b_box, "Trans-Coupling (kₜ) × 10⁻⁵:", self.var_kt_ui, 0.1, 10.0, 0.1)
        self.sidebar.add_slider(b_box, "Chem Steps per Mech Step:", self.var_chem_steps, 1, 20, 1, is_int=True)
        
        s_box = self.sidebar.add_label_frame(self.sidebar.inner_frame, " Global Viscous Solver ")
        self.sidebar.add_slider(s_box, "UI Refresh Rate (Mins):", self.var_plot_interval, 0.1, 5.0, 0.1)
        
        toggle_frame = tk.Frame(s_box, bg=self.styles['bg_panel'])
        toggle_frame.pack(fill=tk.X, pady=10, padx=5)
        ttk.Checkbutton(toggle_frame, text="Langevin Fluctuations", variable=self.use_brownian_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(toggle_frame, text="Show Live Force Vectors", variable=self.show_f_total).pack(side=tk.RIGHT, padx=5)
        
        self.sidebar.add_slider(s_box, "Brownian Mag:", self.var_b_mag, 0.0, 1.0, 0.05)

        btn_frame = tk.Frame(s_box, bg=self.styles['bg_panel'])
        btn_frame.pack(fill=tk.X, pady=(15, 5))
        self.step_btn = self._create_btn(btn_frame, "⏭ Step", self.step_one_frame)
        self.step_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.run_btn = self._create_btn(btn_frame, "▶ Run", self.toggle_continuous_run, bg="#238636")
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.rec_btn = self._create_btn(btn_frame, "⏺ Record", self.trigger_video_toggle, bg="#da3633")
        self.rec_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

    def _embed_graph_canvas(self):
        # Changed to a single, large plot for the mesh only
        self.fig, self.ax_mesh = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.patch.set_facecolor(self.styles['bg_main'])
        self.ax_mesh.set_facecolor(self.styles['bg_panel'])
        self.ax_mesh.axis('off')
        
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        
        self.canvas_driver = FigureCanvasTkAgg(self.fig, master=self.plot_inner)
        self.canvas_driver.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _read_ui_parameters(self) -> bool:
        try:
            self.root.focus_set()
            
            self.m_cfg.grid_type = self.var_grid_type.get()
            self.m_cfg.nx = self.var_nx.get()
            self.m_cfg.ny = self.var_ny.get()
            self.m_cfg.noise_percent = float(self.var_noise.get())
            
            self.p_cfg.K_A = self.var_ka.get()
            self.p_cfg.Gamma_P = self.var_gamma.get()
            self.p_cfg.p_0 = self.var_p0.get()
            self.p_cfg.Lambda = self.var_p_ratio.get() * 2.0 * self.p_cfg.Gamma_P
            self.p_cfg.dt = self.var_dt.get()
            self.p_cfg.brownian_mag = self.var_b_mag.get()
            self.p_cfg.use_brownian = self.use_brownian_var.get()
            
            self.t1_cfg.threshold = self.var_t1_th.get()
            self.t1_cfg.rest_length = self.var_t1_rst.get()
            
            self.bio_cfg.kT0 = self.var_kt_ui.get() * 1e-5
            self.bio_cfg.steps_per_mech = self.var_chem_steps.get()
            
            return True
        except Exception:
            messagebox.showerror("Numerical Error", "Please ensure parameters are valid real numbers.")
            return False

    def trigger_initialization(self):
        try:
            if not self._read_ui_parameters(): return
            
            # CLEAR CACHE: Flush GPU memory pools from previous runs to prevent caching OOM errors
            cp.get_default_memory_pool().free_all_blocks()
            
            self.is_running = False
            self.run_btn.config(text="▶ Run Active Solver")
            if self.exporter.is_recording: self.trigger_video_toggle()

            self.tissue = build_lattice(self.m_cfg.grid_type, self.m_cfg.nx, self.m_cfg.ny, self.m_cfg.noise_percent)
            self.current_time = 0.0
            self._sync_and_redraw("Biochemical Lattice Loaded")
            
        except Exception as e:
            messagebox.showerror("Error", f"Lattice compilation failed:\n{str(e)}\n{traceback.format_exc()}")

    def step_one_frame(self):
        if not self.tissue: return
        if not self._read_ui_parameters(): return
        
        global_dt = 0.1 
        mech_substeps = 2 
        chem_substeps = self.bio_cfg.steps_per_mech 
        
        plot_interval = self.var_plot_interval.get()
        frames_to_skip = max(1, int(round(plot_interval / global_dt)))
        
        swaps = advance_simulation_frame(
            self.tissue, self.p_cfg, self.t1_cfg, self.bio_cfg, 
            self.m_cfg.grid_type, global_dt, mech_substeps, chem_substeps, frames_to_skip
        )
        
        real_time_passed = frames_to_skip * global_dt
        self.current_time += real_time_passed
        
        self._sync_and_redraw(f"Actual Time: {self.current_time:.2f} mins")

    def _sync_and_redraw(self, status_msg):
        render_mesh_to_axis(self.tissue, self.ax_mesh, show_forces=self.show_f_total.get(), title=status_msg)
        self.canvas_driver.draw()
        self.canvas_driver.flush_events()
        
        if self.exporter.is_recording:
            self.exporter.capture_frame(self.fig)

    def trigger_video_toggle(self):
        if not self.exporter.is_recording:
            fps = simpledialog.askinteger("FPS Options", "Frame Rate (FPS):", initialvalue=15, minvalue=1, maxvalue=60)
            if not fps: return
            path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video File", "*.mp4")])
            if path:
                success = self.exporter.start_recording(path, fps)
                if success:
                    self.rec_btn.config(text="⏹ Finalize Video File", fg="#ff4444")
                else:
                    messagebox.showerror("IO Fault", "Failed to lock video codec pipelines.")
        else:
            self.exporter.close_stream()
            self.rec_btn.config(text="⏺ Record Movie File", fg=self.styles['fg_text'])
            messagebox.showinfo("IO Complete", "Compressed video container file written safely.")

    def manual_resolve_t1(self):
        messagebox.showinfo("Scanner Update", f"To run manual T1s with GPU architecture, please Step the simulation forward.")

    def toggle_continuous_run(self):
        if not self.tissue: return
        self.is_running = not self.is_running
        self.run_btn.config(text="⏸ Pause Simulation" if self.is_running else "▶ Run Active Solver")
        if not self.is_running and self.exporter.is_recording:
            self.trigger_video_toggle()
        if self.is_running: self._simulation_loop()

    def _simulation_loop(self):
        if self.is_running:
            self.step_one_frame()
            self.root.after(1, self._simulation_loop)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MainDashboardApp(root)
        root.mainloop()
    except Exception as e:
        print(f"\n[FATAL WORKBENCH CRASH DETECTED]:\n{str(e)}")
        traceback.print_exc()
        input("[CRASH CAPTURED IN STACK] Press Enter to safely close layout terminal...")