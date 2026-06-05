import tkinter as tk
from tkinter import ttk
import math

class ScrollableControlPanel(tk.Frame):
    def __init__(self, parent, ui_styles, *args, **kwargs):
        super().__init__(parent, bg=ui_styles['bg_main'], *args, **kwargs)
        self.styles = ui_styles
        
        self.canvas = tk.Canvas(self, bg=ui_styles['bg_main'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas, bg=ui_styles['bg_main'])

        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=10)
        self.scrollbar.pack(side="right", fill="y", pady=10)
        self._bind_mouse_scrolling()

    def _bind_mouse_scrolling(self):
        def _scroll(event):
            if event.num == 4 or getattr(event, 'delta', 0) > 0: self.canvas.yview_scroll(-1, "units")
            else: self.canvas.yview_scroll(1, "units")
        self.canvas.bind('<Enter>', lambda _: self.canvas.bind_all("<MouseWheel>", _scroll))
        self.canvas.bind('<Leave>', lambda _: self.canvas.unbind_all("<MouseWheel>"))

    def add_label_frame(self, title_text):
        """Creates a padded section box so widgets never intersect the borders."""
        frame = ttk.LabelFrame(self.inner_frame, text=title_text, padding=15)
        frame.pack(fill=tk.X, pady=10, padx=12)
        inner = tk.Frame(frame, bg=self.styles['bg_panel'])
        inner.pack(fill=tk.X, pady=5)
        return inner

    def add_slider(self, parent_frame, label_text, var, min_val, max_val, step, is_int=False):
        """Generates a responsive slider coupled with an editable entry box."""
        container = tk.Frame(parent_frame, bg=self.styles['bg_panel'])
        container.pack(fill=tk.X, pady=12, padx=5)

        header = tk.Frame(container, bg=self.styles['bg_panel'])
        header.pack(fill=tk.X)

        lbl = tk.Label(header, text=label_text, bg=self.styles['bg_panel'], fg=self.styles['fg_text'], font=('Helvetica', 13))
        lbl.pack(side=tk.LEFT)

        # Replaced the static Label with a responsive Entry box
        val_entry = tk.Entry(header, width=6, bg=self.styles['bg_main'], fg=self.styles['accent_blue'], 
                             font=('Courier', 14, 'bold'), borderwidth=1, justify="right", relief=tk.SUNKEN)
        val_entry.pack(side=tk.RIGHT)

        scale = ttk.Scale(container, variable=var, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X, pady=(8, 0))

        decimals = max(0, int(-math.log10(step))) if step < 1 else 0
        _updating = [False] # Prevents recursive looping between entry and slider

        def _update_from_scale(event=None):
            if _updating[0]: return
            _updating[0] = True
            val = scale.get()
            val = round(val / step) * step
            
            if is_int:
                var.set(int(val))
                val_entry.delete(0, tk.END)
                val_entry.insert(0, f"{int(val)}")
            else:
                var.set(val)
                val_entry.delete(0, tk.END)
                val_entry.insert(0, f"{val:.{decimals}f}")
            _updating[0] = False

        def _update_from_entry(event=None):
            if _updating[0]: return
            _updating[0] = True
            try:
                val = float(val_entry.get())
                if is_int: val = int(val)
                var.set(val)
                scale.set(val) # Update slider visual
            except ValueError:
                pass # Ignore if user types invalid text
            _updating[0] = False

        scale.configure(command=_update_from_scale)
        val_entry.bind('<Return>', _update_from_entry)
        val_entry.bind('<FocusOut>', _update_from_entry)
        
        scale.set(var.get())
        _update_from_scale()
        
        return container