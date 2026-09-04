import os
import json
import time
import threading
from queue import Queue, Empty
from collections import deque
from PIL import Image, ImageTk
import numpy as np
import cv2


import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from image_processor import rotate_image, fetch_roi_img, rotate_points_with_pad
from adjust_window import AdjustWindow
from tracker import SimpleTracker
from detector import Detector
from logger import Logger
from anomaly import AnimalBehaviorAnomalyDetector
from rolling_grid import RollingActivityGrid

class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command, colors, width=168, height=42, accent=False):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=colors['panel'],
            highlightthickness=0,
            cursor="hand2"
        )
        self.text = text
        self.command = command
        self.colors = colors
        self.accent = accent
        self.hover = False
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_click)
        self.bind("<Configure>", self._draw)
        self._draw()

    def _rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw(self, _event=None):
        self.delete("all")
        w = max(120, self.winfo_width(), int(self.cget("width")))
        h = max(36, self.winfo_height(), int(self.cget("height")))
        fill = self.colors['green'] if self.accent else self.colors['panel_alt']
        if self.hover:
            fill = self.colors['green_light'] if self.accent else self.colors['green']
        self._rounded_rect(2, 2, w - 2, h - 2, 18, fill=fill, outline=self.colors['green_light'], width=1)
        self.create_text(w // 2, h // 2, text=self.text, fill=self.colors['white'], anchor="center", font=("Segoe UI", 10, "bold"))

    def _on_enter(self, _event):
        self.hover = True
        self._draw()

    def _on_leave(self, _event):
        self.hover = False
        self._draw()

    def _on_click(self, _event):
        if self.command:
            self.command()


class RoundedMetricCard(tk.Canvas):
    def __init__(self, master, title, value, colors, width=150, height=78):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=colors['panel'],
            highlightthickness=0
        )
        self.colors = colors
        self.width = width
        self.height = height
        self.title = title
        self.value = value
        self.bind("<Configure>", self._draw)
        self._draw()

    def _rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw(self, _event=None):
        self.delete("all")
        w = max(120, self.winfo_width(), int(self.cget("width")))
        h = max(70, self.winfo_height(), int(self.cget("height")))
        self._rounded_rect(4, 4, w - 4, h - 4, 16, fill=self.colors['panel_alt'], outline=self.colors['green'], width=1)
        self.create_rectangle(12, 12, 16, h - 12, fill=self.colors['green_light'], outline="")
        self.create_text((w + 16) // 2, h // 2 - 12, text=self.title, fill=self.colors['muted'], anchor="center", font=("Segoe UI", 9))
        self.create_text((w + 16) // 2, h // 2 + 14, text=self.value, fill=self.colors['white'], anchor="center", font=("Segoe UI", 18, "bold"))

    def config(self, **kwargs):
        text = kwargs.pop("text", None)
        if text is not None:
            self.value = text
            self._draw()
        if kwargs:
            super().config(**kwargs)

    configure = config


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Állataktivitás Monitor")
        self.geometry("1320x760")
        self.minsize(1100, 650)
        self.protocol("WM_DELETE_WINDOW", self.on_quit)

        # # # # # # # # # # # # # # # # # # # # 
        # /////////////////////////////////// #
        # # # # # # # # # # # # # # # # # # # # 
        model_filename = 'best.pt'
        model_dir_name = 'models'
        self.parameters_dir_name = 'parameters'
        log_dir_name = 'data'
        # # # # # # # # # # # # # # # # # # # #
        # /////////////////////////////////// #
        # # # # # # # # # # # # # # # # # # # # 

        os.makedirs(model_dir_name, exist_ok=True)
        os.makedirs(self.parameters_dir_name, exist_ok=True)
        os.makedirs(log_dir_name, exist_ok=True)

        model_path = os.path.join(model_dir_name, model_filename)

        self.basename = None
        self.video_path = None
        self.cap = None
        self.rotation = 0
        self.pad_rotation = True
        self.objects = []
        self.detector = Detector(model_path)
        self.logger = Logger(log_dir_name)
        self.tracker = SimpleTracker(max_lost=15, dist_thresh=400)
        self.anomaly_detector = AnimalBehaviorAnomalyDetector()
        self.fps = 25
        self.playing = False
        self.activity_grid = None

  
        self.short_activity_window = deque()
        self.medium_activity_window = deque()
        self.long_activity_window = deque()
        self.short_activity_seconds = 30.0
        self.medium_activity_seconds = 120.0
        self.long_activity_seconds = 600.0

        self.enable_activity_grid = True

        self.crowd_threshold = 8

        self.low_activity_threshold = 10

        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=2)
        self.worker_thread = None
        self.stop_event = threading.Event()

        self.current_total = 0
        self.current_feeder = 0
        self.current_drinker = 0
        self.current_avg_move = 0.0
        self.current_sleeping = 0

        self.history = deque(maxlen=60)
        self.history_buffer = []
        self.last_history_append = 0.0
        self.last_plot_update = 0.0

        self.sleep_detect_window = 300
        self.sleep_detect_threshold = 0.5
        self.avg_move_long_period = deque(maxlen=self.sleep_detect_window)

        self._build_ui()
        self.after(200, self._update_gui)

    def _configure_style(self):
        self.colors = {
            'bg': '#1f2421',
            'panel': '#2b302c',
            'panel_alt': '#343a35',
            'green': '#0f6b4f',
            'green_light': '#19a974',
            'white': '#f4f7f5',
            'muted': '#b8c2bd',
            'canvas': '#111614',
            'grid': '#55615b',
            'red': '#e03131'
        }
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=self.colors['bg'], foreground=self.colors['white'])
        style.configure("TFrame", background=self.colors['bg'])
        style.configure("Header.TFrame", background=self.colors['panel'])
        style.configure("HeaderTitle.TLabel", background=self.colors['panel'], foreground=self.colors['white'], font=("Segoe UI", 16, "bold"))
        style.configure("HeaderInfo.TLabel", background=self.colors['panel'], foreground=self.colors['muted'], font=("Segoe UI", 10))
        style.configure("Panel.TFrame", background=self.colors['panel'])
        style.configure("Panel.TLabelframe", background=self.colors['panel'], foreground=self.colors['white'], relief="solid", borderwidth=1)
        style.configure("Panel.TLabelframe.Label", background=self.colors['panel'], foreground=self.colors['white'], font=("Segoe UI", 10, "bold"))
        style.configure("MetricTitle.TLabel", background=self.colors['panel'], foreground=self.colors['muted'], font=("Segoe UI", 9))
        style.configure("MetricValue.TLabel", background=self.colors['panel'], foreground=self.colors['white'], font=("Segoe UI", 18, "bold"))
        style.configure("TLabel", background=self.colors['bg'], foreground=self.colors['white'])
        style.configure("Panel.TLabel", background=self.colors['panel'], foreground=self.colors['white'])
        style.configure("TButton", padding=(12, 7), background=self.colors['panel_alt'], foreground=self.colors['white'])
        style.map("TButton", background=[("active", self.colors['green']), ("pressed", "#0b513d")])
        style.configure("Accent.TButton", foreground=self.colors['white'], background=self.colors['green'])
        style.map("Accent.TButton", background=[("active", self.colors['green_light']), ("pressed", "#0b513d")])
        style.configure("TPanedwindow", background=self.colors['bg'])
        style.configure("Sash", background=self.colors['green'])

    def _build_ui(self):
        self._configure_style()
        self.configure(bg=self.colors['bg'])

        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(side='top', fill='x')
        ttk.Label(header, text="Állataktivitás Monitor", style="HeaderTitle.TLabel").pack(side='left', padx=(18, 10), pady=14)
        self.lbl_status = ttk.Label(header, text="Nincs betöltött videó", style="HeaderInfo.TLabel")
        self.lbl_status.pack(side='left', pady=14)

        root = ttk.Frame(self)
        root.pack(
            side='top',
            fill='both',
            expand=True,
            padx=12,
            pady=12
        )

        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        main_paned = ttk.PanedWindow(
            root,
            orient='horizontal'
        )

        main_paned.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        controls = ttk.LabelFrame(
            main_paned,
            text="Vezérlés",
            style="Panel.TLabelframe"
        )

        main_paned.add(
            controls,
            weight=1
        )

        controls.rowconfigure(0, weight=1)
        controls.columnconfigure(0, weight=1)

        left_vertical_paned = ttk.PanedWindow(
            controls,
            orient='vertical'
        )

        left_vertical_paned.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=6,
            pady=6
        )

        top_controls = ttk.Frame(
            left_vertical_paned,
            style="Panel.TFrame"
        )

        left_vertical_paned.add(
            top_controls,
            weight=1
        )

        RoundedButton(
            top_controls,
            "Videó betöltése",
            self.import_video,
            self.colors,
            accent=True
        ).pack(fill='x', padx=12, pady=(14, 8))

        RoundedButton(
            top_controls,
            "Zónák beállítása",
            self.open_adjust,
            self.colors
        ).pack(fill='x', padx=12, pady=8)

        ttk.Separator(top_controls).pack(
            fill='x',
            padx=12,
            pady=14
        )

        ttk.Label(
            top_controls,
            text="A panelek mérete egérrel módosítható.",
            style="Panel.TLabel",
            wraplength=180,
            justify='left'
        ).pack(
            fill='x',
            padx=12
        )

        ttk.Separator(top_controls).pack(
            fill='x',
            padx=12,
            pady=14
        )

        self.lbl_zone_card = RoundedMetricCard(
            top_controls,
            "Zónák",
            "0",
            self.colors,
            width=168,
            height=70
        )

        self.lbl_zone_card.pack(
            fill='x',
            padx=12,
            pady=(0, 8)
        )

        self.lbl_fps_card = RoundedMetricCard(
            top_controls,
            "Videó FPS",
            "25",
            self.colors,
            width=168,
            height=70
        )

        self.lbl_fps_card.pack(
            fill='x',
            padx=12,
            pady=(0, 8)
        )

        inspector = ttk.LabelFrame(
            left_vertical_paned,
            text="Állatfigyelő",
            style="Panel.TLabelframe"
        )

        left_vertical_paned.add(
            inspector,
            weight=3
        )

        ttk.Label(
            inspector,
            text="Kiválasztott állat:",
            style="Panel.TLabel"
        ).pack(
            anchor='w',
            padx=8,
            pady=(8, 4)
        )

        self.selected_animal = tk.StringVar()
        self.selected_animal.set("Összes")

        self.animal_dropdown = ttk.Combobox(
            inspector,
            textvariable=self.selected_animal,
            state="readonly",
            values=["Összes"]
        )

        self.animal_dropdown.pack(
            fill='x',
            padx=8,
            pady=(0, 8)
        )

        self.animal_dropdown.bind(
            "<<ComboboxSelected>>",
            self.on_animal_selected
        )

        self.alert_listbox = tk.Listbox(
            inspector,
            height=8,
            bg="#1b211e",
            fg="#ff8080",
            font=("Consolas", 9),
            borderwidth=0,
            highlightthickness=0
        )

        self.alert_listbox.pack(
            fill='both',
            expand=True,
            padx=8,
            pady=(0, 8)
        )

        self.lbl_animal_info = tk.Label(
            inspector,
            text="Nincs kiválasztott állat",
            bg="#2b302c",
            fg="#d0d0d0",
            justify="left",
            anchor="w",
            font=("Segoe UI", 9)
        )

        self.lbl_animal_info.pack(
            fill='x',
            padx=8,
            pady=(0, 8)
        )

        content = ttk.PanedWindow(
            main_paned,
            orient='horizontal'
        )

        main_paned.add(
            content,
            weight=4
        )

        video_panel = ttk.LabelFrame(
            content,
            text="Élőkép és detektálás",
            style="Panel.TLabelframe"
        )

        video_panel.rowconfigure(0, weight=1)
        video_panel.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            video_panel,
            width=760,
            height=600,
            bg=self.colors['canvas'],
            highlightthickness=0
        )

        self.video_canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10
        )

        self.video_tkimg = None

        content.add(
            video_panel,
            weight=3
        )

        side_panel = ttk.Frame(
            content,
            style="Panel.TFrame"
        )

        side_panel.rowconfigure(0, weight=1)
        side_panel.columnconfigure(0, weight=1)

        content.add(
            side_panel,
            weight=1
        )

        vertical_paned = ttk.PanedWindow(
            side_panel,
            orient='vertical'
        )

        vertical_paned.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        metrics = ttk.LabelFrame(
            vertical_paned,
            text="Aktuális mérőszámok",
            style="Panel.TLabelframe"
        )
        vertical_paned.add(
            metrics,
            weight=1
        )
        
        metrics.columnconfigure(0, weight=1)
        metrics.columnconfigure(1, weight=1)
        self.lbl_total = self._create_metric(metrics, "Összes állat", "0", row=0, column=0)
        self.lbl_feeder = self._create_metric(metrics, "Etetőnél", "0", row=0, column=1)
        self.lbl_drinker = self._create_metric(metrics, "Itatónál", "0", row=1, column=0)
        self.lbl_avg = self._create_metric(metrics, "Átlagos mozgás", "0 px", row=1, column=1)
        self.lbl_sleep = self._create_metric(metrics, "Alvó egyedek", "0", row=2, column=0, columnspan=2)
        self.lbl_anomaly = self._create_metric(metrics, "Anomalia", "0", row=3, column=0, columnspan=2)
        plot_panel = ttk.LabelFrame(vertical_paned, text="Idősoros grafikonok", style="Panel.TLabelframe")
        vertical_paned.add(plot_panel,weight=3)
        plot_panel.rowconfigure(0, weight=1)
        plot_panel.columnconfigure(0, weight=1)

        self.plot_canvas = tk.Canvas(plot_panel, bg=self.colors['panel'], highlightthickness=0)
        self.plot_scrollbar = ttk.Scrollbar(plot_panel, orient='vertical', command=self.plot_canvas.yview)
        self.plot_canvas.configure(yscrollcommand=self.plot_scrollbar.set)
        self.plot_canvas.bind("<MouseWheel>", self._on_plot_mousewheel)
        self.plot_canvas.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        self.plot_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)

        self.plot_frame = ttk.Frame(self.plot_canvas, style="Panel.TFrame")
        self.plot_canvas.create_window((0, 0), window=self.plot_frame, anchor='nw')
        self.plot_frame.bind("<Configure>", lambda e: self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all")))

        self.fig = Figure(figsize=(4.2, 10.5), facecolor=self.colors['panel'])
        self.axes = [self.fig.add_subplot(5, 1, i + 1) for i in range(5)]
        self.fig.tight_layout(pad=2.2)
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        plot_widget = self.fig_canvas.get_tk_widget()
        plot_widget.configure(bg=self.colors['panel'], highlightthickness=0)
        plot_widget.bind("<MouseWheel>", self._on_plot_mousewheel)
        plot_widget.pack(fill='both', expand=True)

        self._init_plots()

    def _on_plot_mousewheel(self, event):
        self.plot_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _rounded_rect(self, canvas, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _draw_video_frame(self, image, total, feeder_count, drinker_count, avg_move, sleeping, anomaly_messages=None):
        anomaly_messages = anomaly_messages or []
        canvas_w = max(1, self.video_canvas.winfo_width(), int(self.video_canvas.cget("width")))
        canvas_h = max(1, self.video_canvas.winfo_height(), int(self.video_canvas.cget("height")))
        self.video_canvas.delete("all")
        self._rounded_rect(self.video_canvas, 8, 8, canvas_w - 8, canvas_h - 8, 24, fill="#0b100e", outline=self.colors['green'], width=2)

        image_w, image_h = image.size
        x = canvas_w // 2
        y = canvas_h // 2
        self.video_canvas.create_image(x, y, image=self.video_tkimg)

        left = x - image_w // 2
        top = y - image_h // 2
        right = x + image_w // 2
        bottom = y + image_h // 2
        self._rounded_rect(self.video_canvas, left - 4, top - 4, right + 4, bottom + 4, 18, fill="", outline=self.colors['green_light'], width=2)

        self._rounded_rect(self.video_canvas, left + 14, top + 14, left + 108, top + 44, 14, fill="#12382c", outline=self.colors['green_light'], width=1)
        self.video_canvas.create_oval(left + 28, top + 25, left + 36, top + 33, fill="#31ff9a", outline="")
        self.video_canvas.create_text(left + 48, top + 29, text="ÉLŐ", anchor="w", fill=self.colors['white'], font=("Segoe UI", 10, "bold"))

        status_text = f"Állat: {total}   Etető: {feeder_count}   Itató: {drinker_count}"
        self._rounded_rect(self.video_canvas, left + 122, top + 14, min(right - 14, left + 470), top + 44, 14, fill="#1b211e", outline="#2f473d", width=1)
        self.video_canvas.create_text(left + 140, top + 29, text=status_text, anchor="w", fill=self.colors['white'], font=("Segoe UI", 10))
        # =========================
        # ALERT LIST UPDATE
        # =========================

        self.alert_listbox.delete(0, tk.END)

        for msg in anomaly_messages[-10:]:

            timestamp = time.strftime("%H:%M:%S")

            self.alert_listbox.insert(
                tk.END,
                f"[{timestamp}] {msg}"
            )
        movement_ratio = max(0.0, min(1.0, avg_move / 250.0))
        bar_left = left + 18
        bar_right = right - 18
        bar_bottom = bottom - 18
        self._rounded_rect(self.video_canvas, bar_left, bar_bottom - 20, bar_right, bar_bottom, 9, fill="#1b211e", outline="#2f473d", width=1)
        self._rounded_rect(self.video_canvas, bar_left + 3, bar_bottom - 17, bar_left + 3 + int((bar_right - bar_left - 6) * movement_ratio), bar_bottom - 3, 7, fill=self.colors['green_light'], outline="")
        self.video_canvas.create_text(bar_left, bar_bottom - 30, text=f"Mozgásintenzitás: {avg_move:.1f} px / Alvó: {sleeping}", anchor="w", fill=self.colors['muted'], font=("Segoe UI", 9))

        corner_len = 38
        corner_top = top + 96 if anomaly_messages else top + 62
        for sx, sy in [(left + 16, corner_top), (right - 16, corner_top), (left + 16, bottom - 54), (right - 16, bottom - 54)]:
            if sx < (left + right) / 2:
                self.video_canvas.create_line(sx, sy, sx + corner_len, sy, fill=self.colors['green_light'], width=3)
            else:
                self.video_canvas.create_line(sx, sy, sx - corner_len, sy, fill=self.colors['green_light'], width=3)
            if sy < (top + bottom) / 2:
                self.video_canvas.create_line(sx, sy, sx, sy + corner_len, fill=self.colors['green_light'], width=3)
            else:
                self.video_canvas.create_line(sx, sy, sx, sy - corner_len, fill=self.colors['green_light'], width=3)

    def _create_metric(self, parent, title, value, row=None, column=None, columnspan=1):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        if row is None:
            frame.pack(fill='x', padx=12, pady=(12, 0))
        else:
            frame.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=10, pady=8)
        frame.columnconfigure(0, weight=1)
        card = RoundedMetricCard(frame, title, value, self.colors)
        card.grid(row=0, column=0, sticky="ew")
        return card
    def import_video(self):
        path = filedialog.askopenfilename(title="Videófájl kiválasztása", filetypes=[("Videófájlok", "*.mp4 *.avi *.mov *.mkv"), ("Minden fájl", "*.*")])
        if not path:
            return
        if self.playing:
            self.release_backend()
        if self.cap:
            try: self.cap.release()
            except: pass
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = max(1, int(self.cap.get(cv2.CAP_PROP_FPS)))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.activity_grid = RollingActivityGrid(
            self.frame_width,
            self.frame_height,
            cell_size=140,
            window_seconds=60
        )
        if not self.cap.isOpened():
            messagebox.showerror("Hiba", "A videó nem nyitható meg.")
            return

        # load adjust parameters if exist
        self.basename = os.path.splitext(os.path.basename(self.video_path))[0]
        params_file = f"{self.basename}-parameters.json"
        json_filepath = os.path.join(self.parameters_dir_name, params_file)
        self.objects = []
        if os.path.exists(json_filepath):
            try:
                with open(json_filepath, 'r') as f:
                    params = json.load(f)
                self.rotation = params.get('rotation', 0)
                for obj in params.get('objects', []):
                    for k, v in obj.items():
                        kind = 'feeder' if k.startswith('feeder') else 'drinker'
                        color = (0, 215, 255) if kind == 'feeder' else (255, 100, 30)
                        rect = v[0]
                        self.objects.append({'name': k, 'type': kind, 'rect': [rect['up'], rect['down'], rect['left'], rect['right']], 'color': color})
            except Exception as e:
                print("Failed loading params:", e)

        self.tracker = SimpleTracker(max_lost=15, dist_thresh=400)
        self.anomaly_detector.reset()
        self.short_activity_window.clear()
        self.medium_activity_window.clear()
        self.long_activity_window.clear()
        self.history.clear()
        self.history_buffer = []
        self.avg_move_long_period.clear()
        self.last_history_append = 0.0
        self.last_plot_update = 0.0

        self.stop_event.clear()
        self.playing = True

        # start threads
        self.worker_thread = threading.Thread(target=self._backend_worker, daemon=True)
        self.worker_thread.start()
        threading.Thread(target=self._reader_thread, daemon=True).start()
        self.lbl_status.config(text=f"Betöltve: {self.basename}")

    def on_animal_selected(self, event=None):

        selected = self.selected_animal.get()

        self.alert_listbox.delete(0, tk.END)

        if selected == "Összes":

            for msg in getattr(self, "latest_anomalies", []):

                self.alert_listbox.insert(
                    tk.END,
                    msg
                )

            return

        try:

            oid = int(selected.replace("Kacsa ", ""))

        except:
            return

        found = False

        for aid, metrics in getattr(self, "latest_animal_metrics", {}).items():

            if aid != oid:
                continue

            found = True

            movement = metrics.get("movement", 0)
            feeder = metrics.get("at_feeder", False)
            drinker = metrics.get("at_drinker", False)
            sleeping = metrics.get("is_sleeping", False)
            estimate = getattr(self.anomaly_detector, "latest_summaries", {}).get(aid, {})
            estimate_window = estimate.get("window_seconds")
            estimate_samples = estimate.get("sample_count")
            estimate_text = "Becslési ablak: nincs elég adat"
            if estimate_window is not None and estimate_samples is not None:
                estimate_text = (
                    f"Becslési ablak: {estimate_window:.0f} mp "
                    f"({estimate_samples} minta)"
                )

            info = (
                f"Állat azonosító: {aid}\n"
                f"Mozgás: {movement:.1f}\n"
                f"Etetőnél: {'IGEN' if feeder else 'NEM'}\n"
                f"Itatónál: {'IGEN' if drinker else 'NEM'}\n"
                f"Alszik: {'IGEN' if sleeping else 'NEM'}\n"
                f"{estimate_text}"
            )

            self.lbl_animal_info.config(
                text=info
            )

        if not found:

            self.lbl_animal_info.config(
                text="Nincs adat"
            )
    
    def open_adjust(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showinfo("Információ", "Először tölts be egy videót.")
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showwarning("Figyelmeztetés", "Nem sikerült képkockát olvasni a videóból.")
            return

        self.playing = False

        video_base_path = os.path.join(self.parameters_dir_name, self.basename)
        params = {'rotation': self.rotation, 'objects': []}
        for o in self.objects:
            params['objects'].append({o['name']:[{'up':o['rect'][0],'down':o['rect'][1],'left':o['rect'][2],'right':o['rect'][3]}]})
 
        self.adjust_win = AdjustWindow(self, frame, video_base_path, params)


    def on_adjust_closed(self, saved):

        if saved and self.video_path:
            params_file = f"{self.basename}-parameters.json"
            json_filepath = os.path.join(self.parameters_dir_name, params_file)
            if os.path.exists(json_filepath):
                try:
                    with open(json_filepath,'r') as f:
                        params = json.load(f)
                    self.rotation = params.get('rotation', 0)
                    self.objects = []
                    for obj in params.get('objects', []):
                        for k,v in obj.items():
                            kind = 'feeder' if k.startswith('feeder') else 'drinker'
                            color = (0,215,255) if kind=='feeder' else (255,100,30)
                            rect = v[0]
                            self.objects.append({'name':k,'type':kind,'rect':[rect['up'],rect['down'],rect['left'],rect['right']],'color':color})
                except Exception as e:
                    print("Error reloading params:", e)
            self.anomaly_detector.reset()
        self.playing = True

    def _append_timed_value(self, window, now, value, seconds):
        window.append((now, value))
        cutoff = now - seconds
        while window and window[0][0] < cutoff:
            window.popleft()

    def _timed_average(self, window):
        if not window:
            return 0.0
        return sum(value for _timestamp, value in window) / len(window)

    def _reader_thread(self):
        """Continuously read video frames in real time."""
        while not self.stop_event.is_set() and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break
            try:
                self.frame_queue.put_nowait(frame)
            except:
                pass
            time.sleep(1.0 / self.fps)

    def _backend_worker(self):
        print("Háttérfeldolgozás elindult")
        """Process frames as fast as possible without blocking GUI."""
        while not self.stop_event.is_set():
            try:
                # A feldolgozas tovabb fut a beallitoablak nyitva tartasa kozben is,
                # hogy a grafikon es az idosoros naplo ne szakadjon meg.
                # Emiatt az alabbi szunetelteto logika kommentben maradt.
                # if not self.playing: 
                #     time.sleep(0.1)
                #     continue
                print("Képkocka feldolgozása")
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                centroids = self.detector.detect(frame)
                if len(centroids) == 0:
                    continue

                h, w = frame.shape[:2]
                centroids = rotate_points_with_pad(centroids, self.rotation, w, h)
                tracked_objects = self.tracker.update(centroids)
                active_animals = [
                    (oid, position)
                    for oid, position in tracked_objects.items()
                    if self.tracker.lost.get(oid, 0) == 0
                ]

                total = len(active_animals)
                feeder_count = 0
                drinker_count = 0
                movement_by_id = {
                    oid: self.tracker.get_recent_movement(oid, seconds=5.0)
                    for oid, _position in active_animals
                }
                sleep_movement_by_id = {
                    oid: self.tracker.get_recent_movement(oid, seconds=self.sleep_detect_window)
                    for oid, _position in active_animals
                }
                avg_move = float(np.mean(list(movement_by_id.values()))) if movement_by_id else 0.0
                sleep_limit = None
                if len(self.avg_move_long_period) >= 14:
                    sleep_limit = self.avg_move_long_period[-1] * self.sleep_detect_threshold

                dots = []
                animal_states = []
                rolling_points = []
                for oid, (x, y) in active_animals:
                    in_feeder = any(o['type'] == 'feeder' and o['rect'][0] <= y <= o['rect'][1] and o['rect'][2] <= x <= o['rect'][3] for o in self.objects)
                    in_drinker = any(o['type'] == 'drinker' and o['rect'][0] <= y <= o['rect'][1] and o['rect'][2] <= x <= o['rect'][3] for o in self.objects)
                    movement = movement_by_id.get(oid, 0.0)
                    is_sleeping = sleep_limit is not None and sleep_movement_by_id.get(oid, 0.0) < sleep_limit
                    if in_feeder:
                        color = (0, 140, 255)
                        feeder_count += 1
                    elif in_drinker:
                        color = (255, 255, 0)
                        drinker_count += 1
                    else:
                        color = (80, 80, 80)
                    dots.append((oid, x, y, color))
                    rolling_points.append((x, y))
                    animal_states.append({
                        'id': oid,
                        'movement': movement,
                        'at_feeder': in_feeder,
                        'at_drinker': in_drinker,
                        'is_sleeping': is_sleeping
                    })
                    self.latest_animal_metrics = {
                        s['id']: s
                        for s in animal_states
                    }

                sleeping = sum(
                    1
                    for state in animal_states
                    if state['is_sleeping']
                )

                anomaly_messages = []
                now = time.time()

                self._append_timed_value(
                    self.short_activity_window,
                    now,
                    avg_move,
                    self.short_activity_seconds
                )
                self._append_timed_value(
                    self.medium_activity_window,
                    now,
                    avg_move,
                    self.medium_activity_seconds
                )
                self._append_timed_value(
                    self.long_activity_window,
                    now,
                    avg_move,
                    self.long_activity_seconds
                )

                short_avg = self._timed_average(self.short_activity_window)
                medium_avg = self._timed_average(self.medium_activity_window)
                long_avg = self._timed_average(self.long_activity_window)

                crowded = feeder_count >= self.crowd_threshold

                low_activity = (
                    len(self.medium_activity_window) >= 30
                    and len(self.long_activity_window) >= 60
                    and short_avg < long_avg * 0.35
                    and medium_avg < long_avg * 0.55
                )

                if crowded:
                    anomaly_messages.append(
                        "Torlódás az etetőnél"
                    )

                if low_activity:
                    anomaly_messages.append(
                        "Alacsony aktivitás észlelve"
                    )
                anomalies = self.anomaly_detector.update(animal_states, now=now)
                anomaly_ids = set(anomalies.keys())
                anomaly_messages.extend([
                    result.message
                    for result in sorted(
                        anomalies.values(),
                        key=lambda item: item.score,
                        reverse=True
                    )
                ])

                disp = rotate_image(
                    frame,
                    self.rotation,
                    pad=self.pad_rotation
                )

                if (
                    self.activity_grid is not None
                    and self.enable_activity_grid
                ):
                    self.activity_grid.update(
                        rolling_points
                    )

                if (
                    self.activity_grid is not None
                    and self.enable_activity_grid
                ):
                    disp = self.activity_grid.render(disp)



                for o in self.objects:

                    up, down, leftc, rightc = o['rect']
                    color = o['color']

                    cv2.rectangle(
                        disp,
                        (leftc, up),
                        (rightc, down),
                        color,
                        2
                    )


                for tid, traj in self.tracker.trajectories.items():

                    try:

                        pts = [p for (_, p) in traj]

                        if len(pts) < 2:
                            continue

                        for i in range(1, len(pts)):

                            p1 = pts[i - 1]
                            p2 = pts[i]

                            cv2.line(
                                disp,
                                (int(p1[0]), int(p1[1])),
                                (int(p2[0]), int(p2[1])),
                                (0, 120, 0),
                                1
                            )

                    except:
                        pass



                cv2.putText(
                    disp,
                    f"KACSÁK: {total}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 120, 0),
                    2
                )

                cv2.putText(
                    disp,
                    f"ETETŐNÉL: {feeder_count}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 120, 120),
                    2
                )

                cv2.putText(
                    disp,
                    f"ITATÓNÁL: {drinker_count}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    disp,
                    f"ÁTLAG MOZGÁS: {avg_move:.1f}",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )    
                for (oid, x, y, color) in dots:
                    if oid in anomaly_ids:
                        color = (0, 0, 255)

                    cv2.circle(
                        disp,
                        (int(x), int(y)),
                        6,
                        color,
                        -1
                    )
                if self.pad_rotation:
                    display_for_gui = rotate_image(disp, -self.rotation, pad=False)
                    display_for_gui = fetch_roi_img(display_for_gui, self.frame_width, self.frame_height)
                else:
                    display_for_gui = disp

                self.history_buffer.append({
                    'total': total,
                    'feeder': feeder_count,
                    'drinker': drinker_count,
                    'avg': avg_move,
                    'sleep': sleeping
                })

                if now - self.last_history_append >= 1.0:
                    self.last_history_append = now


                    buf = self.history_buffer
                    avg_total    = sum(x['total']    for x in buf) / len(buf)
                    avg_feeder   = sum(x['feeder']   for x in buf) / len(buf)
                    avg_drinker = sum(x['drinker'] for x in buf) / len(buf)
                    avg_avg      = sum(x['avg']      for x in buf) / len(buf)
                    avg_sleep    = sum(x['sleep']    for x in buf) / len(buf)


                    if len(self.avg_move_long_period) > 0:
                        avg_move_long = (sum(m for m in self.avg_move_long_period) + avg_avg) / len(self.avg_move_long_period)
                    else:
                        avg_move_long = avg_avg  # elso elemnel fallback

                    self.history.append({
                        't': now,
                        'total':    avg_total,
                        'feeder':   avg_feeder,
                        'drinker': avg_drinker,
                        'avg':      avg_avg,
                        'sleep':    avg_sleep,
                    })

                    self.avg_move_long_period.append(avg_move_long)
                    # print("Nagy atlagos mozgas az alvas detektalasahoz:", self.avg_move_long_period[-1])
                    # print("Nagy atlagos mozgas az alvas detektalasahoz:", self.avg_move_long_period[-1])

                    self.history_buffer = []

                    self.logger.log(
                                    self.video_path,
                                    avg_total,
                                    avg_feeder,
                                    avg_drinker,
                                    avg_avg,
                                    avg_sleep
                                )
                

                result = (display_for_gui, total, feeder_count, drinker_count, avg_move, sleeping, anomaly_messages)
                try:
                    self.result_queue.put_nowait(result)
                except:
                    pass
            except Exception as e:
                print("HÁTTÉRRENDSZER HIBA:", e)

    def _update_gui(self):
        if not self.playing:
            self.after(200, self._update_gui)
            return
        try:
            display_frame, total, feeder_count, drinker_count, avg_move, sleeping, anomaly_messages = self.result_queue.get_nowait()
        except Empty:
            self.after(50, self._update_gui)
            return


        disp_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(disp_rgb)
        canvas_w = max(1, self.video_canvas.winfo_width(), int(self.video_canvas.cget("width")))
        canvas_h = max(1, self.video_canvas.winfo_height(), int(self.video_canvas.cget("height")))
        pil.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        self.video_tkimg = ImageTk.PhotoImage(pil)
        self._draw_video_frame(pil, total, feeder_count, drinker_count, avg_move, sleeping, anomaly_messages)


        animal_values = ["Összes"]

        for oid in getattr(self, "tracker").objects.keys():

            animal_values.append(
                f"Kacsa {oid}"
            )

        self.animal_dropdown["values"] = animal_values

        self.latest_anomalies = anomaly_messages

        selected = self.selected_animal.get()

        if selected == "Összes":

            self.alert_listbox.delete(0, tk.END)

            for msg in anomaly_messages[-10:]:

                self.alert_listbox.insert(
                    tk.END,
                    msg
                )
        self.lbl_total.config(text=str(total))
        self.lbl_feeder.config(text=str(feeder_count))
        self.lbl_drinker.config(text=str(drinker_count))
        self.lbl_avg.config(text=f"{avg_move:.2f} px")
        self.lbl_sleep.config(text=str(sleeping))
        self.lbl_anomaly.config(text=str(len(anomaly_messages)))
        self.lbl_zone_card.config(text=str(len(self.objects)))
        self.lbl_fps_card.config(text=str(self.fps))
        if self.basename:
            self.lbl_status.config(text=f"Betöltve: {self.basename}")

        # update plots at 1 Hz
        now = time.time()
        if now - self.last_plot_update >= 1.0:
            self.last_plot_update = now
            self._update_plots()

        self.after(50, self._update_gui)

    def _init_plots(self):
        """
        Letrehozza az osszes plotot, tengelyt, cimet es racsvonalat,
        majd eltarolja a line objektumokat a kesobbi frissiteshez.
        """
        keys = [
            ('Összes állat', 'total'),
            ('Etetőzóna', 'feeder'),
            ('Itatózóna', 'drinker'),
            ('Átlagos mozgás', 'avg'),
            ('Alvó egyedek', 'sleep')
        ]
        colors = ["#19a974", "#ffffff", "#8fd8bd", "#d9fff0", "#7aa897"]

        self.plot_keys = [k for _, k in keys]
        self.lines = []
        self.axs = []

        for ax, (title, key), color in zip(self.axes, keys, colors):
            ax.set_xlim(-60, 0)
            ax.set_xticks([-60, -45, -30, -15, 0])
            ax.set_xticklabels(["60s", "45s", "30s", "15s", "0s"])
            ax.set_title(title, fontsize=9, color=self.colors['white'])
            ax.grid(True, linestyle=':', linewidth=0.5, color=self.colors['grid'])
            ax.set_facecolor(self.colors['panel'])
            ax.tick_params(colors=self.colors['muted'], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])

            line, = ax.plot([], [], lw=1.8, color=color)
            self.lines.append(line)
            self.axs.append(ax)

        self.fig_canvas.draw_idle()

    def _update_plots(self):
        if not self.history:
            return

        now = time.time()

        times = np.array([now - h['t'] for h in self.history])
        valid = times <= 60
        t_plot = -times[valid]

        for ax, line, key in zip(self.axs, self.lines, self.plot_keys):
            vals = np.array([h[key] for h in self.history])
            v_plot = vals[valid]
            line.set_data(t_plot, v_plot)
            ax.relim()
            ax.autoscale(axis='y')
            ax.set_ylim(bottom=0)
        self.fig_canvas.draw_idle()

    
    def release_backend(self):
        self.stop_event.set()
        self.playing = False
        if self.worker_thread is not None:
            self.worker_thread.join()
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=2)
        try:
            if self.cap:
                self.cap.release()
        except:
            pass

    def on_quit(self):
        # if messagebox.askokcancel("Quit", "Do you want to quit?"):
        if True:
            self.logger.close()
            self.release_backend()
            self.destroy()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()

