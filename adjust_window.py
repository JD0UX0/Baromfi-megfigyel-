import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import cv2

from image_processor import rotate_image, add_grid


class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command, colors, width=150, height=40, accent=False):
        super().__init__(master, width=width, height=height, bg=colors['panel'], highlightthickness=0, cursor="hand2")
        self.text = text
        self.command = command
        self.colors = colors
        self.accent = accent
        self.hover = False
        self.enabled = True
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
        w = max(1, self.winfo_width(), int(self.cget("width")))
        h = max(1, self.winfo_height(), int(self.cget("height")))
        inset = 3
        radius = max(1, min(17, (w - inset * 2) // 2, (h - inset * 2) // 2))
        fill = self.colors['green'] if self.accent else self.colors['panel_alt']
        outline = self.colors['green_light']
        text_fill = self.colors['white']
        if not self.enabled:
            fill = self.colors['panel']
            outline = '#536058'
            text_fill = self.colors['muted']
        elif self.hover:
            fill = self.colors['green_light'] if self.accent else self.colors['green']
        self._rounded_rect(inset, inset, w - inset, h - inset, radius, fill=fill, outline=outline, width=1)
        self.create_text(w // 2, h // 2, text=self.text, fill=text_fill, anchor="center", font=("Segoe UI", 10, "bold"))

    def _on_enter(self, _event):
        if not self.enabled:
            return
        self.hover = True
        self._draw()

    def _on_leave(self, _event):
        self.hover = False
        self._draw()

    def _on_click(self, _event):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()


class AdjustWindow(tk.Toplevel):
    def __init__(self, master, frame_bgr, video_basename, params=None):
        super().__init__(master)
        self.title("Megfigyelési zónák beállítása")
        self.geometry("1450x720")
        self.minsize(1100, 620)
        # self.resizable(True, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.video_basename = video_basename
        self.base_frame = frame_bgr.copy()  # BGR numpy
        self.pad_rotation = True
        self.rotation = 0
        self.objects = []  # each: {'name','type','rect':[up,down,left,right],'color'}
        self.selected_object = None
        self.is_ui_builded = False
        self.scale_update_id = None
        self.is_data_updated = False
        if params:
            self.rotation = params.get('rotation', 0)
            for obj in params.get('objects', []):
                for k,v in obj.items():
                    kind = 'feeder' if k.startswith('feeder') else 'drinker'
                    rect = v[0]
                    color = (255,100,30) if kind=='drinker' else (0,215,255)
                    self.objects.append({'name':k,'type':kind,'rect':[rect['up'],rect['down'],rect['left'],rect['right']],'color':color})
        self._build_ui()
        self.update_display()

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
            'feeder': '#ffd43b',
            'drinker': '#1e90ff'
        }
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=self.colors['bg'], foreground=self.colors['white'])
        style.configure("TFrame", background=self.colors['bg'])
        style.configure("Panel.TFrame", background=self.colors['panel'])
        style.configure("Panel.TLabelframe", background=self.colors['panel'], foreground=self.colors['white'], relief="solid", borderwidth=1)
        style.configure("Panel.TLabelframe.Label", background=self.colors['panel'], foreground=self.colors['white'], font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=self.colors['panel'], foreground=self.colors['white'])
        style.configure("Hint.TLabel", background=self.colors['panel'], foreground=self.colors['muted'])
        style.configure("SmallHint.TLabel", background=self.colors['panel'], foreground=self.colors['muted'], font=("Segoe UI", 8))
        style.configure("TButton", padding=(10, 6), background=self.colors['panel_alt'], foreground=self.colors['white'])
        style.map("TButton", background=[("active", self.colors['green']), ("pressed", "#0b513d")])
        style.configure("Accent.TButton", background=self.colors['green'], foreground=self.colors['white'])
        style.map("Accent.TButton", background=[("active", self.colors['green_light']), ("pressed", "#0b513d")])
        style.configure("TRadiobutton", background=self.colors['panel'], foreground=self.colors['white'])
        style.map("TRadiobutton", background=[("active", self.colors['panel'])], foreground=[("active", self.colors['white'])])

    def _build_ui(self):
        self._configure_style()
        self.configure(bg=self.colors['bg'])
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.draw_kind = tk.StringVar(value='feeder')
        self.move_step = tk.IntVar(value=10)
        self.drag_start = None
        self.drag_mode = None
        self.drag_start_image = None
        self.drag_original_rect = None
        self.drag_rect_id = None
        self.display_update_needed = False
        self.selection_buttons = []
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.display_size = (1, 1)
        self.display_image_size = (1, 1)

        left = ttk.LabelFrame(self, text="Zónaszerkesztő", style="Panel.TLabelframe")
        left.grid(row=0, column=0, sticky="ns", padx=12, pady=12)
        left.configure(width=500)
        left.grid_propagate(False)
        right = ttk.LabelFrame(self, text="Előnézet", style="Panel.TLabelframe")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        draw_panel = ttk.Frame(left, style="Panel.TFrame")
        draw_panel.pack(fill='x', padx=10, pady=(12, 8))
        ttk.Label(draw_panel, text="Rajzolás egérrel", style="Hint.TLabel").pack(anchor='w')
        ttk.Radiobutton(draw_panel, text="Etető téglalap", variable=self.draw_kind, value='feeder').pack(anchor='w', pady=(4, 0))
        ttk.Radiobutton(draw_panel, text="Itató téglalap", variable=self.draw_kind, value='drinker').pack(anchor='w')
        ttk.Label(draw_panel, text="A képen tartsd lenyomva a bal egérgombot, húzd ki a keretet, majd engedd el a gombot.", style="Hint.TLabel", wraplength=450, justify='left').pack(anchor='w', pady=(6, 0))

        ttk.Label(left, text="Kép elforgatása").pack(anchor='w', padx=10)
        scale_row = ttk.Frame(left, style="Panel.TFrame")
        scale_row.pack(fill='x', padx=10, pady=6)
        self.btn_rotate_minus = RoundedButton(scale_row, "-", lambda: self._change_rotation(-1), self.colors, width=42, height=36)
        self.btn_rotate_minus.pack(side='left')
        self.scale = ttk.Scale(scale_row, from_=-45, to=45, orient='horizontal', command=self.on_scale)
        self.scale.set(self.rotation)
        self.scale.pack(side='left', fill='x', expand=True, padx=4)
        self.btn_rotate_plus = RoundedButton(scale_row, "+", lambda: self._change_rotation(1), self.colors, width=42, height=36)
        self.btn_rotate_plus.pack(side='left')
        self.lbl_angle = ttk.Label(left, text=f"{int(self.rotation)}°")
        self.lbl_angle.pack()

        ttk.Label(left, text="Kijelölt területek").pack(anchor='w', padx=10, pady=(12, 0))
        list_frame = ttk.Frame(left, style="Panel.TFrame")
        list_frame.pack(fill='x', padx=10, pady=(4, 8))
        self.obj_listbox = tk.Listbox(
            list_frame,
            height=7,
            bg=self.colors['panel_alt'],
            fg=self.colors['white'],
            selectbackground=self.colors['green'],
            selectforeground=self.colors['white'],
            highlightthickness=1,
            highlightbackground=self.colors['green'],
            relief="flat",
            activestyle="none",
            exportselection=False
        )
        self.obj_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.obj_listbox.yview)
        self.obj_listbox.configure(yscrollcommand=self.obj_scroll.set)
        self.obj_listbox.pack(side='left', fill='both', expand=True)
        self.obj_scroll.pack(side='right', fill='y')
        self.obj_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        self.btn_delete_selected = RoundedButton(left, "Kijelölt törlése", self._delete_selected_object, self.colors, width=460, height=34)
        self.btn_delete_selected.pack(fill='x', padx=10, pady=(0, 10))

        adjust_panel = ttk.LabelFrame(left, text="Kijelölt terület állítása", style="Panel.TLabelframe")
        adjust_panel.pack(fill='x', padx=10, pady=(0, 8))
        adjust_controls = ttk.Frame(adjust_panel, style="Panel.TFrame", width=332, height=220)
        adjust_controls.pack(anchor='center', padx=12, pady=(8, 0))
        adjust_controls.grid_propagate(False)
        adjust_controls.columnconfigure(0, minsize=100)
        adjust_controls.columnconfigure(1, minsize=100)
        adjust_controls.columnconfigure(2, minsize=100)

        step_row = ttk.Frame(adjust_controls, style="Panel.TFrame")
        step_row.grid(row=0, column=0, columnspan=3, pady=(0, 8))
        self.step_spinbox = tk.Spinbox(
            step_row,
            from_=1,
            to=100,
            width=5,
            textvariable=self.move_step,
            bg=self.colors['panel_alt'],
            fg=self.colors['white'],
            buttonbackground=self.colors['green'],
            insertbackground=self.colors['white']
        )
        self.step_spinbox.pack(side='left')

        self.btn_up = RoundedButton(adjust_controls, "Fel", lambda: self._move_selected(0, -1), self.colors, width=86, height=30)
        self.btn_up.grid(row=1, column=1, pady=(0, 8))
        self.btn_left = RoundedButton(adjust_controls, "Bal", lambda: self._move_selected(-1, 0), self.colors, width=86, height=30)
        self.btn_left.grid(row=2, column=0, pady=(0, 8))
        self.btn_right = RoundedButton(adjust_controls, "Jobb", lambda: self._move_selected(1, 0), self.colors, width=86, height=30)
        self.btn_right.grid(row=2, column=2, pady=(0, 8))
        self.btn_down = RoundedButton(adjust_controls, "Le", lambda: self._move_selected(0, 1), self.colors, width=86, height=30)
        self.btn_down.grid(row=3, column=1, pady=(0, 10))

        size_row_1 = ttk.Frame(adjust_controls, style="Panel.TFrame")
        size_row_1.grid(row=4, column=0, columnspan=3, pady=(0, 4))
        self.btn_wider = RoundedButton(size_row_1, "Szélesebb", lambda: self._resize_selected(1, 0), self.colors, width=136, height=30)
        self.btn_wider.pack(side='left', padx=(0, 8))
        self.btn_narrower = RoundedButton(size_row_1, "Keskenyebb", lambda: self._resize_selected(-1, 0), self.colors, width=136, height=30)
        self.btn_narrower.pack(side='left')
        size_row_2 = ttk.Frame(adjust_controls, style="Panel.TFrame")
        size_row_2.grid(row=5, column=0, columnspan=3, pady=(0, 8))
        self.btn_taller = RoundedButton(size_row_2, "Magasabb", lambda: self._resize_selected(0, 1), self.colors, width=136, height=30)
        self.btn_taller.pack(side='left', padx=(0, 8))
        self.btn_shorter = RoundedButton(size_row_2, "Alacsonyabb", lambda: self._resize_selected(0, -1), self.colors, width=136, height=30)
        self.btn_shorter.pack(side='left')
        self.selection_buttons = [
            self.btn_delete_selected,
            self.btn_up,
            self.btn_left,
            self.btn_right,
            self.btn_down,
            self.btn_wider,
            self.btn_narrower,
            self.btn_taller,
            self.btn_shorter
        ]
        self.selection_status = ttk.Label(adjust_panel, text="Nincs kijelölt terület.", style="Hint.TLabel")
        self.selection_status.pack(anchor='w', padx=12, pady=(0, 10))
        ttk.Label(left, text="Kattintás: kijelölés, húzás: mozgatás.", style="SmallHint.TLabel").pack(anchor='w', padx=10, pady=(0, 12))

        self.image_canvas = tk.Canvas(right, width=900, height=620, bg=self.colors['canvas'], highlightthickness=0, cursor="crosshair")
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_canvas.bind("<ButtonPress-1>", self._start_draw)
        self.image_canvas.bind("<B1-Motion>", self._update_draw)
        self.image_canvas.bind("<ButtonRelease-1>", self._finish_draw)
        self.image_canvas.bind("<KeyPress>", self._on_key_press)
        self.bind("<KeyPress>", self._on_key_press)
        self.tkimg = None

        self._refresh_object_list()

        self.is_ui_builded = True
        self.image_canvas.focus_set()

    def _change_rotation(self, delta):
        new_val = max(-45, min(45, self.scale.get() + delta))
        self.scale.set(new_val)
        self.on_scale(new_val)
        
    def on_scale(self, val):
        self.rotation = float(val)
        if self.is_ui_builded:
            self.is_data_updated = True
            if self.scale_update_id is not None:
                self.after_cancel(self.scale_update_id)
            self.lbl_angle.config(text=f"{int(self.rotation)}°")
            self.update_display(show_grid=True)
            self.scale_update_id = self.after(10000, self.update_display)


    def add_object(self, kind):
        self.is_data_updated = True
        h,w = self.base_frame.shape[:2]
        up = int(h*0.3); down=int(h*0.5); leftc=int(w*0.3); rightc=int(w*0.5)
        self._create_object(kind, [up, down, leftc, rightc])

    def _create_object(self, kind, rect):
        idx = sum(1 for o in self.objects if o['type']==kind) + 1
        name = f"{kind}-{idx}"
        color = (255,100,30) if kind=='drinker' else (0,215,255)
        o = {'name':name,'type':kind,'rect':rect,'color':color}
        self.objects.append(o)
        self._add_object_row(o)
        self._select_object(o, redraw=False)
        self.update_display()

    def _canvas_to_image_point(self, x, y):
        offset_x, offset_y = self.display_offset
        disp_w, disp_h = self.display_size
        if x < offset_x or y < offset_y or x > offset_x + disp_w or y > offset_y + disp_h:
            return None
        img_x = int((x - offset_x) / self.display_scale)
        img_y = int((y - offset_y) / self.display_scale)
        return img_x, img_y

    def _clamp_canvas_point(self, x, y):
        offset_x, offset_y = self.display_offset
        disp_w, disp_h = self.display_size
        return (
            max(offset_x, min(offset_x + disp_w, x)),
            max(offset_y, min(offset_y + disp_h, y))
        )

    def _find_object_at_point(self, point):
        x, y = point
        for obj in reversed(self.objects):
            up, down, left, right = obj['rect']
            if left <= x <= right and up <= y <= down:
                return obj
        return None

    def _select_object(self, obj, redraw=True):
        self.selected_object = obj
        self._sync_listbox_selection()
        self._update_selection_status()
        self.image_canvas.focus_set()
        if redraw:
            self.update_display()

    def _clamp_rect(self, rect):
        up, down, left, right = rect
        img_w, img_h = self.display_image_size
        width = max(1, right - left)
        height = max(1, down - up)
        left = max(0, min(int(left), max(0, img_w - width)))
        up = max(0, min(int(up), max(0, img_h - height)))
        return [up, up + height, left, left + width]

    def _set_object_rect(self, obj, rect, redraw=True):
        obj['rect'] = self._clamp_rect(rect)
        self.is_data_updated = True
        self._update_selection_status()
        if redraw:
            self.update_display()

    def _format_object_name(self, obj):
        return obj['name'].replace('feeder-', 'Etető ').replace('drinker-', 'Itató ')

    def _refresh_object_list(self):
        self.obj_listbox.delete(0, tk.END)
        for obj in self.objects:
            self.obj_listbox.insert(tk.END, self._format_object_name(obj))
        self._sync_listbox_selection()
        self._update_selection_status()

    def _sync_listbox_selection(self):
        if not hasattr(self, "obj_listbox"):
            return
        self.obj_listbox.selection_clear(0, tk.END)
        if self.selected_object in self.objects:
            index = self.objects.index(self.selected_object)
            self.obj_listbox.selection_set(index)
            self.obj_listbox.activate(index)
            self.obj_listbox.see(index)

    def _update_selection_status(self):
        if not hasattr(self, "selection_status"):
            return
        has_selection = self.selected_object in self.objects
        for button in getattr(self, "selection_buttons", []):
            button.set_enabled(has_selection)
        if has_selection:
            self.selection_status.config(text=f"Kijelölt: {self._format_object_name(self.selected_object)}")
        else:
            self.selection_status.config(text="Nincs kijelölt terület.")

    def _on_listbox_select(self, _event=None):
        selection = self.obj_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if 0 <= index < len(self.objects):
            self._select_object(self.objects[index])

    def _get_move_step(self):
        try:
            return max(1, int(self.move_step.get()))
        except (tk.TclError, ValueError):
            self.move_step.set(1)
            return 1

    def _move_selected(self, dx, dy):
        if self.selected_object not in self.objects:
            self._update_selection_status()
            return
        step = self._get_move_step()
        up, down, left, right = self.selected_object['rect']
        self._set_object_rect(
            self.selected_object,
            [up + dy * step, down + dy * step, left + dx * step, right + dx * step]
        )

    def _resize_selected(self, dw, dh):
        if self.selected_object not in self.objects:
            self._update_selection_status()
            return
        step = self._get_move_step()
        up, down, left, right = self.selected_object['rect']
        center_x = (left + right) // 2
        center_y = (up + down) // 2
        width = max(1, right - left + dw * step)
        height = max(1, down - up + dh * step)
        self._set_object_rect(
            self.selected_object,
            [
                center_y - height // 2,
                center_y + height // 2,
                center_x - width // 2,
                center_x + width // 2
            ]
        )

    def _start_draw(self, event):
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return
        self.image_canvas.focus_set()
        selected = self._find_object_at_point(point)
        if selected is not None:
            self._select_object(selected)
            self.drag_mode = "move"
            self.drag_start_image = point
            self.drag_original_rect = selected['rect'][:]
            self.image_canvas.configure(cursor="fleur")
            return

        self.drag_mode = "draw"
        self.drag_start = self._clamp_canvas_point(event.x, event.y)
        color = self.colors['feeder'] if self.draw_kind.get() == 'feeder' else self.colors['drinker']
        if self.drag_rect_id is not None:
            self.image_canvas.delete(self.drag_rect_id)
        self.drag_rect_id = self.image_canvas.create_rectangle(
            self.drag_start[0],
            self.drag_start[1],
            self.drag_start[0],
            self.drag_start[1],
            outline=color,
            width=3
        )

    def _update_draw(self, event):
        if self.drag_mode == "move" and self.selected_object and self.drag_start_image and self.drag_original_rect:
            point = self._canvas_to_image_point(*self._clamp_canvas_point(event.x, event.y))
            if point is None:
                return
            dx = point[0] - self.drag_start_image[0]
            dy = point[1] - self.drag_start_image[1]
            up, down, left, right = self.drag_original_rect
            self._set_object_rect(self.selected_object, [up + dy, down + dy, left + dx, right + dx])
            return

        if self.drag_start is None or self.drag_rect_id is None:
            return
        x, y = self._clamp_canvas_point(event.x, event.y)
        self.image_canvas.coords(self.drag_rect_id, self.drag_start[0], self.drag_start[1], x, y)

    def _finish_draw(self, event):
        if self.drag_mode == "move":
            self.drag_mode = None
            self.drag_start_image = None
            self.drag_original_rect = None
            self.image_canvas.configure(cursor="crosshair")
            return

        if self.drag_start is None:
            return
        end = self._clamp_canvas_point(event.x, event.y)
        start_img = self._canvas_to_image_point(*self.drag_start)
        end_img = self._canvas_to_image_point(*end)
        if self.drag_rect_id is not None:
            self.image_canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None
        self.drag_start = None
        if start_img is None or end_img is None:
            return

        x1, y1 = start_img
        x2, y2 = end_img
        left, right = sorted((x1, x2))
        up, down = sorted((y1, y2))
        if right - left < 8 or down - up < 8:
            return
        self.is_data_updated = True
        self._create_object(self.draw_kind.get(), [up, down, left, right])

    def _on_key_press(self, event):
        if self.selected_object is None:
            return
        moves = {
            "Left": (0, -1),
            "Right": (0, 1),
            "Up": (-1, 0),
            "Down": (1, 0)
        }
        if event.keysym not in moves:
            return
        step = self._get_move_step()
        if event.state & 0x0001:
            step *= 10
        dy, dx = moves[event.keysym]
        up, down, left, right = self.selected_object['rect']
        self._set_object_rect(
            self.selected_object,
            [up + dy * step, down + dy * step, left + dx * step, right + dx * step]
        )
        return "break"

    def _add_object_row(self, obj):
        self._refresh_object_list()

    def _delete_selected_object(self):
        if self.selected_object in self.objects:
            self._delete_object(self.selected_object)


    def _delete_object(self, obj):
        if obj in self.objects:
            self.is_data_updated = True
            self.objects.remove(obj)
            if self.selected_object is obj:
                self.selected_object = None
            self._refresh_object_list()
            self.update_display()

    def update_display(self, show_grid=False):
        rotated = rotate_image(self.base_frame, self.rotation, pad=True)
        disp = rotated.copy()
        if show_grid:
            disp = add_grid(disp)
        else:
            self.scale_update_id = None
            
            # draw rectangles
            for o in self.objects:
                up, down, leftc, rightc = o['rect']
                color = o['color']
                width = 6 if o is self.selected_object else 4
                cv2.rectangle(disp, (leftc, up), (rightc, down), color, width)
                if o is self.selected_object:
                    cv2.rectangle(disp, (leftc, up), (rightc, down), (244, 247, 245), 1)
        
        # convert to PIL for Tkinter
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(disp_rgb)
        # scale to fit canvas while preserving aspect
        canvas_w = max(1, self.image_canvas.winfo_width(), int(self.image_canvas.cget("width")))
        canvas_h = max(1, self.image_canvas.winfo_height(), int(self.image_canvas.cget("height")))
        original_w, original_h = pil.size
        self.display_image_size = (original_w, original_h)
        pil.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        displayed_w, displayed_h = pil.size
        self.display_scale = displayed_w / original_w if original_w else 1.0
        self.display_size = (displayed_w, displayed_h)
        self.display_offset = ((canvas_w - displayed_w) // 2, (canvas_h - displayed_h) // 2)
        self.tkimg = ImageTk.PhotoImage(pil)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_w//2, canvas_h//2, image=self.tkimg)

    def on_close(self):
        if not self.is_data_updated:
            self.destroy()
            self.master.on_adjust_closed(saved=False)
            return

        ans = messagebox.askyesnocancel("Módosítások mentése", "Szeretnéd menteni a zónabeállításokat?", parent=self)
        if ans is None:
            return  # cancel
        if ans is False:
            self.destroy()
            self.master.on_adjust_closed(saved=False)
            return
        # Yes -> save
        data = {'rotation': int(self.rotation), 'objects': []}
        for o in self.objects:
            data['objects'].append({o['name']:[{'up':int(o['rect'][0]), 'down':int(o['rect'][1]), 'left':int(o['rect'][2]), 'right':int(o['rect'][3])}]})
        fname = f"{self.video_basename}-parameters.json"
        with open(fname, 'w') as f:
            json.dump(data, f, indent=2)
        self.destroy()
        self.master.on_adjust_closed(saved=True)
