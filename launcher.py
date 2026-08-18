"""
DeepSORVF — GUI Launcher
Run this file instead of main.py.
"""

import os
import sys
import time
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox
import traceback
# À ajouter au début de launcher.py, après les imports
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ── Light theme palette ────────────────────────────────────────────────────────
BG_DARK    = "#f0f2f5"
BG_PANEL   = "#ffffff"
BG_CARD    = "#ffffff"
ACCENT     = "#185FA5"
ACCENT_HOV = "#1a6fbd"
SUCCESS    = "#1a7a3a"
WARNING    = "#b08000"
DANGER     = "#c0392b"
TEXT_PRI   = "#1a1a2e"
TEXT_SEC   = "#5a6270"
TEXT_DIM   = "#8c95a1"
BORDER     = "#d0d5dd"
LOG_BG     = "#f7f8fa"
LOG_FG     = "#1a1a2e"

# Path to the academy logo (place logoacad.png next to this file)
LOGO_PATH  = r"C:\Users\alach\Downloads\logoacad.png"


class _QueueWriter:
    def __init__(self, q):
        self._q = q
    def write(self, text):
        if text:
            self._q.put(text)
    def flush(self):
        pass


class DeepSORVFLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Intelligent System for Maritime Surveillance")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(900, 700)
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = 1020, 780
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self._sim_thread = None
        self._stop_event = threading.Event()
        self._log_queue  = queue.Queue()
        self._running    = False
        self._build_ui()
        self._poll_log()

    def _build_ui(self):
        # ── Header ─────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG_PANEL, pady=12)
        hdr.pack(fill="x")

        hdr_inner = tk.Frame(hdr, bg=BG_PANEL)
        hdr_inner.pack(fill="x", padx=24)

        # Left side: logo + academy name + project title
        hdr_left = tk.Frame(hdr_inner, bg=BG_PANEL)
        hdr_left.pack(side="left", anchor="w")

        self._logo_img = None
        _logo_candidates = [
            LOGO_PATH,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "logoacad.png"),
            os.path.join(os.getcwd(), "logoacad.png"),
        ]
        _logo_found = None
        for _lp in _logo_candidates:
            if os.path.isfile(_lp):
                _logo_found = _lp
                break

        if _logo_found:
            try:
                from PIL import Image, ImageTk
                img = Image.open(_logo_found).resize((44, 44), Image.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(img)
            except Exception:
                try:
                    self._logo_img = tk.PhotoImage(file=_logo_found)
                    w = self._logo_img.width()
                    if w > 60:
                        factor = max(1, w // 44)
                        self._logo_img = self._logo_img.subsample(factor)
                except Exception:
                    self._logo_img = None

        if self._logo_img:
            tk.Label(hdr_left, image=self._logo_img, bg=BG_PANEL).pack(side="left", padx=(0, 10))
        else:
            tk.Label(hdr_left, text="⚓", font=("Segoe UI", 22),
                     bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=(0, 10))

        hdr_text = tk.Frame(hdr_left, bg=BG_PANEL)
        hdr_text.pack(side="left")
        tk.Label(hdr_text, text="Tunisian Naval Academy",
                 font=("Segoe UI", 16, "bold"),
                 bg=BG_PANEL, fg=TEXT_PRI).pack(anchor="w")
        tk.Label(hdr_text, text="Intelligent system for maritime surveillance",
                 font=("Segoe UI", 9), bg=BG_PANEL, fg=TEXT_SEC).pack(anchor="w")

        # Right side: name
        hdr_right = tk.Frame(hdr_inner, bg=BG_PANEL)
        hdr_right.pack(side="right", anchor="e")
        tk.Label(hdr_right, text="EV2 Chebbi Ala",
                 font=("Segoe UI", 11, "bold"),
                 bg=BG_PANEL, fg=TEXT_PRI).pack(anchor="e")

        _sep(self)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=24, pady=16)

        # ── Scrollable left sidebar ────────────────────────────────────────
        left_outer = tk.Frame(body, bg=BG_DARK)
        left_outer.pack(side="left", fill="y", padx=(0, 12))

        _sidebar_canvas = tk.Canvas(left_outer, bg=BG_DARK,
                                    highlightthickness=0, width=290)
        _sidebar_canvas.pack(side="left", fill="y", expand=True)

        _sidebar_scroll = tk.Scrollbar(left_outer, orient="vertical",
                                       command=_sidebar_canvas.yview,
                                       bg=BG_PANEL, troughcolor=BG_DARK)
        _sidebar_scroll.pack(side="right", fill="y")
        _sidebar_canvas.config(yscrollcommand=_sidebar_scroll.set)

        left = tk.Frame(_sidebar_canvas, bg=BG_DARK)
        _sidebar_win = _sidebar_canvas.create_window(
            (0, 0), window=left, anchor="nw")

        def _on_sidebar_configure(e):
            _sidebar_canvas.configure(
                scrollregion=_sidebar_canvas.bbox("all"))
        def _on_canvas_resize(e):
            _sidebar_canvas.itemconfig(_sidebar_win, width=e.width)

        left.bind("<Configure>", _on_sidebar_configure)
        _sidebar_canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(e):
            _sidebar_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        left.bind_all("<MouseWheel>", _on_mousewheel)

        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        # Paths
        c = _card(left, "  Paths")
        _label(c, "Dataset Path")
        r = tk.Frame(c, bg=BG_CARD); r.pack(fill="x", pady=(0,8))
        self._data_var = tk.StringVar(value="./clip-01/")
        _entry(r, self._data_var).pack(side="left", fill="x", expand=True)
        _btn_small(r, "Browse", lambda: self._browse_folder(self._data_var)).pack(side="left", padx=(6,0))
        _label(c, "Result Output Path")
        r2 = tk.Frame(c, bg=BG_CARD); r2.pack(fill="x", pady=(0,4))
        self._res_var = tk.StringVar(value="./result/")
        _entry(r2, self._res_var).pack(side="left", fill="x", expand=True)
        _btn_small(r2, "Browse", lambda: self._browse_folder(self._res_var)).pack(side="left", padx=(6,0))

        # Tracking
        c = _card(left, "  Tracking")
        _label(c, "Anti-Occlusion")
        self._anti_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Enable anti-occlusion tracking", variable=self._anti_var,
                       bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_DARK,
                       activebackground=BG_CARD, activeforeground=TEXT_PRI,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0,6))
        _label(c, "Occlusion Overlap Threshold  (0–5)")
        self._rate_var = tk.IntVar(value=0)
        tk.Scale(c, from_=0, to=5, variable=self._rate_var, orient="horizontal",
                 bg=BG_CARD, fg=TEXT_PRI, troughcolor=BG_DARK,
                 highlightbackground=BG_CARD, font=("Segoe UI", 8), length=220).pack(anchor="w")

        # Radar / AIS
        c = _card(left, "  Radar & AIS")
        self._ais_range_var    = _spinrow(c, "AIS Max Range  (nm)",        0.5, 20.0, 0.5,  4.0)
        self._radar_range_var  = _spinrow(c, "Radar Range  (nm)",          0.5, 20.0, 0.5,  4.0)
        self._cpa_m_var        = _spinrow(c, "CPA Distance Alert  (m)",     50, 5000,  50,  400, is_int=True)
        self._cpa_s_var        = _spinrow(c, "CPA Time Alert  (s)",         10,  600,  10,  120, is_int=True)

        # Display & Detection
        c = _card(left, "  Display & Detection")
        self._map_size_var    = _spinrow(c, "Map Size  (px)",              300, 1200,  50,  700, is_int=True)
        self._show_size_var   = _spinrow(c, "Display Height  (px)",        200, 1080,  50,  500, is_int=True)
        self._kolo_conf_var   = _spinrow(c, "KOLOMVERSE Conf",            0.05, 0.95, 0.05, 0.30)

        _label(c, "YOLOX Conf  (C=Clear / R=Rain / F=Fog)")
        row = tk.Frame(c, bg=BG_CARD); row.pack(anchor="w", pady=(0,4))
        self._yolox_clear_var = tk.DoubleVar(value=0.50)
        self._yolox_rain_var  = tk.DoubleVar(value=0.35)
        self._yolox_fog_var   = tk.DoubleVar(value=0.30)
        for var, lbl in [(self._yolox_clear_var,"C"),(self._yolox_rain_var,"R"),(self._yolox_fog_var,"F")]:
            tk.Label(row, text=lbl, bg=BG_CARD, fg=TEXT_SEC, font=("Segoe UI",8)).pack(side="left")
            tk.Spinbox(row, from_=0.05, to=0.95, increment=0.05, textvariable=var, width=5,
                       bg=BG_DARK, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                       buttonbackground=BG_DARK, relief="flat",
                       font=("Consolas", 8)).pack(side="left", padx=(0,8))

        # ── NOUVEAU : Contrôle AIS Fusion ───────────────────────────────────
        _label(c, "AIS Display")
        self._ais_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Enable AIS data fusion (MMSI + info panels)",
                       variable=self._ais_enabled_var,
                       bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_DARK,
                       activebackground=BG_CARD, activeforeground=TEXT_PRI,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0,4))

        # ── NOUVEAU : Contrôle Radar Minimap ─────────────────────────────────
        self._radar_enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(c, text="Show radar minimap overlay",
                       variable=self._radar_enabled_var,
                       bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_DARK,
                       activebackground=BG_CARD, activeforeground=TEXT_PRI,
                       font=("Segoe UI", 9)).pack(anchor="w", pady=(0,8))

        # Control
        c = _card(left, "  Control")
        self._status_dot = tk.Label(c, text="●", font=("Segoe UI",12), bg=BG_CARD, fg=TEXT_DIM)
        self._status_dot.pack(anchor="w", pady=(0,2))
        self._status_lbl = tk.Label(c, text="Ready", font=("Segoe UI",9), bg=BG_CARD, fg=TEXT_SEC)
        self._status_lbl.pack(anchor="w", pady=(0,8))
        self._start_btn = tk.Button(c, text="▶   Start Simulation",
                                    font=("Segoe UI",11,"bold"), bg=ACCENT, fg="white",
                                    activebackground=ACCENT_HOV, activeforeground="white",
                                    relief="flat", cursor="hand2", padx=14, pady=10,
                                    command=self._on_start)
        self._start_btn.pack(fill="x", pady=(0,6))
        self._stop_btn = tk.Button(c, text="■   Stop", font=("Segoe UI",10),
                                   bg=BG_DARK, fg=DANGER, activebackground=DANGER,
                                   activeforeground="white", relief="flat", cursor="hand2",
                                   padx=14, pady=8, state="disabled", command=self._on_stop)
        self._stop_btn.pack(fill="x")

        # Log panel
        lc = tk.Frame(right, bg=BG_PANEL, highlightbackground=BORDER, highlightthickness=1)
        lc.pack(fill="both", expand=True)
        lh = tk.Frame(lc, bg=BG_PANEL); lh.pack(fill="x", padx=12, pady=(10,4))
        tk.Label(lh, text="  Console Output", font=("Segoe UI",10,"bold"),
                 bg=BG_PANEL, fg=TEXT_PRI).pack(side="left")
        _btn_small(lh, "Clear", self._clear_log, fg=TEXT_SEC).pack(side="right")
        tf = tk.Frame(lc, bg=LOG_BG); tf.pack(fill="both", expand=True, padx=1, pady=(0,1))
        self._log = tk.Text(tf, bg=LOG_BG, fg=LOG_FG, font=("Consolas",9),
                            insertbackground=LOG_FG, relief="flat", state="disabled",
                            wrap="word", padx=10, pady=8)
        self._log.pack(side="left", fill="both", expand=True)
        sc = tk.Scrollbar(tf, command=self._log.yview, bg=BG_PANEL, troughcolor=BG_DARK)
        sc.pack(side="right", fill="y")
        self._log.config(yscrollcommand=sc.set)
        self._log.tag_config("ok",   foreground=SUCCESS)
        self._log.tag_config("warn", foreground=WARNING)
        self._log.tag_config("err",  foreground=DANGER)
        self._log.tag_config("dim",  foreground=TEXT_SEC)
        self._log.tag_config("hi",   foreground=ACCENT)

        _sep(self)
        foot = tk.Frame(self, bg=BG_PANEL, pady=6); foot.pack(fill="x")
        tk.Label(foot, text="Tunisian Naval Academy  •  Intelligent System for Maritime Surveillance  •  EV2 Chebbi Ala",
                 font=("Segoe UI",8), bg=BG_PANEL, fg=TEXT_DIM).pack()

        self._log_line("Launcher ready.", "dim")
        self._log_line("Select a dataset path and press  ▶ Start Simulation.", "dim")

    def _browse_folder(self, var):
        p = filedialog.askdirectory(title="Select folder", initialdir=var.get())
        if p: var.set(p)

    def _log_line(self, text, tag=""):
        self._log.config(state="normal")
        self._log.insert("end", time.strftime("[%H:%M:%S] "), "dim")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    def _poll_log(self):
        try:
            while True:
                text = self._log_queue.get_nowait()
                for line in text.splitlines():
                    line = line.strip()
                    if not line: continue
                    if any(k in line for k in ("Error","error","Traceback","ERROR")):
                        tag = "err"
                    elif any(k in line for k in ("Warning","warn")):
                        tag = "warn"
                    elif "Time:" in line or "Stamp:" in line:
                        tag = "ok"
                    elif any(k in line for k in ("Start","loaded","✓","Logger","Saved")):
                        tag = "hi"
                    elif "═" in line or "──" in line:
                        tag = "dim"
                    else:
                        tag = ""
                    self._log_line(line, tag)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_log)

    def _set_status(self, text, color):
        self._status_dot.config(fg=color)
        self._status_lbl.config(text=text)

    def _on_start(self):
        data_path = self._data_var.get().strip()
        res_path  = self._res_var.get().strip()
        if not data_path:
            messagebox.showerror("Missing path", "Please select a dataset path.")
            return
        if not os.path.isdir(data_path):
            messagebox.showerror("Invalid path", f"Dataset path does not exist:\n{data_path}")
            return
        os.makedirs(res_path, exist_ok=True)
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status("Running…", SUCCESS)
        self._stop_event.clear()
        self._running = True

        params = dict(
            data_path      = data_path,
            res_path       = res_path,
            anti           = int(self._anti_var.get()),
            anti_rate      = self._rate_var.get(),
            ais_range_nm   = self._ais_range_var.get(),
            radar_range_nm = self._radar_range_var.get(),
            cpa_warn_m     = self._cpa_m_var.get(),
            cpa_warn_s     = self._cpa_s_var.get(),
            map_size       = self._map_size_var.get(),
            show_size      = self._show_size_var.get(),
            yolox_clear    = self._yolox_clear_var.get(),
            yolox_rain     = self._yolox_rain_var.get(),
            yolox_fog      = self._yolox_fog_var.get(),
            kolo_conf      = self._kolo_conf_var.get(),
            ais_enabled    = self._ais_enabled_var.get(),
            radar_enabled  = self._radar_enabled_var.get(),
        )
        self._log_line("─" * 55, "dim")
        for k, v in params.items():
            self._log_line(f"  {k:<18}: {v}", "hi")
        self._log_line("Starting simulation…", "hi")
        self._log_line("─" * 55, "dim")

        self._sim_thread = threading.Thread(
            target=self._run_simulation, kwargs=params, daemon=True)
        self._sim_thread.start()
        self._watch_thread()

    def _on_stop(self):
        if self._running:
            self._stop_event.set()
            self._log_line("Stop requested…", "warn")
            self._set_status("Stopping…", WARNING)

    def _watch_thread(self):
        if self._sim_thread and self._sim_thread.is_alive():
            self.after(500, self._watch_thread)
        else:
            self._on_sim_done()

    def _on_sim_done(self):
        self._running = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        if self._stop_event.is_set():
            self._set_status("Stopped", WARNING)
            self._log_line("Simulation stopped by user.", "warn")
        else:
            self._set_status("Finished ✓", SUCCESS)
            self._log_line("Simulation completed successfully.", "ok")
        self._log_line("─" * 55, "dim")

    def _run_simulation(self, data_path, res_path,
                        anti, anti_rate,
                        ais_range_nm, radar_range_nm,
                        cpa_warn_m, cpa_warn_s,
                        map_size, show_size,
                        yolox_clear, yolox_rain, yolox_fog,
                        kolo_conf, ais_enabled, radar_enabled):

        old_out, old_err = sys.stdout, sys.stderr
        writer = _QueueWriter(self._log_queue)
        sys.stdout = writer
        sys.stderr = writer

        try:
            proj_root = os.path.dirname(os.path.abspath(__file__))
            if proj_root not in sys.path:
                sys.path.insert(0, proj_root)

            import imutils, numpy as np, pandas as pd, cv2

            from utils.file_read        import read_all, ais_initial, update_time
            from utils.AIS_utils        import AISPRO
            from utils.FUS_utils        import FUSPRO
            from utils.gen_result       import gen_result
            from utils.draw             import DRAW
            from utils.minimap          import MINIMAP
            from utils.detection_logger import DetectionLogger
            import utils.VIS_utils as _vu
            from utils.VIS_utils        import VISPRO, WakeDetector

            video_path, ais_path, result_video, result_metric, \
                initial_time, camera_para = read_all(data_path, res_path)

            # Apply UI thresholds to models
            _vu.yolo.confidence = yolox_clear
            _vu._YOLOX_CONF_CLEAR = yolox_clear
            _vu._YOLOX_CONF_RAIN  = yolox_rain
            _vu._YOLOX_CONF_FOG   = yolox_fog
            if _vu._yolov8 is not None:
                _vu._yolov8.conf = kolo_conf

            ais_max_dis   = ais_range_nm   * 1852
            radar_range_m = radar_range_nm * 1852

            ais_file, timestamp0, time0 = ais_initial(ais_path, initial_time)
            Time = initial_time.copy()

            cap      = cv2.VideoCapture(video_path)
            im_shape = [cap.get(3), cap.get(4)]
            max_dis  = min(im_shape) // 2
            fps      = int(cap.get(5))
            t        = int(1000 / fps)
            clip_name   = os.path.splitext(os.path.basename(result_video))[0]
            _result_dir = os.path.dirname(result_video) or res_path

            AIS = AISPRO(ais_path, ais_file, im_shape, t)
            AIS.max_dis = ais_max_dis

            # Pass ais_enabled to VISPRO
            VIS  = VISPRO(anti, anti_rate, t, ais_enabled=ais_enabled)
            FUS  = FUSPRO(max_dis, im_shape, t)
            DRA  = DRAW(im_shape, t, ais_enabled=ais_enabled)
            WAKE = WakeDetector(shoot_hdir=camera_para[2])
            MM   = MINIMAP(camera_para, im_shape,
                           range_m=radar_range_m, map_size=map_size,
                           result_path=_result_dir, clip_name=clip_name,
                           fps=fps, cpa_warn_m=cpa_warn_m, cpa_warn_s=cpa_warn_s)
            LOGGER = DetectionLogger(result_dir=_result_dir, clip_name=clip_name)

            name        = 'DeepSORVF'
            videoWriter = None
            bin_inf     = pd.DataFrame(columns=['ID', 'mmsi', 'timestamp', 'match'])

            print('Start Time: %s || Stamp: %d || fps: %d' % (time0, timestamp0, fps))
            times = 0; time_i = 0; sum_t = []

            try:
                while not self._stop_event.is_set():
                    _, im = cap.read()
                    if im is None:
                        break
                    start = time.time()
                    Time, timestamp, Time_name = update_time(Time, t)

                    AIS_vis, AIS_cur = AIS.process(camera_para, timestamp, Time_name)
                    Vis_tra, Vis_cur = VIS.feedCap(im, timestamp, AIS_vis, bin_inf, ais_enabled=ais_enabled)
                    Fus_tra, bin_inf = FUS.fusion(AIS_vis, AIS_cur, Vis_tra, Vis_cur, timestamp)
                    wake_angles      = WAKE.update(im, Vis_cur, Vis_tra)

                    end = time.time() - start
                    time_i += end

                    if timestamp % 1000 < t:
                        gen_result(times, Vis_cur, Fus_tra, result_metric, im_shape)
                        yolox_boxes = [
                            (int(r.x1), int(r.y1), int(r.x2), int(r.y2), 'vessel', 1.0)
                            for _, r in Vis_cur.iterrows()
                        ] if len(Vis_cur) else []
                        LOGGER.log_frame(
                            timestamp=timestamp, frame_idx=times, time_name=Time_name,
                            yolox_bboxes=yolox_boxes,
                            maritime_bboxes_kept=VIS.maritime_bboxes,
                            maritime_bboxes_suppressed=VIS.maritime_bboxes_suppressed)
                        times += 1; sum_t.append(time_i)
                        print('Time: %s || Stamp: %d || Process: %.4f || Avg: %.4f' % (
                            Time_name, timestamp, time_i, np.mean(sum_t)))
                        time_i = 0

                    # Draw KOLOMVERSE boxes (orange)
                    for x1, y1, x2, y2, cls_name, conf in VIS.maritime_bboxes:
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(im, f'{cls_name} {float(conf):.2f}',
                                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 165, 255), 2, cv2.LINE_AA)

                    im = DRA.draw_traj(im, AIS_vis, AIS_cur, Vis_tra, Vis_cur, Fus_tra, timestamp)
                    
                    # ── NOUVEAU : Contrôle minimap ───────────────────────────
                    if radar_enabled:
                        im = MM.draw(im, AIS_vis, AIS_cur, Fus_tra, Vis_tra, Vis_cur,
                                     timestamp, wake_angles=wake_angles)

                    result = imutils.resize(im, height=show_size)
                    if videoWriter is None:
                        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
                        videoWriter = cv2.VideoWriter(result_video, fourcc, fps,
                                                      (result.shape[1], result.shape[0]))
                    videoWriter.write(result)
                    cv2.imshow(name, result)
                    cv2.waitKey(1)
                    if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                        break

            except Exception as exc:
                print(f"ERROR: {exc}")
                print(traceback.format_exc())
            finally:
                cap.release()
                if videoWriter: videoWriter.release()
                MM.release()
                cv2.destroyAllWindows()
                LOGGER.close()

        except Exception as exc:
            print(f"ERROR during setup: {exc}")
            print(traceback.format_exc())
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

    def _on_close(self):
        if self._running:
            if messagebox.askyesno("Quit", "Simulation is running. Stop and quit?"):
                self._stop_event.set()
                self.after(800, self.destroy)
        else:
            self.destroy()


# ── Widget helpers ─────────────────────────────────────────────────────────────
def _sep(parent):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")

def _label(parent, text):
    tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"),
             bg=BG_CARD, fg=TEXT_SEC).pack(anchor="w", pady=(6, 2))

def _entry(parent, var):
    return tk.Entry(parent, textvariable=var, bg=BG_DARK, fg=TEXT_PRI,
                    insertbackground=TEXT_PRI, relief="flat",
                    font=("Consolas", 9), width=30)

def _btn_small(parent, text, command, fg=TEXT_PRI):
    return tk.Button(parent, text=text, command=command,
                     bg=BG_DARK, fg=fg, activebackground=BORDER,
                     activeforeground=TEXT_PRI, relief="flat", cursor="hand2",
                     font=("Segoe UI", 8), padx=8, pady=4)

def _card(parent, title):
    outer = tk.Frame(parent, bg=BG_PANEL,
                     highlightbackground=BORDER, highlightthickness=1)
    outer.pack(fill="x", pady=(0, 10))
    tk.Label(outer, text=title, font=("Segoe UI", 10, "bold"),
             bg=BG_PANEL, fg=TEXT_PRI, padx=12, pady=8).pack(anchor="w")
    _sep(outer)
    inner = tk.Frame(outer, bg=BG_CARD, padx=12, pady=10)
    inner.pack(fill="x")
    return inner

def _spinrow(parent, label, from_, to, increment, default, is_int=False):
    _label(parent, label)
    var = tk.IntVar(value=int(default)) if is_int else tk.DoubleVar(value=default)
    tk.Spinbox(parent, from_=from_, to=to, increment=increment,
               textvariable=var, width=9,
               bg=BG_DARK, fg=TEXT_PRI, insertbackground=TEXT_PRI,
               buttonbackground=BG_DARK, relief="flat",
               font=("Consolas", 9)).pack(anchor="w", pady=(0, 4))
    return var


# Patch VIS_utils to use dynamic conf variables
import utils.VIS_utils as _vu_module
if not hasattr(_vu_module, '_YOLOX_CONF_CLEAR'):
    _vu_module._YOLOX_CONF_CLEAR = 0.50
    _vu_module._YOLOX_CONF_RAIN  = 0.35
    _vu_module._YOLOX_CONF_FOG   = 0.30


if __name__ == "__main__":
    app = DeepSORVFLauncher()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()