"""Native AdamLM desktop application. Launch with AdamLM.cmd."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .gui_core import (ROOT, TrainingPlan, build_plan, discover_runs, disk_telemetry,
                       format_duration, format_number, gpu_telemetry, latest_checkpoint,
                       load_metrics, read_json, request_stop, trainer_active)


BG, SURFACE, RAISED = "#0b0f15", "#111720", "#171e29"
LINE, TEXT, MUTED = "#273141", "#f2f5f8", "#8e9bad"
ACCENT, ACCENT_DARK, WARN, BAD = "#60d8ad", "#173c35", "#f1b35b", "#ef7272"


def human_bytes(value):
    if value is None:
        return "—"
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024


class LossChart(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=SURFACE, highlightthickness=0, height=185, **kwargs)
        self.train, self.validation = [], []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, rows):
        self.train = [(r.get("step"), r.get("train_loss")) for r in rows if r.get("step") is not None and r.get("train_loss") is not None]
        self.validation = [(r.get("step"), r.get("validation_loss")) for r in rows if r.get("step") is not None and r.get("validation_loss") is not None]
        self.redraw()

    def add_live(self, step, loss):
        if step is None or loss is None or (self.train and self.train[-1][0] == step):
            return
        self.train.append((step, loss))
        self.train = self.train[-240:]
        self.redraw()

    def redraw(self):
        self.delete("all")
        width, height = max(100, self.winfo_width()), max(100, self.winfo_height())
        left, top, right, bottom = 46, 18, width - 16, height - 29
        all_points = self.train + self.validation
        self.create_text(16, 8, text="LOSS", fill=MUTED, anchor="nw", font=("Segoe UI Semibold", 9))
        self.create_text(width - 16, 8, text="TRAIN  •  VALIDATION", fill=MUTED, anchor="ne", font=("Segoe UI", 9))
        for i in range(4):
            y = top + (bottom - top) * i / 3
            self.create_line(left, y, right, y, fill=LINE)
        if not all_points:
            self.create_text((left + right) / 2, (top + bottom) / 2, text="Loss appears after training reports metrics", fill=MUTED, font=("Segoe UI", 10))
            return
        xs, ys = [p[0] for p in all_points], [p[1] for p in all_points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        if xmax == xmin: xmax += 1
        if ymax == ymin: ymax += 1
        pad = (ymax - ymin) * .08
        ymin, ymax = ymin - pad, ymax + pad
        self.create_text(left - 8, top, text=f"{ymax:.2f}", fill=MUTED, anchor="e", font=("Segoe UI", 8))
        self.create_text(left - 8, bottom, text=f"{ymin:.2f}", fill=MUTED, anchor="e", font=("Segoe UI", 8))
        self.create_text(left, bottom + 11, text=f"step {xmin:,}", fill=MUTED, anchor="nw", font=("Segoe UI", 8))
        self.create_text(right, bottom + 11, text=f"step {xmax:,}", fill=MUTED, anchor="ne", font=("Segoe UI", 8))
        def draw(points, color, width_px):
            coords = []
            for x, y in points:
                coords += [left + (x - xmin) / (xmax - xmin) * (right - left), bottom - (y - ymin) / (ymax - ymin) * (bottom - top)]
            if len(coords) >= 4:
                self.create_line(*coords, fill=color, width=width_px, smooth=True)
            elif coords:
                self.create_oval(coords[0]-2, coords[1]-2, coords[0]+2, coords[1]+2, fill=color, outline="")
        draw(self.train, ACCENT, 2)
        draw(self.validation, WARN, 2)


class AdamLMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AdamLM")
        self.geometry("1280x820")
        self.minsize(1040, 700)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.events = queue.Queue()
        self.runs = []
        self.run_by_label = {}
        self.checkpoint_by_label = {}
        self.current_plan: TrainingPlan | None = None
        self.owned_process = None
        self.owned_log = None
        self.closing = False
        self.last_sample = None
        self.telemetry_pending = False
        self._style()
        self._variables()
        self._build()
        self.refresh_inventory()
        self.after(250, self._drain_events)
        self.after(800, self.poll)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=RAISED, bordercolor=LINE,
                        font=("Segoe UI", 10), focuscolor=ACCENT)
        style.configure("TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Raised.TFrame", background=RAISED)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("SurfaceMuted.TLabel", background=SURFACE, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 24))
        style.configure("Section.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 13))
        style.configure("Metric.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 15))
        style.configure("Accent.TButton", background=ACCENT, foreground="#07120f", padding=(15, 9), borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#83e5c2"), ("disabled", LINE)], foreground=[("disabled", MUTED)])
        style.configure("TButton", background=RAISED, foreground=TEXT, padding=(13, 8), borderwidth=0)
        style.map("TButton", background=[("active", LINE), ("disabled", SURFACE)], foreground=[("disabled", MUTED)])
        style.configure("TEntry", padding=8, insertcolor=TEXT)
        style.configure("TCombobox", padding=7, arrowsize=14)
        style.map("TCombobox", fieldbackground=[("readonly", RAISED)], selectbackground=[("readonly", RAISED)], selectforeground=[("readonly", TEXT)])
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(18, 11), borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT)])

    def _variables(self):
        self.mode = tk.StringVar(value="New continuation stage")
        self.run_choice = tk.StringVar()
        self.dataset = tk.StringVar(value="TinyStories")
        self.checkpoint_choice = tk.StringVar()
        self.stage_name = tk.StringVar(value=f"gui-stage-{time.strftime('%Y%m%d-%H%M%S')}")
        self.additional = tk.StringVar(value="1000000")
        self.duration = tk.StringVar(value="60")
        self.duration_unit = tk.StringVar(value="minutes")
        self.until_stopped = tk.BooleanVar(value=False)
        self.mixture = tk.StringVar(value="tinystories=0.20,wikitext103=0.35,fineweb_edu=0.45")
        self.plan_text = tk.StringVar(value="Select a checkpoint to prepare a continuation stage.")
        self.plan_meta = tk.StringVar(value="")
        self.state_text = tk.StringVar(value="IDLE")
        self.step_text = tk.StringVar(value="—")
        self.tokens_text = tk.StringVar(value="—")
        self.speed_text = tk.StringVar(value="—")
        self.gpu_text = tk.StringVar(value="—")
        self.vram_text = tk.StringVar(value="—")
        self.source_text = tk.StringVar(value="—")
        self.checkpoint_text = tk.StringVar(value="—")
        self.notice = tk.StringVar(value="No training starts until you press Start / Resume Training.")
        self.play_checkpoint = tk.StringVar()
        self.play_mode = tk.StringVar(value="Story completion")
        self.temperature = tk.StringVar(value="0.8")
        self.top_k = tk.StringVar(value="40")
        self.generation_length = tk.StringVar(value="160")

    def _build(self):
        shell = ttk.Frame(self, padding=(28, 20, 28, 24))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="AdamLM", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="LOCAL TRAINING STUDIO", foreground=ACCENT, font=("Segoe UI Semibold", 9)).pack(side="left", padx=(15, 0), pady=(10, 0))
        self.header_state = ttk.Label(header, textvariable=self.state_text, foreground=ACCENT, font=("Segoe UI Semibold", 10))
        self.header_state.pack(side="right", pady=(9, 0))
        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill="both", expand=True, pady=(12, 0))
        self.train_page = ttk.Frame(self.notebook)
        self.play_page = ttk.Frame(self.notebook)
        self.notebook.add(self.train_page, text="TRAINING")
        self.notebook.add(self.play_page, text="PLAYGROUND")
        self._build_training()
        self._build_playground()

    def _metric(self, parent, title, variable, column):
        box = ttk.Frame(parent, style="Surface.TFrame", padding=(16, 13))
        box.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 1, 0))
        ttk.Label(box, text=title.upper(), style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).pack(anchor="w")
        ttk.Label(box, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(5, 0))

    def _build_training(self):
        page = self.train_page
        page.columnconfigure(0, weight=5)
        page.columnconfigure(1, weight=4)
        page.rowconfigure(2, weight=1)
        metrics = ttk.Frame(page, style="Surface.TFrame")
        metrics.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(5, 14))
        for i in range(7): metrics.columnconfigure(i, weight=1)
        for i, item in enumerate((("State", self.state_text), ("Step", self.step_text), ("Processed", self.tokens_text),
                                  ("Throughput / ETA", self.speed_text), ("GPU", self.gpu_text),
                                  ("VRAM / Temp", self.vram_text), ("Dataset / LR", self.source_text))):
            self._metric(metrics, item[0], item[1], i)
        controls = ttk.Frame(page)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        self.start_button = ttk.Button(controls, text="Start / Resume Training", style="Accent.TButton", command=self.start_training)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Graceful Stop", command=self.stop_training)
        self.stop_button.pack(side="left", padx=7)
        ttk.Button(controls, text="View Training Status", command=self.show_status).pack(side="left", padx=7)
        ttk.Button(controls, text="Open Checkpoint Folder", command=self.open_checkpoint_folder).pack(side="left", padx=7)
        ttk.Button(controls, text="Test Model", command=self.test_model).pack(side="left", padx=7)
        ttk.Label(controls, textvariable=self.notice, style="Muted.TLabel").pack(side="right")
        left = ttk.Frame(page)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(page)
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(left, text="Training configuration", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        form = ttk.Frame(left, style="Surface.TFrame", padding=18)
        form.grid(row=1, column=0, sticky="nsew")
        form.columnconfigure(0, weight=1); form.columnconfigure(1, weight=1)
        self._field(form, "ACTION", ttk.Combobox(form, textvariable=self.mode, values=["Resume selected run", "New continuation stage"], state="readonly"), 0, 0, 2)
        self.run_combo = ttk.Combobox(form, textvariable=self.run_choice, state="readonly")
        self._field(form, "EXISTING RUN", self.run_combo, 1, 0)
        self._field(form, "DATASET", ttk.Combobox(form, textvariable=self.dataset, values=["TinyStories", "WikiText-103", "FineWeb-Edu", "Custom mixture"], state="readonly"), 1, 1)
        self.checkpoint_combo = ttk.Combobox(form, textvariable=self.checkpoint_choice, state="readonly")
        self._field(form, "PARENT CHECKPOINT", self.checkpoint_combo, 2, 0)
        stage_box = ttk.Frame(form, style="Surface.TFrame")
        ttk.Entry(stage_box, textvariable=self.stage_name).pack(side="left", fill="x", expand=True)
        self._field(form, "NEW STAGE NAME", stage_box, 2, 1)
        add_box = ttk.Frame(form, style="Surface.TFrame")
        ttk.Entry(add_box, textvariable=self.additional).pack(fill="x")
        presets = ttk.Frame(add_box, style="Surface.TFrame")
        presets.pack(fill="x", pady=(6, 0))
        for label, value in (("1M", 1_000_000), ("10M", 10_000_000), ("50M", 50_000_000), ("100M", 100_000_000)):
            ttk.Button(presets, text=label, command=lambda v=value: self.additional.set(str(v))).pack(side="left", padx=(0, 4))
        self._field(form, "ADDITIONAL TRAINING TOKENS", add_box, 3, 0, 2)
        duration_box = ttk.Frame(form, style="Surface.TFrame")
        ttk.Entry(duration_box, textvariable=self.duration, width=12).pack(side="left", fill="x", expand=True)
        ttk.Combobox(duration_box, textvariable=self.duration_unit, values=["minutes", "hours"], state="readonly", width=10).pack(side="left", padx=(6, 0))
        self._field(form, "MAXIMUM SESSION", duration_box, 4, 0)
        until = ttk.Checkbutton(form, text="Run until stopped", variable=self.until_stopped)
        self._field(form, "SESSION MODE", until, 4, 1)
        self._field(form, "CUSTOM MIXTURE", ttk.Entry(form, textvariable=self.mixture), 5, 0, 2)
        ttk.Label(right, text="Effective plan", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        plan = ttk.Frame(right, style="Surface.TFrame", padding=18)
        plan.grid(row=1, column=0, sticky="nsew")
        plan.columnconfigure(0, weight=1)
        ttk.Label(plan, textvariable=self.plan_text, style="Surface.TLabel", font=("Segoe UI Semibold", 12), wraplength=440, justify="left").grid(row=0, column=0, sticky="ew")
        ttk.Label(plan, textvariable=self.plan_meta, style="SurfaceMuted.TLabel", wraplength=440, justify="left").grid(row=1, column=0, sticky="ew", pady=(8, 14))
        self.chart = LossChart(plan)
        self.chart.grid(row=2, column=0, sticky="nsew", pady=(8, 12))
        plan.rowconfigure(2, weight=1)
        ttk.Label(plan, textvariable=self.checkpoint_text, style="SurfaceMuted.TLabel", wraplength=440, justify="left").grid(row=3, column=0, sticky="ew")
        for variable in (self.mode, self.run_choice, self.dataset, self.checkpoint_choice, self.stage_name,
                         self.additional, self.duration, self.duration_unit, self.until_stopped, self.mixture):
            variable.trace_add("write", lambda *_args: self.after(80, self.preview_plan))
        self.run_choice.trace_add("write", lambda *_args: self.on_run_selected())

    def _field(self, parent, title, widget, row, column, span=1):
        box = ttk.Frame(parent, style="Surface.TFrame")
        box.grid(row=row, column=column, columnspan=span, sticky="ew", padx=(0 if column == 0 else 8, 8 if column == 0 else 0), pady=(0, 12))
        ttk.Label(box, text=title, style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(0, 5))
        widget.pack(in_=box, fill="x")

    def _build_playground(self):
        page = self.play_page
        page.columnconfigure(0, weight=1); page.columnconfigure(1, weight=1); page.rowconfigure(1, weight=1)
        ttk.Label(page, text="Model playground", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(8, 12))
        ttk.Label(page, text="Output is generated locally from the selected AdamLM checkpoint.", style="Muted.TLabel").grid(row=0, column=1, sticky="e", pady=(8, 12))
        left = ttk.Frame(page, style="Surface.TFrame", padding=20)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 9))
        right = ttk.Frame(page, style="Surface.TFrame", padding=20)
        right.grid(row=1, column=1, sticky="nsew", padx=(9, 0))
        for frame in (left, right): frame.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1); right.rowconfigure(1, weight=1)
        ttk.Label(left, text="CHECKPOINT", style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).grid(row=0, column=0, sticky="w")
        cp = ttk.Frame(left, style="Surface.TFrame")
        cp.grid(row=1, column=0, sticky="ew", pady=(6, 14)); cp.columnconfigure(0, weight=1)
        self.play_combo = ttk.Combobox(cp, textvariable=self.play_checkpoint, state="readonly")
        self.play_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(cp, text="Browse", command=self.browse_checkpoint).grid(row=0, column=1, padx=(7, 0))
        settings = ttk.Frame(left, style="Surface.TFrame")
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        for i in range(4): settings.columnconfigure(i, weight=1)
        for col, (label, var, values) in enumerate((("MODE", self.play_mode, ["Story completion", "Conversation"]),
                                                    ("TEMPERATURE", self.temperature, None), ("TOP-K", self.top_k, None),
                                                    ("NEW TOKENS", self.generation_length, None))):
            cell = ttk.Frame(settings, style="Surface.TFrame"); cell.grid(row=0, column=col, sticky="ew", padx=(0, 7 if col < 3 else 0))
            ttk.Label(cell, text=label, style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(0, 5))
            (ttk.Combobox(cell, textvariable=var, values=values, state="readonly") if values else ttk.Entry(cell, textvariable=var)).pack(fill="x")
        ttk.Label(left, text="PROMPT", style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).grid(row=3, column=0, sticky="w")
        self.prompt = tk.Text(left, bg=RAISED, fg=TEXT, insertbackground=TEXT, selectbackground=ACCENT_DARK,
                              relief="flat", font=("Segoe UI", 11), wrap="word", padx=12, pady=12, height=12)
        self.prompt.grid(row=4, column=0, sticky="nsew", pady=(6, 14))
        self.prompt.insert("1.0", "Once upon a time, a curious fox")
        self.generate_button = ttk.Button(left, text="Generate locally", style="Accent.TButton", command=self.generate)
        self.generate_button.grid(row=5, column=0, sticky="w")
        ttk.Label(right, text="GENERATED OUTPUT", style="SurfaceMuted.TLabel", font=("Segoe UI Semibold", 8)).grid(row=0, column=0, sticky="w")
        self.output = tk.Text(right, bg=SURFACE, fg=TEXT, insertbackground=TEXT, selectbackground=ACCENT_DARK,
                              relief="flat", font=("Cascadia Mono", 10), wrap="word", padx=2, pady=10, state="disabled")
        self.output.grid(row=1, column=0, sticky="nsew")

    def refresh_inventory(self):
        self.runs = discover_runs()
        self.run_by_label = {f"{r.name}  ·  {r.status.get('state', 'unknown')}": r for r in self.runs}
        values = list(self.run_by_label)
        self.run_combo.configure(values=values)
        checkpoints = [(f"{r.name}  ·  {r.checkpoint.name}", r.checkpoint) for r in self.runs if r.checkpoint]
        self.checkpoint_by_label = dict(checkpoints)
        cp_values = list(self.checkpoint_by_label)
        self.checkpoint_combo.configure(values=cp_values)
        self.play_combo.configure(values=cp_values)
        if values and self.run_choice.get() not in self.run_by_label:
            self.run_choice.set(values[0])
        if cp_values and self.checkpoint_choice.get() not in self.checkpoint_by_label:
            self.checkpoint_choice.set(cp_values[0])
        if cp_values and self.play_checkpoint.get() not in self.checkpoint_by_label:
            self.play_checkpoint.set(cp_values[0])
        self.preview_plan()

    def selected_run(self):
        return self.run_by_label.get(self.run_choice.get())

    def selected_checkpoint(self, playground=False):
        value = self.play_checkpoint.get() if playground else self.checkpoint_choice.get()
        return self.checkpoint_by_label.get(value, Path(value) if value else None)

    def on_run_selected(self):
        run = self.selected_run()
        if run:
            self.chart.set_data(load_metrics(run.directory))
            if run.checkpoint:
                label = next((key for key, value in self.checkpoint_by_label.items() if value == run.checkpoint), None)
                if label: self.checkpoint_choice.set(label)

    def _dataset_key(self):
        return {"TinyStories": "tinystories", "WikiText-103": "wikitext103", "FineWeb-Edu": "fineweb_edu", "Custom mixture": "mixture"}[self.dataset.get()]

    def _duration_seconds(self):
        if self.until_stopped.get(): return None
        factor = 3600 if self.duration_unit.get() == "hours" else 60
        return float(self.duration.get()) * factor

    def make_plan(self):
        mode = "resume" if self.mode.get().startswith("Resume") else "continuation"
        name = self.stage_name.get().strip()
        if mode == "continuation" and (not name or Path(name).name != name):
            raise ValueError("Stage name must be one folder name")
        run_dir = ROOT / "results" / name if mode == "continuation" else None
        return build_plan(mode=mode, dataset=self._dataset_key(), additional_tokens=int(self.additional.get().replace(",", "")),
                          duration_seconds=self._duration_seconds(), selected_run=self.selected_run(),
                          parent_checkpoint=self.selected_checkpoint(), mixture=self.mixture.get().strip(),
                          run_dir=run_dir)

    def preview_plan(self):
        try:
            plan = self.make_plan()
            self.current_plan = plan
            action = "Resume immutable stage" if plan.mode == "resume" else "Create continuation stage"
            self.plan_text.set(f"{action} · {format_number(plan.remaining_tokens)} tokens remaining")
            eta = format_duration(plan.estimated_seconds) if plan.estimated_seconds is not None else "Unavailable until throughput is measured"
            session = "until stopped" if self.until_stopped.get() else format_duration(self._duration_seconds())
            self.plan_meta.set(
                f"Additional requested  {format_number(plan.additional_tokens)}\n"
                f"Cumulative in stage  {format_number(plan.cumulative_tokens)}\n"
                f"Effective target  {format_number(plan.effective_target)}  ·  {plan.total_updates:,} updates\n"
                f"This session  {session}  ·  estimated completion {eta}\n"
                f"LR schedule  warmup {plan.warmup_steps:,} updates  ·  {plan.learning_rate:g} → {plan.minimum_learning_rate:g}\n"
                f"Model  AdamLM {plan.params:,} parameters  ·  checkpoint reserve ~{human_bytes(plan.checkpoint_bytes)}")
            self.checkpoint_text.set(f"Checkpoint  {plan.checkpoint}\nRun folder  {plan.run_dir}")
            self.start_button.configure(state="disabled" if any(r.active for r in self.runs) else "normal")
        except Exception as exc:
            self.current_plan = None
            self.plan_text.set(str(exc))
            self.plan_meta.set("Existing checkpoint targets and schedules remain unchanged.")
            self.start_button.configure(state="disabled")

    def start_training(self):
        try:
            plan = self.make_plan()
            if any(trainer_active(r.directory) for r in discover_runs()):
                raise RuntimeError("A trainer is already active. Stop it before starting another session.")
            if plan.mode == "continuation" and plan.run_dir.exists() and any((plan.run_dir / "checkpoints").glob("step_*.pt")):
                raise RuntimeError("That stage already has checkpoints. Choose Resume or use a new stage name.")
            plan.run_dir.mkdir(parents=True, exist_ok=True)
            log_path = plan.run_dir / "gui-training.log"
            self.owned_log = log_path.open("a", encoding="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.owned_process = subprocess.Popen(plan.command, cwd=ROOT, stdout=self.owned_log,
                                                  stderr=subprocess.STDOUT, creationflags=flags)
            self.current_plan = plan
            self.notice.set("Trainer is starting…")
            self.state_text.set("STARTING")
            self.start_button.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Cannot start training", str(exc), parent=self)

    def stop_training(self):
        run = self._active_run() or self.selected_run()
        if not run or not trainer_active(run.directory):
            self.notice.set("Training is not running; checkpoints are unchanged.")
            return
        request_stop(run.directory)
        self.notice.set("Graceful stop requested · waiting for a saved checkpoint…")
        self.state_text.set("STOPPING")

    def _active_run(self):
        return next((run for run in self.runs if run.active), None)

    def show_status(self):
        run = self._active_run() or self.selected_run()
        if not run:
            messagebox.showinfo("Training status", "No training run is selected.", parent=self); return
        status = read_json(run.directory / "status.json", {})
        lines = [f"Run: {run.directory}", f"Trainer active: {trainer_active(run.directory)}"]
        lines += [f"{key.replace('_', ' ').title()}: {status.get(key, 'Unavailable')}" for key in
                  ("state", "stage", "dataset", "step", "tokens", "target_tokens", "train_loss", "validation_loss", "learning_rate", "checkpoint", "updated_at", "detail")]
        messagebox.showinfo("Training status", "\n".join(lines), parent=self)

    def open_checkpoint_folder(self):
        run = self._active_run() or self.selected_run()
        folder = (run.directory / "checkpoints") if run else ROOT / "results"
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def test_model(self):
        run = self._active_run() or self.selected_run()
        if run and run.checkpoint:
            label = next((key for key, value in self.checkpoint_by_label.items() if value == run.checkpoint), None)
            if label: self.play_checkpoint.set(label)
        self.notebook.select(self.play_page)

    def browse_checkpoint(self):
        value = filedialog.askopenfilename(parent=self, title="Select AdamLM checkpoint", initialdir=ROOT / "results", filetypes=[("PyTorch checkpoint", "*.pt")])
        if value:
            label = Path(value).name
            self.checkpoint_by_label[label] = Path(value)
            self.play_combo.configure(values=list(self.checkpoint_by_label))
            self.play_checkpoint.set(label)

    def _checkpoint_is_sft(self, checkpoint):
        try:
            run_dir = checkpoint.parent.parent
            return read_json(run_dir / "launcher-config.json", {}).get("stage") == "sft"
        except AttributeError:
            return False

    def generate(self):
        checkpoint = self.selected_checkpoint(playground=True)
        prompt = self.prompt.get("1.0", "end-1c").strip()
        try:
            if not checkpoint or not checkpoint.is_file(): raise ValueError("Select a saved checkpoint")
            if not prompt: raise ValueError("Enter a prompt")
            if any(run.active for run in discover_runs()): raise RuntimeError("Stop training before loading a model into GPU memory.")
            chat = self.play_mode.get() == "Conversation"
            if chat and not self._checkpoint_is_sft(checkpoint):
                raise ValueError("Conversation mode requires a checkpoint from an instruction-tuned SFT stage.")
            temperature, top_k, length = float(self.temperature.get()), int(self.top_k.get()), int(self.generation_length.get())
            if temperature <= 0 or top_k <= 0 or length <= 0: raise ValueError("Generation settings must be positive")
        except Exception as exc:
            messagebox.showerror("Cannot generate", str(exc), parent=self); return
        command = [str(ROOT / ".venv" / "Scripts" / "python.exe"), "scripts/generate.py", str(checkpoint),
                   "--prompt", prompt, "--tokens", str(length), "--temperature", str(temperature), "--top-k", str(top_k)]
        if chat: command.append("--chat")
        self.generate_button.configure(state="disabled", text="Generating…")
        self._set_output("Loading checkpoint and generating locally…")
        def worker():
            try:
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if result.returncode:
                    raise RuntimeError((result.stderr or result.stdout).strip())
                self.events.put(("generated", result.stdout.strip()))
            except Exception as exc:
                self.events.put(("generation_error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _set_output(self, text):
        self.output.configure(state="normal"); self.output.delete("1.0", "end"); self.output.insert("1.0", text); self.output.configure(state="disabled")

    def _drain_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "telemetry": self._apply_poll(value)
                elif kind == "generated":
                    self._set_output(value); self.generate_button.configure(state="normal", text="Generate locally")
                elif kind == "generation_error":
                    self._set_output("Generation failed.\n\n" + value); self.generate_button.configure(state="normal", text="Generate locally")
                elif kind == "closed": self.destroy(); return
        except queue.Empty:
            pass
        if self.winfo_exists(): self.after(250, self._drain_events)

    def poll(self):
        if not self.telemetry_pending:
            self.telemetry_pending = True
            selected = self.selected_run()
            selected_dir = selected.directory if selected else None
            def worker():
                runs = discover_runs(); active = next((r for r in runs if r.active), None)
                shown = active or next((r for r in runs if r.directory == selected_dir), None)
                payload = {"runs": runs, "run": shown, "gpu": gpu_telemetry(), "disk": disk_telemetry()}
                self.events.put(("telemetry", payload))
            threading.Thread(target=worker, daemon=True).start()
        if self.winfo_exists(): self.after(1000, self.poll)

    def _apply_poll(self, payload):
        self.telemetry_pending = False
        self.runs = payload["runs"]
        if self.closing and self.owned_process is not None and self.owned_process.poll() is None and self.current_plan:
            request_stop(self.current_plan.run_dir)
        run = payload["run"]
        gpu, disk = payload["gpu"], payload["disk"]
        now = time.monotonic()
        if run:
            status = read_json(run.directory / "status.json", {})
            state = status.get("state", "unknown")
            display_state = "STOPPING" if (run.directory / "STOP_REQUESTED").exists() and run.active else state.upper().replace("_", " ")
            self.state_text.set(display_state)
            self.header_state.configure(foreground=BAD if state == "error" else (WARN if state in {"disk_pause", "interrupted"} else ACCENT))
            step, tokens, target = status.get("step"), status.get("tokens"), status.get("target_tokens") or run.target
            self.step_text.set(f"{step:,}" if isinstance(step, int) else "—")
            self.tokens_text.set(f"{format_number(tokens)} / {format_number(target)}")
            speed = None
            if run.active and isinstance(tokens, int):
                if self.last_sample and self.last_sample[0] == run.directory and tokens >= self.last_sample[1]:
                    elapsed = now - self.last_sample[2]
                    if elapsed > 0 and tokens > self.last_sample[1]: speed = (tokens - self.last_sample[1]) / elapsed
                self.last_sample = (run.directory, tokens, now)
            if speed:
                remaining = max(0, (target or 0) - tokens)
                self.speed_text.set(f"{format_number(speed)}/s · {format_duration(remaining/speed)}")
            else: self.speed_text.set("Measuring…" if run.active else "—")
            lr = status.get("learning_rate")
            self.source_text.set(f"{status.get('dataset', run.launcher.get('dataset', '—'))} · {lr:.2e}" if isinstance(lr, (int, float)) else status.get("dataset", run.launcher.get("dataset", "—")))
            checkpoint = latest_checkpoint(run.directory)
            self.checkpoint_text.set(f"Latest checkpoint  {checkpoint or 'Unavailable'}\nAvailable disk  {human_bytes(disk.get('free'))}")
            self.chart.add_live(step, status.get("train_loss"))
            if run.active:
                self.notice.set("Graceful stop pending · saving safely…" if (run.directory / "STOP_REQUESTED").exists() else "Training is active · dashboard uses live metrics")
            elif self.owned_process is not None and self.owned_process.poll() is not None:
                if self.owned_log: self.owned_log.close(); self.owned_log = None
                self.owned_process = None
                self.notice.set("Training has stopped; the latest complete checkpoint is available.")
                self.refresh_inventory()
        else:
            self.state_text.set("IDLE"); self.step_text.set("—"); self.tokens_text.set("—"); self.speed_text.set("—"); self.source_text.set("—")
        if gpu:
            self.gpu_text.set(f"{gpu['utilization']:.0f}%")
            self.vram_text.set(f"{gpu['memory_used_mib']:.0f}/{gpu['memory_total_mib']:.0f} MiB · {gpu['temperature_c']:.0f}°C")
        else:
            self.gpu_text.set("Unavailable"); self.vram_text.set("Unavailable")
        self.stop_button.configure(state="normal" if run and run.active else "disabled")
        self.start_button.configure(state="disabled" if any(r.active for r in self.runs) or self.current_plan is None else "normal")
        if self.closing and not any(r.active for r in self.runs):
            self.notice.set("Training stopped safely. Closing AdamLM…")
            self.after(350, self.destroy)

    def on_close(self):
        owned_active = self.owned_process is not None and self.owned_process.poll() is None
        if owned_active:
            self.closing = True
            if self.current_plan: request_stop(self.current_plan.run_dir)
            self.notice.set("Closing after the trainer saves and reports that it has stopped…")
            self.state_text.set("STOPPING")
            return
        self.destroy()


def main():
    app = AdamLMApp()
    # Tk's mainloop does not return to the interpreter while idle, so a
    # console Ctrl+C would otherwise sit unhandled until the next UI event.
    # A periodic no-op tick lets Python service signals promptly, and the
    # handler reuses on_close so Ctrl+C behaves exactly like the window's
    # close button -- including waiting for an owned trainer to save.
    def pump():
        app.after(200, pump)

    app.after(200, pump)
    signal.signal(signal.SIGINT, lambda *_: app.after(0, app.on_close))
    app.mainloop()


if __name__ == "__main__":
    main()
