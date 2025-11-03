"""
🎮 HYPERREALISTISCHE TIFF-SIMULATOR GUI V4.0 - ADVANCED EDITION
===============================================================

ERWEITERTE GUI mit maximalen Einstellungsmöglichkeiten!

Features:
✅ Alle physikalischen Parameter einstellbar
✅ Live-Preview für Frames
✅ Erweiterte Photophysik-Steuerung
✅ Optimierte Performance-Engine
✅ Tooltips für alle Parameter
✅ Schöneres modernes Design
✅ Batch-Modus mit Custom-Presets
✅ Echtzeit Progress-Tracking

Version: 4.0 - Oktober 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pathlib import Path
from datetime import datetime

try:
    from tiff_simulator_v3 import (
        TDI_PRESET, TETRASPECS_PRESET, TIFFSimulatorOptimized, save_tiff
    )
    from metadata_exporter import MetadataExporter
    from rf_trainer import RandomForestTrainer, RFTrainingConfig
    from track_analysis import TrackAnalysisOrchestrator, TrackMateXMLParser
    from adaptive_rf_trainer import quick_train_adaptive_rf
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Bitte stelle sicher, dass alle Dateien im gleichen Ordner sind:")
    print("   - tiff_simulator_v3.py (optimized)")
    print("   - metadata_exporter.py")
    print("   - batch_simulator.py")
    print("   - track_analysis.py")
    print("\n   UND dass alle Dependencies installiert sind:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


class ToolTip:
    """Tooltip-Widget für Hilfe-Texte."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip, text=self.text, justify=tk.LEFT,
            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
            font=("Arial", 9)
        )
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ScrollableFrame(ttk.Frame):
    """Frame mit Scrollbar."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self, bg='#f5f5f5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mausrad-Scrolling
        canvas.bind_all("<MouseWheel>",
                       lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))


class TIFFSimulatorGUI_V4:
    """
    ERWEITERTE Hauptfenster für TIFF-Simulation.

    Version 4.0 mit maximalen Einstellungsmöglichkeiten!
    """

    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Hyperrealistischer TIFF Simulator V4.0 - ADVANCED")
        self.root.geometry("1100x850")
        self.root.resizable(True, True)

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Variablen
        self._init_variables()

        # GUI aufbauen
        self._create_widgets()

        # Defaults setzen
        self._apply_detector_preset()
        self._update_mode_info()

        # Thread für Simulation
        self.simulation_thread = None
        self.is_running = False

    def _init_variables(self):
        """Initialisiert alle Tkinter-Variablen."""

        # ===== BASIC SETTINGS =====
        self.detector_var = tk.StringVar(value="TDI-G0")
        self.mode_var = tk.StringVar(value="single")
        self.sim_mode_var = tk.StringVar(value="polyzeit")

        # Output
        self.output_dir = tk.StringVar(value=str(Path.home() / "Desktop"))
        self.filename = tk.StringVar(value="simulation.tif")

        # ===== IMAGE PARAMETERS =====
        self.image_width = tk.IntVar(value=256)  # REALISTISCH
        self.image_height = tk.IntVar(value=256)
        self.num_spots = tk.IntVar(value=30)

        # NEU: Spot Range für Batch
        self.num_spots_min = tk.IntVar(value=10)
        self.num_spots_max = tk.IntVar(value=20)

        # ===== TIME SERIES PARAMETERS =====
        self.t_poly = tk.DoubleVar(value=60.0)
        self.d_initial = tk.DoubleVar(value=0.24)  # KORRIGIERT: Realistische Werte!
        self.num_frames = tk.IntVar(value=200)  # REALISTISCH
        self.frame_rate = tk.DoubleVar(value=20.0)
        self.exposure_substeps = tk.IntVar(value=3)

        # ===== PHOTOPHYSICS (NEU!) =====
        self.enable_photophysics = tk.BooleanVar(value=False)
        self.on_mean_frames = tk.DoubleVar(value=4.0)
        self.off_mean_frames = tk.DoubleVar(value=6.0)
        self.bleach_prob = tk.DoubleVar(value=0.002)

        # ===== NOISE & PSF (NEU!) =====
        self.background_mean = tk.DoubleVar(value=100.0)
        self.background_std = tk.DoubleVar(value=15.0)
        self.read_noise_std = tk.DoubleVar(value=1.5)
        self.spot_intensity_sigma = tk.DoubleVar(value=0.25)
        self.frame_jitter_sigma = tk.DoubleVar(value=0.10)
        self.max_intensity = tk.DoubleVar(value=260.0)

        # ===== ASTIGMATISM & 3D (NEU!) =====
        self.z_amp_um = tk.DoubleVar(value=0.7)
        self.z_max_um = tk.DoubleVar(value=0.6)
        self.astig_z0_um = tk.DoubleVar(value=0.5)
        self.astig_Ax = tk.DoubleVar(value=1.0)
        self.astig_Ay = tk.DoubleVar(value=-0.5)

        # ===== Z-STACK =====
        self.z_min = tk.DoubleVar(value=-1.0)
        self.z_max = tk.DoubleVar(value=1.0)
        self.z_step = tk.DoubleVar(value=0.1)

        # ===== BATCH (KOMPLETT NEU!) =====
        self.batch_mode_enabled = tk.BooleanVar(value=False)  # Single vs Batch
        self.batch_poly_times = tk.StringVar(value="0, 30, 60, 90, 120")
        self.batch_repeats = tk.IntVar(value=3)
        self.batch_astig = tk.BooleanVar(value=False)

        # ===== TRACK ANALYSIS (NEU V4.1!) =====
        self.analysis_xml_path = tk.StringVar(value="")
        self.analysis_mode = tk.StringVar(value="single")  # "single" or "batch"
        self.analysis_rf_model = tk.StringVar(value="")
        self.analysis_frame_rate = tk.DoubleVar(value=20.0)
        self.analysis_output_dir = tk.StringVar(value="")
        self.analysis_recursive = tk.BooleanVar(value=True)
        self.analysis_adaptive_training = tk.BooleanVar(value=True)  # NEU: Adaptive RF Training
        self.analysis_n_tracks_training = tk.IntVar(value=200)  # Anzahl Tracks für Training
        self.analysis_preview_text = tk.StringVar(value="Keine XML geladen...")
        self.analysis_status = tk.StringVar(value="Bereit")
        self.batch_use_spot_range = tk.BooleanVar(value=True)
        self.batch_subfolder_per_repeat = tk.BooleanVar(value=True)
        # Legacy (falls benötigt)
        self.batch_preset = tk.StringVar(value="quick")
        self.batch_detector = tk.StringVar(value="TDI-G0")
        self.batch_custom_times = tk.StringVar(value="")

        # ===== RANDOM-FOREST TRAINING =====
        self.batch_train_rf = tk.BooleanVar(value=False)
        self.batch_rf_window = tk.IntVar(value=48)
        self.batch_rf_step = tk.IntVar(value=16)
        self.batch_rf_estimators = tk.IntVar(value=1024)
        self.batch_rf_max_depth = tk.IntVar(value=28)
        self.batch_rf_min_leaf = tk.IntVar(value=3)
        self.batch_rf_min_split = tk.IntVar(value=6)
        self.batch_rf_max_samples = tk.DoubleVar(value=0.85)
        self.batch_rf_max_windows_per_class = tk.IntVar(value=100_000)
        self.batch_rf_max_windows_per_track = tk.IntVar(value=600)

        # ===== EXPORT =====
        self.export_metadata = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_txt = tk.BooleanVar(value=True)
        self.export_csv = tk.BooleanVar(value=True)

    def _create_widgets(self):
        """Erstellt alle GUI-Elemente."""

        # ====================================================================
        # HEADER
        # ====================================================================
        header_frame = tk.Frame(self.root, bg="#1a1a2e", height=90)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        tk.Label(
            header_frame,
            text="🔬 Hyperrealistischer TIFF Simulator V4.0",
            font=("Arial", 20, "bold"),
            bg="#1a1a2e",
            fg="white"
        ).pack(pady=5)

        tk.Label(
            header_frame,
            text="⚡ ADVANCED EDITION - Optimiert für maximale Performance & Flexibilität",
            font=("Arial", 11),
            bg="#1a1a2e",
            fg="#16c79a"
        ).pack()

        tk.Label(
            header_frame,
            text="✨ Mit erweiterten Photophysik-Parametern, Live-Preview & Batch-Processing",
            font=("Arial", 9),
            bg="#1a1a2e",
            fg="#a8dadc"
        ).pack(pady=2)

        # ====================================================================
        # SCROLLBARER HAUPTBEREICH
        # ====================================================================
        self.scrollable_container = ScrollableFrame(self.root)
        self.scrollable_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame = self.scrollable_container.scrollable_frame

        # ====================================================================
        # DETEKTOR PRESET
        # ====================================================================
        detector_frame = ttk.LabelFrame(main_frame, text="📷 Detektor Konfiguration", padding=10)
        detector_frame.pack(fill=tk.X, padx=5, pady=5)

        btn_frame = tk.Frame(detector_frame)
        btn_frame.pack()

        tdi_btn = ttk.Radiobutton(
            btn_frame,
            text="🔵 TDI-G0 (0.108 µm/px)",
            variable=self.detector_var,
            value="TDI-G0",
            command=self._apply_detector_preset
        )
        tdi_btn.pack(side=tk.LEFT, padx=10)
        ToolTip(tdi_btn, "TDI Line Scan Camera\n260 counts max, QE=0.85")

        tetra_btn = ttk.Radiobutton(
            btn_frame,
            text="🟢 Tetraspecs (0.160 µm/px)",
            variable=self.detector_var,
            value="Tetraspecs",
            command=self._apply_detector_preset
        )
        tetra_btn.pack(side=tk.LEFT, padx=10)
        ToolTip(tetra_btn, "sCMOS Camera\n300 counts max, QE=0.90")

        # ====================================================================
        # NOTEBOOK für Parameter
        # ====================================================================
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Basis-Parameter
        self.basic_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_tab, text="📊 Basis-Parameter")
        self._create_basic_tab()

        # Tab 2: Erweiterte Physik
        self.physics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.physics_tab, text="⚛️ Erweiterte Physik")
        self._create_physics_tab()

        # Tab 3: Photophysik & Noise
        self.photophysics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.photophysics_tab, text="💡 Photophysik & Noise")
        self._create_photophysics_tab()

        # Tab 4: 3D & Astigmatismus
        self.astig_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.astig_tab, text="📐 3D & Astigmatismus")
        self._create_astigmatism_tab()

        # Tab 5: Batch Simulation
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="📦 Batch-Modus")
        self._create_batch_tab()

        # Tab 6: Export & Metadata
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="💾 Export")
        self._create_export_tab()

        # Tab 7: Track Analysis (NEU V4.1!)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="🔬 Track Analysis")
        self._create_analysis_tab()

        # ====================================================================
        # PROGRESS BAR & STATUS
        # ====================================================================
        progress_frame = tk.Frame(self.root, bg='#f5f5f5')
        progress_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress.pack(fill=tk.X, pady=5)

        # Status Label
        self.status_label = tk.Label(
            progress_frame,
            text="⚙️ Bereit - V4.0 Optimized Engine aktiv",
            font=("Arial", 10),
            fg="#27ae60",
            bg="#ecf0f1",
            relief=tk.SUNKEN,
            height=2
        )
        self.status_label.pack(fill=tk.X, pady=5)

        # Buttons
        button_frame = tk.Frame(progress_frame, bg='#f5f5f5')
        button_frame.pack(pady=5)

        self.start_button = tk.Button(
            button_frame,
            text="🚀 SIMULATION STARTEN",
            font=("Arial", 12, "bold"),
            bg="#16c79a",
            fg="white",
            activebackground="#11b08a",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            width=25,
            height=2,
            command=self._start_simulation
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            button_frame,
            text="❌ Beenden",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=10)

    def _create_basic_tab(self):
        """Tab für Basis-Parameter."""

        # Simulationsmodus
        mode_frame = ttk.LabelFrame(self.basic_tab, text="Simulationsmodus", padding=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Radiobutton(
            mode_frame,
            text="⏱️ Polymerisationszeit (2D)",
            variable=self.sim_mode_var,
            value="polyzeit",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="⏱️📺 Polymerisationszeit + Astigmatismus (3D)",
            variable=self.sim_mode_var,
            value="polyzeit_astig",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="📊 z-Stack Kalibrierung",
            variable=self.sim_mode_var,
            value="z_stack",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)

        # Info-Text
        self.mode_info_text = scrolledtext.ScrolledText(
            mode_frame,
            height=4,
            width=80,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#e8f4f8",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.mode_info_text.pack(pady=5, fill=tk.X)

        # Bild-Parameter
        img_frame = ttk.LabelFrame(self.basic_tab, text="🖼️ Bild-Parameter", padding=10)
        img_frame.pack(fill=tk.X, padx=10, pady=5)

        # Größe
        size_frame = tk.Frame(img_frame)
        size_frame.pack(fill=tk.X, pady=2)
        tk.Label(size_frame, text="Breite [px]:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        width_spin = ttk.Spinbox(size_frame, from_=32, to=1024, increment=32,
                   textvariable=self.image_width, width=10)
        width_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(width_spin, "Bildbreite in Pixeln\nGrößere Bilder = länger")

        tk.Label(size_frame, text="Höhe [px]:", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=(20,0))
        height_spin = ttk.Spinbox(size_frame, from_=32, to=1024, increment=32,
                   textvariable=self.image_height, width=10)
        height_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(height_spin, "Bildhöhe in Pixeln")

        # Spots
        spots_frame = tk.Frame(img_frame)
        spots_frame.pack(fill=tk.X, pady=2)
        tk.Label(spots_frame, text="Anzahl Spots:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        spots_spin = ttk.Spinbox(spots_frame, from_=1, to=200, increment=1,
                   textvariable=self.num_spots, width=10)
        spots_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(spots_spin, "Anzahl simulierter Fluorophore\nMehr Spots = realistischer aber langsamer")

        # Zeitreihen-Parameter
        time_frame = ttk.LabelFrame(self.basic_tab, text="⏱️ Zeitreihen-Parameter", padding=10)
        time_frame.pack(fill=tk.X, padx=10, pady=5)

        # Polyzeit
        t_frame = tk.Frame(time_frame)
        t_frame.pack(fill=tk.X, pady=2)
        tk.Label(t_frame, text="Polyzeit [min]:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        poly_spin = ttk.Spinbox(t_frame, from_=0, to=240, increment=10,
                   textvariable=self.t_poly, width=10,
                   command=self._update_d_estimate)
        poly_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(poly_spin, "Polymerisationszeit in Minuten\nBestimmt Gel-Vernetzung & D-Wert")
        self.d_info_label = tk.Label(t_frame, text="", font=("Arial", 9), fg="#27ae60")
        self.d_info_label.pack(side=tk.LEFT, padx=10)

        # Frames
        frames_frame = tk.Frame(time_frame)
        frames_frame.pack(fill=tk.X, pady=2)
        tk.Label(frames_frame, text="Anzahl Frames:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        frames_spin = ttk.Spinbox(frames_frame, from_=10, to=5000, increment=10,
                   textvariable=self.num_frames, width=10)
        frames_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(frames_spin, "Anzahl Zeitpunkte\nV4.0 optimiert für große Werte!")

        # Frame Rate
        rate_frame = tk.Frame(time_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        tk.Label(rate_frame, text="Frame Rate [Hz]:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        rate_spin = ttk.Spinbox(rate_frame, from_=1, to=100, increment=1,
                   textvariable=self.frame_rate, width=10)
        rate_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(rate_spin, "Aufnahmefrequenz in Hz\nBestimmt Zeitauflösung")

        # D_initial
        d_frame = tk.Frame(time_frame)
        d_frame.pack(fill=tk.X, pady=2)
        tk.Label(d_frame, text="D_initial [µm²/s]:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        d_spin = ttk.Spinbox(d_frame, from_=0.01, to=2.0, increment=0.01,
                   textvariable=self.d_initial, width=10,
                   format='%.3f', command=self._update_d_estimate)
        d_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(d_spin, "Initialer Diffusionskoeffizient (t=0 min)\nRealistische Werte: 0.15-0.30 µm²/s\nLiteratur: D₀ ≈ 0.24 µm²/s (2.4e-13 m²/s)")

        # Exposure substeps
        sub_frame = tk.Frame(time_frame)
        sub_frame.pack(fill=tk.X, pady=2)
        tk.Label(sub_frame, text="Exposure Substeps:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        sub_spin = ttk.Spinbox(sub_frame, from_=1, to=10, increment=1,
                   textvariable=self.exposure_substeps, width=10)
        sub_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(sub_spin, "Motion Blur Substeps\n3-5 = realistisch, 1 = schnell")

        # Output
        output_frame = ttk.LabelFrame(self.basic_tab, text="💾 Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        # Ordner
        dir_frame = tk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        tk.Label(dir_frame, text="Speicherort:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=self.output_dir, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="📁", width=3, command=self._browse_dir).pack(side=tk.LEFT)

        # Dateiname
        file_frame = tk.Frame(output_frame)
        file_frame.pack(fill=tk.X, pady=2)
        tk.Label(file_frame, text="Dateiname:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(file_frame, textvariable=self.filename, width=50).pack(side=tk.LEFT, padx=5)

        # Initial Updates
        self._update_d_estimate()

    def _create_physics_tab(self):
        """Tab für erweiterte Physik-Parameter."""

        tk.Label(
            self.physics_tab,
            text="⚛️ Erweiterte Physikalische Parameter",
            font=("Arial", 12, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        # PSF-Parameter
        psf_frame = ttk.LabelFrame(self.physics_tab, text="🔬 PSF (Point Spread Function)", padding=10)
        psf_frame.pack(fill=tk.X, padx=10, pady=5)

        # Max Intensity
        int_frame = tk.Frame(psf_frame)
        int_frame.pack(fill=tk.X, pady=2)
        tk.Label(int_frame, text="Max Intensity [counts]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        int_spin = ttk.Spinbox(int_frame, from_=50, to=1000, increment=10,
                   textvariable=self.max_intensity, width=10)
        int_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(int_spin, "Maximale Photonenzahl pro Spot\nTDI-G0: ~260, Tetraspecs: ~300")

        # Spot Intensity Sigma
        spot_sig_frame = tk.Frame(psf_frame)
        spot_sig_frame.pack(fill=tk.X, pady=2)
        tk.Label(spot_sig_frame, text="Spot Intensity Sigma:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spot_sig_spin = ttk.Spinbox(spot_sig_frame, from_=0.0, to=1.0, increment=0.05,
                   textvariable=self.spot_intensity_sigma, width=10, format='%.3f')
        spot_sig_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(spot_sig_spin, "Lognormale Variabilität der Spot-Helligkeit\n0.25 = realistisch")

        # Frame Jitter
        jitter_frame = tk.Frame(psf_frame)
        jitter_frame.pack(fill=tk.X, pady=2)
        tk.Label(jitter_frame, text="Frame Jitter Sigma:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        jitter_spin = ttk.Spinbox(jitter_frame, from_=0.0, to=0.5, increment=0.01,
                   textvariable=self.frame_jitter_sigma, width=10, format='%.3f')
        jitter_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(jitter_spin, "Frame-zu-Frame Intensitätsschwankung\n0.10 = realistisch")

        # Background
        bg_frame = ttk.LabelFrame(self.physics_tab, text="🌫️ Background & Noise", padding=10)
        bg_frame.pack(fill=tk.X, padx=10, pady=5)

        # Background Mean
        bg_mean_frame = tk.Frame(bg_frame)
        bg_mean_frame.pack(fill=tk.X, pady=2)
        tk.Label(bg_mean_frame, text="Background Mean [counts]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        bg_mean_spin = ttk.Spinbox(bg_mean_frame, from_=0, to=500, increment=10,
                   textvariable=self.background_mean, width=10)
        bg_mean_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(bg_mean_spin, "Mittlerer Background-Level\n100 = typisch")

        # Background Std
        bg_std_frame = tk.Frame(bg_frame)
        bg_std_frame.pack(fill=tk.X, pady=2)
        tk.Label(bg_std_frame, text="Background Std [counts]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        bg_std_spin = ttk.Spinbox(bg_std_frame, from_=0, to=100, increment=1,
                   textvariable=self.background_std, width=10)
        bg_std_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(bg_std_spin, "Background-Rauschen\n15 = realistisch")

        # Read Noise
        read_frame = tk.Frame(bg_frame)
        read_frame.pack(fill=tk.X, pady=2)
        tk.Label(read_frame, text="Read Noise Std [counts]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        read_spin = ttk.Spinbox(read_frame, from_=0.0, to=10.0, increment=0.1,
                   textvariable=self.read_noise_std, width=10, format='%.1f')
        read_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(read_spin, "Kamera-Ausleserauschen\nTDI: 1.2, sCMOS: 1.8")

    def _create_photophysics_tab(self):
        """Tab für Photophysik (Blinking, Bleaching)."""

        tk.Label(
            self.photophysics_tab,
            text="💡 Photophysik: Blinking & Bleaching",
            font=("Arial", 12, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        # Enable Photophysik
        enable_frame = tk.Frame(self.photophysics_tab)
        enable_frame.pack(pady=10)

        enable_check = ttk.Checkbutton(
            enable_frame,
            text="✅ Photophysik aktivieren (Blinking & Bleaching)",
            variable=self.enable_photophysics
        )
        enable_check.pack()
        ToolTip(enable_check, "Aktiviert ON/OFF-Blinking und Photobleaching\nRealistischer aber komplexer")

        # Blinking-Parameter
        blink_frame = ttk.LabelFrame(self.photophysics_tab, text="💫 Blinking (ON/OFF)", padding=10)
        blink_frame.pack(fill=tk.X, padx=10, pady=5)

        # ON mean
        on_frame = tk.Frame(blink_frame)
        on_frame.pack(fill=tk.X, pady=2)
        tk.Label(on_frame, text="ON Mean Duration [frames]:", width=28, anchor=tk.W).pack(side=tk.LEFT)
        on_spin = ttk.Spinbox(on_frame, from_=1.0, to=20.0, increment=0.5,
                   textvariable=self.on_mean_frames, width=10, format='%.1f')
        on_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(on_spin, "Mittlere Dauer im ON-Zustand\n4 frames = typisch")

        # OFF mean
        off_frame = tk.Frame(blink_frame)
        off_frame.pack(fill=tk.X, pady=2)
        tk.Label(off_frame, text="OFF Mean Duration [frames]:", width=28, anchor=tk.W).pack(side=tk.LEFT)
        off_spin = ttk.Spinbox(off_frame, from_=1.0, to=20.0, increment=0.5,
                   textvariable=self.off_mean_frames, width=10, format='%.1f')
        off_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(off_spin, "Mittlere Dauer im OFF-Zustand\n6 frames = typisch")

        # Bleaching
        bleach_frame = ttk.LabelFrame(self.photophysics_tab, text="💥 Photobleaching", padding=10)
        bleach_frame.pack(fill=tk.X, padx=10, pady=5)

        # Bleach Probability
        bleach_prob_frame = tk.Frame(bleach_frame)
        bleach_prob_frame.pack(fill=tk.X, pady=2)
        tk.Label(bleach_prob_frame, text="Bleach Probability [per frame]:", width=28, anchor=tk.W).pack(side=tk.LEFT)
        bleach_spin = ttk.Spinbox(bleach_prob_frame, from_=0.0, to=0.05, increment=0.0001,
                   textvariable=self.bleach_prob, width=10, format='%.4f')
        bleach_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(bleach_spin, "Wahrscheinlichkeit für irreversibles Bleaching\n0.002 = 0.2% pro Frame")

        # Info
        info_text = scrolledtext.ScrolledText(
            self.photophysics_tab,
            height=6,
            width=80,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#fff3cd",
            relief=tk.FLAT
        )
        info_text.pack(padx=10, pady=10, fill=tk.X)
        info_text.insert(1.0,
            "PHOTOPHYSIK-MODELL:\n\n"
            "• Blinking: 2-Zustands-Modell (ON/OFF) mit geometrischen Dauern\n"
            "• Bleaching: Irreversibles Photobleaching während ON-Zustand\n"
            "• Physikalisch korrekt basierend auf Single-Molecule-Experimenten\n\n"
            "EMPFEHLUNG: Für maximale Realität aktivieren!"
        )
        info_text.config(state=tk.DISABLED)

    def _create_astigmatism_tab(self):
        """Tab für 3D & Astigmatismus."""

        tk.Label(
            self.astig_tab,
            text="📐 3D-Lokalisierung & Astigmatismus",
            font=("Arial", 12, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        # z-Bereich
        z_range_frame = ttk.LabelFrame(self.astig_tab, text="📏 z-Bereich", padding=10)
        z_range_frame.pack(fill=tk.X, padx=10, pady=5)

        # z_amp
        zamp_frame = tk.Frame(z_range_frame)
        zamp_frame.pack(fill=tk.X, pady=2)
        tk.Label(zamp_frame, text="z Amplitude [µm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        zamp_spin = ttk.Spinbox(zamp_frame, from_=0.1, to=2.0, increment=0.1,
                   textvariable=self.z_amp_um, width=10, format='%.2f')
        zamp_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(zamp_spin, "Intensitätsabfall-Skala in z\n0.7 µm = realistisch")

        # z_max
        zmax_frame = tk.Frame(z_range_frame)
        zmax_frame.pack(fill=tk.X, pady=2)
        tk.Label(zmax_frame, text="z Max [µm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        zmax_spin = ttk.Spinbox(zmax_frame, from_=0.1, to=2.0, increment=0.1,
                   textvariable=self.z_max_um, width=10, format='%.2f')
        zmax_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(zmax_spin, "Maximale z-Position\n0.6 µm = typisch")

        # Astigmatismus-Koeffizienten
        astig_coef_frame = ttk.LabelFrame(self.astig_tab, text="🔍 Astigmatismus-Koeffizienten", padding=10)
        astig_coef_frame.pack(fill=tk.X, padx=10, pady=5)

        # z0
        z0_frame = tk.Frame(astig_coef_frame)
        z0_frame.pack(fill=tk.X, pady=2)
        tk.Label(z0_frame, text="z0 [µm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        z0_spin = ttk.Spinbox(z0_frame, from_=0.1, to=2.0, increment=0.1,
                   textvariable=self.astig_z0_um, width=10, format='%.2f')
        z0_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(z0_spin, "Charakteristische z-Skala\n0.5 µm = Standard")

        # Ax
        ax_frame = tk.Frame(astig_coef_frame)
        ax_frame.pack(fill=tk.X, pady=2)
        tk.Label(ax_frame, text="A_x (x-Koeffizient):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ax_spin = ttk.Spinbox(ax_frame, from_=-2.0, to=2.0, increment=0.1,
                   textvariable=self.astig_Ax, width=10, format='%.2f')
        ax_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(ax_spin, "Astigmatismus x-Koeffizient\n+1.0 = Standard")

        # Ay
        ay_frame = tk.Frame(astig_coef_frame)
        ay_frame.pack(fill=tk.X, pady=2)
        tk.Label(ay_frame, text="A_y (y-Koeffizient):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ay_spin = ttk.Spinbox(ay_frame, from_=-2.0, to=2.0, increment=0.1,
                   textvariable=self.astig_Ay, width=10, format='%.2f')
        ay_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(ay_spin, "Astigmatismus y-Koeffizient\n-0.5 = Standard")

        # z-Stack Parameter
        zstack_frame = ttk.LabelFrame(self.astig_tab, text="📊 z-Stack Kalibrierung", padding=10)
        zstack_frame.pack(fill=tk.X, padx=10, pady=5)

        # z_min
        zmin_frame = tk.Frame(zstack_frame)
        zmin_frame.pack(fill=tk.X, pady=2)
        tk.Label(zmin_frame, text="z_min [µm]:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        zmin_spin = ttk.Spinbox(zmin_frame, from_=-2.0, to=0.0, increment=0.1,
                   textvariable=self.z_min, width=10, format='%.2f',
                   command=self._update_z_slices)
        zmin_spin.pack(side=tk.LEFT, padx=5)

        # z_max
        zmax_stack_frame = tk.Frame(zstack_frame)
        zmax_stack_frame.pack(fill=tk.X, pady=2)
        tk.Label(zmax_stack_frame, text="z_max [µm]:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        zmax_stack_spin = ttk.Spinbox(zmax_stack_frame, from_=0.0, to=2.0, increment=0.1,
                   textvariable=self.z_max, width=10, format='%.2f',
                   command=self._update_z_slices)
        zmax_stack_spin.pack(side=tk.LEFT, padx=5)

        # z_step
        zstep_frame = tk.Frame(zstack_frame)
        zstep_frame.pack(fill=tk.X, pady=2)
        tk.Label(zstep_frame, text="z_step [µm]:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        zstep_spin = ttk.Spinbox(zstep_frame, from_=0.01, to=0.5, increment=0.01,
                   textvariable=self.z_step, width=10, format='%.3f',
                   command=self._update_z_slices)
        zstep_spin.pack(side=tk.LEFT, padx=5)
        self.z_slices_label = tk.Label(zstep_frame, text="", font=("Arial", 9), fg="#27ae60")
        self.z_slices_label.pack(side=tk.LEFT, padx=10)

        self._update_z_slices()

    def _create_batch_tab(self):
        """KOMPLETT ÜBERARBEITETER Batch-Tab mit voller Funktionalität!"""

        tk.Label(
            self.batch_tab,
            text="📦 Batch-Modus: Automatisierte Serien",
            font=("Arial", 14, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        # Enable Checkbox
        enable_frame = tk.Frame(self.batch_tab)
        enable_frame.pack(pady=5)

        ttk.Checkbutton(
            enable_frame,
            text="✅ Batch-Modus aktivieren (Multiple TIFFs generieren)",
            variable=self.batch_mode_enabled,
            command=self._toggle_batch_mode
        ).pack()

        # ====================================================================
        # POLYMERISATIONSZEITEN
        # ====================================================================
        time_frame = ttk.LabelFrame(self.batch_tab, text="⏱️ Polymerisationszeiten", padding=10)
        time_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(
            time_frame,
            text="Geben Sie beliebig viele Zeiten ein (Komma-separiert):",
            font=("Arial", 10)
        ).pack(anchor=tk.W, pady=2)

        times_entry = tk.Entry(time_frame, textvariable=self.batch_poly_times, width=60)
        times_entry.pack(fill=tk.X, pady=5)

        tk.Label(
            time_frame,
            text="Beispiel: 0, 10, 30, 60, 90, 120, 180",
            font=("Arial", 9),
            fg="#7f8c8d"
        ).pack(anchor=tk.W)

        # Quick Presets
        preset_frame = tk.Frame(time_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(preset_frame, text="Quick Presets:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        ttk.Button(
            preset_frame,
            text="Schnell (3)",
            command=lambda: self.batch_poly_times.set("30, 60, 90")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            preset_frame,
            text="Standard (5)",
            command=lambda: self.batch_poly_times.set("0, 30, 60, 90, 120")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            preset_frame,
            text="Vollständig (7)",
            command=lambda: self.batch_poly_times.set("0, 10, 30, 60, 90, 120, 180")
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            preset_frame,
            text="Dicht (12)",
            command=lambda: self.batch_poly_times.set("0, 5, 10, 15, 30, 45, 60, 75, 90, 105, 120, 180")
        ).pack(side=tk.LEFT, padx=2)

        # ====================================================================
        # SPOT-RANGE
        # ====================================================================
        spot_frame = ttk.LabelFrame(self.batch_tab, text="🎯 Spot-Anzahl", padding=10)
        spot_frame.pack(fill=tk.X, padx=10, pady=5)

        spot_check = ttk.Checkbutton(
            spot_frame,
            text="✨ Randomisierte Spot-Anzahl aktivieren (realistische Variation)",
            variable=self.batch_use_spot_range
        )
        spot_check.pack(anchor=tk.W, pady=2)

        range_frame = tk.Frame(spot_frame)
        range_frame.pack(fill=tk.X, pady=5)

        tk.Label(range_frame, text="Min:", width=10, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(range_frame, from_=1, to=100, increment=1,
                   textvariable=self.num_spots_min, width=10).pack(side=tk.LEFT, padx=5)

        tk.Label(range_frame, text="Max:", width=10, anchor=tk.W).pack(side=tk.LEFT, padx=(20,0))
        ttk.Spinbox(range_frame, from_=1, to=100, increment=1,
                   textvariable=self.num_spots_max, width=10).pack(side=tk.LEFT, padx=5)

        tk.Label(
            spot_frame,
            text="💡 Jedes TIFF bekommt zufällige Spot-Anzahl in diesem Bereich",
            font=("Arial", 9),
            fg="#27ae60"
        ).pack(anchor=tk.W, pady=2)

        # ====================================================================
        # WIEDERHOLUNGEN & ORDNERSTRUKTUR
        # ====================================================================
        repeat_frame = ttk.LabelFrame(self.batch_tab, text="🔄 Wiederholungen", padding=10)
        repeat_frame.pack(fill=tk.X, padx=10, pady=5)

        rep_row = tk.Frame(repeat_frame)
        rep_row.pack(fill=tk.X, pady=2)
        tk.Label(rep_row, text="Wiederholungen:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(rep_row, from_=1, to=20, increment=1,
                   textvariable=self.batch_repeats, width=10).pack(side=tk.LEFT, padx=5)

        folder_check = ttk.Checkbutton(
            repeat_frame,
            text="📁 Jede Wiederholung in eigenem Unterordner (repeat_1, repeat_2, ...)",
            variable=self.batch_subfolder_per_repeat
        )
        folder_check.pack(anchor=tk.W, pady=5)

        # ====================================================================
        # ASTIGMATISMUS
        # ====================================================================
        astig_frame = ttk.LabelFrame(self.batch_tab, text="📐 3D-Modus", padding=10)
        astig_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(
            astig_frame,
            text="🔺 Astigmatismus aktivieren (3D z-Lokalisierung)",
            variable=self.batch_astig
        ).pack(anchor=tk.W, pady=2)

        tk.Label(
            astig_frame,
            text="Wenn aktiviert: Alle TIFFs mit astigmatischer PSF für 3D-Tracking",
            font=("Arial", 9),
            fg="#7f8c8d"
        ).pack(anchor=tk.W)

        # ====================================================================
        # RANDOM-FOREST TRAINING
        # ====================================================================
        rf_frame = ttk.LabelFrame(self.batch_tab, text="🌲 Random-Forest KI-Training", padding=10)
        rf_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(
            rf_frame,
            text="🌲 Parallel ein Premium-Random-Forest trainieren",
            variable=self.batch_train_rf,
            command=self._toggle_rf_options
        ).pack(anchor=tk.W, pady=2)

        self.rf_option_widgets = []

        rf_row1 = tk.Frame(rf_frame)
        rf_row1.pack(fill=tk.X, pady=2)
        tk.Label(rf_row1, text="Fenstergröße (Frames):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spin_window = ttk.Spinbox(rf_row1, from_=10, to=600, increment=2,
                                  textvariable=self.batch_rf_window, width=8)
        spin_window.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_window)

        tk.Label(rf_row1, text="Schrittweite:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        spin_step = ttk.Spinbox(rf_row1, from_=1, to=300, increment=1,
                                textvariable=self.batch_rf_step, width=8)
        spin_step.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_step)

        rf_row2 = tk.Frame(rf_frame)
        rf_row2.pack(fill=tk.X, pady=2)
        tk.Label(rf_row2, text="Bäume (n_estimators):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spin_estimators = ttk.Spinbox(rf_row2, from_=256, to=4096, increment=64,
                                      textvariable=self.batch_rf_estimators, width=8)
        spin_estimators.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_estimators)

        tk.Label(rf_row2, text="Max. Tiefe (0 = ∞):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        spin_depth = ttk.Spinbox(rf_row2, from_=0, to=80, increment=1,
                                 textvariable=self.batch_rf_max_depth, width=8)
        spin_depth.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_depth)

        rf_row3 = tk.Frame(rf_frame)
        rf_row3.pack(fill=tk.X, pady=2)
        tk.Label(rf_row3, text="Min. Samples/Leaf:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spin_leaf = ttk.Spinbox(rf_row3, from_=1, to=20, increment=1,
                                textvariable=self.batch_rf_min_leaf, width=8)
        spin_leaf.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_leaf)

        tk.Label(rf_row3, text="Min. Samples/Split:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        spin_split = ttk.Spinbox(rf_row3, from_=2, to=40, increment=1,
                                 textvariable=self.batch_rf_min_split, width=8)
        spin_split.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_split)

        rf_row4 = tk.Frame(rf_frame)
        rf_row4.pack(fill=tk.X, pady=2)
        tk.Label(rf_row4, text="Baum-Subsampling (max_samples):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spin_max_samples = ttk.Spinbox(
            rf_row4,
            from_=0.1,
            to=1.0,
            increment=0.05,
            format="%.2f",
            textvariable=self.batch_rf_max_samples,
            width=8,
        )
        spin_max_samples.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_max_samples)

        tk.Label(rf_row4, text="Fenster/Klasse (0 = ∞):", width=18, anchor=tk.W).pack(side=tk.LEFT)
        spin_class_cap = ttk.Spinbox(
            rf_row4,
            from_=0,
            to=500000,
            increment=5000,
            textvariable=self.batch_rf_max_windows_per_class,
            width=10,
        )
        spin_class_cap.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_class_cap)

        rf_row5 = tk.Frame(rf_frame)
        rf_row5.pack(fill=tk.X, pady=2)
        tk.Label(rf_row5, text="Fenster/Track (0 = ∞):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        spin_track_cap = ttk.Spinbox(
            rf_row5,
            from_=0,
            to=5000,
            increment=50,
            textvariable=self.batch_rf_max_windows_per_track,
            width=8,
        )
        spin_track_cap.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_track_cap)

        tk.Label(rf_row5, text="Mehrfachläufe nutzen denselben Wald und sammeln neue Fenster.",
                 anchor=tk.W, justify=tk.LEFT).pack(side=tk.LEFT, padx=5)

        tk.Label(
            rf_frame,
            text="💾 Modell & Feature-Tabelle landen automatisch im Batch-Output-Verzeichnis",
            font=("Arial", 9),
            fg="#16a085"
        ).pack(anchor=tk.W, pady=3)

        self._toggle_rf_options()

        # ====================================================================
        # ZUSAMMENFASSUNG
        # ====================================================================
        summary_frame = ttk.LabelFrame(self.batch_tab, text="📊 Batch-Zusammenfassung", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)

        self.batch_summary_label = tk.Label(
            summary_frame,
            text="",
            font=("Arial", 10),
            fg="#2c3e50",
            justify=tk.LEFT
        )
        self.batch_summary_label.pack(anchor=tk.W, pady=5)

        ttk.Button(
            summary_frame,
            text="🔄 Zusammenfassung aktualisieren",
            command=self._update_batch_summary
        ).pack(pady=5)

        # Info Box
        info_box = tk.Frame(self.batch_tab, bg="#e8f4f8", relief=tk.SOLID, bd=1)
        info_box.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            info_box,
            text="💡 BATCH-MODUS FEATURES:\n\n"
                 "✅ Unbegrenzt viele Polymerisationszeiten\n"
                 "✅ Randomisierte Spot-Anzahl für Realismus\n"
                 "✅ Automatische Unterordner pro Wiederholung\n"
                 "✅ Automatische Dateinamen (zeit, spots, wiederholung)\n"
                 "✅ Echtzeit Progress-Tracking\n"
                 "✅ Optional: Random-Forest-KI für Diffusionsklassifikation\n"
                 "✅ Vollständige Metadata für jedes TIFF\n\n"
                 "Klicken Sie 'SIMULATION STARTEN' um den Batch zu beginnen!",
            font=("Arial", 9),
            bg="#e8f4f8",
            fg="#2c3e50",
            justify=tk.LEFT,
            padx=10,
            pady=10
        ).pack(fill=tk.X)

        # Initial Summary
        self._update_batch_summary()

    def _toggle_batch_mode(self):
        """Toggelt zwischen Single und Batch Mode."""
        if self.batch_mode_enabled.get():
            self._update_status("📦 Batch-Modus aktiviert", "#3498db")
            self.notebook.select(4)  # Switch to Batch tab
        else:
            self._update_status("📄 Single-Modus aktiviert", "#27ae60")

    def _update_batch_summary(self):
        """Aktualisiert die Batch-Zusammenfassung."""
        try:
            # Parse Zeiten
            times_str = self.batch_poly_times.get().strip()
            times = [float(t.strip()) for t in times_str.split(',') if t.strip()]

            repeats = self.batch_repeats.get()

            # Spot-Range
            if self.batch_use_spot_range.get():
                spots_info = f"{self.num_spots_min.get()}-{self.num_spots_max.get()} (randomisiert)"
            else:
                spots_info = f"{self.num_spots.get()} (fix)"

            # Astigmatismus
            astig_info = "JA (3D)" if self.batch_astig.get() else "NEIN (2D)"

            # Gesamtzahl TIFFs
            total_tiffs = len(times) * repeats

            # Geschätzte Zeit (sehr grob)
            frames = self.num_frames.get()
            img_size = self.image_width.get()
            time_per_tiff = (frames * img_size) / 20000  # Sekunden (grobe Schätzung)
            total_time_min = (time_per_tiff * total_tiffs) / 60

            summary = (
                f"📊 BATCH-KONFIGURATION:\n\n"
                f"⏱️  Polyzeiten: {len(times)} Zeiten ({', '.join(str(int(t)) for t in times[:5])}{'...' if len(times) > 5 else ''} min)\n"
                f"🔄 Wiederholungen: {repeats}x\n"
                f"🎯 Spots: {spots_info}\n"
                f"📐 Astigmatismus: {astig_info}\n"
                f"📁 Unterordner: {'JA' if self.batch_subfolder_per_repeat.get() else 'NEIN'}\n"
                f"🌲 Random Forest: {'AKTIV' if self.batch_train_rf.get() else 'aus'}\n\n"
                f"═══════════════════\n"
                f"📦 Gesamt TIFFs: {total_tiffs}\n"
                f"⏱️  Geschätzte Zeit: ~{total_time_min:.0f} Minuten\n"
                f"💾 Speicherbedarf: ~{total_tiffs * 40} MB"
            )

            self.batch_summary_label.config(text=summary)

        except Exception as e:
            self.batch_summary_label.config(
                text=f"❌ Fehler beim Parsen: {str(e)}\nBitte Eingabe prüfen!"
            )

    def _toggle_rf_options(self):
        """Aktiviert oder deaktiviert RF-Parameterfelder."""
        state = tk.NORMAL if self.batch_train_rf.get() else tk.DISABLED
        for widget in getattr(self, 'rf_option_widgets', []):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _get_batch_rf_kwargs(self) -> dict:
        """Erstellt kwargs für Random-Forest-Training."""
        if not self.batch_train_rf.get():
            return {}

        max_depth = self.batch_rf_max_depth.get()
        if max_depth <= 0:
            max_depth = None

        max_samples = self.batch_rf_max_samples.get()
        if max_samples <= 0:
            max_samples = None
        else:
            max_samples = min(1.0, float(max_samples))

        return {
            'enable_rf': True,
            'rf_config': {
                'window_size': max(5, self.batch_rf_window.get()),
                'step_size': max(1, self.batch_rf_step.get()),
                'n_estimators': max(100, self.batch_rf_estimators.get()),
                'max_depth': max_depth,
                'min_samples_leaf': max(1, self.batch_rf_min_leaf.get()),
                'min_samples_split': max(2, self.batch_rf_min_split.get()),
                'random_state': 42,
                'max_samples': max_samples,
                'max_windows_per_class': max(0, self.batch_rf_max_windows_per_class.get()),
                'max_windows_per_track': max(0, self.batch_rf_max_windows_per_track.get()),
            }
        }

    def _create_export_tab(self):
        """Tab für Export-Optionen."""

        tk.Label(
            self.export_tab,
            text="💾 Export & Metadata-Formate",
            font=("Arial", 12, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        export_frame = ttk.LabelFrame(self.export_tab, text="📋 Metadata Export", padding=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(
            export_frame,
            text="📄 JSON (maschinenlesbar, vollständig)",
            variable=self.export_json
        ).pack(anchor=tk.W, pady=2)

        ttk.Checkbutton(
            export_frame,
            text="📝 TXT (menschenlesbar, Zusammenfassung)",
            variable=self.export_txt
        ).pack(anchor=tk.W, pady=2)

        ttk.Checkbutton(
            export_frame,
            text="📊 CSV (tabellarisch, für Batch-Analysen)",
            variable=self.export_csv
        ).pack(anchor=tk.W, pady=2)

    def _apply_detector_preset(self):
        """Wendet Detektor-Preset an."""
        detector = self.detector_var.get()

        if detector == "TDI-G0":
            self.max_intensity.set(260.0)
            self.background_mean.set(100.0)
            self.background_std.set(15.0)
            self.read_noise_std.set(1.2)
            self._update_status("📷 TDI-G0 Preset geladen")
        else:
            self.max_intensity.set(300.0)
            self.background_mean.set(100.0)
            self.background_std.set(15.0)
            self.read_noise_std.set(1.8)
            self._update_status("📷 Tetraspecs Preset geladen")

    def _update_mode_info(self):
        """Aktualisiert Mode-Info-Text."""
        mode = self.sim_mode_var.get()

        info_texts = {
            "polyzeit": (
                "⏱️ POLYMERISATIONSZEIT (2D)\n\n"
                "Simuliert Brownsche Bewegung bei verschiedenen Polymerisationszeiten.\n"
                "Spots bewegen sich in 2D ohne z-Information.\n\n"
                "Anwendung: MSD-Analyse, Zeitabhängigkeit von D(t)"
            ),
            "polyzeit_astig": (
                "⏱️📺 POLYMERISATIONSZEIT MIT ASTIGMATISMUS (3D)\n\n"
                "Simuliert 3D-Diffusion mit astigmatischer PSF.\n"
                "Spots werden elliptisch je nach z-Position.\n\n"
                "Anwendung: 3D-Lokalisierung mit ThunderSTORM, etc."
            ),
            "z_stack": (
                "📊 Z-STACK KALIBRIERUNG\n\n"
                "Erstellt einen z-Stack mit statischen Spots für PSF-Kalibrierung.\n"
                "Spots bewegen sich NICHT, nur z variiert.\n\n"
                "Anwendung: Kalibrierung von 3D-Tracking-Software"
            )
        }

        text = info_texts.get(mode, "")

        self.mode_info_text.config(state=tk.NORMAL)
        self.mode_info_text.delete(1.0, tk.END)
        self.mode_info_text.insert(1.0, text)
        self.mode_info_text.config(state=tk.DISABLED)

    def _update_d_estimate(self):
        """Zeigt geschätztes D."""
        try:
            from tiff_simulator_v3 import get_time_dependent_D

            t = self.t_poly.get()
            d_init = self.d_initial.get()

            d_normal = get_time_dependent_D(t, d_init, "normal")
            d_sub = get_time_dependent_D(t, d_init, "subdiffusion")

            self.d_info_label.config(
                text=f"→ D_normal ≈ {d_normal:.3f}, D_sub ≈ {d_sub:.3f} µm²/s"
            )
        except:
            pass

    def _update_z_slices(self):
        """Berechnet z-Slices."""
        try:
            z_min = self.z_min.get()
            z_max = self.z_max.get()
            z_step = self.z_step.get()

            if z_step > 0 and z_max > z_min:
                n_slices = int((z_max - z_min) / z_step) + 1
                self.z_slices_label.config(text=f"→ {n_slices} Slices")
            else:
                self.z_slices_label.config(text="❌ Ungültig")
        except:
            pass

    def _browse_dir(self):
        """Ordner-Dialog."""
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)

    def _set_status_ui(self, message: str, color: str = "#27ae60"):
        """Status-Update (UI-Thread)."""
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def _update_status(self, message: str, color: str = "#27ae60"):
        """Aktualisiert Status (thread-safe)."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._set_status_ui(message, color)
        else:
            self.root.after(0, lambda: self._set_status_ui(message, color))

    def _set_progress_ui(self, value: int):
        """Progress-Update (UI-Thread)."""
        self.progress['value'] = value
        self.root.update()

    def _update_progress(self, value: int):
        """Aktualisiert Progress (thread-safe)."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._set_progress_ui(value)
        else:
            self.root.after(0, lambda: self._set_progress_ui(value))

    def _start_simulation(self):
        """Startet Simulation in separatem Thread."""
        if self.is_running:
            messagebox.showwarning("Warnung", "Simulation läuft bereits!")
            return

        # Validierung
        if not self._validate_parameters():
            return

        # Starte Thread
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)

        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.start()

    def _validate_parameters(self) -> bool:
        """Validiert Parameter."""
        errors = []

        if self.image_width.get() < 32 or self.image_height.get() < 32:
            errors.append("Bildgröße muss mindestens 32×32 sein!")

        if self.num_spots.get() < 1:
            errors.append("Mindestens 1 Spot erforderlich!")

        if not os.path.exists(self.output_dir.get()):
            errors.append(f"Output-Ordner existiert nicht!")

        if errors:
            messagebox.showerror("Validierungsfehler", "\n".join(errors))
            return False

        return True

    def _run_simulation(self):
        """Führt Simulation aus - Single ODER Batch!"""
        try:
            # Check: Single oder Batch?
            if self.batch_mode_enabled.get():
                self._run_batch_simulation_integrated()
            else:
                self._run_single_simulation_integrated()

            # Erfolg
            self.root.after(0, lambda: messagebox.showinfo(
                "Erfolg! 🎉",
                f"Simulation erfolgreich abgeschlossen!\n\n"
                f"Output: {self.output_dir.get()}"
            ))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Fehler",
                f"Simulation fehlgeschlagen:\n\n{str(e)}"
            ))

        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self._update_progress(0))
            self.root.after(0, lambda: self._update_status("✅ Fertig!"))

    def _run_single_simulation_integrated(self):
        """Führt Single Simulation aus."""
        # Erstelle Custom-Detektor mit aktuellen Parametern
        detector = self._create_custom_detector()

        self._update_status("🔬 Initialisiere V4.0 Optimized Engine...")
        self._update_progress(10)

        # Modus
        sim_mode = self.sim_mode_var.get()
        astigmatism = (sim_mode in ("polyzeit_astig", "z_stack"))

        # Simulator (OPTIMIERT)
        sim = TIFFSimulatorOptimized(
            detector=detector,
            mode=sim_mode,
            t_poly_min=self.t_poly.get(),
            astigmatism=astigmatism
        )

        # Progress Callback
        def progress_cb(current, total, status):
            progress = int(10 + (current / total) * 80)
            self._update_progress(progress)
            self._update_status(f"⚡ {status}")

        # Generiere
        if sim_mode == "z_stack":
            tiff_stack = sim.generate_z_stack(
                image_size=(self.image_height.get(), self.image_width.get()),
                num_spots=self.num_spots.get(),
                z_range_um=(self.z_min.get(), self.z_max.get()),
                z_step_um=self.z_step.get(),
                progress_callback=progress_cb
            )
        else:
            tiff_stack = sim.generate_tiff(
                image_size=(self.image_height.get(), self.image_width.get()),
                num_spots=self.num_spots.get(),
                num_frames=self.num_frames.get(),
                frame_rate_hz=self.frame_rate.get(),
                d_initial=self.d_initial.get(),
                exposure_substeps=self.exposure_substeps.get(),
                enable_photophysics=self.enable_photophysics.get(),
                progress_callback=progress_cb
            )

        # Speichern
        self._update_status("💾 Speichere TIFF...")
        self._update_progress(90)

        filepath = Path(self.output_dir.get()) / self.filename.get()
        save_tiff(str(filepath), tiff_stack)

        # Metadata
        if self.export_json.get() or self.export_txt.get() or self.export_csv.get():
            self._update_status("📋 Exportiere Metadata...")
            exporter = MetadataExporter(self.output_dir.get())
            metadata = sim.get_metadata()
            base_name = filepath.stem

            if self.export_json.get():
                exporter.export_json(metadata, base_name)
            if self.export_txt.get():
                exporter.export_txt(metadata, base_name)
            if self.export_csv.get():
                exporter.export_csv_row(metadata, base_name)

        self._update_progress(100)

    def _run_batch_simulation_integrated(self):
        """NEUE FUNKTION: Batch-Modus vollständig integriert!"""
        import re
        import numpy as np

        self._update_status("📦 Starte Batch-Modus...")
        self._update_progress(5)

        rf_kwargs = self._get_batch_rf_kwargs()
        rf_enabled = bool(rf_kwargs)
        rf_trainer = None
        rf_info = None

        # Parse Polyzeiten
        times_str = self.batch_poly_times.get().strip()
        poly_times = []
        for t in re.split(r'[,;\s]+', times_str):
            if t:
                try:
                    poly_times.append(float(t))
                except:
                    pass

        if not poly_times:
            raise ValueError("Keine gültigen Polymerisationszeiten eingegeben!")

        poly_times = sorted(poly_times)  # Sortieren

        # Parameter
        repeats = self.batch_repeats.get()
        use_spot_range = self.batch_use_spot_range.get()
        spot_min = self.num_spots_min.get()
        spot_max = self.num_spots_max.get()
        astigmatism = self.batch_astig.get()
        subfolder_per_repeat = self.batch_subfolder_per_repeat.get()

        # Output-Verzeichnis
        base_dir = Path(self.output_dir.get())
        base_dir.mkdir(parents=True, exist_ok=True)

        if rf_enabled:
            rf_config_dict = rf_kwargs.get('rf_config', {})
            rf_config = RFTrainingConfig(**rf_config_dict)
            rf_trainer = RandomForestTrainer(base_dir, rf_config)
            self._update_status("🌲 RF-Training aktiviert – Features werden gesammelt...", "#16a085")

        # Gesamtzahl Tasks
        total_tasks = len(poly_times) * repeats
        current_task = 0

        # Erstelle Detector
        detector = self._create_custom_detector()

        # Batch-Loop
        for repeat in range(1, repeats + 1):
            # Unterordner?
            if subfolder_per_repeat:
                output_dir = base_dir / f"repeat_{repeat}"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = base_dir
                output_dir.mkdir(parents=True, exist_ok=True)

            for t_poly in poly_times:
                current_task += 1
                progress = int((current_task / total_tasks) * 90) + 5

                # Spot-Anzahl
                if use_spot_range:
                    num_spots = np.random.randint(spot_min, spot_max + 1)
                else:
                    num_spots = self.num_spots.get()

                # Filename
                astig_suffix = "_3d" if astigmatism else "_2d"
                filename = f"t{int(t_poly):03d}min_spots{num_spots:02d}_r{repeat}{astig_suffix}.tif"
                filepath = output_dir / filename

                # Status
                status_msg = f"📦 Batch [{current_task}/{total_tasks}]: t={t_poly:.0f}min, spots={num_spots}, repeat={repeat}"
                self._update_status(status_msg)
                self._update_progress(progress)

                # Simulator
                sim_mode = "polyzeit_astig" if astigmatism else "polyzeit"
                sim = TIFFSimulatorOptimized(
                    detector=detector,
                    mode=sim_mode,
                    t_poly_min=t_poly,
                    astigmatism=astigmatism
                )

                # Generiere TIFF
                tiff_stack = sim.generate_tiff(
                    image_size=(self.image_height.get(), self.image_width.get()),
                    num_spots=num_spots,
                    num_frames=self.num_frames.get(),
                    frame_rate_hz=self.frame_rate.get(),
                    d_initial=self.d_initial.get(),
                    exposure_substeps=self.exposure_substeps.get(),
                    enable_photophysics=self.enable_photophysics.get(),
                    progress_callback=None  # Kein Sub-Progress
                )

                # Speichern
                save_tiff(str(filepath), tiff_stack)

                # Metadata
                metadata = None
                if self.export_json.get() or self.export_txt.get() or self.export_csv.get() or rf_trainer:
                    metadata = sim.get_metadata()

                if metadata is not None and (self.export_json.get() or self.export_txt.get() or self.export_csv.get()):
                    exporter = MetadataExporter(str(output_dir))
                    base_name = filepath.stem

                    if self.export_json.get():
                        exporter.export_json(metadata, base_name)
                    if self.export_txt.get():
                        exporter.export_txt(metadata, base_name)
                    if self.export_csv.get():
                        exporter.export_csv_row(metadata, base_name)

                if rf_trainer and metadata is not None:
                    # Aktualisiere Random-Forest mit den frisch simulierten Trajektorien
                    try:
                        rf_trainer.update_with_metadata(metadata)
                    except Exception as rf_err:
                        print(f"⚠️ RF-Update Warnung: {rf_err}")

        if rf_trainer:
            rf_info = rf_trainer.finalize()

        self._update_progress(100)
        self._update_status(f"✅ Batch fertig! {total_tasks} TIFFs erstellt.")

        if rf_trainer:
            def _notify_rf_completion():
                if rf_info and rf_info.get('model_path'):
                    msg = (
                        "Random-Forest Training abgeschlossen!\n\n"
                        f"Fenster: {rf_info.get('samples', 0)}\n"
                        f"Modell: {rf_info.get('model_path')}"
                    )
                    messagebox.showinfo("Random Forest", msg)
                    self._update_status(f"🌲 RF gespeichert: {rf_info.get('model_path')}", "#16a085")
                else:
                    self._update_status("🌲 Keine gültigen Trajektorienfenster für RF gefunden.", "#e67e22")

            self.root.after(0, _notify_rf_completion)

    def _create_custom_detector(self):
        """Erstellt Custom-Detektor mit aktuellen GUI-Parametern."""
        from tiff_simulator_v3 import DetectorPreset

        base_preset = TDI_PRESET if self.detector_var.get() == "TDI-G0" else TETRASPECS_PRESET

        # Custom Detector mit GUI-Werten
        custom = DetectorPreset(
            name=base_preset.name,
            max_intensity=self.max_intensity.get(),
            background_mean=self.background_mean.get(),
            background_std=self.background_std.get(),
            pixel_size_um=base_preset.pixel_size_um,
            fwhm_um=base_preset.fwhm_um,
            metadata={
                **base_preset.metadata,
                "read_noise_std": self.read_noise_std.get(),
                "spot_intensity_sigma": self.spot_intensity_sigma.get(),
                "frame_jitter_sigma": self.frame_jitter_sigma.get(),
                "on_mean_frames": self.on_mean_frames.get(),
                "off_mean_frames": self.off_mean_frames.get(),
                "bleach_prob_per_frame": self.bleach_prob.get(),
                "z_amp_um": self.z_amp_um.get(),
                "z_max_um": self.z_max_um.get(),
                "astig_z0_um": self.astig_z0_um.get(),
                "astig_coeffs": {"A_x": self.astig_Ax.get(), "B_x": 0.0,
                               "A_y": self.astig_Ay.get(), "B_y": 0.0}
            }
        )

        return custom

    # ========================================================================
    # TRACK ANALYSIS TAB (NEU V4.1!)
    # ========================================================================

    def _create_analysis_tab(self):
        """Erstellt den Track Analysis Tab."""

        tk.Label(
            self.analysis_tab,
            text="🔬 TRACK ANALYSIS - Experimentelle Daten analysieren",
            font=("Arial", 14, "bold"),
            fg="#1a1a2e"
        ).pack(pady=10)

        tk.Label(
            self.analysis_tab,
            text="Analysiere TrackMate XML-Dateien mit trainiertem RF-Modell",
            font=("Arial", 10),
            fg="#7f8c8d"
        ).pack(pady=2)

        # ====================================================================
        # INPUT SECTION
        # ====================================================================
        input_frame = ttk.LabelFrame(self.analysis_tab, text="📁 Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Single/Batch Mode
        mode_frame = tk.Frame(input_frame)
        mode_frame.pack(fill=tk.X, pady=5)

        tk.Label(mode_frame, text="Modus:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        ttk.Radiobutton(
            mode_frame,
            text="📄 Einzelne XML-Datei",
            variable=self.analysis_mode,
            value="single",
            command=self._on_analysis_mode_change
        ).pack(side=tk.LEFT, padx=5)

        ttk.Radiobutton(
            mode_frame,
            text="📁 Ganzer Ordner",
            variable=self.analysis_mode,
            value="batch",
            command=self._on_analysis_mode_change
        ).pack(side=tk.LEFT, padx=5)

        # File/Folder Selection
        path_frame = tk.Frame(input_frame)
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(path_frame, text="XML/Ordner:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        xml_entry = tk.Entry(path_frame, textvariable=self.analysis_xml_path, width=50)
        xml_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.analysis_browse_btn = ttk.Button(
            path_frame,
            text="Browse...",
            command=self._browse_analysis_xml
        )
        self.analysis_browse_btn.pack(side=tk.LEFT, padx=5)

        # Recursive Checkbox (nur für Batch)
        self.analysis_recursive_check = ttk.Checkbutton(
            input_frame,
            text="📂 Rekursiv (Unterordner durchsuchen)",
            variable=self.analysis_recursive
        )
        self.analysis_recursive_check.pack(anchor=tk.W, pady=5)

        # ====================================================================
        # PREVIEW SECTION
        # ====================================================================
        preview_frame = ttk.LabelFrame(self.analysis_tab, text="👀 Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.analysis_preview_label = tk.Label(
            preview_frame,
            textvariable=self.analysis_preview_text,
            font=("Courier", 10),
            justify=tk.LEFT,
            anchor=tk.NW,
            bg="#f9f9f9",
            fg="#2c3e50",
            padx=10,
            pady=10
        )
        self.analysis_preview_label.pack(fill=tk.BOTH, expand=True)

        # ====================================================================
        # SETTINGS SECTION
        # ====================================================================
        settings_frame = ttk.LabelFrame(self.analysis_tab, text="⚙️ Einstellungen", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)

        # RF Model
        model_frame = tk.Frame(settings_frame)
        model_frame.pack(fill=tk.X, pady=5)

        tk.Label(model_frame, text="RF-Modell:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        model_entry = tk.Entry(model_frame, textvariable=self.analysis_rf_model, width=50)
        model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(
            model_frame,
            text="Browse...",
            command=self._browse_rf_model
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            model_frame,
            text="🔍 Auto-Detect",
            command=self._auto_detect_rf_model
        ).pack(side=tk.LEFT, padx=2)

        # Frame Rate
        fr_frame = tk.Frame(settings_frame)
        fr_frame.pack(fill=tk.X, pady=5)

        tk.Label(fr_frame, text="Frame Rate [Hz]:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        ttk.Spinbox(
            fr_frame,
            from_=1.0,
            to=100.0,
            increment=1.0,
            textvariable=self.analysis_frame_rate,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        ToolTip(fr_frame.winfo_children()[-1], "Frame-Rate deiner Aufnahmen\nWichtig für korrekte D-Wert Berechnung!")

        # Adaptive Training (NEU!)
        adaptive_frame = tk.Frame(settings_frame)
        adaptive_frame.pack(fill=tk.X, pady=5)

        adaptive_check = ttk.Checkbutton(
            adaptive_frame,
            text="🤖 Adaptive RF Training",
            variable=self.analysis_adaptive_training
        )
        adaptive_check.pack(side=tk.LEFT)

        ToolTip(
            adaptive_check,
            "INTELLIGENT: Schätzt Polymerisationsgrad aus Daten\n"
            "und trainiert RF speziell darauf!\n\n"
            "✓ Optimal für experimentelle Bedingungen\n"
            "✓ Nur ~2-3 Min für 200 Tracks\n"
            "✓ Deutlich bessere Klassifikation!"
        )

        # Anzahl Tracks für Training (nur sichtbar wenn adaptive training an)
        tracks_frame = tk.Frame(settings_frame)
        tracks_frame.pack(fill=tk.X, pady=2)

        tk.Label(tracks_frame, text="  Training Tracks:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(
            tracks_frame,
            from_=50,
            to=500,
            increment=50,
            textvariable=self.analysis_n_tracks_training,
            width=10
        ).pack(side=tk.LEFT, padx=5)

        ToolTip(tracks_frame.winfo_children()[-1], "Anzahl Tracks für Quick-Training\n200 = guter Kompromiss")

        # Output Directory
        out_frame = tk.Frame(settings_frame)
        out_frame.pack(fill=tk.X, pady=5)

        tk.Label(out_frame, text="Output Dir:", width=15, anchor=tk.W).pack(side=tk.LEFT)

        out_entry = tk.Entry(out_frame, textvariable=self.analysis_output_dir, width=50)
        out_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        ttk.Button(
            out_frame,
            text="Browse...",
            command=self._browse_analysis_output
        ).pack(side=tk.LEFT, padx=5)

        # ====================================================================
        # ACTION BUTTONS
        # ====================================================================
        btn_frame = tk.Frame(self.analysis_tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.analysis_start_btn = ttk.Button(
            btn_frame,
            text="🚀 ANALYSE STARTEN",
            command=self._start_analysis,
            style='Accent.TButton'
        )
        self.analysis_start_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="📖 Hilfe",
            command=self._show_analysis_help
        ).pack(side=tk.LEFT, padx=5)

        # Status
        status_frame = tk.Frame(self.analysis_tab, bg="#e8f4f8", relief=tk.SOLID, bd=1)
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            status_frame,
            text="Status:",
            font=("Arial", 10, "bold"),
            bg="#e8f4f8"
        ).pack(side=tk.LEFT, padx=10)

        tk.Label(
            status_frame,
            textvariable=self.analysis_status,
            font=("Arial", 10),
            bg="#e8f4f8",
            fg="#16a085"
        ).pack(side=tk.LEFT, padx=5)

        # Initial state
        self._on_analysis_mode_change()

    def _on_analysis_mode_change(self):
        """Update UI based on analysis mode."""
        is_batch = self.analysis_mode.get() == "batch"

        # Show/hide recursive checkbox
        if is_batch:
            self.analysis_recursive_check.pack(anchor=tk.W, pady=5)
        else:
            self.analysis_recursive_check.pack_forget()

    def _browse_analysis_xml(self):
        """Browse for XML file or folder."""
        if self.analysis_mode.get() == "single":
            path = filedialog.askopenfilename(
                title="Wähle TrackMate XML-Datei",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")]
            )
        else:
            path = filedialog.askdirectory(
                title="Wähle Ordner mit XML-Dateien"
            )

        if path:
            self.analysis_xml_path.set(path)
            self._update_analysis_preview()

    def _browse_rf_model(self):
        """Browse for RF model file."""
        path = filedialog.askopenfilename(
            title="Wähle RF-Modell",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.analysis_rf_model.set(path)
            self.analysis_status.set(f"RF-Modell geladen: {Path(path).name}")

    def _auto_detect_rf_model(self):
        """Auto-detect RF model in common locations."""
        # Check in output_dir
        search_paths = [
            Path(self.output_dir.get()),
            Path.cwd(),
            Path.home() / "Desktop",
        ]

        for search_path in search_paths:
            if search_path.exists():
                # Search for .joblib files
                models = list(search_path.rglob("random_forest_diffusion.joblib"))
                if models:
                    self.analysis_rf_model.set(str(models[0]))
                    self.analysis_status.set(f"✅ RF-Modell gefunden: {models[0].name}")
                    messagebox.showinfo(
                        "Auto-Detect",
                        f"RF-Modell gefunden:\n{models[0]}"
                    )
                    return

        messagebox.showwarning(
            "Auto-Detect",
            "Kein RF-Modell gefunden.\nBitte manuell auswählen oder zuerst trainieren:\n"
            "Batch-Modus → RF Training aktivieren"
        )

    def _browse_analysis_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(
            title="Wähle Output-Verzeichnis"
        )
        if path:
            self.analysis_output_dir.set(path)

    def _update_analysis_preview(self):
        """Update preview with XML statistics."""
        xml_path = self.analysis_xml_path.get()

        if not xml_path:
            self.analysis_preview_text.set("Keine XML geladen...")
            return

        xml_path = Path(xml_path)

        if self.analysis_mode.get() == "single":
            # Single file preview
            if not xml_path.exists():
                self.analysis_preview_text.set("❌ Datei nicht gefunden!")
                return

            try:
                parser = TrackMateXMLParser(xml_path)
                tracks = parser.parse()
                stats = parser.get_preview_stats()

                preview = f"""
📄 XML-Datei: {xml_path.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 STATISTIKEN:
   Tracks:        {stats['num_tracks']}
   Gesamt Frames: {stats['total_frames']}

   Mean Length:   {stats['mean_length']:.1f} Frames
   Median Length: {stats['median_length']:.1f} Frames
   Min Length:    {stats['min_length']} Frames
   Max Length:    {stats['max_length']} Frames

✅ Bereit zur Analyse!
"""
                self.analysis_preview_text.set(preview.strip())
                self.analysis_status.set(f"✅ Preview: {stats['num_tracks']} Tracks")

            except Exception as e:
                self.analysis_preview_text.set(f"❌ Fehler beim Parsen:\n{str(e)}")
                self.analysis_status.set("❌ Parse-Fehler")

        else:
            # Batch mode preview
            if not xml_path.exists():
                self.analysis_preview_text.set("❌ Ordner nicht gefunden!")
                return

            # Count XML files
            if self.analysis_recursive.get():
                xml_files = list(xml_path.rglob("*.xml"))
            else:
                xml_files = list(xml_path.glob("*.xml"))

            preview = f"""
📁 Ordner: {xml_path.name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GEFUNDEN:
   XML-Dateien: {len(xml_files)}
   Rekursiv:    {'Ja' if self.analysis_recursive.get() else 'Nein'}

🔍 Erste 10 Dateien:
"""
            for i, xml in enumerate(xml_files[:10]):
                preview += f"\n   {i+1}. {xml.name}"

            if len(xml_files) > 10:
                preview += f"\n   ... und {len(xml_files) - 10} weitere"

            preview += "\n\n✅ Bereit zur Batch-Analyse!"

            self.analysis_preview_text.set(preview.strip())
            self.analysis_status.set(f"✅ {len(xml_files)} XML-Dateien gefunden")

    def _start_analysis(self):
        """Start the track analysis."""

        # Validation
        xml_path = self.analysis_xml_path.get()
        if not xml_path:
            messagebox.showerror("Fehler", "Bitte wähle eine XML-Datei oder einen Ordner!")
            return

        rf_model = self.analysis_rf_model.get()
        if not rf_model or not Path(rf_model).exists():
            messagebox.showerror("Fehler", "Bitte wähle ein gültiges RF-Modell!")
            return

        # Set output dir if not set
        if not self.analysis_output_dir.get():
            xml_p = Path(xml_path)
            if self.analysis_mode.get() == "single":
                default_out = xml_p.parent / f"{xml_p.stem}_analysis"
            else:
                default_out = xml_p / "analysis_results"
            self.analysis_output_dir.set(str(default_out))

        # Confirm
        msg = f"Analyse starten?\n\n"
        msg += f"Input: {Path(xml_path).name}\n"
        msg += f"Modell: {Path(rf_model).name}\n"
        msg += f"Frame Rate: {self.analysis_frame_rate.get()} Hz\n"
        msg += f"Output: {Path(self.analysis_output_dir.get()).name}"

        if not messagebox.askyesno("Analyse starten?", msg):
            return

        # Disable button
        self.analysis_start_btn.config(state=tk.DISABLED)
        self.analysis_status.set("🔄 Analyse läuft...")

        # Run in thread
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()

    def _run_analysis_thread(self):
        """Run analysis in background thread."""
        try:
            rf_model_path = Path(self.analysis_rf_model.get())

            # ADAPTIVE TRAINING: Trainiere RF auf geschätzten Polygrad
            if self.analysis_adaptive_training.get():
                self.root.after(0, lambda: self.analysis_status.set("🤖 Adaptive RF Training..."))

                # Quick-Train adaptiver RF
                trainer, estimate = quick_train_adaptive_rf(
                    xml_path=Path(self.analysis_xml_path.get()),
                    detector=TDI_PRESET,  # Verwende TDI als Standard
                    frame_rate_hz=self.analysis_frame_rate.get(),
                    n_tracks_total=self.analysis_n_tracks_training.get(),
                    output_dir=Path(self.analysis_output_dir.get()),
                    verbose=False,  # Kein Print in GUI
                    cleanup_temp=True
                )

                # Nutze den frisch trainierten RF
                rf_model_path = Path(self.analysis_output_dir.get()) / f"rf_adaptive_t{estimate.t_poly_min:.0f}min.joblib"

                # Status Update mit Polygrad-Info
                polygrad_info = f" (t={estimate.t_poly_min:.0f}min)"
                self.root.after(0, lambda: self.analysis_status.set(f"🔬 Analysiere{polygrad_info}..."))

            orchestrator = TrackAnalysisOrchestrator(rf_model_path)

            if self.analysis_mode.get() == "single":
                # Single file
                result = orchestrator.analyze_xml(
                    Path(self.analysis_xml_path.get()),
                    Path(self.analysis_output_dir.get()),
                    self.analysis_frame_rate.get()
                )

                def _success():
                    self.analysis_status.set("✅ Analyse abgeschlossen!")
                    self.analysis_start_btn.config(state=tk.NORMAL)
                    messagebox.showinfo(
                        "Erfolg! 🎉",
                        f"Analyse abgeschlossen!\n\n"
                        f"Tracks: {result.num_tracks}\n"
                        f"Output: {self.analysis_output_dir.get()}\n\n"
                        f"Öffne die Excel-Datei für Details!"
                    )

                self.root.after(0, _success)

            else:
                # Batch
                results = orchestrator.batch_analyze_folder(
                    Path(self.analysis_xml_path.get()),
                    Path(self.analysis_output_dir.get()),
                    self.analysis_frame_rate.get(),
                    self.analysis_recursive.get()
                )

                def _success():
                    self.analysis_status.set("✅ Batch-Analyse abgeschlossen!")
                    self.analysis_start_btn.config(state=tk.NORMAL)
                    messagebox.showinfo(
                        "Erfolg! 🎉",
                        f"Batch-Analyse abgeschlossen!\n\n"
                        f"Erfolgreich: {len(results)} XMLs\n"
                        f"Output: {self.analysis_output_dir.get()}"
                    )

                self.root.after(0, _success)

        except Exception as e:
            def _error():
                self.analysis_status.set("❌ Fehler!")
                self.analysis_start_btn.config(state=tk.NORMAL)
                messagebox.showerror(
                    "Fehler",
                    f"Analyse fehlgeschlagen:\n\n{str(e)}"
                )

            self.root.after(0, _error)

    def _show_analysis_help(self):
        """Show help dialog for track analysis."""
        help_text = """
🔬 TRACK ANALYSIS HILFE

WORKFLOW:
1. RF-Modell trainieren (Batch-Tab → RF Training)
2. XML-Datei(en) aus TrackMate exportieren
3. Hier: XML + Modell auswählen
4. Analyse starten

OUTPUT:
- Excel: Ein Sheet pro Track (Frame-by-Frame)
- CSV: Aggregierte Statistiken
- PDFs: Pie Chart, D/Alpha Boxplots

WICHTIG:
- Frame Rate muss korrekt sein!
- Mindestens 30 Frames pro Track
- RF-Modell muss trainiert sein

Mehr Info: Siehe TRACK_ANALYSIS_GUIDE.md
"""
        messagebox.showinfo("Track Analysis Hilfe", help_text)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = TIFFSimulatorGUI_V4(root)
    root.mainloop()
