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
from typing import List, Dict
import numpy as np

try:
    from tiff_simulator_v3 import (
        TDI_PRESET,
        TETRASPECS_PRESET,
        TIFFSimulatorOptimized,
        save_tiff,
        evaluate_z_profile
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

    # ========================================================================
    # UI LAYOUT CONSTANTS
    # ========================================================================
    # Widget Widths
    WIDGET_WIDTH_SMALL = 10
    WIDGET_WIDTH_MEDIUM = 15
    WIDGET_WIDTH_LARGE = 18
    WIDGET_WIDTH_XLARGE = 25
    WIDGET_WIDTH_ENTRY = 30
    WIDGET_WIDTH_LONG_ENTRY = 50

    # Padding
    PADDING_TINY = 1
    PADDING_SMALL = 2
    PADDING_MEDIUM = 5
    PADDING_LARGE = 10
    PADDING_XLARGE = 15

    # Fonts
    FONT_SMALL = ("Arial", 9)
    FONT_NORMAL = ("Arial", 10)
    FONT_HEADER = ("Arial", 12, "bold")
    FONT_LARGE_HEADER = ("Arial", 14, "bold")
    FONT_TITLE = ("Arial", 16, "bold")

    # Colors
    COLOR_SUCCESS = "#27ae60"
    COLOR_INFO = "#16a085"
    COLOR_WARNING = "#f39c12"
    COLOR_ERROR = "#e74c3c"
    COLOR_MUTED = "#7f8c8d"
    COLOR_PRIMARY = "#3498db"
    COLOR_BACKGROUND = "#f5f5f5"

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

        # z-Profil Beobachter registrieren & initiales Summary anzeigen
        self._register_z_profile_traces()
        self._update_z_stack_summary()

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
        self.wavelength_nm = tk.DoubleVar(value=580.0)

        # ===== ASTIGMATISM & 3D (NEU!) =====
        self.z_amp_um = tk.DoubleVar(value=1.2)       # OPTIMIERT: Größer für weiteren z-Range
        self.z_max_um = tk.DoubleVar(value=0.6)
        self.astig_z0_um = tk.DoubleVar(value=0.7)    # OPTIMIERT: Bessere Spreizung der Astigmatismus-Kurve
        self.astig_ax = tk.DoubleVar(value=1.5)       # OPTIMIERT: Stärkerer Astigmatismus für bessere Sichtbarkeit
        self.astig_ay = tk.DoubleVar(value=-1.2)      # OPTIMIERT: Stärkerer Astigmatismus für bessere Sichtbarkeit
        self.astig_focus_offset_um = tk.DoubleVar(value=0.25)
        self.spherical_aberration_strength = tk.DoubleVar(value=0.35)
        self.axial_intensity_floor = tk.DoubleVar(value=0.15)
        self.refractive_index_correction = tk.DoubleVar(value=1.0)  # Legacy: Einfacher Faktor

        # ===== ERWEITERTE BRECHUNGSINDEX-KORREKTUR (NEU!) =====
        self.use_advanced_refractive_correction = tk.BooleanVar(value=True)  # Aktiviert erweiterte Korrektur
        self.n_oil = tk.DoubleVar(value=1.518)        # Brechungsindex Immersionsöl
        self.n_glass = tk.DoubleVar(value=1.523)      # Brechungsindex Deckglas
        self.n_polymer = tk.DoubleVar(value=1.54)     # Brechungsindex Polymer/Medium (KORRIGIERT: war 1.47)
        self.NA = tk.DoubleVar(value=1.45)            # Numerische Apertur (KORRIGIERT: war 1.50)
        self.d_glass_um = tk.DoubleVar(value=170.0)   # Deckglas-Dicke [µm]

        # ===== ILLUMINATION GRADIENT (NEU!) =====
        self.illumination_gradient_strength = tk.DoubleVar(value=0.0)  # NEU
        self.illumination_gradient_type = tk.StringVar(value="radial")  # NEU

        # ===== COMONOMER-BESCHLEUNIGUNGSFAKTOR (NEU!) =====
        self.polymerization_acceleration_factor = tk.DoubleVar(value=1.0)  # NEU

        # ===== Z-STACK (REALISTISCH FÜR TETRASPECS MESSUNG) =====
        self.z_min = tk.DoubleVar(value=-0.5)   # Realistisch: von -0.5 µm
        self.z_max = tk.DoubleVar(value=0.5)    # bis +0.5 µm (1 µm total range)
        self.z_step = tk.DoubleVar(value=0.02)  # Feine Steps: 0.02 µm (61 slices)
        self.z_stack_summary_var = tk.StringVar(value="")

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
        self.batch_folder_structure = tk.StringVar(value="by_repeat")  # "by_repeat", "by_polytime", or "flat"
        # Legacy (falls benötigt)
        self.batch_preset = tk.StringVar(value="quick")
        self.batch_detector = tk.StringVar(value="TDI-G0")
        self.batch_custom_times = tk.StringVar(value="")

        # ===== RF-TRAINING VARIABLEN ENTFERNT =====
        # RF-Training ist jetzt nur noch im dedizierten "RF Training" Tab!
        # Batch-Modus fokussiert sich auf TIFF-Generierung.

        # ===== DEDIZIERTES RF-TRAINING (NEU!) =====
        self.rf_dedicated_output_dir = tk.StringVar(value=str(Path.home() / "Desktop" / "rf_training"))
        # Trajectory Generation
        self.rf_dedicated_num_frames = tk.IntVar(value=300)  # Lange Tracks!
        self.rf_dedicated_tracks_per_type = tk.IntVar(value=1000)  # Viele Tracks pro Typ
        self.rf_dedicated_poly_times = tk.StringVar(value="0, 30, 60, 90, 120, 180")  # Alle Polygrade
        self.rf_dedicated_frame_rate = tk.DoubleVar(value=20.0)
        self.rf_dedicated_d_initial = tk.DoubleVar(value=0.24)
        # RF Hyperparameters (GROSSE Werte für bestes Modell!)
        self.rf_dedicated_n_estimators = tk.IntVar(value=2048)  # Reduziert für bessere Generalisierung
        self.rf_dedicated_max_depth = tk.IntVar(value=20)      # Weniger Tiefe = weniger Overfitting
        self.rf_dedicated_min_leaf = tk.IntVar(value=5)        # Mehr Samples pro Blatt = Regularisierung
        self.rf_dedicated_min_split = tk.IntVar(value=10)      # Höherer Split-Threshold
        self.rf_dedicated_max_samples = tk.DoubleVar(value=0.7) # Weniger Samples pro Baum
        self.rf_dedicated_window_sizes = tk.StringVar(value="32, 48, 64, 96")
        self.rf_dedicated_step_fraction = tk.DoubleVar(value=0.5)

        # ===== EXPORT =====
        self.export_metadata = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_txt = tk.BooleanVar(value=True)
        self.export_csv = tk.BooleanVar(value=True)

    # ========================================================================
    # WIDGET FACTORY HELPERS (Eliminates Code Duplication)
    # ========================================================================

    def _create_parameter_row(self, parent, label_text, widget_type="spinbox",
                              variable=None, tooltip=None, **widget_kwargs):
        """
        Creates a standardized parameter row: Label + Widget.

        Args:
            parent: Parent frame
            label_text: Text for the label
            widget_type: "spinbox", "entry", "checkbox", "scale"
            variable: Tkinter variable to bind
            tooltip: Tooltip text (optional)
            **widget_kwargs: Additional kwargs for the widget

        Returns:
            frame, widget: The created frame and widget
        """
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=self.PADDING_SMALL)

        # Label
        label = tk.Label(
            frame,
            text=label_text,
            width=self.WIDGET_WIDTH_XLARGE,
            anchor=tk.W
        )
        label.pack(side=tk.LEFT)

        # Widget
        if widget_type == "spinbox":
            widget = ttk.Spinbox(
                frame,
                textvariable=variable,
                width=self.WIDGET_WIDTH_MEDIUM,
                **widget_kwargs
            )
        elif widget_type == "entry":
            widget = tk.Entry(
                frame,
                textvariable=variable,
                width=widget_kwargs.pop("width", self.WIDGET_WIDTH_ENTRY),
                **widget_kwargs
            )
        elif widget_type == "checkbox":
            widget = ttk.Checkbutton(
                frame,
                variable=variable,
                **widget_kwargs
            )
        elif widget_type == "scale":
            widget = ttk.Scale(
                frame,
                variable=variable,
                **widget_kwargs
            )
        else:
            raise ValueError(f"Unknown widget type: {widget_type}")

        widget.pack(side=tk.LEFT, padx=self.PADDING_MEDIUM)

        # Tooltip
        if tooltip:
            ToolTip(frame, tooltip)

        return frame, widget

    def _create_labeled_frame(self, parent, title, **kwargs):
        """Creates a standardized labeled frame."""
        frame = ttk.LabelFrame(parent, text=title, padding=self.PADDING_LARGE)
        frame.pack(fill=tk.X, padx=self.PADDING_MEDIUM, pady=self.PADDING_MEDIUM, **kwargs)
        return frame

    def _create_section_header(self, parent, text, emoji="", font=None):
        """Creates a standardized section header label."""
        if font is None:
            font = self.FONT_LARGE_HEADER

        full_text = f"{emoji} {text}" if emoji else text
        label = tk.Label(
            parent,
            text=full_text,
            font=font,
            fg=self.COLOR_INFO
        )
        label.pack(pady=self.PADDING_MEDIUM)
        return label

    # ========================================================================
    # MAIN UI CREATION
    # ========================================================================

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
            font=("Arial", 20, "bold"),  # Custom title font
            bg="#1a1a2e",
            fg="white"
        ).pack(pady=self.PADDING_MEDIUM)

        tk.Label(
            header_frame,
            text="⚡ ADVANCED EDITION - Optimiert für maximale Performance & Flexibilität",
            font=("Arial", 11),  # Custom subtitle font
            bg="#1a1a2e",
            fg=self.COLOR_INFO
        ).pack()

        tk.Label(
            header_frame,
            text="✨ Mit erweiterten Photophysik-Parametern, Live-Preview & Batch-Processing",
            font=self.FONT_SMALL,
            bg="#1a1a2e",
            fg="#a8dadc"
        ).pack(pady=self.PADDING_SMALL)

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

        # Tab 7: RF Training (DEDICATED MODE - NEU!)
        self.rf_training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.rf_training_tab, text="🌲 RF Training")
        self._create_rf_training_tab()

        # Tab 8: Track Analysis
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

        # Comonomer-Beschleunigungsfaktor (NEU!)
        comonomer_frame = tk.Frame(time_frame)
        comonomer_frame.pack(fill=tk.X, pady=2)
        tk.Label(comonomer_frame, text="Comonomer-Faktor:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        comonomer_spin = ttk.Spinbox(comonomer_frame, from_=0.5, to=2.0, increment=0.1,
                   textvariable=self.polymerization_acceleration_factor, width=10, format='%.2f')
        comonomer_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(comonomer_spin, "Beschleunigungsfaktor für Polymerisation\n1.0 = Standard\n1.5 = 50% schnellere Vernetzung (reaktives Comonomer)\n0.7 = 30% langsamere Vernetzung")

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

        # Emissionswellenlänge
        wavelength_frame = tk.Frame(psf_frame)
        wavelength_frame.pack(fill=tk.X, pady=2)
        tk.Label(wavelength_frame, text="Emission λ [nm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        wavelength_spin = ttk.Spinbox(
            wavelength_frame,
            from_=450.0,
            to=700.0,
            increment=5.0,
            textvariable=self.wavelength_nm,
            width=10,
            format='%.1f'
        )
        wavelength_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(wavelength_spin, "Emissionswellenlänge der Fluorophore\n580 nm = Tetraspeck/TMR\n488 nm für GFP, 647 nm für Cy5")

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

        # ILLUMINATION GRADIENT (NEU!)
        illum_frame = ttk.LabelFrame(self.physics_tab, text="💡 Ausleuchtungsgradient", padding=10)
        illum_frame.pack(fill=tk.X, padx=10, pady=5)

        # Gradient Strength
        grad_strength_frame = tk.Frame(illum_frame)
        grad_strength_frame.pack(fill=tk.X, pady=2)
        tk.Label(grad_strength_frame, text="Gradient Stärke [counts]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        grad_strength_spin = ttk.Spinbox(grad_strength_frame, from_=0.0, to=50.0, increment=1.0,
                   textvariable=self.illumination_gradient_strength, width=10, format='%.1f')
        grad_strength_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(grad_strength_spin, "Stärke des Ausleuchtungsgradienten\n0 = aus, 5-20 = subtil, >20 = stark")

        # Gradient Type
        grad_type_frame = tk.Frame(illum_frame)
        grad_type_frame.pack(fill=tk.X, pady=2)
        tk.Label(grad_type_frame, text="Gradient Typ:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        grad_type_combo = ttk.Combobox(grad_type_frame, textvariable=self.illumination_gradient_type,
                                       values=["radial", "linear_x", "linear_y", "corner"],
                                       width=10, state="readonly")
        grad_type_combo.pack(side=tk.LEFT, padx=5)
        ToolTip(grad_type_combo, "Typ des Gradienten:\nradial = vom Zentrum\nlinear_x/y = entlang Achse\ncorner = von Ecke")

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
                   textvariable=self.astig_ax, width=10, format='%.2f')
        ax_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(ax_spin, "Astigmatismus x-Koeffizient\n+1.0 = Standard")

        # Ay
        ay_frame = tk.Frame(astig_coef_frame)
        ay_frame.pack(fill=tk.X, pady=2)
        tk.Label(ay_frame, text="A_y (y-Koeffizient):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ay_spin = ttk.Spinbox(ay_frame, from_=-2.0, to=2.0, increment=0.1,
                   textvariable=self.astig_ay, width=10, format='%.2f')
        ay_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(ay_spin, "Astigmatismus y-Koeffizient\n-0.5 = Standard")

        # Fokusversatz
        focus_frame = tk.Frame(astig_coef_frame)
        focus_frame.pack(fill=tk.X, pady=2)
        tk.Label(focus_frame, text="Fokusversatz Δz [µm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        focus_spin = ttk.Spinbox(
            focus_frame,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.astig_focus_offset_um,
            width=10,
            format='%.2f'
        )
        focus_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(focus_spin, "Astigmatische Fokusverschiebung zwischen x- und y-Brennlinie\n0.25 µm = realistischer ThunderSTORM-Wert")

        # Sphärische Aberration
        aberr_frame = tk.Frame(astig_coef_frame)
        aberr_frame.pack(fill=tk.X, pady=2)
        tk.Label(aberr_frame, text="Sphärische Aberration:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        aberr_spin = ttk.Spinbox(
            aberr_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.spherical_aberration_strength,
            width=10,
            format='%.2f'
        )
        aberr_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(aberr_spin, "Stärke der Brechungsindex-bedingten Aberration\n0.35 = stark für Polymer vs. Öl")

        # Axiale Intensitätsuntergrenze
        axial_frame = tk.Frame(astig_coef_frame)
        axial_frame.pack(fill=tk.X, pady=2)
        tk.Label(axial_frame, text="Axiale Intensität (min):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        axial_spin = ttk.Spinbox(
            axial_frame,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.axial_intensity_floor,
            width=10,
            format='%.2f'
        )
        axial_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(axial_spin, "Minimaler Intensitätsfaktor entlang der z-Achse\nVerhindert völliges Auslöschen entfernter Slices")

        # Brechungsindex-Korrektur (Legacy, einfacher Faktor)
        ri_frame = tk.Frame(astig_coef_frame)
        ri_frame.pack(fill=tk.X, pady=2)
        tk.Label(ri_frame, text="Brechungsindex-Korrektur:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ri_spin = ttk.Spinbox(ri_frame, from_=0.5, to=1.5, increment=0.01,
                   textvariable=self.refractive_index_correction, width=10, format='%.3f')
        ri_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(ri_spin, "LEGACY: Einfacher Faktor\n1.0 = keine Korrektur\n0.876 = Wasser/Öl-Immersion\n(Für physikalisch korrekte Korrektur siehe unten)")

        # ====================================================================
        # ERWEITERTE BRECHUNGSINDEX-KORREKTUR (NEU!)
        # ====================================================================
        advanced_ri_frame = ttk.LabelFrame(self.astig_tab, text="🔬 ERWEITERTE Brechungsindex-Korrektur (TIRF)", padding=10)
        advanced_ri_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            advanced_ri_frame,
            text="⚡ Physikalisch korrekte z-Korrektur für TIRF-Mikroskopie",
            font=("Arial", 9, "bold"),
            fg="#16a085"
        ).pack(anchor=tk.W, pady=(0, 5))

        # Aktivierungs-Checkbox
        enable_advanced_check = ttk.Checkbutton(
            advanced_ri_frame,
            text="✅ Erweiterte Korrektur aktivieren (berücksichtigt n_oil, n_glass, n_polymer, NA, d_glass)",
            variable=self.use_advanced_refractive_correction
        )
        enable_advanced_check.pack(anchor=tk.W, pady=5)
        ToolTip(enable_advanced_check, "Aktiviert die physikalisch korrekte Brechungsindex-Korrektur\nmit allen optischen Parametern (Öl, Glas, Polymer, NA)")

        # n_oil
        n_oil_frame = tk.Frame(advanced_ri_frame)
        n_oil_frame.pack(fill=tk.X, pady=2)
        tk.Label(n_oil_frame, text="n_oil (Immersionsöl):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        n_oil_spin = ttk.Spinbox(n_oil_frame, from_=1.40, to=1.60, increment=0.001,
                   textvariable=self.n_oil, width=10, format='%.4f')
        n_oil_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(n_oil_spin, "Brechungsindex Immersionsöl\n1.518 = Olympus TIRF-Öl (Standard)\n1.515 = Nikon Type F\n1.512 = Zeiss Immersol")

        # n_glass
        n_glass_frame = tk.Frame(advanced_ri_frame)
        n_glass_frame.pack(fill=tk.X, pady=2)
        tk.Label(n_glass_frame, text="n_glass (Deckglas):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        n_glass_spin = ttk.Spinbox(n_glass_frame, from_=1.45, to=1.60, increment=0.001,
                   textvariable=self.n_glass, width=10, format='%.4f')
        n_glass_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(n_glass_spin, "Brechungsindex Deckglas\n1.523 = High Precision Coverslide (Standard)\n1.515 = Standard-Glas")

        # n_polymer
        n_polymer_frame = tk.Frame(advanced_ri_frame)
        n_polymer_frame.pack(fill=tk.X, pady=2)
        tk.Label(n_polymer_frame, text="n_polymer (Medium/Probe):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        n_polymer_spin = ttk.Spinbox(n_polymer_frame, from_=1.30, to=1.60, increment=0.001,
                   textvariable=self.n_polymer, width=10, format='%.4f')
        n_polymer_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(n_polymer_spin, "Brechungsindex Polymer/Medium\n1.47 = Hydrogel (Standard)\n1.49 = PMMA\n1.33 = Wasser\n1.45 = Polyacrylamid")

        # NA
        na_frame = tk.Frame(advanced_ri_frame)
        na_frame.pack(fill=tk.X, pady=2)
        tk.Label(na_frame, text="NA (Numerische Apertur):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        na_spin = ttk.Spinbox(na_frame, from_=1.0, to=1.65, increment=0.01,
                   textvariable=self.NA, width=10, format='%.3f')
        na_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(na_spin, "Numerische Apertur des Objektivs\n1.50 = Olympus UPLAPO100XOHR (Standard)\n1.49 = Nikon CFI Apo TIRF\n1.46 = Zeiss Alpha Plan-Apochromat")

        # d_glass
        d_glass_frame = tk.Frame(advanced_ri_frame)
        d_glass_frame.pack(fill=tk.X, pady=2)
        tk.Label(d_glass_frame, text="d_glass (Deckglas-Dicke) [µm]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        d_glass_spin = ttk.Spinbox(d_glass_frame, from_=100, to=200, increment=1,
                   textvariable=self.d_glass_um, width=10, format='%.1f')
        d_glass_spin.pack(side=tk.LEFT, padx=5)
        ToolTip(d_glass_spin, "Deckglas-Dicke in Mikrometern\n170 µm = #1.5 High Precision (Standard)\n160 µm = #1.5 Normal\n150 µm = #1.0")

        # Info-Label
        tk.Label(
            advanced_ri_frame,
            text="💡 Diese Werte beeinflussen die z-Kalibrierung bei Astigmatismus-Simulationen und z-Stacks",
            font=("Arial", 8, "italic"),
            fg="#7f8c8d"
        ).pack(anchor=tk.W, pady=(5, 0))

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

        summary_frame = tk.Frame(zstack_frame)
        summary_frame.pack(fill=tk.X, pady=(8, 2))
        self.z_stack_summary_label = tk.Label(
            summary_frame,
            textvariable=self.z_stack_summary_var,
            font=("Arial", 9),
            fg="#2c3e50",
            justify=tk.LEFT,
            wraplength=420
        )
        self.z_stack_summary_label.pack(anchor=tk.W)

        button_frame = tk.Frame(zstack_frame)
        button_frame.pack(fill=tk.X, pady=(4, 0))
        preset_btn = ttk.Button(button_frame, text="⚙️ ThunderSTORM Preset", command=self._apply_thunderstorm_preset)
        preset_btn.pack(side=tk.LEFT, padx=2)
        preview_btn = ttk.Button(button_frame, text="📈 Profil-Vorschau", command=self._show_z_profile_preview)
        preview_btn.pack(side=tk.LEFT, padx=2)

        if hasattr(self, 'z_stack_widgets'):
            self.z_stack_widgets.extend([zmin_spin, zmax_stack_spin, zstep_spin, preset_btn, preview_btn])

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

        # Ordnerstruktur-Optionen
        folder_label = tk.Label(repeat_frame, text="📁 Ordnerstruktur:", anchor=tk.W, font=("Segoe UI", 9, "bold"))
        folder_label.pack(anchor=tk.W, pady=(10, 5))

        ttk.Radiobutton(
            repeat_frame,
            text="Nach Wiederholungen (repeat_1/, repeat_2/, ...)",
            variable=self.batch_folder_structure,
            value="by_repeat"
        ).pack(anchor=tk.W, padx=20)

        ttk.Radiobutton(
            repeat_frame,
            text="Nach Polymerisationszeit (t030min/, t060min/, ...)",
            variable=self.batch_folder_structure,
            value="by_polytime"
        ).pack(anchor=tk.W, padx=20)

        ttk.Radiobutton(
            repeat_frame,
            text="Alle Dateien in einem Ordner (keine Unterordner)",
            variable=self.batch_folder_structure,
            value="flat"
        ).pack(anchor=tk.W, padx=20, pady=(0, 5))

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
        # RF-TRAINING FRAME ENTFERNT
        # ====================================================================
        # RF-Training ist jetzt NUR im dedizierten "RF Training" Tab!
        # Der Batch-Modus fokussiert sich auf effiziente TIFF-Generierung.

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
        """Toggelt zwischen Single und Batch Mode und aktualisiert Control-States."""
        self._update_control_states()  # Update UI states
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

            # Folder structure description
            folder_mode = self.batch_folder_structure.get()
            if folder_mode == "by_repeat":
                folder_desc = "Nach Wiederholungen (repeat_1/, repeat_2/, ...)"
            elif folder_mode == "by_polytime":
                folder_desc = "Nach Polyzeit (t030min/, t060min/, ...)"
            else:
                folder_desc = "Flach (alle in einem Ordner)"

            summary = (
                f"📊 BATCH-KONFIGURATION:\n\n"
                f"⏱️  Polyzeiten: {len(times)} Zeiten ({', '.join(str(int(t)) for t in times[:5])}{'...' if len(times) > 5 else ''} min)\n"
                f"🔄 Wiederholungen: {repeats}x\n"
                f"🎯 Spots: {spots_info}\n"
                f"📐 Astigmatismus: {astig_info}\n"
                f"📁 Ordnerstruktur: {folder_desc}\n"
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

    # ========================================================================
    # DYNAMIC ENABLE/DISABLE LOGIC (Smart UI)
    # ========================================================================

    def _update_control_states(self):
        """
        Aktualisiert Enable/Disable Status aller Controls basierend auf gewählten Modi.
        Macht die GUI benutzerfreundlicher durch Ausgrauen irrelevanter Optionen.
        """
        sim_mode = self.sim_mode_var.get()
        batch_enabled = self.batch_mode_enabled.get()
        photophysics_enabled = self.enable_photophysics.get()
        advanced_refractive = self.use_advanced_refractive_correction.get()

        # ====================================================================
        # BASIC TAB: Time-Series vs z-Stack Controls
        # ====================================================================
        # z-Stack spezifische Controls
        z_stack_mode = (sim_mode == "z_stack")
        # Time-Series spezifische Controls
        time_series_mode = sim_mode in ("polyzeit", "polyzeit_astig")

        # Store widget references if not already done
        if not hasattr(self, '_control_widgets_initialized'):
            self._init_control_widget_references()

        # z-Stack: nur z-Parameter aktiv
        z_state = tk.NORMAL if z_stack_mode else tk.DISABLED
        for widget in getattr(self, 'z_stack_widgets', []):
            try:
                widget.configure(state=z_state)
            except tk.TclError:
                pass

        # Time-Series: nur Time-Series Parameter aktiv
        ts_state = tk.NORMAL if time_series_mode else tk.DISABLED
        for widget in getattr(self, 'time_series_widgets', []):
            try:
                widget.configure(state=ts_state)
            except tk.TclError:
                pass

        # ====================================================================
        # PHOTOPHYSICS TAB: nur aktiv wenn Checkbox an
        # ====================================================================
        photo_state = tk.NORMAL if photophysics_enabled else tk.DISABLED
        for widget in getattr(self, 'photophysics_widgets', []):
            try:
                widget.configure(state=photo_state)
            except tk.TclError:
                pass

        # ====================================================================
        # ASTIGMATISM TAB: Advanced Refractive Correction
        # ====================================================================
        adv_refr_state = tk.NORMAL if advanced_refractive else tk.DISABLED
        for widget in getattr(self, 'advanced_refractive_widgets', []):
            try:
                widget.configure(state=adv_refr_state)
            except tk.TclError:
                pass

        # ====================================================================
        # BATCH TAB: Batch-spezifische Controls
        # ====================================================================
        batch_state = tk.NORMAL if batch_enabled else tk.DISABLED
        for widget in getattr(self, 'batch_specific_widgets', []):
            try:
                widget.configure(state=batch_state)
            except tk.TclError:
                pass

    def _init_control_widget_references(self):
        """
        Initialisiert Listen von Widgets die zusammen aktiviert/deaktiviert werden.
        Wird beim ersten Aufruf von _update_control_states() ausgeführt.
        """
        # Diese Listen werden später in den Tab-Creation Methoden befüllt
        self.z_stack_widgets = []
        self.time_series_widgets = []
        self.photophysics_widgets = []
        self.advanced_refractive_widgets = []
        self.batch_specific_widgets = []
        self._control_widgets_initialized = True

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
        """Aktualisiert Mode-Info-Text und Control-States."""
        self._update_control_states()  # Update UI states
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

    def _compute_z_positions(self) -> np.ndarray:
        """Hilfsfunktion: berechnet alle z-Stufen basierend auf GUI-Werten."""
        try:
            z_min = self.z_min.get()
            z_max = self.z_max.get()
            z_step = self.z_step.get()
        except tk.TclError:
            return np.array([], dtype=np.float32)

        if z_step <= 0 or z_max <= z_min:
            return np.array([], dtype=np.float32)

        return np.arange(z_min, z_max + 0.5 * z_step, z_step, dtype=np.float32)

    def _update_z_slices(self):
        """Aktualisiert die Anzeige zur Anzahl der z-Slices und Summary."""
        try:
            z_positions = self._compute_z_positions()
            if z_positions.size > 0:
                self.z_slices_label.config(text=f"→ {z_positions.size} Slices")
            else:
                self.z_slices_label.config(text="❌ Ungültig")
        except Exception:
            self.z_slices_label.config(text="❌ Fehler")

        self._update_z_stack_summary()

    def _browse_dir(self):
        """Ordner-Dialog."""
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)

    def _update_z_stack_summary(self):
        """Aktualisiert das Info-Panel für die z-Stack-Physik."""
        if not hasattr(self, "z_stack_summary_var"):
            return

        z_positions = self._compute_z_positions()
        if z_positions.size == 0:
            self.z_stack_summary_var.set("Bitte z_min < z_max und z_step > 0 wählen.")
            if hasattr(self, "z_stack_summary_label"):
                self.z_stack_summary_label.config(fg="#c0392b")
            return

        try:
            detector = self._create_custom_detector()
            profile = evaluate_z_profile(detector, z_positions, astigmatism=True)

            z_corrected = profile.get("z_corrected", z_positions)
            intensity = profile.get("intensity_scale", np.ones_like(z_positions))
            sigma_x = profile.get("sigma_x", np.ones_like(z_positions))
            sigma_y = profile.get("sigma_y", np.ones_like(z_positions))
            refr_comp = profile.get("refractive_components") or {}
            depth_factor = np.asarray(refr_comp.get("depth_factor"), dtype=np.float32) \
                if isinstance(refr_comp.get("depth_factor"), np.ndarray) else np.ones_like(z_positions)

            sample_min = float(np.min(z_corrected)) if z_corrected.size else float(z_positions[0])
            sample_max = float(np.max(z_corrected)) if z_corrected.size else float(z_positions[-1])
            intensity_min = float(np.min(intensity)) if intensity.size else 1.0
            intensity_max = float(np.max(intensity)) if intensity.size else 1.0
            ratio = sigma_x / np.maximum(sigma_y, 1e-6)
            ratio_min = float(np.min(ratio)) if ratio.size else 1.0
            ratio_max = float(np.max(ratio)) if ratio.size else 1.0

            rayleigh = profile.get("rayleigh_range_um")
            use_adv = profile.get("use_advanced_refractive_correction", False)
            besseling_factor = refr_comp.get("besseling_factor")
            if isinstance(depth_factor, np.ndarray) and depth_factor.size > 0:
                depth_min = float(np.min(depth_factor))
                depth_max = float(np.max(depth_factor))
            else:
                depth_min = depth_max = 1.0

            summary_lines = [
                f"Stage z: {z_positions[0]:+.2f} … {z_positions[-1]:+.2f} µm ({z_positions.size} Slices)",
                f"Probe z (korrigiert): {sample_min:+.2f} … {sample_max:+.2f} µm",
                f"Intensitätsskala: {intensity_min:.2f} – {intensity_max:.2f}",
                f"σx/σy Verhältnis: {ratio_min:.2f} – {ratio_max:.2f}",
                f"Refraktive Korrektur: {'aktiv' if use_adv else 'Legacy'}"
            ]

            if besseling_factor is not None:
                summary_lines.append(f"Besseling-Faktor: {float(besseling_factor):.3f}")
            summary_lines.append(f"Deckglas-Faktor: {depth_min:.3f} – {depth_max:.3f}")

            if rayleigh and np.isfinite(rayleigh):
                summary_lines.append(f"Rayleigh-Range: {rayleigh:.2f} µm")

            self.z_stack_summary_var.set("\n".join(summary_lines))
            if hasattr(self, "z_stack_summary_label"):
                self.z_stack_summary_label.config(fg="#2c3e50")
        except Exception as exc:
            self.z_stack_summary_var.set(f"Fehler bei Profilberechnung: {exc}")
            if hasattr(self, "z_stack_summary_label"):
                self.z_stack_summary_label.config(fg="#c0392b")

    def _register_z_profile_traces(self):
        """Registriert trace-Handler für alle Parameter, die das z-Profil beeinflussen."""
        if getattr(self, "_z_profile_traces_registered", False):
            return

        vars_to_trace = [
            self.z_min,
            self.z_max,
            self.z_step,
            self.astig_z0_um,
            self.astig_ax,
            self.astig_ay,
            self.astig_focus_offset_um,
            self.spherical_aberration_strength,
            self.axial_intensity_floor,
            self.use_advanced_refractive_correction,
            self.refractive_index_correction,
            self.n_oil,
            self.n_glass,
            self.n_polymer,
            self.NA,
            self.d_glass_um,
            self.wavelength_nm,
            self.z_amp_um,
            self.z_max_um
        ]

        for var in vars_to_trace:
            if hasattr(var, "trace_add"):
                var.trace_add("write", lambda *args: self._update_z_stack_summary())

        self._z_profile_traces_registered = True

    def _apply_thunderstorm_preset(self):
        """Setzt empfohlene Parameter für ThunderSTORM-Kalibrierung."""
        self.z_min.set(-0.6)
        self.z_max.set(0.6)
        self.z_step.set(0.02)
        self.astig_focus_offset_um.set(0.28)
        self.spherical_aberration_strength.set(0.32)
        self.axial_intensity_floor.set(0.18)
        self.use_advanced_refractive_correction.set(True)
        self.wavelength_nm.set(580.0)
        self._update_z_slices()
        self._update_status("🎯 ThunderSTORM Preset übernommen")

    def _show_z_profile_preview(self):
        """Öffnet eine Matplotlib-Vorschau des axialen PSF-Profils."""
        z_positions = self._compute_z_positions()
        if z_positions.size == 0:
            messagebox.showwarning(
                "Ungültiger Bereich",
                "Bitte gültigen z-Bereich konfigurieren bevor die Vorschau geöffnet wird."
            )
            return

        try:
            detector = self._create_custom_detector()
            profile = evaluate_z_profile(detector, z_positions, astigmatism=True)
        except Exception as exc:
            messagebox.showerror("Profilfehler", f"Profil konnte nicht berechnet werden:\n{exc}")
            return

        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            messagebox.showerror("Matplotlib fehlt", f"Bitte Matplotlib installieren:\n{exc}")
            return

        stage = z_positions
        sample = profile.get("z_corrected", stage)
        intensity = np.clip(profile.get("intensity_scale", np.ones_like(stage)), 0.0, 1.2)
        sigma_x = profile.get("sigma_x", np.ones_like(stage))
        sigma_y = profile.get("sigma_y", np.ones_like(stage))
        refr_comp = profile.get("refractive_components") or {}
        depth_factor = refr_comp.get("depth_factor", np.ones_like(stage))
        if not isinstance(depth_factor, np.ndarray):
            depth_factor = np.ones_like(stage)
        ratio = sigma_x / np.maximum(sigma_y, 1e-6)

        top = tk.Toplevel(self.root)
        top.title("Z-Profil Vorschau")
        top.geometry("720x560")

        fig = Figure(figsize=(6.5, 5.2), dpi=100)
        ax1 = fig.add_subplot(211)
        ax1.plot(stage, intensity, label="Intensität (normiert)", color="#27ae60")
        ax1.set_ylabel("Intensität")
        ax1.set_xlabel("Stage z [µm]")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc="upper left")

        ax1_twin = ax1.twinx()
        ax1_twin.plot(stage, sample, label="Probe z (korr.)", color="#2980b9", linestyle="--")
        ax1_twin.set_ylabel("Probe z [µm]")
        ax1_twin.legend(loc="upper right")

        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.plot(stage, sigma_x, label="σx [px]", color="#8e44ad")
        ax2.plot(stage, sigma_y, label="σy [px]", color="#c0392b")
        ax2.plot(stage, ratio, label="σx/σy", color="#f39c12", linestyle=":")
        ax2.plot(stage, depth_factor, label="Deckglas-Faktor", color="#16a085", linestyle="--")
        ax2.set_xlabel("Stage z [µm]")
        ax2.set_ylabel("PSF-Breite / Verhältnis")
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc="best")

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        top.canvas = canvas
        top.figure = fig

        ttk.Button(top, text="Schließen", command=top.destroy).pack(pady=6)

    def _set_status_ui(self, message: str, color: str = None):
        """Status-Update (UI-Thread)."""
        if color is None:
            color = self.COLOR_SUCCESS
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def _update_status(self, message: str, color: str = None):
        """
        Aktualisiert Status-Nachricht (thread-safe).

        Args:
            message: Status-Text
            color: Farbe (default: COLOR_SUCCESS)
        """
        if color is None:
            color = self.COLOR_SUCCESS
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
        """Batch-Modus: Generiert mehrere TIFFs mit verschiedenen Parametern."""
        import re
        import numpy as np

        self._update_status("📦 Starte Batch-Modus...")
        self._update_progress(5)

        # RF-Training wurde aus Batch entfernt - verwende den dedizierten RF Tab!

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
        folder_structure = self.batch_folder_structure.get()  # "by_repeat", "by_polytime", or "flat"

        # Output-Verzeichnis
        base_dir = Path(self.output_dir.get())
        base_dir.mkdir(parents=True, exist_ok=True)

        # Gesamtzahl Tasks
        total_tasks = len(poly_times) * repeats
        current_task = 0

        # Erstelle Detector
        detector = self._create_custom_detector()

        # Batch-Loop
        for repeat in range(1, repeats + 1):
            for t_poly in poly_times:
                current_task += 1
                progress = int((current_task / total_tasks) * 90) + 5

                # Ordnerstruktur basierend auf gewähltem Modus
                if folder_structure == "by_repeat":
                    output_dir = base_dir / f"repeat_{repeat}"
                elif folder_structure == "by_polytime":
                    output_dir = base_dir / f"t{int(t_poly):03d}min"
                else:  # flat
                    output_dir = base_dir
                output_dir.mkdir(parents=True, exist_ok=True)

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

                # Metadata Export
                if self.export_json.get() or self.export_txt.get() or self.export_csv.get():
                    metadata = sim.get_metadata()
                    exporter = MetadataExporter(str(output_dir))
                    base_name = filepath.stem

                    if self.export_json.get():
                        exporter.export_json(metadata, base_name)
                    if self.export_txt.get():
                        exporter.export_txt(metadata, base_name)
                    if self.export_csv.get():
                        exporter.export_csv_row(metadata, base_name)

        self._update_progress(100)
        self._update_status(f"✅ Batch fertig! {total_tasks} TIFFs erstellt.")

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
                "wavelength_nm": self.wavelength_nm.get(),
                "on_mean_frames": self.on_mean_frames.get(),
                "off_mean_frames": self.off_mean_frames.get(),
                "bleach_prob_per_frame": self.bleach_prob.get(),
                "z_amp_um": self.z_amp_um.get(),
                "z_max_um": self.z_max_um.get(),
                "astig_z0_um": self.astig_z0_um.get(),
                "astig_coeffs": {"A_x": self.astig_ax.get(), "B_x": 0.0,
                               "A_y": self.astig_ay.get(), "B_y": 0.0},
                "astig_focus_offset_um": self.astig_focus_offset_um.get(),
                "spherical_aberration_strength": self.spherical_aberration_strength.get(),
                "axial_intensity_floor": self.axial_intensity_floor.get(),
                # Legacy: Einfache Brechungsindex-Korrektur
                "refractive_index_correction": self.refractive_index_correction.get(),
                # ERWEITERTE Brechungsindex-Korrektur (NEU!)
                "use_advanced_refractive_correction": self.use_advanced_refractive_correction.get(),
                "n_oil": self.n_oil.get(),
                "n_glass": self.n_glass.get(),
                "n_polymer": self.n_polymer.get(),
                "NA": self.NA.get(),
                "d_glass_um": self.d_glass_um.get(),
                # Illumination Gradient
                "illumination_gradient_strength": self.illumination_gradient_strength.get(),
                "illumination_gradient_type": self.illumination_gradient_type.get(),
                # Comonomer-Beschleunigungsfaktor
                "polymerization_acceleration_factor": self.polymerization_acceleration_factor.get()
            }
        )

        return custom

    # ========================================================================
    # RF TRAINING TAB (DEDICATED MODE - NEU!)
    # ========================================================================

    def _create_rf_training_tab(self):
        """Erstellt den dedizierten RF-Training Tab."""

        tk.Label(
            self.rf_training_tab,
            text="🌲 DEDIZIERTES RF-TRAINING",
            font=("Arial", 16, "bold"),
            fg="#16a085"
        ).pack(pady=10)

        tk.Label(
            self.rf_training_tab,
            text="⚡ Trainiere einen extrem leistungsstarken Random Forest OHNE TIFF-Generierung",
            font=("Arial", 10, "bold"),
            fg="#27ae60"
        ).pack()

        tk.Label(
            self.rf_training_tab,
            text="🎯 Generiert nur Trajektorien + Metadata → Schneller & effizienter!",
            font=("Arial", 9),
            fg="#7f8c8d"
        ).pack(pady=2)

        # Scrollable Frame
        scroll_frame = ScrollableFrame(self.rf_training_tab)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        container = scroll_frame.scrollable_frame

        # ====================================================================
        # OUTPUT DIRECTORY
        # ====================================================================
        output_frame = ttk.LabelFrame(container, text="📁 Output-Verzeichnis", padding=10)
        output_frame.pack(fill=tk.X, padx=5, pady=5)

        dir_frame = tk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=5)

        tk.Entry(dir_frame, textvariable=self.rf_dedicated_output_dir, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_rf_output_dir).pack(side=tk.LEFT, padx=5)

        # ====================================================================
        # TRAJECTORY GENERATION
        # ====================================================================
        traj_frame = ttk.LabelFrame(container, text="🎬 Trajektorien-Generierung", padding=10)
        traj_frame.pack(fill=tk.X, padx=5, pady=5)

        # Num Frames
        frames_frame = tk.Frame(traj_frame)
        frames_frame.pack(fill=tk.X, pady=2)
        tk.Label(frames_frame, text="Track-Länge (Frames):", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(frames_frame, from_=100, to=1000, increment=50,
                   textvariable=self.rf_dedicated_num_frames, width=15).pack(side=tk.LEFT, padx=5)
        ToolTip(frames_frame, "Lange Tracks = mehr Features pro Track\n300-500 empfohlen")

        # Tracks per Type
        tracks_frame = tk.Frame(traj_frame)
        tracks_frame.pack(fill=tk.X, pady=2)
        tk.Label(tracks_frame, text="Tracks pro Diffusionsart:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(tracks_frame, from_=100, to=10000, increment=100,
                   textvariable=self.rf_dedicated_tracks_per_type, width=15).pack(side=tk.LEFT, padx=5)
        ToolTip(tracks_frame, "1000-5000 empfohlen für starkes Modell\nJe mehr desto besser!")

        # Poly Times
        poly_frame = tk.Frame(traj_frame)
        poly_frame.pack(fill=tk.X, pady=2)
        tk.Label(poly_frame, text="Polymerisationszeiten [min]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(poly_frame, textvariable=self.rf_dedicated_poly_times, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ToolTip(poly_frame, "Komma-separiert: z.B. '0, 30, 60, 90, 120, 180'\nMehr Werte = robusteres Modell über alle Polygrade!")

        # Frame Rate
        framerate_frame = tk.Frame(traj_frame)
        framerate_frame.pack(fill=tk.X, pady=2)
        tk.Label(framerate_frame, text="Frame Rate [Hz]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(framerate_frame, from_=1, to=100, increment=1,
                   textvariable=self.rf_dedicated_frame_rate, width=15, format='%.1f').pack(side=tk.LEFT, padx=5)

        # D_initial
        d_frame = tk.Frame(traj_frame)
        d_frame.pack(fill=tk.X, pady=2)
        tk.Label(d_frame, text="D_initial [µm²/s]:", width=25, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(d_frame, from_=0.01, to=2.0, increment=0.01,
                   textvariable=self.rf_dedicated_d_initial, width=15, format='%.3f').pack(side=tk.LEFT, padx=5)

        # ====================================================================
        # RF HYPERPARAMETERS
        # ====================================================================
        rf_hyper_frame = ttk.LabelFrame(container, text="🌲 Random Forest Hyperparameter", padding=10)
        rf_hyper_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            rf_hyper_frame,
            text="💪 POWER-DEFAULTS: 4096 Bäume, Multi-Window, Poly-Feature integriert!",
            font=("Arial", 9, "bold"),
            fg="#16a085"
        ).pack(pady=5)

        # N Estimators
        estimators_frame = tk.Frame(rf_hyper_frame)
        estimators_frame.pack(fill=tk.X, pady=2)
        tk.Label(estimators_frame, text="Anzahl Bäume (n_estimators):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(estimators_frame, from_=512, to=8192, increment=512,
                   textvariable=self.rf_dedicated_n_estimators, width=15).pack(side=tk.LEFT, padx=5)
        ToolTip(estimators_frame, "4096-8192 = Extrem stark!\n2048-4096 = Sehr gut (Standard)\n1024 = Gut")

        # Max Depth
        depth_frame = tk.Frame(rf_hyper_frame)
        depth_frame.pack(fill=tk.X, pady=2)
        tk.Label(depth_frame, text="Max. Tiefe (max_depth):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(depth_frame, from_=10, to=50, increment=5,
                   textvariable=self.rf_dedicated_max_depth, width=15).pack(side=tk.LEFT, padx=5)
        ToolTip(depth_frame, "30-40 = Sehr flexibel\n20-30 = Gut regularisiert (Standard)")

        # Min Leaf
        leaf_frame = tk.Frame(rf_hyper_frame)
        leaf_frame.pack(fill=tk.X, pady=2)
        tk.Label(leaf_frame, text="Min Samples/Blatt:", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(leaf_frame, from_=1, to=20, increment=1,
                   textvariable=self.rf_dedicated_min_leaf, width=15).pack(side=tk.LEFT, padx=5)

        # Min Split
        split_frame = tk.Frame(rf_hyper_frame)
        split_frame.pack(fill=tk.X, pady=2)
        tk.Label(split_frame, text="Min Samples/Split:", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(split_frame, from_=2, to=40, increment=2,
                   textvariable=self.rf_dedicated_min_split, width=15).pack(side=tk.LEFT, padx=5)

        # Max Samples
        samples_frame = tk.Frame(rf_hyper_frame)
        samples_frame.pack(fill=tk.X, pady=2)
        tk.Label(samples_frame, text="Max Samples/Baum (0-1):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(samples_frame, from_=0.5, to=1.0, increment=0.05,
                   textvariable=self.rf_dedicated_max_samples, width=15, format='%.2f').pack(side=tk.LEFT, padx=5)

        # Window Sizes
        windows_frame = tk.Frame(rf_hyper_frame)
        windows_frame.pack(fill=tk.X, pady=2)
        tk.Label(windows_frame, text="Window-Größen (Frames):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(windows_frame, textvariable=self.rf_dedicated_window_sizes, width=20).pack(side=tk.LEFT, padx=5)
        ToolTip(windows_frame, "Komma-separiert: '32, 48, 64, 96'\nMulti-Window = robuster!")

        # Step Fraction
        step_frame = tk.Frame(rf_hyper_frame)
        step_frame.pack(fill=tk.X, pady=2)
        tk.Label(step_frame, text="Sliding Step Fraction (0-1):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(step_frame, from_=0.1, to=1.0, increment=0.1,
                   textvariable=self.rf_dedicated_step_fraction, width=15, format='%.2f').pack(side=tk.LEFT, padx=5)
        ToolTip(step_frame, "0.5 = 50% Overlap (Standard)\n1.0 = Kein Overlap")

        # Info Box
        info_frame = ttk.LabelFrame(container, text="ℹ️ Trainings-Strategie", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=10)

        tk.Label(
            info_frame,
            text="✨ Der RF wird über ALLE Polymerisationszeiten hinweg trainiert!\n"
                 "→ Robuste Klassifikation für alle Diffusionsarten, unabhängig vom Polygrad",
            font=("Arial", 9),
            fg="#2c3e50",
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=5)

        # ====================================================================
        # TRAINING BUTTON
        # ====================================================================
        button_frame = tk.Frame(container)
        button_frame.pack(pady=20)

        self.rf_training_button = tk.Button(
            button_frame,
            text="🚀 RF-TRAINING STARTEN",
            font=("Arial", 14, "bold"),
            bg="#16a085",
            fg="white",
            padx=30,
            pady=15,
            command=self._start_dedicated_rf_training
        )
        self.rf_training_button.pack()

        # Info
        tk.Label(
            container,
            text="ℹ️  Das Training generiert KEINE TIFFs - nur Trajektorien-Metadata für maximale Geschwindigkeit!\n"
                 "📦 Output: rf_model.joblib, feature_names.json, RF_USAGE_GUIDE.md",
            font=("Arial", 9),
            fg="#7f8c8d",
            justify=tk.CENTER
        ).pack(pady=10)

    # ========================================================================
    # TRACK ANALYSIS TAB
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

    # ========================================================================
    # RF TRAINING HELPERS
    # ========================================================================

    def _browse_rf_output_dir(self):
        """Browse for RF training output directory."""
        path = filedialog.askdirectory(
            title="Wähle Output-Verzeichnis für RF-Training"
        )
        if path:
            self.rf_dedicated_output_dir.set(path)

    def _start_dedicated_rf_training(self):
        """Startet das dedizierte RF-Training in einem separaten Thread."""
        if self.is_running:
            messagebox.showwarning("Training läuft", "Es läuft bereits ein Training!")
            return

        # Validation
        output_dir = Path(self.rf_dedicated_output_dir.get())
        if not output_dir.parent.exists():
            messagebox.showerror("Fehler", f"Parent-Ordner existiert nicht:\n{output_dir.parent}")
            return

        # Parse poly times
        try:
            poly_times_str = self.rf_dedicated_poly_times.get()
            poly_times = [float(t.strip()) for t in poly_times_str.split(",") if t.strip()]
            if not poly_times:
                raise ValueError("Keine gültigen Polymerisationszeiten!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Ungültige Polymerisationszeiten:\n{e}")
            return

        # Parse window sizes
        try:
            window_str = self.rf_dedicated_window_sizes.get()
            window_sizes = [int(w.strip()) for w in window_str.split(",") if w.strip()]
            if not window_sizes:
                raise ValueError("Keine gültigen Window-Größen!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Ungültige Window-Größen:\n{e}")
            return

        # Disable button
        self.rf_training_button.config(state=tk.DISABLED)
        self.is_running = True

        # Start training thread
        self.simulation_thread = threading.Thread(
            target=self._run_dedicated_rf_training,
            args=(output_dir, poly_times, window_sizes),
            daemon=True
        )
        self.simulation_thread.start()

    def _run_dedicated_rf_training(self, output_dir: Path, poly_times: List[float], window_sizes: List[int]):
        """Führt das dedizierte RF-Training aus (läuft in separatem Thread)."""
        import time
        start_time = time.time()

        try:
            self._update_status("🌲 Starte dediziertes RF-Training...")
            self._update_progress(5)

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Import dependencies
            from tiff_simulator_v3 import TIFFSimulatorOptimized, DetectorPreset, TrajectoryGenerator
            from rf_trainer import RandomForestTrainer, RFTrainingConfig

            # Get parameters from GUI
            num_frames = self.rf_dedicated_num_frames.get()
            tracks_per_type = self.rf_dedicated_tracks_per_type.get()
            frame_rate = self.rf_dedicated_frame_rate.get()
            d_initial = self.rf_dedicated_d_initial.get()

            # Create detector
            detector = self._create_custom_detector()

            # RF Config
            rf_config = RFTrainingConfig(
                window_sizes=tuple(window_sizes),
                step_size_fraction=self.rf_dedicated_step_fraction.get(),
                n_estimators=self.rf_dedicated_n_estimators.get(),
                max_depth=self.rf_dedicated_max_depth.get(),
                min_samples_leaf=self.rf_dedicated_min_leaf.get(),
                min_samples_split=self.rf_dedicated_min_split.get(),
                max_samples=self.rf_dedicated_max_samples.get(),
            )

            self._update_status(f"🎬 Generiere Trajektorien für {len(poly_times)} Polymerisationsgrade...")
            self._update_progress(10)

            # Initialize RF Trainer (standard, poly-feature handling wird danach gemacht)
            rf_trainer = RandomForestTrainer(output_dir, rf_config)

            total_tracks = len(poly_times) * 4 * tracks_per_type  # 4 diffusion types
            processed_tracks = 0

            # Generate trajectories for each poly time
            for poly_idx, t_poly in enumerate(poly_times):
                # Create trajectory generator
                traj_gen = TrajectoryGenerator(
                    D_initial=d_initial,
                    t_poly_min=t_poly,
                    frame_rate_hz=frame_rate,
                    pixel_size_um=detector.pixel_size_um,
                    enable_switching=True,
                    polymerization_acceleration_factor=detector.metadata.get("polymerization_acceleration_factor", 1.0)
                )

                # Generate tracks for each diffusion type
                for diffusion_type in ["normal", "subdiffusion", "confined", "superdiffusion"]:
                    # Generate multiple tracks
                    for track_idx in range(tracks_per_type):
                        # Detailed status update (every 10 tracks)
                        if track_idx % 10 == 0:
                            status_msg = f"🎬 Trajektorien: {diffusion_type} @ t={t_poly}min | Track {track_idx+1}/{tracks_per_type} | Gesamt: {processed_tracks}/{total_tracks}"
                            self._update_status(status_msg)
                            tracks_progress = 10 + int((processed_tracks / total_tracks) * 65)
                            self._update_progress(tracks_progress)

                        # Random start position
                        start_pos = (
                            np.random.uniform(10, 100),  # x [µm]
                            np.random.uniform(10, 100),  # y [µm]
                            np.random.uniform(-0.5, 0.5)  # z [µm]
                        )

                        # Generate trajectory
                        trajectory, switch_log = traj_gen.generate_trajectory(
                            start_pos,
                            num_frames,
                            diffusion_type=diffusion_type
                        )

                        # Create metadata (similar to TIFF metadata)
                        metadata = {
                            "diffusion": {
                                "t_poly_min": t_poly,
                                "frame_rate_hz": frame_rate,
                                "D_initial": d_initial,
                            },
                            "trajectories": [{
                                "positions": trajectory,
                                "diffusion_type": diffusion_type,
                                "switch_log": switch_log,
                                "num_switches": len(switch_log)
                            }]
                        }

                        # Feed to RF trainer
                        rf_trainer.update_with_metadata(metadata)
                        processed_tracks += 1

            # Finalize and train RF
            self._update_status(f"🧮 Berechne Features aus {processed_tracks} Trajektorien...")
            self._update_progress(75)

            # Feature calculation happens during update_with_metadata
            self._update_status("✅ Features berechnet! Starte Random Forest Training...")
            self._update_progress(78)

            self._update_status(f"🌲 Trainiere Random Forest ({rf_config.n_estimators} Bäume)...")
            self._update_progress(80)

            # This is where the heavy computation happens
            rf_info = rf_trainer.finalize()

            self._update_status("💾 Speichere Modell und Metriken...")
            self._update_progress(92)

            self._update_status("📝 Exportiere Dokumentation...")
            self._update_progress(95)

            # Export usage guide
            self._export_rf_usage_guide(output_dir, rf_info, rf_config)

            # Done
            elapsed = time.time() - start_time
            self._update_progress(100)
            self._update_status(f"✅ RF-Training abgeschlossen! ({elapsed:.1f}s)")

            # Show summary
            def _show_summary():
                summary = f"""
🎉 RF-TRAINING ERFOLGREICH!

📊 STATISTIKEN:
   Samples: {rf_info.get('samples', 0)}
   Labels: {rf_info.get('labels', {})}

🌲 MODELL:
   Bäume: {rf_config.n_estimators}
   Tiefe: {rf_config.max_depth}
   Training Accuracy: {rf_info.get('training_accuracy', 0):.4f}
   OOB Score: {rf_info.get('oob_score', 0):.4f}

📦 OUTPUT:
   {output_dir}

✅ Dateien:
   - rf_model.joblib (Trainiertes Modell)
   - rf_training_features.csv (Feature-Daten)
   - rf_training_summary.json (Metriken)
   - RF_USAGE_GUIDE.md (Dokumentation)

⏱️  Zeit: {elapsed:.1f}s
"""
                messagebox.showinfo("RF-Training abgeschlossen", summary)

            self.root.after(0, _show_summary)

        except Exception as e:
            import traceback
            error_msg = f"❌ FEHLER beim RF-Training:\n{e}\n\n{traceback.format_exc()}"
            print(error_msg)
            self._update_status(f"❌ Fehler: {e}")

            def _show_error():
                messagebox.showerror("RF-Training Fehler", str(e))

            self.root.after(0, _show_error)

        finally:
            self.is_running = False
            self.root.after(0, lambda: self.rf_training_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self._update_progress(0))

    def _export_rf_usage_guide(self, output_dir: Path, rf_info: Dict, rf_config):
        """Exportiert umfassende Nutzungs-Dokumentation für das RF-Modell."""
        guide_path = output_dir / "RF_USAGE_GUIDE.md"

        feature_names = rf_info.get("feature_names", [])

        guide_content = f"""# 🌲 Random Forest Model - Usage Guide

## 📦 Modell-Übersicht

Dieses RF-Modell wurde mit **{rf_info.get('samples', 0)} Samples** trainiert und erreicht:
- **Training Accuracy**: {rf_info.get('training_accuracy', 0):.4f}
- **OOB Score**: {rf_info.get('oob_score', 0):.4f}

### Klassifizierte Diffusionstypen:
{self._format_labels_md(rf_info.get('labels', {}))}

### Modell-Parameter:
- **Anzahl Bäume**: {rf_config.n_estimators}
- **Max. Tiefe**: {rf_config.max_depth}
- **Min Samples/Blatt**: {rf_config.min_samples_leaf}
- **Min Samples/Split**: {rf_config.min_samples_split}
- **Window-Größen**: {rf_config.window_sizes}

---

## 🔢 Features (Total: {len(feature_names)})

Das Modell verwendet **{len(feature_names)} Features** pro Track-Fenster, kategorisiert nach Typ:

### 📋 Feature-Übersicht

| # | Feature Name | Kategorie | Beschreibung |
|---|--------------|-----------|--------------|
| 1-6 | mean/std/median/mad/max/min_step_xy | Step Stats | Laterale Schrittweiten-Statistiken |
| 7-8 | mean/std_step_z | Step Stats | Axiale Schrittweiten-Statistiken |
| 9-12 | msd_lag1/2/4/8 | MSD | Mean Squared Displacement bei verschiedenen Lags |
| 13 | msd_loglog_slope | **Anomal** | Anomaler Exponent α (Normal=1.0, Sub<1.0, Super>1.0) |
| 14 | straightness | Geometry | End-to-End-Distanz / Pfadlänge |
| 15 | confinement_radius | Confinement | Maximale Distanz vom Schwerpunkt |
| 16-17 | radius/asymmetry_of_gyration | Geometry | Gyrationsradius und Asymmetrie |
| 18-19 | turning_angle_mean/std | Direction | Wendewinkiel-Statistiken |
| 20-21 | step_p90/p10 | Step Stats | 90. und 10. Perzentil der Schritte |
| 22 | bounding_box_area | Geometry | XY Bounding Box Fläche |
| 23 | axial_range | Geometry | Z-Range (max - min) |
| 24 | directional_persistence | Direction | Richtungspersistenz |
| 25 | velocity_autocorr | Dynamics | Geschwindigkeits-Autokorrelation |
| 26-27 | step_skewness/kurtosis | Distribution | Schiefe und Kurtosis der Schrittverteilung |
| **28-29** | **msd_ratio_lag4/8** | **Anomal 🔥** | **MSD(lag)/(lag\*MSD(1)) - Schlüssel für Sub/Super!** |
| **30-32** | **d_eff_lag1/4/8** | **Anomal 🔥** | **Effektive Diffusionskoeffizienten** |
| **33** | **d_eff_variation** | **Anomal 🔥** | **Variation in D_eff (hoch bei anomaler Diffusion)** |
| **34** | **displacement_kurtosis** | **Anomal 🔥** | **Non-Gaussianität (Normal≈3.0)** |

### 🎯 Features 28-34: Anomale Diffusion Detektoren

Diese **7 neuen Features** sind speziell für die Erkennung von anomaler Diffusion optimiert:

**MSD-Ratios (28-29)**:
- Normale Diffusion: `MSD(lag) = 4*D*lag` → Ratio ≈ 1.0
- Subdiffusion: Ratio < 1.0 (MSD wächst sublinear)
- Superdiffusion: Ratio > 1.0 (MSD wächst superlinear)

**Effektive D-Koeffizienten (30-33)**:
- `D_eff(lag) = MSD(lag) / (4*lag)`
- Normal: D_eff ist **konstant** → niedrige Variation
- Anomal: D_eff **variiert** mit lag → hohe Variation

**Displacement Kurtosis (34)**:
- Misst Abweichung von Gauß-Verteilung
- Normal ≈ 3.0, Anomal weicht ab

{self._format_feature_list_md(feature_names)}

---

## 🚀 Nutzung in Python

### 1. Modell laden

```python
import joblib
import numpy as np
from pathlib import Path

# Lade trainiertes Modell
model_path = Path("{output_dir}") / "rf_model.joblib"
rf_model = joblib.load(model_path)

print(f"Modell geladen: {{type(rf_model)}}")
print(f"Classes: {{rf_model.classes_}}")
```

### 2. Features aus Trajektorie berechnen

```python
def calculate_features(positions, frame_rate_hz=20.0):
    \"\"\"
    Berechnet alle {len(feature_names)} Features aus einer Trajektorie.

    Parameters:
    -----------
    positions : np.ndarray
        Trajektorie (N_frames, 3) [µm] - (x, y, z)
    frame_rate_hz : float
        Frame-Rate [Hz]

    Returns:
    --------
    np.ndarray : Feature-Vektor (shape: ({len(feature_names)},))
    \"\"\"

    # Berechne Schritte (lateral x,y)
    steps_xy = np.sqrt(np.sum(np.diff(positions[:, :2], axis=0)**2, axis=1))
    steps_z = np.abs(np.diff(positions[:, 2]))

    # 1-6: Step Statistics (XY)
    mean_step_xy = np.mean(steps_xy)
    std_step_xy = np.std(steps_xy)
    median_step_xy = np.median(steps_xy)
    mad_step_xy = np.median(np.abs(steps_xy - median_step_xy))
    max_step_xy = np.max(steps_xy)
    min_step_xy = np.min(steps_xy)

    # 7-8: Step Statistics (Z)
    mean_step_z = np.mean(steps_z)
    std_step_z = np.std(steps_z)

    # 9-12: Mean Squared Displacement (MSD)
    def msd_at_lag(pos, lag):
        displacements = pos[lag:] - pos[:-lag]
        return np.mean(np.sum(displacements**2, axis=1))

    msd_lag1 = msd_at_lag(positions, 1)
    msd_lag2 = msd_at_lag(positions, 2)
    msd_lag4 = msd_at_lag(positions, min(4, len(positions)//2))
    msd_lag8 = msd_at_lag(positions, min(8, len(positions)//2))

    # 13: MSD Log-Log Slope (Anomaler Exponent)
    lags = np.array([1, 2, 4, 8])
    msds = np.array([msd_lag1, msd_lag2, msd_lag4, msd_lag8])
    if np.all(msds > 0):
        msd_loglog_slope = np.polyfit(np.log(lags), np.log(msds), 1)[0]
    else:
        msd_loglog_slope = 0.0

    # 14: Straightness
    start_end_dist = np.linalg.norm(positions[-1] - positions[0])
    path_length = np.sum(steps_xy)
    straightness = start_end_dist / max(path_length, 1e-9)

    # 15: Confinement Radius
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    confinement_radius = np.max(distances)

    # 16-17: Radius of Gyration + Asymmetry
    rg_squared = np.mean(distances**2)
    radius_of_gyration = np.sqrt(rg_squared)
    gyration_asymmetry = np.std(distances) / max(radius_of_gyration, 1e-9)

    # 18-19: Turning Angles
    if len(steps_xy) > 1:
        vectors = np.diff(positions[:, :2], axis=0)
        dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)
        magnitudes = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
        angles = np.arccos(np.clip(dot_products / np.maximum(magnitudes, 1e-9), -1, 1))
        turning_angle_mean = np.mean(angles)
        turning_angle_std = np.std(angles)
    else:
        turning_angle_mean = 0.0
        turning_angle_std = 0.0

    # 20-21: Step Percentiles
    step_p90 = np.percentile(steps_xy, 90)
    step_p10 = np.percentile(steps_xy, 10)

    # 22: Bounding Box Area
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    bounding_box_area = x_range * y_range

    # 23: Axial Range (Z)
    axial_range = np.ptp(positions[:, 2])

    # 24: Directional Persistence
    if len(positions) > 2:
        velocities = np.diff(positions[:, :2], axis=0)
        speed = np.linalg.norm(velocities, axis=1)
        if np.any(speed > 0):
            directions = velocities / np.maximum(speed[:, None], 1e-9)
            persistence = np.mean([np.dot(directions[i], directions[i+1])
                                  for i in range(len(directions)-1)])
        else:
            persistence = 0.0
    else:
        persistence = 0.0
    directional_persistence = persistence

    # 25: Velocity Autocorrelation
    if len(steps_xy) > 1:
        velocities = steps_xy * frame_rate_hz
        mean_vel = np.mean(velocities)
        acf = np.correlate(velocities - mean_vel, velocities - mean_vel, mode='valid')[0]
        acf /= max(np.var(velocities) * len(velocities), 1e-9)
        velocity_autocorr = acf
    else:
        velocity_autocorr = 0.0

    # 26-27: Step Skewness + Kurtosis
    from scipy import stats
    step_skewness = float(stats.skew(steps_xy))
    step_kurtosis = float(stats.kurtosis(steps_xy))

    # ============================================
    # NEUE FEATURES (28-34) für Anomale Diffusion
    # ============================================

    # 28-29: MSD Ratios (Schlüssel-Features!)
    # Für normale Diffusion: MSD(lag) = 4*D*lag → Ratio = 1.0
    # Subdiffusion: Ratio < 1.0 (sublinear)
    # Superdiffusion: Ratio > 1.0 (superlinear)
    msd_ratio_lag4 = (msd_lag4 / (4.0 * msd_lag1 + 1e-12)) if msd_lag1 > 0 else 1.0
    msd_ratio_lag8 = (msd_lag8 / (8.0 * msd_lag1 + 1e-12)) if msd_lag1 > 0 else 1.0

    # 30-32: Effektive Diffusionskoeffizienten
    # D_eff = MSD(lag) / (4 * lag) für 2D-Diffusion
    # Anomale Diffusion zeigt lag-abhängiges D!
    d_eff_lag1 = msd_lag1 / 4.0
    d_eff_lag4 = msd_lag4 / 16.0
    d_eff_lag8 = msd_lag8 / 32.0

    # 33: Variation im effektiven D
    # Normale Diffusion: D_eff konstant → Variation niedrig
    # Anomale Diffusion: D_eff variiert mit lag → Variation hoch
    d_eff_variation = float(np.std([d_eff_lag1, d_eff_lag4, d_eff_lag8]))

    # 34: Displacement Kurtosis (Non-Gaussianity)
    # Normale Diffusion: ~3.0 (Gauß-Verteilung)
    # Anomale Diffusion: weicht ab
    displacements_xy = steps_xy  # Already computed as step distances
    if len(displacements_xy) > 3:
        disp_mean = np.mean(displacements_xy)
        disp_std = np.std(displacements_xy) + 1e-12
        displacement_kurtosis = float(np.mean(((displacements_xy - disp_mean) / disp_std) ** 4))
    else:
        displacement_kurtosis = 3.0

    # Feature-Vektor zusammenbauen (ALLE 34 Features!)
    features = [
        mean_step_xy, std_step_xy, median_step_xy, mad_step_xy,
        max_step_xy, min_step_xy, mean_step_z, std_step_z,
        msd_lag1, msd_lag2, msd_lag4, msd_lag8,
        msd_loglog_slope, straightness, confinement_radius,
        radius_of_gyration, gyration_asymmetry,
        turning_angle_mean, turning_angle_std,
        step_p90, step_p10, bounding_box_area, axial_range,
        directional_persistence, velocity_autocorr,
        step_skewness, step_kurtosis,
        # NEUE Features für anomale Diffusion
        msd_ratio_lag4, msd_ratio_lag8,
        d_eff_lag1, d_eff_lag4, d_eff_lag8,
        d_eff_variation, displacement_kurtosis
    ]

    return np.array(features, dtype=np.float32)
```

### Erklärung der neuen Features (28-34):

#### **MSD-Ratios** (Features 28-29) - SEHR WICHTIG!
Für normale 2D-Diffusion gilt: `MSD(lag) = 4*D*lag`

- **Normal Diffusion**: `msd_ratio_lag4 ≈ 1.0` (linear)
- **Subdiffusion**: `msd_ratio_lag4 < 1.0` (MSD wächst langsamer als linear)
- **Superdiffusion**: `msd_ratio_lag4 > 1.0` (MSD wächst schneller als linear)

```python
# Beispiel-Werte:
# Normal:        msd_ratio_lag4 ≈ 0.98 - 1.02
# Subdiffusion:  msd_ratio_lag4 ≈ 0.5 - 0.8
# Superdiffusion: msd_ratio_lag4 ≈ 1.2 - 1.8
```

#### **Effektive Diffusionskoeffizienten** (Features 30-33)
`D_eff(lag) = MSD(lag) / (4 * lag)`

- **Normal**: D_eff ist konstant über alle lags
- **Anomal**: D_eff ändert sich mit lag!

Die **Variation** (Feature 33) ist der Schlüssel:
```python
# Normal:        d_eff_variation ≈ 0.001 - 0.01 (sehr klein)
# Subdiffusion:  d_eff_variation ≈ 0.05 - 0.15 (mittel)
# Superdiffusion: d_eff_variation ≈ 0.1 - 0.3 (groß)
```

#### **Displacement Kurtosis** (Feature 34)
Misst die "Schwere der Verteilungsenden":
- **Normal (Gauß)**: ≈ 3.0
- **Anomal**: weicht von 3.0 ab

```python
# Confined: oft > 3.0 (heavy tails)
# Superdiffusion: oft < 3.0 (light tails)
```

### 3. Vorhersage für neue Trajektorie

```python
# Beispiel-Trajektorie laden (z.B. aus TrackMate XML oder CSV)
trajectory = np.array([...])  # Shape: (N_frames, 3)

# Features berechnen
features = calculate_features(
    trajectory,
    frame_rate_hz=20.0
)

# Vorhersage
prediction = rf_model.predict([features])[0]
probabilities = rf_model.predict_proba([features])[0]

print(f"Predicted Label: {{prediction}}")
print(f"Probabilities: {{dict(zip(rf_model.classes_, probabilities))}}")
```

---

## 📊 Feature-Importanzen

```python
import matplotlib.pyplot as plt

# Feature-Importanzen aus Modell
importances = rf_model.feature_importances_
feature_names = {feature_names}

# Sortieren
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('RF Feature Importances')
plt.tight_layout()
plt.show()
```

---

## 🔗 Integration in andere Software

### TrackMate (Fiji/ImageJ)

1. Exportiere Tracks als XML
2. Parse XML mit Python (z.B. `xml.etree.ElementTree`)
3. Berechne Features pro Track
4. Klassifiziere mit RF-Modell
5. Füge Klassifikation zurück zu TrackMate hinzu

### MATLAB

```matlab
% Lade Modell (erfordert Python-Bridge)
py.joblib.load("{output_dir}/rf_model.joblib");

% Oder: Exportiere Entscheidungsbaum-Regeln und implementiere in MATLAB
```

---

## 📝 Lizenz & Zitation

Dieses Modell wurde generiert mit dem **Hyperrealistischen TIFF Simulator V4.0**.

Bei Verwendung in Publikationen bitte zitieren:
- TIFF Simulator V4.0 (2025)
- Random Forest Classifier für Diffusions-Klassifikation

---

**Erstellt am**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Modell-Pfad**: `{output_dir}/rf_model.joblib`
"""

        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)

        print(f"✅ RF Usage Guide exportiert: {guide_path}")

    def _format_labels_md(self, labels_dict):
        """Formatiert Labels für Markdown."""
        lines = []
        for label, count in sorted(labels_dict.items()):
            lines.append(f"- **{label}**: {count} Samples")
        return "\n".join(lines) if lines else "- (Keine Labels)"

    def _format_feature_list_md(self, feature_names):
        """Formatiert Feature-Liste für Markdown."""
        lines = []
        for idx, name in enumerate(feature_names, 1):
            lines.append(f"{idx}. `{name}`")
        return "\n".join(lines)

    # ========================================================================
    # ANALYSIS TAB HELPERS
    # ========================================================================

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
