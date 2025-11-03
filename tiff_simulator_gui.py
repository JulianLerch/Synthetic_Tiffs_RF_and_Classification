"""
ðŸŽ® HYPERREALISTISCHE TIFF-SIMULATOR GUI V3.0
=============================================

Benutzerfreundliche OberflÃ¤che mit:
âœ… Scrollbares Tab-Interface
âœ… Echtzeit Progress Bar
âœ… Batch-Modus Integration
âœ… Parameter-Validierung
âœ… Metadata-Export
âœ… Wissenschaftlich prÃ¤zise Dokumentation

Version: 3.0 - Oktober 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
from pathlib import Path
from datetime import datetime

try:
    from tiff_simulator_v3 import (
        TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    )
    from metadata_exporter import MetadataExporter
    from batch_simulator import BatchSimulator, PresetBatches
except ImportError as e:
    print(f"âŒ Import Error: {e}")
    print("   Bitte stelle sicher, dass alle Dateien im gleichen Ordner sind:")
    print("   - tiff_simulator_v3.py")
    print("   - metadata_exporter.py")
    print("   - batch_simulator.py")
    exit(1)


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


class TIFFSimulatorGUI:
    """Hauptfenster fÃ¼r TIFF-Simulation mit Progress Bar."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ðŸ”¬ Hyperrealistischer TIFF Simulator V3.0")
        self.root.geometry("900x750")
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
        
        # Thread fÃ¼r Simulation
        self.simulation_thread = None
        self.is_running = False
    
    def _init_variables(self):
        """Initialisiert alle Tkinter-Variablen."""
        
        # Detektor & Modus
        self.detector_var = tk.StringVar(value="TDI-G0")
        self.mode_var = tk.StringVar(value="single")
        self.sim_mode_var = tk.StringVar(value="polyzeit")
        
        # Output
        self.output_dir = tk.StringVar(value=str(Path.home() / "Desktop"))
        self.filename = tk.StringVar(value="simulation.tif")
        
        # Bild-Parameter
        self.image_width = tk.IntVar(value=128)
        self.image_height = tk.IntVar(value=128)
        self.num_spots = tk.IntVar(value=10)
        
        # Zeitreihen-Parameter
        self.t_poly = tk.DoubleVar(value=60.0)
        self.d_initial = tk.DoubleVar(value=4.0)
        self.num_frames = tk.IntVar(value=100)
        self.frame_rate = tk.DoubleVar(value=20.0)
        self.exposure_substeps = tk.IntVar(value=3)
        self.enable_photophysics = tk.BooleanVar(value=False)
        
        # z-Stack Parameter
        self.z_min = tk.DoubleVar(value=-1.0)
        self.z_max = tk.DoubleVar(value=1.0)
        self.z_step = tk.DoubleVar(value=0.1)
        
        # Batch Parameter
        self.batch_preset = tk.StringVar(value="quick")
        self.batch_repeats = tk.IntVar(value=3)
        self.batch_detector = tk.StringVar(value="TDI-G0")
        self.batch_custom_times = tk.StringVar(value="")  # e.g. 10,30,60
        self.batch_astig = tk.BooleanVar(value=False)
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

        # Export-Optionen
        self.export_metadata = tk.BooleanVar(value=True)
        self.export_json = tk.BooleanVar(value=True)
        self.export_txt = tk.BooleanVar(value=True)
        self.export_csv = tk.BooleanVar(value=True)
    
    def _create_widgets(self):
        """Erstellt alle GUI-Elemente."""
        
        # ====================================================================
        # HEADER
        # ====================================================================
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(
            header_frame,
            text="ðŸ”¬ Hyperrealistischer TIFF Simulator V3.0",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack(pady=5)
        
        tk.Label(
            header_frame,
            text="âœ¨ Mit Batch-Modus, Progress Bar & Metadata-Export",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#3498db"
        ).pack()
        
        # ====================================================================
        # SCROLLBARER HAUPTBEREICH
        # ====================================================================
        self.scrollable_container = ScrollableFrame(self.root)
        self.scrollable_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        main_frame = self.scrollable_container.scrollable_frame
        
        # ====================================================================
        # DETEKTOR PRESET
        # ====================================================================
        detector_frame = ttk.LabelFrame(main_frame, text="ðŸ“· Detektor", padding=10)
        detector_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = tk.Frame(detector_frame)
        btn_frame.pack()
        
        ttk.Radiobutton(
            btn_frame,
            text="ðŸ”µ TDI-G0 (~260 counts, 0.108 Âµm/px)",
            variable=self.detector_var,
            value="TDI-G0",
            command=self._apply_detector_preset
        ).pack(side=tk.LEFT, padx=10)
        
        ttk.Radiobutton(
            btn_frame,
            text="ðŸŸ¢ Tetraspecs (~300 counts, 0.160 Âµm/px)",
            variable=self.detector_var,
            value="Tetraspecs",
            command=self._apply_detector_preset
        ).pack(side=tk.LEFT, padx=10)
        
        # ====================================================================
        # SINGLE vs BATCH
        # ====================================================================
        mode_frame = ttk.LabelFrame(main_frame, text="ðŸŽ¯ Modus", padding=10)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="ðŸ“„ Single Simulation (1 TIFF)",
            variable=self.mode_var,
            value="single",
            command=self._toggle_mode_panels
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            mode_frame,
            text="ðŸ”„ Batch Simulation (mehrere TIFFs automatisch)",
            variable=self.mode_var,
            value="batch",
            command=self._toggle_mode_panels
        ).pack(anchor=tk.W, pady=2)
        
        # ====================================================================
        # NOTEBOOK fÃ¼r Parameter
        # ====================================================================
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Single Simulation
        self.single_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text="ðŸ“„ Single Simulation")
        self._create_single_tab()
        
        # Tab 2: Batch Simulation
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text="ðŸ”„ Batch Simulation")
        self._create_batch_tab()
        
        # Tab 3: Export-Optionen
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.export_tab, text="ðŸ’¾ Export & Metadata")
        self._create_export_tab()
        
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
            text="âš™ï¸ Bereit",
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
            text="ðŸš€ SIMULATION STARTEN",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
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
            text="âŒ Beenden",
            command=self.root.quit
        ).pack(side=tk.LEFT, padx=10)
        
        # Initial: Single-Modus
        self._toggle_mode_panels()
    
    def _create_single_tab(self):
        """Tab fÃ¼r Single Simulation."""
        
        # Simulationsmodus
        mode_frame = ttk.LabelFrame(self.single_tab, text="Simulationsmodus", padding=10)
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="â±ï¸ Polymerisationszeit (2D)",
            variable=self.sim_mode_var,
            value="polyzeit",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            mode_frame,
            text="â±ï¸ðŸ”º Polymerisationszeit + Astigmatismus (3D)",
            variable=self.sim_mode_var,
            value="polyzeit_astig",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            mode_frame,
            text="ðŸ“Š z-Stack Kalibrierung",
            variable=self.sim_mode_var,
            value="z_stack",
            command=self._update_mode_info
        ).pack(anchor=tk.W, pady=2)
        
        # Info-Text
        self.mode_info_text = scrolledtext.ScrolledText(
            mode_frame,
            height=4,
            width=70,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#ecf0f1",
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.mode_info_text.pack(pady=5, fill=tk.X)
        
        # Bild-Parameter
        img_frame = ttk.LabelFrame(self.single_tab, text="ðŸ–¼ï¸ Bild-Parameter", padding=10)
        img_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # GrÃ¶ÃŸe
        size_frame = tk.Frame(img_frame)
        size_frame.pack(fill=tk.X, pady=2)
        tk.Label(size_frame, text="Breite [px]:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(size_frame, from_=32, to=512, increment=32, 
                   textvariable=self.image_width, width=10).pack(side=tk.LEFT, padx=5)
        tk.Label(size_frame, text="HÃ¶he [px]:", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=(20,0))
        ttk.Spinbox(size_frame, from_=32, to=512, increment=32,
                   textvariable=self.image_height, width=10).pack(side=tk.LEFT, padx=5)
        
        # Spots
        spots_frame = tk.Frame(img_frame)
        spots_frame.pack(fill=tk.X, pady=2)
        tk.Label(spots_frame, text="Anzahl Spots:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(spots_frame, from_=1, to=100, increment=1,
                   textvariable=self.num_spots, width=10).pack(side=tk.LEFT, padx=5)
        
        # Zeitreihen-Parameter
        time_frame = ttk.LabelFrame(self.single_tab, text="â±ï¸ Zeitreihen", padding=10)
        time_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Polyzeit
        t_frame = tk.Frame(time_frame)
        t_frame.pack(fill=tk.X, pady=2)
        tk.Label(t_frame, text="Polyzeit [min]:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(t_frame, from_=0, to=240, increment=10,
                   textvariable=self.t_poly, width=10,
                   command=self._update_d_estimate).pack(side=tk.LEFT, padx=5)
        self.d_info_label = tk.Label(t_frame, text="", font=("Arial", 9), fg="#27ae60")
        self.d_info_label.pack(side=tk.LEFT, padx=10)
        
        # Frames
        frames_frame = tk.Frame(time_frame)
        frames_frame.pack(fill=tk.X, pady=2)
        tk.Label(frames_frame, text="Anzahl Frames:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(frames_frame, from_=10, to=2000, increment=10,
                   textvariable=self.num_frames, width=10).pack(side=tk.LEFT, padx=5)
        
        # Frame Rate
        rate_frame = tk.Frame(time_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        tk.Label(rate_frame, text="Frame Rate [Hz]:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(rate_frame, from_=1, to=100, increment=1,
                   textvariable=self.frame_rate, width=10).pack(side=tk.LEFT, padx=5)

        # D_initial
        d_frame = tk.Frame(time_frame)
        d_frame.pack(fill=tk.X, pady=2)
        tk.Label(d_frame, text="D_initial [ÂµmÂ²/s]:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(d_frame, from_=0.01, to=10.0, increment=0.05,
                   textvariable=self.d_initial, width=10,
                   format='%.2f', command=self._update_d_estimate).pack(side=tk.LEFT, padx=5)

        # Exposure substeps (Motion Blur)
        sub_frame = tk.Frame(time_frame)
        sub_frame.pack(fill=tk.X, pady=2)
        tk.Label(sub_frame, text="Exposure Substeps:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(sub_frame, from_=1, to=8, increment=1,
                   textvariable=self.exposure_substeps, width=10).pack(side=tk.LEFT, padx=5)
        
        # z-Stack Parameter
        z_frame = ttk.LabelFrame(self.single_tab, text="ðŸ“Š z-Stack", padding=10)
        z_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # z_min
        zmin_frame = tk.Frame(z_frame)
        zmin_frame.pack(fill=tk.X, pady=2)
        tk.Label(zmin_frame, text="z_min [Âµm]:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(zmin_frame, from_=-2.0, to=0.0, increment=0.1,
                   textvariable=self.z_min, width=10,
                   command=self._update_z_slices).pack(side=tk.LEFT, padx=5)
        
        # z_max
        zmax_frame = tk.Frame(z_frame)
        zmax_frame.pack(fill=tk.X, pady=2)
        tk.Label(zmax_frame, text="z_max [Âµm]:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(zmax_frame, from_=0.0, to=2.0, increment=0.1,
                   textvariable=self.z_max, width=10,
                   command=self._update_z_slices).pack(side=tk.LEFT, padx=5)
        
        # z_step
        zstep_frame = tk.Frame(z_frame)
        zstep_frame.pack(fill=tk.X, pady=2)
        tk.Label(zstep_frame, text="z_step [Âµm]:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(zstep_frame, from_=0.01, to=0.5, increment=0.01,
                   textvariable=self.z_step, width=10,
                   command=self._update_z_slices).pack(side=tk.LEFT, padx=5)
        self.z_slices_label = tk.Label(zstep_frame, text="", font=("Arial", 9), fg="#27ae60")
        self.z_slices_label.pack(side=tk.LEFT, padx=10)
        
        # Output
        output_frame = ttk.LabelFrame(self.single_tab, text="ðŸ’¾ Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Ordner
        dir_frame = tk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        tk.Label(dir_frame, text="Speicherort:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="ðŸ“", width=3, command=self._browse_dir).pack(side=tk.LEFT)
        
        # Dateiname
        file_frame = tk.Frame(output_frame)
        file_frame.pack(fill=tk.X, pady=2)
        tk.Label(file_frame, text="Dateiname:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(file_frame, textvariable=self.filename, width=40).pack(side=tk.LEFT, padx=5)
        
        # Initial Updates
        self._update_d_estimate()
        self._update_z_slices()
    
    def _create_batch_tab(self):
        """Tab fÃ¼r Batch Simulation."""
        
        # Preset Auswahl
        preset_frame = ttk.LabelFrame(self.batch_tab, text="ðŸŽ¯ Batch Preset", padding=10)
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            preset_frame,
            text="WÃ¤hle eine vordefinierte Batch-Konfiguration:",
            font=("Arial", 10)
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Radiobutton(
            preset_frame,
            text="ðŸš€ Quick Test (3 TIFFs, ~2 Min)",
            variable=self.batch_preset,
            value="quick"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            preset_frame,
            text="ðŸŽ“ Masterthesis (60+ TIFFs, ~45 Min)",
            variable=self.batch_preset,
            value="thesis"
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Radiobutton(
            preset_frame,
            text="ðŸ“„ Publication Quality (30 TIFFs, ~2 Std)",
            variable=self.batch_preset,
            value="publication"
        ).pack(anchor=tk.W, pady=2)
        
        # Info
        info_text = scrolledtext.ScrolledText(
            preset_frame,
            height=8,
            width=70,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#e8f4f8",
            relief=tk.FLAT
        )
        info_text.pack(pady=5, fill=tk.X)
        
        info_text.insert(1.0, 
            "BATCH-MODUS INFO:\n\n"
            "â€¢ Quick Test: 3 Polyzeiten (30, 60, 90 min), 64Ã—64 px, 50 Frames\n"
            "â€¢ Masterthesis: VollstÃ¤ndige Studie mit 6 Polyzeiten, TDI vs Tetraspecs,\n"
            "  3D-Simulationen, z-Stack Kalibrierung (3 Wiederholungen)\n"
            "â€¢ Publication Quality: Hohe AuflÃ¶sung (256Ã—256), 50 Spots, 500 Frames,\n"
            "  5 Wiederholungen fÃ¼r Statistik\n\n"
            "Alle TIFFs werden mit Metadata (JSON, TXT, CSV) exportiert!"
        )
        info_text.config(state=tk.DISABLED)
        
        # Custom Times (überschreibt Preset, falls gesetzt)
        custom_frame = ttk.LabelFrame(self.batch_tab, text="?? Custom Times", padding=10)
        custom_frame.pack(fill=tk.X, padx=10, pady=5)

        row_custom = tk.Frame(custom_frame)
        row_custom.pack(fill=tk.X, pady=2)
        tk.Label(row_custom, text="Zeiten [min] (z.B. 10,30,60):", width=30, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(row_custom, textvariable=self.batch_custom_times, width=30).pack(side=tk.LEFT, padx=5)

        row_det = tk.Frame(custom_frame)
        row_det.pack(fill=tk.X, pady=2)
        tk.Label(row_det, text="Detektor:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Radiobutton(row_det, text="TDI-G0", variable=self.batch_detector, value="TDI-G0").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(row_det, text="Tetraspecs", variable=self.batch_detector, value="Tetraspecs").pack(side=tk.LEFT, padx=5)

        row_ast = tk.Frame(custom_frame)
        row_ast.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(row_ast, text="Astigmatismus (3D) aktiv", variable=self.batch_astig).pack(anchor=tk.W)

        row_rep = tk.Frame(custom_frame)
        row_rep.pack(fill=tk.X, pady=2)
        tk.Label(row_rep, text="Wiederholungen:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(row_rep, from_=1, to=20, increment=1, textvariable=self.batch_repeats, width=10).pack(side=tk.LEFT, padx=5)

        hint = tk.Label(custom_frame, text="Hinweis: Bild-/Zeit-Parameter (Größe, Spots, Frames, Hz) werden aus dem Single-Tab übernommen.", font=("Arial", 9), fg="#7f8c8d")
        hint.pack(anchor=tk.W, pady=4)

        # Output
        output_frame = ttk.LabelFrame(self.batch_tab, text="ðŸ’¾ Output", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        dir_frame = tk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        tk.Label(dir_frame, text="Batch Output:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="ðŸ“", width=3, command=self._browse_dir).pack(side=tk.LEFT)
    
        # Random-Forest Training
        rf_frame = ttk.LabelFrame(self.batch_tab, text="ðŸŒ² Random-Forest Training", padding=10)
        rf_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Checkbutton(
            rf_frame,
            text="Random-Forest während des Batch-Laufs mittrainieren",
            variable=self.batch_train_rf,
            command=self._toggle_rf_options
        ).pack(anchor=tk.W, pady=2)

        self.rf_option_widgets = []

        row_rf1 = tk.Frame(rf_frame)
        row_rf1.pack(fill=tk.X, pady=2)
        tk.Label(row_rf1, text="Fenstergröße (Frames):", width=28, anchor=tk.W).pack(side=tk.LEFT)
        spin_window = ttk.Spinbox(row_rf1, from_=10, to=600, increment=2, textvariable=self.batch_rf_window, width=8)
        spin_window.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_window)

        tk.Label(row_rf1, text="Schrittweite:", width=15, anchor=tk.W).pack(side=tk.LEFT)
        spin_step = ttk.Spinbox(row_rf1, from_=1, to=300, increment=1, textvariable=self.batch_rf_step, width=8)
        spin_step.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_step)

        row_rf2 = tk.Frame(rf_frame)
        row_rf2.pack(fill=tk.X, pady=2)
        tk.Label(row_rf2, text="Bäume (n_estimators):", width=28, anchor=tk.W).pack(side=tk.LEFT)
        spin_estimators = ttk.Spinbox(row_rf2, from_=256, to=4096, increment=64, textvariable=self.batch_rf_estimators, width=8)
        spin_estimators.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_estimators)

        tk.Label(row_rf2, text="Max. Tiefe (0 = ∞):", width=15, anchor=tk.W).pack(side=tk.LEFT)
        spin_depth = ttk.Spinbox(row_rf2, from_=0, to=80, increment=1, textvariable=self.batch_rf_max_depth, width=8)
        spin_depth.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_depth)

        row_rf3 = tk.Frame(rf_frame)
        row_rf3.pack(fill=tk.X, pady=2)
        tk.Label(row_rf3, text="Min. Samples/Leaf:", width=28, anchor=tk.W).pack(side=tk.LEFT)
        spin_leaf = ttk.Spinbox(row_rf3, from_=1, to=20, increment=1, textvariable=self.batch_rf_min_leaf, width=8)
        spin_leaf.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_leaf)

        tk.Label(row_rf3, text="Min. Samples/Split:", width=18, anchor=tk.W).pack(side=tk.LEFT)
        spin_split = ttk.Spinbox(row_rf3, from_=2, to=40, increment=1, textvariable=self.batch_rf_min_split, width=8)
        spin_split.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_split)

        row_rf4 = tk.Frame(rf_frame)
        row_rf4.pack(fill=tk.X, pady=2)
        tk.Label(row_rf4, text="Baum-Subsampling (max_samples):", width=28, anchor=tk.W).pack(side=tk.LEFT)
        spin_max_samples = ttk.Spinbox(row_rf4, from_=0.1, to=1.0, increment=0.05, format="%.2f",
                                       textvariable=self.batch_rf_max_samples, width=8)
        spin_max_samples.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_max_samples)

        tk.Label(row_rf4, text="Fenster/Klasse (0 = ∞):", width=18, anchor=tk.W).pack(side=tk.LEFT)
        spin_class_cap = ttk.Spinbox(row_rf4, from_=0, to=500000, increment=5000,
                                     textvariable=self.batch_rf_max_windows_per_class, width=10)
        spin_class_cap.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_class_cap)

        row_rf5 = tk.Frame(rf_frame)
        row_rf5.pack(fill=tk.X, pady=2)
        tk.Label(row_rf5, text="Fenster/Track (0 = ∞):", width=28, anchor=tk.W).pack(side=tk.LEFT)
        spin_track_cap = ttk.Spinbox(row_rf5, from_=0, to=5000, increment=50,
                                     textvariable=self.batch_rf_max_windows_per_track, width=8)
        spin_track_cap.pack(side=tk.LEFT, padx=5)
        self.rf_option_widgets.append(spin_track_cap)

        tk.Label(row_rf5, text="Mehr Wiederholungen erweitern den gleichen Wald (kein Reset).",
                 anchor=tk.W, justify=tk.LEFT).pack(side=tk.LEFT, padx=5)

        hint_rf = tk.Label(
            rf_frame,
            text="Hinweis: Das Modell wird nach dem Batch-Lauf als random_forest_diffusion.joblib gespeichert.",
            font=("Arial", 9),
            fg="#16a085"
        )
        hint_rf.pack(anchor=tk.W, pady=4)

        self._toggle_rf_options()

    def _create_export_tab(self):
        """Tab fÃ¼r Export-Optionen."""

        export_frame = ttk.LabelFrame(self.export_tab, text="ðŸ“‹ Metadata Export", padding=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            export_frame,
            text="WÃ¤hle welche Metadata-Formate exportiert werden sollen:",
            font=("Arial", 10)
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Checkbutton(
            export_frame,
            text="ðŸ“„ JSON (maschinenlesbar, vollstÃ¤ndig)",
            variable=self.export_json
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(
            export_frame,
            text="ðŸ“ TXT (menschenlesbar, Zusammenfassung)",
            variable=self.export_txt
        ).pack(anchor=tk.W, pady=2)
        
        ttk.Checkbutton(
            export_frame,
            text="ðŸ“Š CSV (tabellarisch, fÃ¼r Batch-Analysen)",
            variable=self.export_csv
        ).pack(anchor=tk.W, pady=2)
        
        # Info
        info_frame = tk.Frame(export_frame, bg="#e8f4f8", relief=tk.SOLID, bd=1)
        info_frame.pack(fill=tk.X, pady=10, padx=5)
        
        tk.Label(
            info_frame,
            text="ðŸ’¡ EMPFEHLUNG:\n\n"
                 "Alle Formate aktivieren fÃ¼r maximale FlexibilitÃ¤t!\n\n"
                 "â€¢ JSON: FÃ¼r Software-Integration (Python, R, MATLAB)\n"
                 "â€¢ TXT: FÃ¼r Dokumentation und Reports\n"
                 "â€¢ CSV: FÃ¼r Excel und Batch-Analysen",
            font=("Arial", 9),
            bg="#e8f4f8",
            fg="#2c3e50",
            justify=tk.LEFT,
            padx=10,
            pady=10
        ).pack(fill=tk.X)
    


    def _toggle_rf_options(self):
        """Aktiviert oder deaktiviert RF-Parameterfelder."""
        state = tk.NORMAL if self.batch_train_rf.get() else tk.DISABLED
        for widget in getattr(self, 'rf_option_widgets', []):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _get_batch_rf_kwargs(self) -> dict:
        """Erstellt kwargs für BatchSimulator bzgl. RF-Training."""
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
        
    def _toggle_mode_panels(self):
        """Zeigt/versteckt Panels je nach Modus."""
        
        mode = self.mode_var.get()
        
        if mode == "single":
            self.notebook.select(0)  # Single tab
        else:
            self.notebook.select(1)  # Batch tab
    
    def _update_mode_info(self):
        """Aktualisiert Mode-Info-Text."""
        
        mode = self.sim_mode_var.get()
        
        info_texts = {
            "polyzeit": (
                "â±ï¸ POLYMERISATIONSZEIT (2D)\n\n"
                "Simuliert Brownsche Bewegung bei verschiedenen Polymerisationszeiten.\n"
                "Spots bewegen sich in 2D ohne z-Information.\n\n"
                "Anwendung: MSD-Analyse, ZeitabhÃ¤ngigkeit von D(t)"
            ),
            "polyzeit_astig": (
                "â±ï¸ðŸ”º POLYMERISATIONSZEIT MIT ASTIGMATISMUS (3D)\n\n"
                "Simuliert 3D-Diffusion mit astigmatischer PSF.\n"
                "Spots werden elliptisch je nach z-Position.\n\n"
                "Anwendung: 3D-Lokalisierung mit ThunderSTORM, etc."
            ),
            "z_stack": (
                "ðŸ“Š Z-STACK KALIBRIERUNG\n\n"
                "Erstellt einen z-Stack mit statischen Spots fÃ¼r PSF-Kalibrierung.\n"
                "Spots bewegen sich NICHT, nur z variiert.\n\n"
                "Anwendung: Kalibrierung von 3D-Tracking-Software"
            )
        }
        
        text = info_texts.get(mode, "")
        
        self.mode_info_text.config(state=tk.NORMAL)
        self.mode_info_text.delete(1.0, tk.END)
        self.mode_info_text.insert(1.0, text)
        self.mode_info_text.config(state=tk.DISABLED)
    
    def _apply_detector_preset(self):
        """Wendet Detektor-Preset an."""
        
        detector = self.detector_var.get()
        
        if detector == "TDI-G0":
            self._update_status("ðŸ“· TDI-G0 Preset: ~260 counts, 0.108 Âµm/px")
        else:
            self._update_status("ðŸ“· Tetraspecs Preset: ~300 counts, 0.160 Âµm/px")
    
    def _update_d_estimate(self):
        """Zeigt geschÃ¤tztes D."""
        
        try:
            from tiff_simulator_v3 import get_time_dependent_D
            
            t = self.t_poly.get()
            d_init = self.d_initial.get()
            
            d_normal = get_time_dependent_D(t, d_init, "normal")
            d_sub = get_time_dependent_D(t, d_init, "subdiffusion")
            
            self.d_info_label.config(
                text=f"â†’ D_normal â‰ˆ {d_normal:.3f}, D_sub â‰ˆ {d_sub:.3f} ÂµmÂ²/s"
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
                self.z_slices_label.config(text=f"â†’ {n_slices} Slices")
            else:
                self.z_slices_label.config(text="âŒ UngÃ¼ltig")
        except:
            pass
    
    def _browse_dir(self):
        """Ordner-Dialog."""
        
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
    
    def _set_status_ui(self, message: str, color: str = "#27ae60"):
        """Interne Helferfunktion: fÃ¼hrt den eigentlichen UI-Status-Update aus."""
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def _update_status(self, message: str, color: str = "#27ae60"):
        """Aktualisiert Status, thread-sicher."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._set_status_ui(message, color)
        else:
            # In den Tk-Hauptthread schedulen
            self.root.after(0, lambda: self._set_status_ui(message, color))
    
    def _set_progress_ui(self, value: int):
        """Interne Helferfunktion: fÃ¼hrt den eigentlichen Progress-Update aus."""
        self.progress['value'] = value
        self.root.update()

    def _update_progress(self, value: int):
        """Aktualisiert Progress Bar, thread-sicher."""
        import threading
        if threading.current_thread() is threading.main_thread():
            self._set_progress_ui(value)
        else:
            self.root.after(0, lambda: self._set_progress_ui(value))
    
    def _start_simulation(self):
        """Startet Simulation in separatem Thread."""
        
        if self.is_running:
            messagebox.showwarning("Warnung", "Simulation lÃ¤uft bereits!")
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
            errors.append("BildgrÃ¶ÃŸe muss mindestens 32Ã—32 sein!")
        
        if self.num_spots.get() < 1:
            errors.append("Mindestens 1 Spot erforderlich!")
        
        if not os.path.exists(self.output_dir.get()):
            errors.append(f"Output-Ordner existiert nicht!")
        
        if errors:
            messagebox.showerror("Validierungsfehler", "\n".join(errors))
            return False
        
        return True
    
    def _run_simulation(self):
        """FÃ¼hrt Simulation aus (in Thread)."""
        
        try:
            mode = self.mode_var.get()
            
            if mode == "single":
                self._run_single_simulation()
            else:
                self._run_batch_simulation()
            
            # Erfolg
            self.root.after(0, lambda: messagebox.showinfo(
                "Erfolg! ðŸŽ‰",
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
            self.root.after(0, lambda: self._update_status("âœ… Fertig!"))
    
    def _run_single_simulation(self):
        """FÃ¼hrt Single Simulation aus."""
        
        self._update_status("ðŸ”¬ Initialisiere...")
        self._update_progress(10)
        
        # Detektor
        detector = TDI_PRESET if self.detector_var.get() == "TDI-G0" else TETRASPECS_PRESET
        
        # Modus
        sim_mode = self.sim_mode_var.get()
        # Astigmatismus auch fÃ¼r z-Stack aktivieren, damit z-abhÃ¤ngige PSF greift
        astigmatism = (sim_mode in ("polyzeit_astig", "z_stack"))
        
        # Simulator
        sim = TIFFSimulator(
            detector=detector,
            mode=sim_mode,
            t_poly_min=self.t_poly.get(),
            astigmatism=astigmatism
        )
        
        self._update_status("âœ¨ Generiere TIFF...")
        self._update_progress(30)
        
        # Generiere
        if sim_mode == "z_stack":
            tiff_stack = sim.generate_z_stack(
                image_size=(self.image_height.get(), self.image_width.get()),
                num_spots=self.num_spots.get(),
                z_range_um=(self.z_min.get(), self.z_max.get()),
                z_step_um=self.z_step.get()
            )
        else:
            tiff_stack = sim.generate_tiff(
                image_size=(self.image_height.get(), self.image_width.get()),
                num_spots=self.num_spots.get(),
                num_frames=self.num_frames.get(),
                frame_rate_hz=self.frame_rate.get(),
                d_initial=self.d_initial.get(),
                exposure_substeps=self.exposure_substeps.get(),
                enable_photophysics=False
            )
        
        self._update_progress(70)
        
        # Speichern
        self._update_status("ðŸ’¾ Speichere...")
        filepath = Path(self.output_dir.get()) / self.filename.get()
        save_tiff(str(filepath), tiff_stack)
        
        self._update_progress(85)
        
        # Metadata
        if self.export_json.get() or self.export_txt.get() or self.export_csv.get():
            self._update_status("ðŸ“‹ Exportiere Metadata...")
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
    
    def _run_batch_simulation(self):
        """FÃ¼hrt Batch Simulation aus."""

        self._update_status("ðŸ”„ Initialisiere Batch...")
        self._update_progress(5)

        batch_kwargs = self._get_batch_rf_kwargs()
        rf_enabled = bool(batch_kwargs.get('enable_rf'))

        # Prüfe, ob Custom Times gesetzt wurden
        custom_times_str = self.batch_custom_times.get().strip()
        if custom_times_str:
            from batch_simulator import BatchSimulator
            from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET
            batch = BatchSimulator(self.output_dir.get(), **batch_kwargs)

            # Parse Zeiten
            import re
            parts = re.split(r'[,:;\s]+', custom_times_str)
            times = []
            for p in parts:
                if p:
                    try:
                        times.append(float(p))
                    except:
                        pass
            # Wähle Detektor
            det = TDI_PRESET if self.batch_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            # Füge Aufgaben hinzu (2D oder 3D)
            if self.batch_astig.get():
                for t in times:
                    filename = f"{det.name.lower()}_3d_t{int(t)}min.tif"
                    task = {
                        'detector': det,
                        'mode': 'polyzeit_astig',
                        't_poly_min': t,
                        'astigmatism': True,
                        'filename': filename,
                        'image_size': (self.image_height.get(), self.image_width.get()),
                        'num_spots': self.num_spots.get(),
                        'num_frames': self.num_frames.get(),
                        'frame_rate_hz': self.frame_rate.get()
                    }
                    for _ in range(self.batch_repeats.get()):
                        batch.add_task(task.copy())
            else:
                for t in times:
                    filename = f"{det.name.lower()}_t{int(t)}min.tif"
                    task = {
                        'detector': det,
                        'mode': 'polyzeit',
                        't_poly_min': t,
                        'astigmatism': False,
                        'filename': filename,
                        'image_size': (self.image_height.get(), self.image_width.get()),
                        'num_spots': self.num_spots.get(),
                        'num_frames': self.num_frames.get(),
                        'frame_rate_hz': self.frame_rate.get()
                    }
                    for _ in range(self.batch_repeats.get()):
                        batch.add_task(task.copy())
        else:
            # Wähle Preset
            preset = self.batch_preset.get()
            if preset == "quick":
                batch = PresetBatches.quick_test(self.output_dir.get(), **batch_kwargs)
            elif preset == "thesis":
                batch = PresetBatches.masterthesis_full(self.output_dir.get(), **batch_kwargs)
            else:
                batch = PresetBatches.publication_quality(self.output_dir.get(), **batch_kwargs)

        total_tasks = len(batch.tasks)

        # Progress Callback
        def progress_callback(current, total, status):
            progress = int((current / total) * 100)
            self.root.after(0, lambda: self._update_progress(progress))
            self.root.after(0, lambda: self._update_status(
                f"ðŸ”„ {current}/{total}: {status}"
            ))
        
        # Run Batch
        stats = batch.run(progress_callback=progress_callback)

        self._update_progress(100)

        if rf_enabled:
            rf_stats = stats.get('rf', {}) if isinstance(stats, dict) else {}
            if rf_stats.get('trained'):
                model_path = rf_stats.get('model_path', 'unbekannt')
                message = (
                    "Random-Forest Training abgeschlossen!\n\n"
                    f"Fenster: {rf_stats.get('samples', 0)}\n"
                    f"Modell: {model_path}"
                )
                self.root.after(0, lambda: messagebox.showinfo("Random Forest", message))
                self.root.after(0, lambda: self._update_status(f"ðŸŒ² RF gespeichert: {model_path}"))
            else:
                self.root.after(0, lambda: self._update_status("ðŸŒ² Keine gültigen Trajektorienfenster für RF gefunden."))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = TIFFSimulatorGUI(root)
    root.mainloop()


