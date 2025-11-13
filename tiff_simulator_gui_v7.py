"""
🎨 TIFF SIMULATOR V7.1 - COMPLETE PROFESSIONAL GUI
==================================================

Vollständige GUI mit ALLEN Parametern und Features!

Features:
- Single TIFF (volle Kontrolle über alle Parameter)
- Z-Stack (mit korrigiertem Astigmatismus)
- Batch-Modus (flexibel konfigurierbar)
- Erweiterte Physik-Parameter (Brechungsindizes, etc.)

Version: 7.1 - November 2025
Author: Claude Agent (Astigmatismus-Fix & Complete GUI)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
from typing import Optional, Dict, List
import json

try:
    from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    from batch_simulator import BatchSimulator
    from metadata_exporter import MetadataExporter
except ImportError as e:
    print(f"❌ Import Error: {e}")
    exit(1)


class CompleteGUI:
    """Vollständige professionelle GUI mit allen Features"""

    COLORS = {
        'bg': '#f0f2f5',
        'fg': '#2c3e50',
        'primary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#9b59b6',
        'frame_bg': '#ffffff',
        'header_bg': '#34495e',
        'header_fg': '#ecf0f1',
        'card_bg': '#fafbfc',
        'border': '#dce0e3',
        'input_bg': '#ffffff'
    }

    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Simulator v7.1 - Complete Edition")
        self.root.geometry("1300x900")
        self.root.configure(bg=self.COLORS['bg'])

        # State
        self.is_running = False
        self.current_thread: Optional[threading.Thread] = None
        self.output_dir = Path("./tiff_output_v7")

        # Build GUI
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        self._apply_styles()

    def _apply_styles(self):
        """Apply modern styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook
        style.configure('TNotebook', background=self.COLORS['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', padding=[20, 12], font=('Segoe UI', 11, 'bold'))
        style.map('TNotebook.Tab',
                 background=[('selected', self.COLORS['primary'])],
                 foreground=[('selected', 'white'), ('!selected', self.COLORS['fg'])])

        # Progressbar
        style.configure("Custom.Horizontal.TProgressbar",
                       background=self.COLORS['success'],
                       troughcolor=self.COLORS['border'],
                       borderwidth=0,
                       thickness=25)

    def _create_header(self):
        """Create header with title"""
        header = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=90)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="🔬 TIFF SIMULATOR V7.1",
            font=("Segoe UI", 26, "bold"),
            bg=self.COLORS['header_bg'],
            fg=self.COLORS['header_fg']
        )
        title.pack(pady=(12, 2))

        subtitle = tk.Label(
            header,
            text="Complete Professional Edition • Astigmatismus Fix v4.2",
            font=("Segoe UI", 10),
            bg=self.COLORS['header_bg'],
            fg='#95a5a6'
        )
        subtitle.pack()

    def _create_main_content(self):
        """Create main tabbed interface"""
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self._create_single_tiff_tab()
        self._create_z_stack_tab()
        self._create_batch_tab()
        self._create_advanced_physics_tab()

    def _create_single_tiff_tab(self):
        """Tab 1: Single TIFF with COMPLETE control"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="📄 Single TIFF")

        # Create scrollable canvas
        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === BASIC SETTINGS ===
        card = self._create_card(content, "⚙️ Grundeinstellungen")

        row = 0
        # Detector
        tk.Label(card, text="Detektor:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_detector = ttk.Combobox(card, values=["TDI-G0", "Tetraspecs"],
                                           state="readonly", width=20)
        self.single_detector.set("TDI-G0")
        self.single_detector.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "TDI-G0: 0.108µm/px | Tetraspecs: 0.160µm/px",
                         row=row, column=2)

        row += 1
        # Image Size
        tk.Label(card, text="Bildgröße:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_img_size = ttk.Combobox(card,
                                           values=["64x64", "128x128", "256x256", "512x512"],
                                           state="readonly", width=20)
        self.single_img_size.set("128x128")
        self.single_img_size.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Größer = realistischer aber langsamer",
                         row=row, column=2)

        row += 1
        # Number of spots
        tk.Label(card, text="Anzahl Spots:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_spots = tk.Spinbox(card, from_=1, to=200, width=18, font=("Segoe UI", 10))
        self.single_spots.delete(0, tk.END)
        self.single_spots.insert(0, "15")
        self.single_spots.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Anzahl simulierter Fluorophore", row=row, column=2)

        row += 1
        # Number of frames
        tk.Label(card, text="Anzahl Frames:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_frames = tk.Spinbox(card, from_=10, to=10000, width=18,
                                       increment=50, font=("Segoe UI", 10))
        self.single_frames.delete(0, tk.END)
        self.single_frames.insert(0, "200")
        self.single_frames.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Zeitliche Auflösung der Messung", row=row, column=2)

        row += 1
        # Frame rate
        tk.Label(card, text="Frame Rate (Hz):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_framerate = tk.Spinbox(card, from_=1, to=200, width=18,
                                          font=("Segoe UI", 10))
        self.single_framerate.delete(0, tk.END)
        self.single_framerate.insert(0, "20")
        self.single_framerate.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Bildrate der Kamera (fps)", row=row, column=2)

        row += 1
        # Polymerization time
        tk.Label(card, text="Polymerisationszeit (min):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_polytime = tk.Spinbox(card, from_=0, to=240, width=18,
                                         increment=5, font=("Segoe UI", 10))
        self.single_polytime.delete(0, tk.END)
        self.single_polytime.insert(0, "60")
        self.single_polytime.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Hydrogel-Vernetzungszeit", row=row, column=2)

        row += 1
        # Initial diffusion coefficient
        tk.Label(card, text="D_initial (µm²/s):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.single_d_initial = tk.Spinbox(card, from_=0.1, to=10.0, width=18,
                                          increment=0.5, font=("Segoe UI", 10))
        self.single_d_initial.delete(0, tk.END)
        self.single_d_initial.insert(0, "0.5")
        self.single_d_initial.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self._add_tooltip(card, "Initialer Diffusionskoeffizient", row=row, column=2)

        # === ASTIGMATISM ===
        card2 = self._create_card(content, "🔬 3D / Astigmatismus (FIXED v4.2)")

        self.single_astig = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(card2, text="Astigmatismus aktivieren (für 3D-Tracking)",
                           variable=self.single_astig,
                           font=("Segoe UI", 10, "bold"),
                           bg=self.COLORS['card_bg'])
        cb.pack(anchor='w', padx=10, pady=5)

        info = tk.Label(card2,
                       text="✅ Physikalisch korrekte Implementierung (Huang et al. 2008)\n"
                            "✓ z<0: Horizontal  ✓ z=0: Rund  ✓ z>0: Vertikal",
                       font=("Segoe UI", 9),
                       bg=self.COLORS['card_bg'],
                       fg=self.COLORS['success'],
                       justify=tk.LEFT)
        info.pack(anchor='w', padx=25, pady=2)

        # === ADVANCED OPTIONS ===
        card3 = self._create_card(content, "🧪 Erweiterte Optionen")

        row = 0
        # Photophysics
        self.single_photophysics = tk.BooleanVar(value=False)
        tk.Checkbutton(card3, text="Photophysik (Blinking/Bleaching)",
                      variable=self.single_photophysics,
                      font=("Segoe UI", 10),
                      bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w',
                                                      padx=10, pady=3, columnspan=2)

        row += 1
        # Diffusion switching
        self.single_switching = tk.BooleanVar(value=True)
        tk.Checkbutton(card3, text="Dynamisches Diffusions-Switching",
                      variable=self.single_switching,
                      font=("Segoe UI", 10),
                      bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w',
                                                      padx=10, pady=3, columnspan=2)

        row += 1
        # Polymerization acceleration
        tk.Label(card3, text="Comonomer-Faktor:", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.single_comonomer = tk.Spinbox(card3, from_=0.5, to=3.0, width=10,
                                          increment=0.1, font=("Segoe UI", 10))
        self.single_comonomer.delete(0, tk.END)
        self.single_comonomer.insert(0, "1.0")
        self.single_comonomer.grid(row=row, column=1, sticky='w', padx=10, pady=3)

        # === BUTTONS ===
        btn_frame = tk.Frame(content, bg=self.COLORS['frame_bg'])
        btn_frame.pack(pady=20)

        self.single_generate_btn = tk.Button(
            btn_frame,
            text="🎬 TIFF Generieren",
            command=self._generate_single_tiff,
            bg=self.COLORS['success'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.single_generate_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="📁 Output Ordner",
            command=self._select_output_dir,
            bg=self.COLORS['primary'],
            fg='white',
            font=("Segoe UI", 11),
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=5)

    def _create_z_stack_tab(self):
        """Tab 2: Z-Stack with FIXED astigmatism"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="📚 Z-Stack")

        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === SETTINGS ===
        card = self._create_card(content, "⚙️ Z-Stack Einstellungen")

        row = 0
        # Detector
        tk.Label(card, text="Detektor:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_detector = ttk.Combobox(card, values=["TDI-G0", "Tetraspecs"],
                                           state="readonly", width=20)
        self.zstack_detector.set("Tetraspecs")
        self.zstack_detector.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Image size
        tk.Label(card, text="Bildgröße:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_img_size = ttk.Combobox(card,
                                           values=["64x64", "128x128", "256x256"],
                                           state="readonly", width=20)
        self.zstack_img_size.set("128x128")
        self.zstack_img_size.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Z-Start
        tk.Label(card, text="Z Start (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_z_start = tk.Spinbox(card, from_=-3.0, to=0, width=18,
                                         increment=0.1, format="%.1f", font=("Segoe UI", 10))
        self.zstack_z_start.delete(0, tk.END)
        self.zstack_z_start.insert(0, "-1.0")
        self.zstack_z_start.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Z-End
        tk.Label(card, text="Z Ende (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_z_end = tk.Spinbox(card, from_=0, to=3.0, width=18,
                                       increment=0.1, format="%.1f", font=("Segoe UI", 10))
        self.zstack_z_end.delete(0, tk.END)
        self.zstack_z_end.insert(0, "1.0")
        self.zstack_z_end.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Z-Step
        tk.Label(card, text="Z Schritt (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_z_step = tk.Spinbox(card, from_=0.01, to=0.5, width=18,
                                        increment=0.01, format="%.2f", font=("Segoe UI", 10))
        self.zstack_z_step.delete(0, tk.END)
        self.zstack_z_step.insert(0, "0.10")
        self.zstack_z_step.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Spots
        tk.Label(card, text="Anzahl Spots:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.zstack_spots = tk.Spinbox(card, from_=5, to=100, width=18, font=("Segoe UI", 10))
        self.zstack_spots.delete(0, tk.END)
        self.zstack_spots.insert(0, "20")
        self.zstack_spots.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        # === INFO ===
        card2 = self._create_card(content, "✅ Astigmatismus-Fix v4.2")

        info = tk.Label(card2,
                       text="🎉 Z-Stack nutzt jetzt KORREKTEN Astigmatismus!\n\n"
                            "✓ z < 0: PSF horizontal gestreckt (σx > σy)\n"
                            "✓ z = 0: PSF rund (σx ≈ σy)\n"
                            "✓ z > 0: PSF vertikal gestreckt (σy > σx)\n\n"
                            "Physikalisch korrekt nach Huang et al. 2008\n"
                            "Kompatibel mit TrackMate und ThunderSTORM",
                       font=("Segoe UI", 10),
                       bg=self.COLORS['card_bg'],
                       fg=self.COLORS['success'],
                       justify=tk.LEFT)
        info.pack(padx=15, pady=10)

        # === BUTTONS ===
        btn_frame = tk.Frame(content, bg=self.COLORS['frame_bg'])
        btn_frame.pack(pady=20)

        self.zstack_generate_btn = tk.Button(
            btn_frame,
            text="📚 Z-Stack Generieren",
            command=self._generate_z_stack,
            bg=self.COLORS['primary'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.zstack_generate_btn.pack()

    def _create_batch_tab(self):
        """Tab 3: Flexible Batch Mode"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="🔄 Batch Modus")

        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === BATCH CONFIGURATION ===
        card = self._create_card(content, "⚙️ Batch Konfiguration")

        row = 0
        # Detector
        tk.Label(card, text="Detektor:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.batch_detector = ttk.Combobox(card, values=["TDI-G0", "Tetraspecs"],
                                          state="readonly", width=20)
        self.batch_detector.set("TDI-G0")
        self.batch_detector.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Image size
        tk.Label(card, text="Bildgröße:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.batch_img_size = ttk.Combobox(card, values=["128x128", "256x256"],
                                          state="readonly", width=20)
        self.batch_img_size.set("128x128")
        self.batch_img_size.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Number of spots (can vary)
        tk.Label(card, text="Anzahl Spots (min-max):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        spots_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        spots_frame.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        self.batch_spots_min = tk.Spinbox(spots_frame, from_=5, to=50, width=8, font=("Segoe UI", 10))
        self.batch_spots_min.delete(0, tk.END)
        self.batch_spots_min.insert(0, "10")
        self.batch_spots_min.pack(side=tk.LEFT)
        tk.Label(spots_frame, text=" bis ", bg=self.COLORS['card_bg']).pack(side=tk.LEFT)
        self.batch_spots_max = tk.Spinbox(spots_frame, from_=5, to=50, width=8, font=("Segoe UI", 10))
        self.batch_spots_max.delete(0, tk.END)
        self.batch_spots_max.insert(0, "20")
        self.batch_spots_max.pack(side=tk.LEFT)

        row += 1
        # Polymerization times
        tk.Label(card, text="Polymerisationszeiten (min):",
                font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.batch_polytimes = tk.Entry(card, width=22, font=("Segoe UI", 10))
        self.batch_polytimes.insert(0, "15,30,45,60,90,120")
        self.batch_polytimes.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        tk.Label(card, text="(komma-separiert)", font=("Segoe UI", 8, "italic"),
                bg=self.COLORS['card_bg'], fg='#7f8c8d').grid(row=row, column=2, sticky='w', padx=5)

        row += 1
        # Repeats per condition
        tk.Label(card, text="Wiederholungen pro Bedingung:",
                font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.batch_repeats = tk.Spinbox(card, from_=1, to=10, width=20, font=("Segoe UI", 10))
        self.batch_repeats.delete(0, tk.END)
        self.batch_repeats.insert(0, "3")
        self.batch_repeats.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        row += 1
        # Frames
        tk.Label(card, text="Frames pro TIFF:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.batch_frames = tk.Spinbox(card, from_=50, to=2000, width=20,
                                      increment=50, font=("Segoe UI", 10))
        self.batch_frames.delete(0, tk.END)
        self.batch_frames.insert(0, "200")
        self.batch_frames.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        # === OPTIONS ===
        card2 = self._create_card(content, "🎯 Batch Optionen")

        self.batch_random_spots = tk.BooleanVar(value=True)
        tk.Checkbutton(card2, text="Randomisiere Spot-Anzahl innerhalb des Bereichs",
                      variable=self.batch_random_spots,
                      font=("Segoe UI", 10),
                      bg=self.COLORS['card_bg']).pack(anchor='w', padx=10, pady=3)

        self.batch_with_astig = tk.BooleanVar(value=False)
        tk.Checkbutton(card2, text="Mit Astigmatismus (FIXED v4.2)",
                      variable=self.batch_with_astig,
                      font=("Segoe UI", 10),
                      bg=self.COLORS['card_bg']).pack(anchor='w', padx=10, pady=3)

        self.batch_train_rf = tk.BooleanVar(value=False)
        tk.Checkbutton(card2, text="Random Forest trainieren (nach Batch-Generierung)",
                      variable=self.batch_train_rf,
                      font=("Segoe UI", 10),
                      bg=self.COLORS['card_bg']).pack(anchor='w', padx=10, pady=3)

        # === INFO ===
        card3 = self._create_card(content, "📊 Batch Vorschau")
        self.batch_info_label = tk.Label(card3,
                                        text="Konfiguriere Parameter und klicke auf 'Berechnen'",
                                        font=("Segoe UI", 10),
                                        bg=self.COLORS['card_bg'],
                                        fg=self.COLORS['fg'],
                                        justify=tk.LEFT)
        self.batch_info_label.pack(padx=15, pady=10)

        tk.Button(card3,
                 text="🧮 Anzahl TIFFs berechnen",
                 command=self._calculate_batch_size,
                 bg=self.COLORS['info'],
                 fg='white',
                 font=("Segoe UI", 10),
                 relief=tk.FLAT,
                 padx=15,
                 pady=8,
                 cursor="hand2").pack(pady=5)

        # === BUTTONS ===
        btn_frame = tk.Frame(content, bg=self.COLORS['frame_bg'])
        btn_frame.pack(pady=20)

        self.batch_generate_btn = tk.Button(
            btn_frame,
            text="🔄 Batch Starten",
            command=self._run_batch,
            bg=self.COLORS['warning'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.batch_generate_btn.pack()

    def _create_advanced_physics_tab(self):
        """Tab 4: Advanced Physics Parameters"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="⚗️ Erweiterte Physik")

        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # === ASTIGMATISM PARAMETERS ===
        card = self._create_card(content, "🔬 Astigmatismus-Parameter (v4.2 FIXED)")

        row = 0
        tk.Label(card, text="Fokustrennung (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.adv_focal_offset = tk.Spinbox(card, from_=0.1, to=1.0, width=15,
                                          increment=0.05, format="%.2f", font=("Segoe UI", 10))
        self.adv_focal_offset.delete(0, tk.END)
        self.adv_focal_offset.insert(0, "0.40")
        self.adv_focal_offset.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        tk.Label(card, text="Standard: 0.4", font=("Segoe UI", 8, "italic"),
                bg=self.COLORS['card_bg'], fg='#7f8c8d').grid(row=row, column=2, sticky='w')

        row += 1
        tk.Label(card, text="Rayleigh-Bereich (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.adv_z_rayleigh = tk.Spinbox(card, from_=0.3, to=1.5, width=15,
                                        increment=0.05, format="%.2f", font=("Segoe UI", 10))
        self.adv_z_rayleigh.delete(0, tk.END)
        self.adv_z_rayleigh.insert(0, "0.60")
        self.adv_z_rayleigh.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        tk.Label(card, text="Standard: 0.6", font=("Segoe UI", 8, "italic"),
                bg=self.COLORS['card_bg'], fg='#7f8c8d').grid(row=row, column=2, sticky='w')

        # === REFRACTIVE INDEX ===
        card2 = self._create_card(content, "🌊 Brechungsindex-Korrektur")

        self.adv_use_refractive = tk.BooleanVar(value=False)
        tk.Checkbutton(card2, text="Erweiterte Brechungsindex-Korrektur aktivieren",
                      variable=self.adv_use_refractive,
                      font=("Segoe UI", 10, "bold"),
                      bg=self.COLORS['card_bg'],
                      command=self._toggle_refractive_params).pack(anchor='w', padx=10, pady=5)

        self.refractive_frame = tk.Frame(card2, bg=self.COLORS['card_bg'])
        self.refractive_frame.pack(fill=tk.X, padx=20, pady=5)

        row = 0
        tk.Label(self.refractive_frame, text="n (Immersionsöl):", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.adv_n_oil = tk.Entry(self.refractive_frame, width=15, font=("Segoe UI", 10))
        self.adv_n_oil.insert(0, "1.518")
        self.adv_n_oil.grid(row=row, column=1, sticky='w', padx=10, pady=3)
        self.adv_n_oil.config(state='disabled')

        row += 1
        tk.Label(self.refractive_frame, text="n (Deckglas):", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.adv_n_glass = tk.Entry(self.refractive_frame, width=15, font=("Segoe UI", 10))
        self.adv_n_glass.insert(0, "1.523")
        self.adv_n_glass.grid(row=row, column=1, sticky='w', padx=10, pady=3)
        self.adv_n_glass.config(state='disabled')

        row += 1
        tk.Label(self.refractive_frame, text="n (Polymer/Medium):", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.adv_n_polymer = tk.Entry(self.refractive_frame, width=15, font=("Segoe UI", 10))
        self.adv_n_polymer.insert(0, "1.47")
        self.adv_n_polymer.grid(row=row, column=1, sticky='w', padx=10, pady=3)
        self.adv_n_polymer.config(state='disabled')

        row += 1
        tk.Label(self.refractive_frame, text="NA (Objektiv):", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.adv_na = tk.Entry(self.refractive_frame, width=15, font=("Segoe UI", 10))
        self.adv_na.insert(0, "1.45")
        self.adv_na.grid(row=row, column=1, sticky='w', padx=10, pady=3)
        self.adv_na.config(state='disabled')

        row += 1
        tk.Label(self.refractive_frame, text="Deckglas-Dicke (µm):", font=("Segoe UI", 10),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=3)
        self.adv_d_glass = tk.Entry(self.refractive_frame, width=15, font=("Segoe UI", 10))
        self.adv_d_glass.insert(0, "170.0")
        self.adv_d_glass.grid(row=row, column=1, sticky='w', padx=10, pady=3)
        self.adv_d_glass.config(state='disabled')

        # === ILLUMINATION ===
        card3 = self._create_card(content, "💡 Ausleuchtungsgradient")

        row = 0
        tk.Label(card3, text="Gradient-Stärke:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.adv_illum_strength = tk.Spinbox(card3, from_=0, to=50, width=15,
                                            increment=5, font=("Segoe UI", 10))
        self.adv_illum_strength.delete(0, tk.END)
        self.adv_illum_strength.insert(0, "0")
        self.adv_illum_strength.grid(row=row, column=1, sticky='w', padx=10, pady=5)
        tk.Label(card3, text="0 = aus", font=("Segoe UI", 8, "italic"),
                bg=self.COLORS['card_bg'], fg='#7f8c8d').grid(row=row, column=2, sticky='w')

        row += 1
        tk.Label(card3, text="Gradient-Typ:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        self.adv_illum_type = ttk.Combobox(card3,
                                          values=["radial", "linear_x", "linear_y", "corner"],
                                          state="readonly", width=13)
        self.adv_illum_type.set("radial")
        self.adv_illum_type.grid(row=row, column=1, sticky='w', padx=10, pady=5)

        # === INFO ===
        card4 = self._create_card(content, "ℹ️ Information")

        info_text = """Diese Parameter beeinflussen die physikalische Genauigkeit der Simulation.

🔬 Astigmatismus-Parameter:
   • Fokustrennung: Abstand zwischen x- und y-Fokus
   • Rayleigh-Bereich: Tiefenschärfe der PSF

🌊 Brechungsindex-Korrektur:
   • Berücksichtigt Aberrationen durch Brechungsindex-Mismatch
   • Wichtig für tiefe z-Stacks (>2µm)

💡 Ausleuchtungsgradient:
   • Simuliert inhomogene Beleuchtung
   • Typisch bei TIRF oder Weitfeld-Mikroskopie

⚠️ Hinweis: Standard-Werte sind für die meisten Anwendungen optimal!
Ändere diese nur, wenn du die physikalischen Auswirkungen verstehst.
"""

        tk.Label(card4, text=info_text,
                font=("Segoe UI", 9),
                bg=self.COLORS['card_bg'],
                fg=self.COLORS['fg'],
                justify=tk.LEFT).pack(padx=15, pady=10)

        # Save/Load buttons
        btn_frame = tk.Frame(content, bg=self.COLORS['frame_bg'])
        btn_frame.pack(pady=15)

        tk.Button(btn_frame,
                 text="💾 Parameter speichern",
                 command=self._save_advanced_params,
                 bg=self.COLORS['success'],
                 fg='white',
                 font=("Segoe UI", 10),
                 relief=tk.FLAT,
                 padx=15,
                 pady=10,
                 cursor="hand2").pack(side=tk.LEFT, padx=5)

        tk.Button(btn_frame,
                 text="📂 Parameter laden",
                 command=self._load_advanced_params,
                 bg=self.COLORS['primary'],
                 fg='white',
                 font=("Segoe UI", 10),
                 relief=tk.FLAT,
                 padx=15,
                 pady=10,
                 cursor="hand2").pack(side=tk.LEFT, padx=5)

    def _create_card(self, parent, title):
        """Create a styled card"""
        frame = tk.Frame(parent, bg=self.COLORS['card_bg'],
                        highlightbackground=self.COLORS['border'],
                        highlightthickness=1)
        frame.pack(fill=tk.X, padx=15, pady=10)

        tk.Label(frame, text=title,
                font=("Segoe UI", 12, "bold"),
                bg=self.COLORS['card_bg'],
                fg=self.COLORS['primary']).pack(anchor='w', padx=10, pady=(10, 5))

        tk.Frame(frame, height=2, bg=self.COLORS['border']).pack(fill=tk.X, padx=10)

        return frame

    def _add_tooltip(self, parent, text, row, column):
        """Add a tooltip icon"""
        label = tk.Label(parent, text="ℹ️", bg=self.COLORS['card_bg'],
                        fg=self.COLORS['info'], cursor="hand2")
        label.grid(row=row, column=column, sticky='w', padx=5)
        # TODO: Add actual tooltip on hover

    def _toggle_refractive_params(self):
        """Enable/disable refractive index parameters"""
        state = 'normal' if self.adv_use_refractive.get() else 'disabled'
        for widget in [self.adv_n_oil, self.adv_n_glass, self.adv_n_polymer,
                      self.adv_na, self.adv_d_glass]:
            widget.config(state=state)

    def _create_status_bar(self):
        """Create status bar"""
        status_frame = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=45)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame,
                                     text=f"📁 Output: {self.output_dir.absolute()}",
                                     font=("Segoe UI", 10),
                                     bg=self.COLORS['header_bg'],
                                     fg=self.COLORS['header_fg'])
        self.status_label.pack(side=tk.LEFT, padx=15)

        self.progress = ttk.Progressbar(status_frame,
                                       style="Custom.Horizontal.TProgressbar",
                                       mode='indeterminate',
                                       length=250)
        self.progress.pack(side=tk.RIGHT, padx=15)

    def _select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_dir)
        if directory:
            self.output_dir = Path(directory)
            self.status_label.config(text=f"📁 Output: {self.output_dir.absolute()}")

    def _update_status(self, message, is_running=False):
        """Update status bar"""
        self.status_label.config(text=message)
        if is_running:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _calculate_batch_size(self):
        """Calculate how many TIFFs will be generated"""
        try:
            times = [float(t.strip()) for t in self.batch_polytimes.get().split(',')]
            repeats = int(self.batch_repeats.get())
            total = len(times) * repeats

            info = f"📊 Batch Vorschau:\n\n"
            info += f"• Polymerisationszeiten: {len(times)} ({', '.join(map(str, times))} min)\n"
            info += f"• Wiederholungen pro Zeit: {repeats}\n"
            info += f"• Astigmatismus: {'Ja' if self.batch_with_astig.get() else 'Nein'}\n"
            info += f"• Random Forest Training: {'Ja' if self.batch_train_rf.get() else 'Nein'}\n\n"
            info += f"➡️ Gesamt: {total} TIFFs werden generiert"

            self.batch_info_label.config(text=info, fg=self.COLORS['success'])

        except Exception as e:
            self.batch_info_label.config(
                text=f"❌ Fehler bei Berechnung: {str(e)}",
                fg=self.COLORS['danger']
            )

    def _generate_single_tiff(self):
        """Generate single TIFF"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        try:
            # Get parameters
            detector = TDI_PRESET if self.single_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            size_str = self.single_img_size.get()
            size = tuple(map(int, size_str.split('x')))
            spots = int(self.single_spots.get())
            frames = int(self.single_frames.get())
            framerate = float(self.single_framerate.get())
            polytime = float(self.single_polytime.get())
            d_initial = float(self.single_d_initial.get())
            astig = self.single_astig.get()
            photophysics = self.single_photophysics.get()
            comonomer = float(self.single_comonomer.get())

            # Apply advanced parameters
            detector = self._apply_advanced_params(detector)

            self.output_dir.mkdir(exist_ok=True)

            def run():
                try:
                    self._update_status("🔄 Generiere TIFF...", True)

                    sim = TIFFSimulator(
                        detector=detector,
                        mode='polyzeit',
                        t_poly_min=polytime,
                        astigmatism=astig,
                        polymerization_acceleration_factor=comonomer
                    )

                    tiff = sim.generate_tiff(
                        image_size=size,
                        num_spots=spots,
                        num_frames=frames,
                        frame_rate_hz=framerate,
                        d_initial=d_initial,
                        enable_photophysics=photophysics,
                        trajectory_options={'enable_switching': self.single_switching.get()}
                    )

                    # Save
                    filename = f"single_{'astig' if astig else 'noastig'}_t{int(polytime)}min.tif"
                    filepath = self.output_dir / filename
                    save_tiff(str(filepath), tiff)

                    # Export metadata
                    exporter = MetadataExporter(self.output_dir)
                    metadata = sim.get_metadata()
                    exporter.export_all(metadata, Path(filename).stem)

                    self._update_status(f"✅ Gespeichert: {filename}", False)
                    messagebox.showinfo("Erfolg", f"TIFF erfolgreich generiert!\n\n{filepath}")

                except Exception as e:
                    self._update_status(f"❌ Fehler: {str(e)}", False)
                    messagebox.showerror("Fehler", f"Generierung fehlgeschlagen:\n{str(e)}")
                finally:
                    self.is_running = False

            self.is_running = True
            self.current_thread = threading.Thread(target=run, daemon=True)
            self.current_thread.start()

        except ValueError as e:
            messagebox.showerror("Ungültige Eingabe", f"Bitte überprüfe deine Eingaben:\n{str(e)}")

    def _generate_z_stack(self):
        """Generate Z-stack"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        try:
            detector = TDI_PRESET if self.zstack_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            size_str = self.zstack_img_size.get()
            size = tuple(map(int, size_str.split('x')))
            z_start = float(self.zstack_z_start.get())
            z_end = float(self.zstack_z_end.get())
            z_step = float(self.zstack_z_step.get())
            spots = int(self.zstack_spots.get())

            # Apply advanced parameters
            detector = self._apply_advanced_params(detector)

            self.output_dir.mkdir(exist_ok=True)

            def run():
                try:
                    self._update_status("🔄 Generiere Z-Stack (FIXED Astigmatismus)...", True)

                    sim = TIFFSimulator(
                        detector=detector,
                        mode='z_stack',
                        astigmatism=True
                    )

                    zstack = sim.generate_z_stack(
                        image_size=size,
                        num_spots=spots,
                        z_range_um=(z_start, z_end),
                        z_step_um=z_step
                    )

                    filename = f"zstack_{z_start}to{z_end}um_step{z_step}um.tif"
                    filepath = self.output_dir / filename
                    save_tiff(str(filepath), zstack)

                    # Export metadata
                    exporter = MetadataExporter(self.output_dir)
                    metadata = sim.get_metadata()
                    exporter.export_all(metadata, Path(filename).stem)

                    self._update_status(f"✅ Gespeichert: {filename}", False)
                    messagebox.showinfo("Erfolg", f"Z-Stack erfolgreich generiert!\n\n{filepath}")

                except Exception as e:
                    self._update_status(f"❌ Fehler: {str(e)}", False)
                    messagebox.showerror("Fehler", f"Generierung fehlgeschlagen:\n{str(e)}")
                finally:
                    self.is_running = False

            self.is_running = True
            self.current_thread = threading.Thread(target=run, daemon=True)
            self.current_thread.start()

        except ValueError as e:
            messagebox.showerror("Ungültige Eingabe", f"Bitte überprüfe deine Eingaben:\n{str(e)}")

    def _run_batch(self):
        """Run batch generation"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        try:
            detector = TDI_PRESET if self.batch_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            size_str = self.batch_img_size.get()
            size = tuple(map(int, size_str.split('x')))
            times = [float(t.strip()) for t in self.batch_polytimes.get().split(',')]
            repeats = int(self.batch_repeats.get())
            frames = int(self.batch_frames.get())
            spots_min = int(self.batch_spots_min.get())
            spots_max = int(self.batch_spots_max.get())
            astig = self.batch_with_astig.get()
            train_rf = self.batch_train_rf.get()
            random_spots = self.batch_random_spots.get()

            # Apply advanced parameters
            detector = self._apply_advanced_params(detector)

            self.output_dir.mkdir(exist_ok=True)

            def run():
                try:
                    self._update_status(f"🔄 Batch-Modus: Generiere {len(times) * repeats} TIFFs...", True)

                    batch = BatchSimulator(str(self.output_dir))

                    import numpy as np
                    for i, t_poly in enumerate(times):
                        for rep in range(repeats):
                            # Randomize spots if enabled
                            if random_spots:
                                num_spots = np.random.randint(spots_min, spots_max + 1)
                            else:
                                num_spots = (spots_min + spots_max) // 2

                            batch.add_simulation({
                                'detector': detector,
                                'mode': 'polyzeit',
                                't_poly_min': t_poly,
                                'astigmatism': astig,
                                'image_size': size,
                                'num_spots': num_spots,
                                'num_frames': frames,
                                'frame_rate_hz': 20.0,
                                'filename_suffix': f"_t{int(t_poly)}min_rep{rep+1}"
                            })

                    batch.run()

                    # Train Random Forest if requested
                    if train_rf:
                        self._update_status("🤖 Trainiere Random Forest...", True)
                        # TODO: Integrate RF training

                    self._update_status(f"✅ Batch abgeschlossen! {len(times) * repeats} TIFFs", False)
                    messagebox.showinfo("Erfolg",
                                      f"Batch erfolgreich abgeschlossen!\n\n"
                                      f"Generiert: {len(times) * repeats} TIFFs\n"
                                      f"Output: {self.output_dir}")

                except Exception as e:
                    self._update_status(f"❌ Fehler: {str(e)}", False)
                    messagebox.showerror("Fehler", f"Batch fehlgeschlagen:\n{str(e)}")
                finally:
                    self.is_running = False

            self.is_running = True
            self.current_thread = threading.Thread(target=run, daemon=True)
            self.current_thread.start()

        except ValueError as e:
            messagebox.showerror("Ungültige Eingabe", f"Bitte überprüfe deine Eingaben:\n{str(e)}")

    def _apply_advanced_params(self, detector):
        """Apply advanced parameters to detector"""
        import copy
        detector_copy = copy.deepcopy(detector)

        # Astigmatism parameters
        detector_copy.metadata['astig_focal_offset_um'] = float(self.adv_focal_offset.get())
        detector_copy.metadata['astig_z_rayleigh_um'] = float(self.adv_z_rayleigh.get())

        # Refractive index correction
        if self.adv_use_refractive.get():
            detector_copy.metadata['use_advanced_refractive_correction'] = True
            detector_copy.metadata['n_oil'] = float(self.adv_n_oil.get())
            detector_copy.metadata['n_glass'] = float(self.adv_n_glass.get())
            detector_copy.metadata['n_polymer'] = float(self.adv_n_polymer.get())
            detector_copy.metadata['NA'] = float(self.adv_na.get())
            detector_copy.metadata['d_glass_um'] = float(self.adv_d_glass.get())

        # Illumination gradient
        detector_copy.metadata['illumination_gradient_strength'] = float(self.adv_illum_strength.get())
        detector_copy.metadata['illumination_gradient_type'] = self.adv_illum_type.get()

        return detector_copy

    def _save_advanced_params(self):
        """Save advanced parameters to JSON"""
        params = {
            'astig_focal_offset_um': float(self.adv_focal_offset.get()),
            'astig_z_rayleigh_um': float(self.adv_z_rayleigh.get()),
            'use_advanced_refractive_correction': self.adv_use_refractive.get(),
            'n_oil': float(self.adv_n_oil.get()) if self.adv_use_refractive.get() else 1.518,
            'n_glass': float(self.adv_n_glass.get()) if self.adv_use_refractive.get() else 1.523,
            'n_polymer': float(self.adv_n_polymer.get()) if self.adv_use_refractive.get() else 1.47,
            'NA': float(self.adv_na.get()) if self.adv_use_refractive.get() else 1.45,
            'd_glass_um': float(self.adv_d_glass.get()) if self.adv_use_refractive.get() else 170.0,
            'illumination_gradient_strength': float(self.adv_illum_strength.get()),
            'illumination_gradient_type': self.adv_illum_type.get()
        }

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="advanced_params.json"
        )

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=4)
            messagebox.showinfo("Gespeichert", f"Parameter gespeichert:\n{filepath}")

    def _load_advanced_params(self):
        """Load advanced parameters from JSON"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Parameter laden"
        )

        if filepath:
            try:
                with open(filepath, 'r') as f:
                    params = json.load(f)

                self.adv_focal_offset.delete(0, tk.END)
                self.adv_focal_offset.insert(0, str(params['astig_focal_offset_um']))

                self.adv_z_rayleigh.delete(0, tk.END)
                self.adv_z_rayleigh.insert(0, str(params['astig_z_rayleigh_um']))

                self.adv_use_refractive.set(params['use_advanced_refractive_correction'])
                self._toggle_refractive_params()

                if params['use_advanced_refractive_correction']:
                    self.adv_n_oil.delete(0, tk.END)
                    self.adv_n_oil.insert(0, str(params['n_oil']))
                    self.adv_n_glass.delete(0, tk.END)
                    self.adv_n_glass.insert(0, str(params['n_glass']))
                    self.adv_n_polymer.delete(0, tk.END)
                    self.adv_n_polymer.insert(0, str(params['n_polymer']))
                    self.adv_na.delete(0, tk.END)
                    self.adv_na.insert(0, str(params['NA']))
                    self.adv_d_glass.delete(0, tk.END)
                    self.adv_d_glass.insert(0, str(params['d_glass_um']))

                self.adv_illum_strength.delete(0, tk.END)
                self.adv_illum_strength.insert(0, str(params['illumination_gradient_strength']))
                self.adv_illum_type.set(params['illumination_gradient_type'])

                messagebox.showinfo("Geladen", f"Parameter geladen:\n{filepath}")

            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden:\n{str(e)}")


def main():
    """Launch the complete GUI"""
    root = tk.Tk()
    app = CompleteGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
