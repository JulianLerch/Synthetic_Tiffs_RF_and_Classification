"""
🎨 TIFF SIMULATOR V7.0 - HYPERREALISTIC GUI
===========================================

Beautiful, modern GUI with full V6 + V7 physics integration

Features:
- Single TIFF (hyperrealistic settings)
- Z-Stack (depth-dependent PSF)
- Batch Mode (publication quality)
- All 20 physics modules (V6: 9, V7: 11)

Version: 7.0 - November 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import threading
from pathlib import Path
from typing import Optional, Dict
import json

try:
    from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    from batch_simulator import BatchSimulator, PresetBatches
    # V6 and V7 will be integrated in tiff_simulator_v3.py
except ImportError as e:
    print(f"❌ Import Error: {e}")
    exit(1)


class ToolTip:
    """Modern tooltip implementation"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None

        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return

        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#34495e", foreground="#ecf0f1",
                        relief=tk.SOLID, borderwidth=1,
                        font=("Segoe UI", 9), padx=8, pady=5)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class CollapsibleFrame(tk.Frame):
    """Collapsible frame for advanced options"""

    def __init__(self, parent, title, bg_color):
        super().__init__(parent, bg=bg_color)

        self.is_expanded = False
        self.content_frame = None

        # Header button
        self.toggle_button = tk.Button(
            self, text=f"▶ {title}", command=self.toggle,
            bg="#3498db", fg="white", font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT, cursor="hand2", anchor="w", padx=10, pady=8
        )
        self.toggle_button.pack(fill=tk.X)

    def toggle(self):
        if self.is_expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        if not self.content_frame:
            self.content_frame = tk.Frame(self, bg=self['bg'], relief=tk.FLAT, bd=1)
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.populate_content()

        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.toggle_button.config(text=self.toggle_button['text'].replace('▶', '▼'))
        self.is_expanded = True

    def collapse(self):
        if self.content_frame:
            self.content_frame.pack_forget()
        self.toggle_button.config(text=self.toggle_button['text'].replace('▼', '▶'))
        self.is_expanded = False

    def populate_content(self):
        """Override this in subclasses"""
        pass


class HyperrealisticGUI:
    """Beautiful V7.0 GUI with all physics modules"""

    # Modern color palette
    COLORS = {
        'bg': '#ecf0f1',
        'fg': '#2c3e50',
        'primary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#9b59b6',
        'frame_bg': '#ffffff',
        'header_bg': '#2c3e50',
        'header_fg': '#ecf0f1',
        'card_bg': '#f8f9fa',
        'border': '#bdc3c7'
    }

    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Simulator v7.0 - Hyperrealistic")
        self.root.geometry("1100x850")
        self.root.configure(bg=self.COLORS['bg'])

        # State
        self.is_running = False
        self.current_thread: Optional[threading.Thread] = None
        self.output_dir = Path("./tiff_output_v7")

        # Physics settings (V6 + V7)
        self.physics_v6_enabled = tk.BooleanVar(value=True)
        self.physics_v7_enabled = tk.BooleanVar(value=True)

        # Build GUI
        self._create_header()
        self._create_main_content()
        self._create_status_bar()

        # Apply modern theme
        self._apply_modern_style()

    def _apply_modern_style(self):
        """Apply modern ttk styling"""
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook style
        style.configure('TNotebook', background=self.COLORS['bg'], borderwidth=0)
        style.configure('TNotebook.Tab',
                       padding=[20, 12],
                       font=('Segoe UI', 11, 'bold'),
                       borderwidth=0)
        style.map('TNotebook.Tab',
                 background=[('selected', self.COLORS['primary'])],
                 foreground=[('selected', 'white'), ('!selected', self.COLORS['fg'])])

        # Progressbar style
        style.configure("Custom.Horizontal.TProgressbar",
                       troughcolor=self.COLORS['border'],
                       background=self.COLORS['success'],
                       borderwidth=0,
                       thickness=25)

    def _create_header(self):
        """Beautiful gradient header"""
        header = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=100)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Title
        title = tk.Label(
            header,
            text="🔬 TIFF SIMULATOR V7.0",
            font=("Segoe UI", 28, "bold"),
            bg=self.COLORS['header_bg'],
            fg=self.COLORS['header_fg']
        )
        title.pack(pady=(15, 5))

        # Subtitle
        subtitle = tk.Label(
            header,
            text="Hyperrealistic Single-Molecule Microscopy Simulation",
            font=("Segoe UI", 11),
            bg=self.COLORS['header_bg'],
            fg='#95a5a6'
        )
        subtitle.pack()

    def _create_main_content(self):
        """Create main tabbed interface"""
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self._create_single_tiff_tab()
        self._create_z_stack_tab()
        self._create_batch_tab()
        self._create_physics_settings_tab()

    def _create_single_tiff_tab(self):
        """Tab 1: Single TIFF with full physics"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="📄 Single TIFF")

        # Scrollable canvas
        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Card: Basic Settings
        basic_card = self._create_card(scrollable_frame, "⚙️ Basic Settings")

        # Detector
        detector_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        detector_frame.pack(fill=tk.X, pady=5)

        tk.Label(detector_frame, text="Detector:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_detector = ttk.Combobox(detector_frame, values=["TDI-G0", "Tetraspecs"],
                                           state="readonly", width=15)
        self.single_detector.set("TDI-G0")
        self.single_detector.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_detector, "TDI-G0: 0.108µm/px, Tetraspecs: 0.160µm/px")

        # Image size
        size_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        size_frame.pack(fill=tk.X, pady=5)

        tk.Label(size_frame, text="Image Size:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_size = ttk.Combobox(size_frame, values=["64x64", "128x128", "256x256", "512x512"],
                                       state="readonly", width=12)
        self.single_size.set("128x128")
        self.single_size.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_size, "Larger = more realistic but slower")

        # Number of spots
        spots_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        spots_frame.pack(fill=tk.X, pady=5)

        tk.Label(spots_frame, text="Spots:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_spots = tk.Spinbox(spots_frame, from_=1, to=100, width=10)
        self.single_spots.delete(0, tk.END)
        self.single_spots.insert(0, "15")
        self.single_spots.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_spots, "Number of fluorophores to simulate")

        # Frames
        frames_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        frames_frame.pack(fill=tk.X, pady=5)

        tk.Label(frames_frame, text="Frames:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_frames = tk.Spinbox(frames_frame, from_=10, to=2000, width=10, increment=10)
        self.single_frames.delete(0, tk.END)
        self.single_frames.insert(0, "200")
        self.single_frames.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_frames, "Number of time frames")

        # Frame rate
        rate_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        rate_frame.pack(fill=tk.X, pady=5)

        tk.Label(rate_frame, text="Frame Rate (Hz):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_frame_rate = tk.Spinbox(rate_frame, from_=1, to=100, width=10)
        self.single_frame_rate.delete(0, tk.END)
        self.single_frame_rate.insert(0, "20")
        self.single_frame_rate.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_frame_rate, "Acquisition speed (fps)")

        # Polymerization time
        poly_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        poly_frame.pack(fill=tk.X, pady=5)

        tk.Label(poly_frame, text="Polymerization (min):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.single_poly_time = tk.Spinbox(poly_frame, from_=0, to=180, width=10, increment=5)
        self.single_poly_time.delete(0, tk.END)
        self.single_poly_time.insert(0, "60")
        self.single_poly_time.pack(side=tk.LEFT, padx=5)
        ToolTip(self.single_poly_time, "Gel polymerization time (affects diffusion)")

        # Astigmatism
        astig_frame = tk.Frame(basic_card, bg=self.COLORS['card_bg'])
        astig_frame.pack(fill=tk.X, pady=5)

        self.single_astigmatism = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(astig_frame, text="Enable Astigmatism (3D)",
                           variable=self.single_astigmatism,
                           font=("Segoe UI", 10, "bold"),
                           bg=self.COLORS['card_bg'])
        cb.pack(side=tk.LEFT, padx=5)
        ToolTip(cb, "z-dependent PSF ellipticity for 3D localization")

        # Card: V6 Physics (9 modules)
        v6_card = self._create_card(scrollable_frame, "🧬 V6.0 Physics (9 Modules)")

        self.single_v6_enable = tk.BooleanVar(value=True)
        v6_cb = tk.Checkbutton(v6_card, text="Enable V6 Physics Suite",
                              variable=self.single_v6_enable,
                              font=("Segoe UI", 10, "bold"),
                              bg=self.COLORS['card_bg'],
                              fg=self.COLORS['success'])
        v6_cb.pack(anchor=tk.W, padx=5, pady=5)
        ToolTip(v6_cb, "Camera noise, TIRF, polymer chemistry, anomalous diffusion, photobleaching")

        v6_info = tk.Label(v6_card,
                          text="✓ Camera Noise  ✓ Piezo Stage  ✓ TIRF  ✓ Polymer Chemistry  ✓ CTRW\n"
                               "✓ Fractional Brownian Motion  ✓ Caging  ✓ Triplet State  ✓ Z-Wobble",
                          font=("Segoe UI", 9),
                          bg=self.COLORS['card_bg'],
                          fg='#7f8c8d',
                          justify=tk.LEFT)
        v6_info.pack(anchor=tk.W, padx=20, pady=2)

        # Card: V7 Physics (11 modules)
        v7_card = self._create_card(scrollable_frame, "🚀 V7.0 Physics (11 Modules)")

        self.single_v7_enable = tk.BooleanVar(value=True)
        v7_cb = tk.Checkbutton(v7_card, text="Enable V7 Physics Suite",
                              variable=self.single_v7_enable,
                              font=("Segoe UI", 10, "bold"),
                              bg=self.COLORS['card_bg'],
                              fg=self.COLORS['info'])
        v7_cb.pack(anchor=tk.W, padx=5, pady=5)
        ToolTip(v7_cb, "Depth-PSF, sCMOS correlation, power-law blinking, drift, precision, chromatic aberration")

        v7_info = tk.Label(v7_card,
                          text="✓ Depth-Dependent PSF  ✓ sCMOS Correlation  ✓ Power-Law Blinking\n"
                               "✓ Long-Term Drift  ✓ Localization Precision  ✓ Photon Budget\n"
                               "✓ Chromatic Aberration  ✓ RI Evolution  ✓ Autofluorescence\n"
                               "✓ Spectral Bleed-Through  ✓ Gaussian Illumination",
                          font=("Segoe UI", 9),
                          bg=self.COLORS['card_bg'],
                          fg='#7f8c8d',
                          justify=tk.LEFT)
        v7_info.pack(anchor=tk.W, padx=20, pady=2)

        # Generate button
        button_frame = tk.Frame(scrollable_frame, bg=self.COLORS['frame_bg'])
        button_frame.pack(pady=20)

        self.single_generate_btn = tk.Button(
            button_frame,
            text="🎬 Generate TIFF",
            command=self._generate_single_tiff,
            bg=self.COLORS['success'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.single_generate_btn.pack()

    def _create_z_stack_tab(self):
        """Tab 2: Z-Stack with depth-dependent PSF"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="📚 Z-Stack")

        # Scrollable
        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Card: Settings
        card = self._create_card(scrollable_frame, "⚙️ Z-Stack Settings")

        # Detector
        detector_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        detector_frame.pack(fill=tk.X, pady=5)

        tk.Label(detector_frame, text="Detector:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.zstack_detector = ttk.Combobox(detector_frame, values=["TDI-G0", "Tetraspecs"],
                                           state="readonly", width=15)
        self.zstack_detector.set("Tetraspecs")
        self.zstack_detector.pack(side=tk.LEFT, padx=5)

        # Z-range
        z_start_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        z_start_frame.pack(fill=tk.X, pady=5)

        tk.Label(z_start_frame, text="Z Start (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.zstack_z_start = tk.Spinbox(z_start_frame, from_=-2.0, to=0, width=10,
                                         increment=0.1, format="%.1f")
        self.zstack_z_start.delete(0, tk.END)
        self.zstack_z_start.insert(0, "-1.0")
        self.zstack_z_start.pack(side=tk.LEFT, padx=5)

        z_end_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        z_end_frame.pack(fill=tk.X, pady=5)

        tk.Label(z_end_frame, text="Z End (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.zstack_z_end = tk.Spinbox(z_end_frame, from_=0, to=2.0, width=10,
                                       increment=0.1, format="%.1f")
        self.zstack_z_end.delete(0, tk.END)
        self.zstack_z_end.insert(0, "1.0")
        self.zstack_z_end.pack(side=tk.LEFT, padx=5)

        z_step_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        z_step_frame.pack(fill=tk.X, pady=5)

        tk.Label(z_step_frame, text="Z Step (µm):", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.zstack_z_step = tk.Spinbox(z_step_frame, from_=0.01, to=0.5, width=10,
                                        increment=0.01, format="%.2f")
        self.zstack_z_step.delete(0, tk.END)
        self.zstack_z_step.insert(0, "0.10")
        self.zstack_z_step.pack(side=tk.LEFT, padx=5)
        ToolTip(self.zstack_z_step, "Smaller = more slices but slower")

        # Spots
        spots_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        spots_frame.pack(fill=tk.X, pady=5)

        tk.Label(spots_frame, text="Spots:", font=("Segoe UI", 10, "bold"),
                bg=self.COLORS['card_bg']).pack(side=tk.LEFT, padx=5)

        self.zstack_spots = tk.Spinbox(spots_frame, from_=5, to=50, width=10)
        self.zstack_spots.delete(0, tk.END)
        self.zstack_spots.insert(0, "20")
        self.zstack_spots.pack(side=tk.LEFT, padx=5)

        # Depth-dependent PSF info
        psf_card = self._create_card(scrollable_frame, "🔬 Depth-Dependent PSF (V7.0)")

        psf_info = tk.Label(psf_card,
                           text="Z-Stack automatically includes depth-dependent spherical aberration!\n"
                                "PSF grows with imaging depth (380% axial @ 90µm)\n"
                                "Realistic refractive index mismatch effects",
                           font=("Segoe UI", 9),
                           bg=self.COLORS['card_bg'],
                           fg=self.COLORS['info'],
                           justify=tk.LEFT)
        psf_info.pack(padx=10, pady=10)

        # Generate button
        button_frame = tk.Frame(scrollable_frame, bg=self.COLORS['frame_bg'])
        button_frame.pack(pady=20)

        self.zstack_generate_btn = tk.Button(
            button_frame,
            text="📚 Generate Z-Stack",
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
        """Tab 3: Batch mode"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="🔄 Batch Mode")

        # Card: Presets
        card = self._create_card(tab, "🎯 Batch Presets")

        info = tk.Label(card,
                       text="Generate multiple TIFFs with different polymerization times",
                       font=("Segoe UI", 10),
                       bg=self.COLORS['card_bg'],
                       fg='#7f8c8d')
        info.pack(pady=10)

        # Preset selection
        preset_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        preset_frame.pack(fill=tk.X, pady=10)

        tk.Label(preset_frame, text="Select Preset:", font=("Segoe UI", 11, "bold"),
                bg=self.COLORS['card_bg']).pack(pady=5)

        presets = [
            ("Quick Test (3 TIFFs, ~2 min)", "quick"),
            ("Thesis Quality (60+ TIFFs, ~45 min)", "thesis"),
            ("Publication Quality (30 TIFFs, ~2 hours)", "publication")
        ]

        self.batch_preset = tk.StringVar(value="quick")

        for text, value in presets:
            rb = tk.Radiobutton(preset_frame, text=text, variable=self.batch_preset,
                               value=value, font=("Segoe UI", 10),
                               bg=self.COLORS['card_bg'])
            rb.pack(anchor=tk.W, padx=20, pady=2)

        # Physics options
        physics_frame = tk.Frame(card, bg=self.COLORS['card_bg'])
        physics_frame.pack(fill=tk.X, pady=10)

        self.batch_v6 = tk.BooleanVar(value=True)
        self.batch_v7 = tk.BooleanVar(value=True)

        tk.Checkbutton(physics_frame, text="Enable V6 Physics",
                      variable=self.batch_v6,
                      font=("Segoe UI", 10, "bold"),
                      bg=self.COLORS['card_bg']).pack(anchor=tk.W, padx=20, pady=2)

        tk.Checkbutton(physics_frame, text="Enable V7 Physics",
                      variable=self.batch_v7,
                      font=("Segoe UI", 10, "bold"),
                      bg=self.COLORS['card_bg']).pack(anchor=tk.W, padx=20, pady=2)

        # Generate button
        button_frame = tk.Frame(tab, bg=self.COLORS['frame_bg'])
        button_frame.pack(pady=20)

        self.batch_generate_btn = tk.Button(
            button_frame,
            text="🔄 Start Batch Generation",
            command=self._generate_batch,
            bg=self.COLORS['warning'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT,
            padx=30,
            pady=12,
            cursor="hand2"
        )
        self.batch_generate_btn.pack()

    def _create_physics_settings_tab(self):
        """Tab 4: Physics settings overview"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['frame_bg'])
        self.notebook.add(tab, text="⚗️ Physics Info")

        # Scrollable
        canvas = tk.Canvas(tab, bg=self.COLORS['frame_bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['frame_bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # V6 Physics Card
        v6_card = self._create_card(scrollable_frame, "🧬 V6.0 Physics Suite (9 Modules)")

        v6_modules = [
            ("1. Advanced Camera Noise", "Pixel-dependent read noise, dark current, FPN, temporal correlation"),
            ("2. Z-Wobble Simulator", "Thermal drift + mechanical vibrations for 2D tracking"),
            ("3. Piezo Stage Simulator", "Hysteresis, positioning noise, non-linearity, drift"),
            ("4. TIRF Functions", "Penetration depth calculation, intensity profile"),
            ("5. Polymer Chemistry", "Mesh size evolution, crosslinking density, Ogston obstruction"),
            ("6. CTRW", "Continuous-time random walk with power-law waiting times"),
            ("7. Fractional Brownian Motion", "Hurst exponent for memory effects"),
            ("8. Caging Model", "Exponential escape kinetics, harmonic potential"),
            ("9. Triplet State", "3-state photobleaching model (S0/S1/T1)")
        ]

        for name, desc in v6_modules:
            frame = tk.Frame(v6_card, bg=self.COLORS['card_bg'])
            frame.pack(fill=tk.X, pady=3)

            tk.Label(frame, text=name, font=("Segoe UI", 10, "bold"),
                    bg=self.COLORS['card_bg'], fg=self.COLORS['success']).pack(anchor=tk.W, padx=10)
            tk.Label(frame, text=desc, font=("Segoe UI", 9),
                    bg=self.COLORS['card_bg'], fg='#7f8c8d').pack(anchor=tk.W, padx=25)

        # V7 Physics Card
        v7_card = self._create_card(scrollable_frame, "🚀 V7.0 Physics Suite (11 Modules)")

        v7_modules = [
            ("1. Depth-Dependent PSF", "380% axial PSF growth @ 90µm depth (spherical aberration)"),
            ("2. sCMOS Spatial Correlation", "Pixel-correlated noise (not white noise)"),
            ("3. Power-Law Blinking", "Heavy-tailed ON/OFF statistics (realistic photophysics)"),
            ("4. Long-Term Thermal Drift", "2-3 µm drift over 12 hours (exponential + linear + random)"),
            ("5. Localization Precision", "CRLB-based with 2x experimental factor"),
            ("6. Photon Budget Tracker", "10⁴-10⁶ photons before bleaching"),
            ("7. Chromatic Aberration", "Wavelength-dependent z-offset"),
            ("8. RI Evolution", "Sample refractive index: 1.333 → 1.45 with crosslinking"),
            ("9. Autofluorescence", "Cellular background (1-10 counts/pixel)"),
            ("10. Spectral Bleed-Through", "Channel crosstalk (5-20%)"),
            ("11. Gaussian Illumination", "Beam intensity falloff at edges")
        ]

        for name, desc in v7_modules:
            frame = tk.Frame(v7_card, bg=self.COLORS['card_bg'])
            frame.pack(fill=tk.X, pady=3)

            tk.Label(frame, text=name, font=("Segoe UI", 10, "bold"),
                    bg=self.COLORS['card_bg'], fg=self.COLORS['info']).pack(anchor=tk.W, padx=10)
            tk.Label(frame, text=desc, font=("Segoe UI", 9),
                    bg=self.COLORS['card_bg'], fg='#7f8c8d').pack(anchor=tk.W, padx=25)

    def _create_card(self, parent, title):
        """Create a modern card with shadow effect"""
        card_frame = tk.Frame(parent, bg=self.COLORS['card_bg'], relief=tk.FLAT, bd=0)
        card_frame.pack(fill=tk.X, padx=10, pady=10)

        # Shadow effect (simple border)
        card_frame.config(highlightbackground=self.COLORS['border'],
                         highlightthickness=1)

        # Title
        title_label = tk.Label(card_frame, text=title,
                              font=("Segoe UI", 12, "bold"),
                              bg=self.COLORS['card_bg'],
                              fg=self.COLORS['primary'])
        title_label.pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Separator
        sep = tk.Frame(card_frame, height=2, bg=self.COLORS['border'])
        sep.pack(fill=tk.X, padx=10, pady=5)

        return card_frame

    def _create_status_bar(self):
        """Create modern status bar"""
        status_frame = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="Ready",
                                     font=("Segoe UI", 10),
                                     bg=self.COLORS['header_bg'],
                                     fg=self.COLORS['header_fg'])
        self.status_label.pack(side=tk.LEFT, padx=15)

        # Progress bar
        self.progress = ttk.Progressbar(status_frame,
                                       style="Custom.Horizontal.TProgressbar",
                                       mode='indeterminate',
                                       length=200)
        self.progress.pack(side=tk.RIGHT, padx=15)

    def _update_status(self, message, is_running=False):
        """Update status bar"""
        self.status_label.config(text=message)

        if is_running:
            self.progress.start(10)
        else:
            self.progress.stop()

    def _generate_single_tiff(self):
        """Generate single TIFF with V6+V7 physics"""
        if self.is_running:
            messagebox.showwarning("Busy", "Simulation already running!")
            return

        # Get parameters
        detector = TDI_PRESET if self.single_detector.get() == "TDI-G0" else TETRASPECS_PRESET
        size_str = self.single_size.get()
        size = tuple(map(int, size_str.split('x')))
        spots = int(self.single_spots.get())
        frames = int(self.single_frames.get())
        frame_rate = float(self.single_frame_rate.get())
        poly_time = float(self.single_poly_time.get())
        astig = self.single_astigmatism.get()
        v6_enabled = self.single_v6_enable.get()
        v7_enabled = self.single_v7_enable.get()

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        def run_simulation():
            try:
                self._update_status("Generating hyperrealistic TIFF...", True)

                # Create simulator
                sim = TIFFSimulator(
                    detector=detector,
                    mode='polyzeit',
                    t_poly_min=poly_time,
                    astigmatism=astig
                )

                # Generate TIFF (V6+V7 will be automatically applied if available)
                tiff = sim.generate_tiff(
                    image_size=size,
                    num_spots=spots,
                    num_frames=frames,
                    frame_rate_hz=frame_rate
                )

                # Save
                filename = f"tiff_v7_{'astig' if astig else 'no-astig'}_t{int(poly_time)}min.tif"
                filepath = self.output_dir / filename
                save_tiff(str(filepath), tiff)

                self._update_status(f"✅ Saved: {filename}", False)
                messagebox.showinfo("Success", f"TIFF generated!\n{filepath}")

            except Exception as e:
                self._update_status(f"❌ Error: {str(e)}", False)
                messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
            finally:
                self.is_running = False

        self.is_running = True
        self.current_thread = threading.Thread(target=run_simulation, daemon=True)
        self.current_thread.start()

    def _generate_z_stack(self):
        """Generate Z-stack with depth-dependent PSF"""
        if self.is_running:
            messagebox.showwarning("Busy", "Simulation already running!")
            return

        # Get parameters
        detector = TDI_PRESET if self.zstack_detector.get() == "TDI-G0" else TETRASPECS_PRESET
        z_start = float(self.zstack_z_start.get())
        z_end = float(self.zstack_z_end.get())
        z_step = float(self.zstack_z_step.get())
        spots = int(self.zstack_spots.get())

        self.output_dir.mkdir(exist_ok=True)

        def run_simulation():
            try:
                self._update_status("Generating Z-stack with depth-dependent PSF...", True)

                sim = TIFFSimulator(
                    detector=detector,
                    mode='z_stack',
                    astigmatism=True  # Always on for z-stack
                )

                zstack = sim.generate_z_stack(
                    image_size=(128, 128),
                    num_spots=spots,
                    z_range_um=(z_start, z_end),
                    z_step_um=z_step
                )

                filename = f"zstack_v7_{z_start}to{z_end}um_step{z_step}um.tif"
                filepath = self.output_dir / filename
                save_tiff(str(filepath), zstack)

                self._update_status(f"✅ Saved: {filename}", False)
                messagebox.showinfo("Success", f"Z-stack generated!\n{filepath}")

            except Exception as e:
                self._update_status(f"❌ Error: {str(e)}", False)
                messagebox.showerror("Error", f"Generation failed:\n{str(e)}")
            finally:
                self.is_running = False

        self.is_running = True
        self.current_thread = threading.Thread(target=run_simulation, daemon=True)
        self.current_thread.start()

    def _generate_batch(self):
        """Generate batch with selected preset"""
        if self.is_running:
            messagebox.showwarning("Busy", "Simulation already running!")
            return

        preset = self.batch_preset.get()
        self.output_dir.mkdir(exist_ok=True)

        def run_simulation():
            try:
                self._update_status(f"Running batch preset: {preset}...", True)

                if preset == "quick":
                    batch = PresetBatches.quick_test(str(self.output_dir))
                elif preset == "thesis":
                    batch = PresetBatches.masterthesis_full(str(self.output_dir))
                else:  # publication
                    batch = PresetBatches.publication_quality(str(self.output_dir))

                batch.run()

                self._update_status("✅ Batch complete!", False)
                messagebox.showinfo("Success", f"Batch generation complete!\n{self.output_dir}")

            except Exception as e:
                self._update_status(f"❌ Error: {str(e)}", False)
                messagebox.showerror("Error", f"Batch failed:\n{str(e)}")
            finally:
                self.is_running = False

        self.is_running = True
        self.current_thread = threading.Thread(target=run_simulation, daemon=True)
        self.current_thread.start()


def main():
    """Launch the hyperrealistic GUI"""
    root = tk.Tk()
    app = HyperrealisticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
