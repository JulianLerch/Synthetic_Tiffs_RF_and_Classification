"""
🎨 TIFF SIMULATOR - MODERNE GUI
=================================

Benutzerfreundliche Oberfläche zur Generierung synthetischer TIFF-Mikroskopie-Daten

Features:
- Single TIFF Simulation (mit/ohne Astigmatismus)
- Z-Stack Generierung
- Batch-Modus für Parameterserien

Version: 5.0 - November 2025
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from typing import Optional

try:
    from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    from batch_simulator import BatchSimulator, PresetBatches
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Bitte stelle sicher, dass alle erforderlichen Dateien vorhanden sind:")
    print("  - tiff_simulator_v3.py")
    print("  - batch_simulator.py")
    exit(1)


class ModernTIFFSimulatorGUI:
    """Moderne GUI für TIFF Simulator mit 3 Hauptfunktionen"""

    # Farbschema
    COLORS = {
        'bg': '#f5f6fa',
        'fg': '#2c3e50',
        'primary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'frame_bg': '#ffffff',
        'header_bg': '#34495e',
        'header_fg': '#ecf0f1'
    }

    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Simulator v5.0")
        self.root.geometry("900x750")
        self.root.configure(bg=self.COLORS['bg'])

        # Simulationsstatus
        self.is_running = False
        self.current_thread: Optional[threading.Thread] = None

        # Output-Verzeichnis
        self.output_dir = Path("./tiff_output")

        # GUI aufbauen
        self._create_header()
        self._create_notebook()
        self._create_status_bar()

    def _create_header(self):
        """Erstellt den Header"""
        header = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=80)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="🔬 TIFF Simulator",
            font=("Segoe UI", 24, "bold"),
            bg=self.COLORS['header_bg'],
            fg=self.COLORS['header_fg']
        )
        title.pack(pady=20)

    def _create_notebook(self):
        """Erstellt das Tab-System"""
        notebook_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        notebook_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.COLORS['bg'])
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Segoe UI', 10))
        style.map('TNotebook.Tab', background=[('selected', self.COLORS['primary'])])

        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Single TIFF
        self._create_single_tiff_tab()

        # Tab 2: Z-Stack
        self._create_z_stack_tab()

        # Tab 3: Batch Mode
        self._create_batch_tab()

    def _create_single_tiff_tab(self):
        """Tab für einzelne TIFF-Generierung"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg'])
        self.notebook.add(tab, text="📄 Einzelnes TIFF")

        # Scrollbarer Container
        canvas = tk.Canvas(tab, bg=self.COLORS['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === DETEKTOR-SEKTION ===
        self._create_section(scrollable_frame, "🔍 Detektor-Einstellungen")
        detector_frame = self._create_frame(scrollable_frame)

        tk.Label(detector_frame, text="Detektor:", font=("Segoe UI", 10), bg=self.COLORS['frame_bg']).grid(
            row=0, column=0, sticky='w', padx=10, pady=5
        )
        self.single_detector = tk.StringVar(value="TDI-G0")
        detector_combo = ttk.Combobox(
            detector_frame,
            textvariable=self.single_detector,
            values=["TDI-G0", "Tetraspecs"],
            state='readonly',
            width=25
        )
        detector_combo.grid(row=0, column=1, sticky='w', padx=10, pady=5)

        # === BILD-PARAMETER ===
        self._create_section(scrollable_frame, "📐 Bild-Parameter")
        img_frame = self._create_frame(scrollable_frame)

        params = [
            ("Bildgröße (Pixel):", "single_img_size", "128"),
            ("Anzahl Spots:", "single_num_spots", "15"),
            ("Anzahl Frames:", "single_num_frames", "200"),
            ("Frame Rate (Hz):", "single_frame_rate", "20.0"),
        ]

        self.single_vars = {}
        for idx, (label, var_name, default) in enumerate(params):
            tk.Label(img_frame, text=label, font=("Segoe UI", 10), bg=self.COLORS['frame_bg']).grid(
                row=idx, column=0, sticky='w', padx=10, pady=5
            )
            var = tk.StringVar(value=default)
            self.single_vars[var_name] = var
            entry = tk.Entry(img_frame, textvariable=var, width=28, font=("Segoe UI", 10))
            entry.grid(row=idx, column=1, sticky='w', padx=10, pady=5)

        # === POLYMERISATION ===
        self._create_section(scrollable_frame, "⏱️ Polymerisation")
        poly_frame = self._create_frame(scrollable_frame)

        tk.Label(poly_frame, text="Polymerisationszeit (min):", font=("Segoe UI", 10), bg=self.COLORS['frame_bg']).grid(
            row=0, column=0, sticky='w', padx=10, pady=5
        )
        self.single_poly_time = tk.StringVar(value="60.0")
        tk.Entry(poly_frame, textvariable=self.single_poly_time, width=28, font=("Segoe UI", 10)).grid(
            row=0, column=1, sticky='w', padx=10, pady=5
        )

        # === ASTIGMATISMUS ===
        self._create_section(scrollable_frame, "🔬 3D / Astigmatismus")
        astig_frame = self._create_frame(scrollable_frame)

        self.single_astig = tk.BooleanVar(value=False)
        tk.Checkbutton(
            astig_frame,
            text="Astigmatismus aktivieren (für 3D-Tracking)",
            variable=self.single_astig,
            font=("Segoe UI", 10),
            bg=self.COLORS['frame_bg']
        ).grid(row=0, column=0, sticky='w', padx=10, pady=10)

        # === BUTTONS ===
        button_frame = tk.Frame(scrollable_frame, bg=self.COLORS['bg'])
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        self.single_generate_btn = tk.Button(
            button_frame,
            text="🚀 TIFF Generieren",
            command=self._generate_single_tiff,
            bg=self.COLORS['success'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.single_generate_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="📁 Output wählen",
            command=self._select_output_dir,
            bg=self.COLORS['primary'],
            fg='white',
            font=("Segoe UI", 10),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

    def _create_z_stack_tab(self):
        """Tab für Z-Stack Generierung"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg'])
        self.notebook.add(tab, text="📚 Z-Stack")

        # Scrollbarer Container
        canvas = tk.Canvas(tab, bg=self.COLORS['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # === DETEKTOR ===
        self._create_section(scrollable_frame, "🔍 Detektor-Einstellungen")
        detector_frame = self._create_frame(scrollable_frame)

        tk.Label(detector_frame, text="Detektor:", font=("Segoe UI", 10), bg=self.COLORS['frame_bg']).grid(
            row=0, column=0, sticky='w', padx=10, pady=5
        )
        self.zstack_detector = tk.StringVar(value="TDI-G0")
        ttk.Combobox(
            detector_frame,
            textvariable=self.zstack_detector,
            values=["TDI-G0", "Tetraspecs"],
            state='readonly',
            width=25
        ).grid(row=0, column=1, sticky='w', padx=10, pady=5)

        # === Z-STACK PARAMETER ===
        self._create_section(scrollable_frame, "📏 Z-Stack Parameter")
        zstack_frame = self._create_frame(scrollable_frame)

        params = [
            ("z-Start (µm):", "z_start", "-1.0"),
            ("z-Ende (µm):", "z_end", "1.0"),
            ("z-Schritt (µm):", "z_step", "0.1"),
            ("Bildgröße (Pixel):", "z_img_size", "128"),
            ("Anzahl Spots:", "z_num_spots", "20"),
        ]

        self.zstack_vars = {}
        for idx, (label, var_name, default) in enumerate(params):
            tk.Label(zstack_frame, text=label, font=("Segoe UI", 10), bg=self.COLORS['frame_bg']).grid(
                row=idx, column=0, sticky='w', padx=10, pady=5
            )
            var = tk.StringVar(value=default)
            self.zstack_vars[var_name] = var
            tk.Entry(zstack_frame, textvariable=var, width=28, font=("Segoe UI", 10)).grid(
                row=idx, column=1, sticky='w', padx=10, pady=5
            )

        # Info
        info_frame = tk.Frame(scrollable_frame, bg='#e8f4f8', relief=tk.RIDGE, borderwidth=2)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            info_frame,
            text="ℹ️ Z-Stacks werden automatisch mit Astigmatismus generiert",
            font=("Segoe UI", 9, "italic"),
            bg='#e8f4f8',
            fg='#2980b9'
        ).pack(padx=10, pady=8)

        # === BUTTONS ===
        button_frame = tk.Frame(scrollable_frame, bg=self.COLORS['bg'])
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        self.zstack_generate_btn = tk.Button(
            button_frame,
            text="🚀 Z-Stack Generieren",
            command=self._generate_z_stack,
            bg=self.COLORS['success'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.zstack_generate_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="📁 Output wählen",
            command=self._select_output_dir,
            bg=self.COLORS['primary'],
            fg='white',
            font=("Segoe UI", 10),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

    def _create_batch_tab(self):
        """Tab für Batch-Modus"""
        tab = tk.Frame(self.notebook, bg=self.COLORS['bg'])
        self.notebook.add(tab, text="🔄 Batch-Modus")

        # Info
        info_frame = tk.Frame(tab, bg='#fff9e6', relief=tk.RIDGE, borderwidth=2)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            info_frame,
            text="Generiere mehrere TIFFs automatisch mit vordefinierten Parametern",
            font=("Segoe UI", 11, "bold"),
            bg='#fff9e6',
            fg='#856404'
        ).pack(padx=10, pady=5)

        tk.Label(
            info_frame,
            text="Wähle ein Preset aus und starte die Batch-Generierung",
            font=("Segoe UI", 9),
            bg='#fff9e6',
            fg='#856404'
        ).pack(padx=10, pady=(0, 10))

        # === PRESET-AUSWAHL ===
        self._create_section(tab, "📦 Batch-Presets")
        preset_frame = self._create_frame(tab)

        self.batch_preset = tk.StringVar(value="quick")

        presets = [
            ("🚀 Quick Test", "quick", "3 TIFFs, ~2 Minuten"),
            ("🎓 Thesis Quality", "thesis", "~60 TIFFs, ~45 Minuten"),
            ("📄 Publication Quality", "publication", "~30 TIFFs, ~2 Stunden"),
        ]

        for idx, (title, value, desc) in enumerate(presets):
            frame = tk.Frame(preset_frame, bg=self.COLORS['frame_bg'])
            frame.pack(fill=tk.X, padx=10, pady=5)

            tk.Radiobutton(
                frame,
                text=title,
                variable=self.batch_preset,
                value=value,
                font=("Segoe UI", 11, "bold"),
                bg=self.COLORS['frame_bg']
            ).pack(anchor='w')

            tk.Label(
                frame,
                text=desc,
                font=("Segoe UI", 9, "italic"),
                bg=self.COLORS['frame_bg'],
                fg='#7f8c8d'
            ).pack(anchor='w', padx=25)

        # === CUSTOM PARAMETER ===
        self._create_section(tab, "⚙️ Custom Batch (Optional)")
        custom_frame = self._create_frame(tab)

        tk.Label(
            custom_frame,
            text="Polymerisationszeiten (komma-separiert, leer=Preset nutzen):",
            font=("Segoe UI", 10),
            bg=self.COLORS['frame_bg']
        ).grid(row=0, column=0, sticky='w', padx=10, pady=5)

        self.batch_custom_times = tk.StringVar(value="")
        tk.Entry(
            custom_frame,
            textvariable=self.batch_custom_times,
            width=40,
            font=("Segoe UI", 10)
        ).grid(row=0, column=1, sticky='w', padx=10, pady=5)

        tk.Label(
            custom_frame,
            text="Beispiel: 30,60,90,120",
            font=("Segoe UI", 8, "italic"),
            bg=self.COLORS['frame_bg'],
            fg='#95a5a6'
        ).grid(row=1, column=1, sticky='w', padx=10)

        # === BUTTONS ===
        button_frame = tk.Frame(tab, bg=self.COLORS['bg'])
        button_frame.pack(fill=tk.X, padx=10, pady=20)

        self.batch_generate_btn = tk.Button(
            button_frame,
            text="🚀 Batch Starten",
            command=self._run_batch,
            bg=self.COLORS['success'],
            fg='white',
            font=("Segoe UI", 12, "bold"),
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.batch_generate_btn.pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="📁 Output wählen",
            command=self._select_output_dir,
            bg=self.COLORS['primary'],
            fg='white',
            font=("Segoe UI", 10),
            padx=15,
            pady=10,
            relief=tk.FLAT,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)

    def _create_status_bar(self):
        """Erstellt die Status-Bar"""
        self.status_frame = tk.Frame(self.root, bg=self.COLORS['header_bg'], height=40)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(
            self.status_frame,
            text=f"📁 Output: {self.output_dir.absolute()}",
            font=("Segoe UI", 9),
            bg=self.COLORS['header_bg'],
            fg=self.COLORS['header_fg'],
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(
            self.status_frame,
            mode='indeterminate',
            length=200
        )

    def _create_section(self, parent, title):
        """Erstellt eine Sektion mit Titel"""
        frame = tk.Frame(parent, bg=self.COLORS['bg'])
        frame.pack(fill=tk.X, padx=10, pady=(15, 5))

        tk.Label(
            frame,
            text=title,
            font=("Segoe UI", 12, "bold"),
            bg=self.COLORS['bg'],
            fg=self.COLORS['fg']
        ).pack(anchor='w')

    def _create_frame(self, parent):
        """Erstellt einen weißen Content-Frame"""
        frame = tk.Frame(parent, bg=self.COLORS['frame_bg'], relief=tk.RIDGE, borderwidth=1)
        frame.pack(fill=tk.X, padx=10, pady=5)
        return frame

    def _select_output_dir(self):
        """Wählt Output-Verzeichnis"""
        directory = filedialog.askdirectory(initialdir=self.output_dir)
        if directory:
            self.output_dir = Path(directory)
            self.status_label.config(text=f"📁 Output: {self.output_dir.absolute()}")

    def _update_status(self, message: str, is_running: bool = False):
        """Aktualisiert Status"""
        self.status_label.config(text=message)
        if is_running:
            self.progress.pack(side=tk.RIGHT, padx=10)
            self.progress.start(10)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _disable_buttons(self):
        """Deaktiviert alle Generate-Buttons"""
        self.single_generate_btn.config(state=tk.DISABLED)
        self.zstack_generate_btn.config(state=tk.DISABLED)
        self.batch_generate_btn.config(state=tk.DISABLED)
        self.is_running = True

    def _enable_buttons(self):
        """Aktiviert alle Generate-Buttons"""
        self.single_generate_btn.config(state=tk.NORMAL)
        self.zstack_generate_btn.config(state=tk.NORMAL)
        self.batch_generate_btn.config(state=tk.NORMAL)
        self.is_running = False

    # =====================================================================
    # GENERIERUNGS-FUNKTIONEN
    # =====================================================================

    def _generate_single_tiff(self):
        """Generiert einzelnes TIFF"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        try:
            # Parameter auslesen
            detector = TDI_PRESET if self.single_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            img_size = int(self.single_vars['single_img_size'].get())
            num_spots = int(self.single_vars['single_num_spots'].get())
            num_frames = int(self.single_vars['single_num_frames'].get())
            frame_rate = float(self.single_vars['single_frame_rate'].get())
            poly_time = float(self.single_poly_time.get())
            astig = self.single_astig.get()

            # Thread starten
            def run():
                try:
                    self._update_status("🔄 Generiere TIFF...", is_running=True)

                    # Simulator erstellen
                    mode = 'polyzeit_astig' if astig else 'polyzeit'
                    sim = TIFFSimulator(
                        detector=detector,
                        mode=mode,
                        t_poly_min=poly_time,
                        astigmatism=astig
                    )

                    # TIFF generieren
                    tiff_stack = sim.generate_tiff(
                        image_size=(img_size, img_size),
                        num_spots=num_spots,
                        num_frames=num_frames,
                        frame_rate_hz=frame_rate
                    )

                    # Speichern
                    self.output_dir.mkdir(exist_ok=True, parents=True)
                    astig_suffix = "_astig" if astig else ""
                    filename = f"{detector.name.lower()}_t{int(poly_time)}min{astig_suffix}.tif"
                    output_path = self.output_dir / filename
                    save_tiff(str(output_path), tiff_stack)

                    # Metadaten exportieren
                    from metadata_exporter import MetadataExporter
                    exporter = MetadataExporter(self.output_dir)
                    metadata = sim.get_metadata()
                    exporter.export_all(metadata, Path(filename).stem)

                    self._update_status(f"✅ Erfolgreich: {filename}", is_running=False)
                    messagebox.showinfo("Erfolg", f"TIFF erfolgreich erstellt!\n\n{output_path}")

                except Exception as e:
                    self._update_status(f"❌ Fehler: {str(e)}", is_running=False)
                    messagebox.showerror("Fehler", f"Fehler bei der Generierung:\n{str(e)}")
                finally:
                    self._enable_buttons()

            self._disable_buttons()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()

        except ValueError as e:
            messagebox.showerror("Ungültige Eingabe", f"Bitte überprüfe deine Eingaben:\n{str(e)}")

    def _generate_z_stack(self):
        """Generiert Z-Stack"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        try:
            # Parameter auslesen
            detector = TDI_PRESET if self.zstack_detector.get() == "TDI-G0" else TETRASPECS_PRESET
            z_start = float(self.zstack_vars['z_start'].get())
            z_end = float(self.zstack_vars['z_end'].get())
            z_step = float(self.zstack_vars['z_step'].get())
            img_size = int(self.zstack_vars['z_img_size'].get())
            num_spots = int(self.zstack_vars['z_num_spots'].get())

            # Thread starten
            def run():
                try:
                    self._update_status("🔄 Generiere Z-Stack...", is_running=True)

                    # Simulator erstellen
                    sim = TIFFSimulator(
                        detector=detector,
                        mode='z_stack',
                        astigmatism=True
                    )

                    # Z-Stack generieren
                    tiff_stack = sim.generate_z_stack(
                        image_size=(img_size, img_size),
                        num_spots=num_spots,
                        z_range_um=(z_start, z_end),
                        z_step_um=z_step
                    )

                    # Speichern
                    self.output_dir.mkdir(exist_ok=True, parents=True)
                    filename = f"{detector.name.lower()}_zstack.tif"
                    output_path = self.output_dir / filename
                    save_tiff(str(output_path), tiff_stack)

                    # Metadaten exportieren
                    from metadata_exporter import MetadataExporter
                    exporter = MetadataExporter(self.output_dir)
                    metadata = sim.get_metadata()
                    exporter.export_all(metadata, Path(filename).stem)

                    self._update_status(f"✅ Erfolgreich: {filename}", is_running=False)
                    messagebox.showinfo("Erfolg", f"Z-Stack erfolgreich erstellt!\n\n{output_path}")

                except Exception as e:
                    self._update_status(f"❌ Fehler: {str(e)}", is_running=False)
                    messagebox.showerror("Fehler", f"Fehler bei der Generierung:\n{str(e)}")
                finally:
                    self._enable_buttons()

            self._disable_buttons()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()

        except ValueError as e:
            messagebox.showerror("Ungültige Eingabe", f"Bitte überprüfe deine Eingaben:\n{str(e)}")

    def _run_batch(self):
        """Startet Batch-Simulation"""
        if self.is_running:
            messagebox.showwarning("Läuft bereits", "Eine Simulation läuft bereits!")
            return

        # Thread starten
        def run():
            try:
                self._update_status("🔄 Batch-Simulation läuft...", is_running=True)

                # Custom Times?
                custom_times_str = self.batch_custom_times.get().strip()
                if custom_times_str:
                    # Custom Batch
                    times = [float(t.strip()) for t in custom_times_str.split(',')]
                    batch = BatchSimulator(str(self.output_dir))
                    batch.add_polyzeit_series(
                        times=times,
                        detector=TDI_PRESET,
                        repeats=1,
                        image_size=(128, 128),
                        num_spots=15,
                        num_frames=200
                    )
                else:
                    # Preset Batch
                    preset = self.batch_preset.get()
                    if preset == 'quick':
                        batch = PresetBatches.quick_test(str(self.output_dir))
                    elif preset == 'thesis':
                        batch = PresetBatches.masterthesis_full(str(self.output_dir))
                    else:
                        batch = PresetBatches.publication_quality(str(self.output_dir))

                # Batch ausführen
                batch.run()

                self._update_status("✅ Batch abgeschlossen!", is_running=False)
                messagebox.showinfo("Erfolg", f"Batch-Simulation erfolgreich abgeschlossen!\n\nOutput: {self.output_dir}")

            except Exception as e:
                self._update_status(f"❌ Fehler: {str(e)}", is_running=False)
                messagebox.showerror("Fehler", f"Fehler beim Batch:\n{str(e)}")
            finally:
                self._enable_buttons()

        self._disable_buttons()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()


def main():
    """Startet die GUI"""
    root = tk.Tk()
    app = ModernTIFFSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
