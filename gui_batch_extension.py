# GUI V4.1 - BATCH-MODUS ERWEITERUNG
# ====================================
# Diese Code-Snippets erweitern die GUI um vollständigen Batch-Support

# 1. NEUE VARIABLEN (_init_variables ergänzen):
# ============================================

        # ===== IMAGE PARAMETERS (AKTUALISIERT) =====
        self.image_width = tk.IntVar(value=256)  # REALISTISCH
        self.image_height = tk.IntVar(value=256)
        self.num_spots = tk.IntVar(value=30)

        # NEU: Spot Range für Batch
        self.num_spots_min = tk.IntVar(value=10)
        self.num_spots_max = tk.IntVar(value=20)

        # ===== TIME SERIES (AKTUALISIERT) =====
        self.num_frames = tk.IntVar(value=200)  # REALISTISCH

        # ===== BATCH (KOMPLETT NEU!) =====
        self.batch_mode_enabled = tk.BooleanVar(value=False)  # Single vs Batch
        self.batch_poly_times = tk.StringVar(value="0, 30, 60, 90, 120")
        self.batch_repeats = tk.IntVar(value=3)
        self.batch_astig = tk.BooleanVar(value=False)
        self.batch_use_spot_range = tk.BooleanVar(value=True)
        self.batch_subfolder_per_repeat = tk.BooleanVar(value=True)


# 2. NEUER BATCH-TAB (_create_batch_tab ersetzen):
# ================================================

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
                f"📁 Unterordner: {'JA' if self.batch_subfolder_per_repeat.get() else 'NEIN'}\n\n"
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


# 3. BATCH-AUSFÜHRUNG (_run_simulation erweitern):
# =================================================

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


    def _run_batch_simulation_integrated(self):
        """NEUE FUNKTION: Batch-Modus vollständig integriert!"""
        import re
        from pathlib import Path
        import numpy as np

        self._update_status("📦 Starte Batch-Modus...")
        self._update_progress(5)

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
                if self.export_json.get() or self.export_txt.get() or self.export_csv.get():
                    exporter = MetadataExporter(str(output_dir))
                    metadata = sim.get_metadata()
                    base_name = filepath.stem

                    if self.export_json.get():
                        exporter.export_json(metadata, base_name)
                    if self.export_txt.get():
                        exporter.export_txt(metadata, base_name)
                    if self.export_csv.get():
                        exporter.export_csv_row(metadata, base_name)

        self._update_progress(100)
        self._update_status(f"✅ Batch fertig! {total_tasks} TIFFs erstellt.")


    def _run_single_simulation_integrated(self):
        """Führt Single Simulation aus (wie vorher)."""
        # ... (bestehender Code bleibt gleich)
        # Siehe vorherige Implementierung
        pass


# 4. ZUSÄTZLICHE HELPER-FUNKTION:
# ================================

    def _create_custom_detector(self):
        """Erstellt Custom-Detektor mit aktuellen GUI-Parametern."""
        from tiff_simulator_v3 import DetectorPreset, TDI_PRESET, TETRASPECS_PRESET

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
                               "A_y": self.astig_Ay.get(), "B_y": 0.0},
                # NEU: Illumination Gradient und Brechungsindex-Korrektur
                "illumination_gradient_strength": self.illumination_gradient_strength.get(),
                "illumination_gradient_type": self.illumination_gradient_type.get(),
                "refractive_index_correction": self.refractive_index_correction.get()
            }
        )

        return custom
