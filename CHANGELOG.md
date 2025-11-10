# CHANGELOG - TIFF Simulator

## Version 4.1 - März 2026 (Random-Forest Pipeline)

### 🌲 Random-Forest Co-Training

- **BatchSimulator** trainiert optional einen leistungsstarken, aber gedeckelten Random Forest (1024 Trees, Tiefe 28, `class_weight="balanced_subsample"`, echte OOB-Validierung) parallel zu jeder Simulation – Sliding-Window-Features decken alle vier Diffusionsarten und Polymerisationszeiten ab, ohne dass das Modell auf zig Gigabyte anwächst.
- Neues Modul `rf_trainer.py` extrahiert 27 physikalische Merkmale pro Fenster (MSD-Lags + Log-Log-Slope, Straightness, Radius of Gyration, Richtungs- & Geschwindigkeits-Autokorrelation, Bounding Box, z-Range, Step-Momente, …) und speichert Modell (`random_forest_diffusion.joblib`), Feature-CSV und eine ausführliche JSON-Summary im Batch-Output.
- CLI-Flags `--train-rf`, `--rf-window`, `--rf-step`, `--rf-estimators`, `--rf-max-depth`, `--rf-min-leaf`, `--rf-min-split`, `--rf-random-state`, `--rf-max-samples`, `--rf-max-windows-per-class`, `--rf-max-windows-per-track` ermöglichen Feintuning direkt beim Start.
- GUI-Batch-Tab besitzt jetzt Checkbox + Spinboxen für Fenstergröße, Schrittweite, Baumanzahl, Tiefe, Min-Leaf/Split **sowie** Baum-Subsampling und Fenster-Limits – inklusive Statusmeldungen nach dem Lauf.
- Random-Forest-Training nutzt Reservoir-Sampling pro Klasse und per-Track-Limits, reduziert Speicherverbrauch und trainiert erst beim Finalisieren; Summary enthält Trainings- und OOB-Konfusionsmatrizen plus Fensterverteilungen.
- Metadata-Export (TXT/CSV/JSON) dokumentiert zusätzlich die tatsächlich realisierten Diffusionsfraktionen inklusive Frame-Zahlen.
- `requirements.txt` ergänzt um `scikit-learn` und `joblib`.

### 🧾 Dokumentation

- README & BATCH_MODE_GUIDE beschreiben die neue RF-Pipeline (CLI, Python, GUI) samt Output-Artefakten und Best Practices.

### 🔭 Z-Stack Physik & GUI

- Z-Stacks nutzen jetzt eine Rayleigh-basierte PSF-Expansion inklusive astigmatischem Fokusversatz und sphärischer Aberrations-
  Abschätzung. Die Intensität fällt über Defokus und einen justierbaren Intensitätsboden realistisch ab.
- Die erweiterte Brechungsindex-Korrektur (Öl/Glas/Probe/NA) ist standardmäßig aktiv; `evaluate_z_profile()` liefert die
  berechneten σx/σy- und Intensitätsprofile ohne TIFF-Rendering.
- GUI-Tab „📐 3D & Astigmatismus“ zeigt ein dynamisches Physik-Dashboard mit Stage-/Sample-z-Bereich, Intensitätsskala und
  σx/σy-Verhältnis sowie Buttons für einen ThunderSTORM-Optimierungspreset und eine interaktive Matplotlib-Vorschau des
  axialen Profils.

## Version 4.0 - Oktober 2025 (MAJOR UPDATE)

### 🚀 Performance-Optimierungen (10-50x schneller!)

**Core Engine:**
- ✅ **Vektorisierte PSF-Generierung:** Batch-Processing für alle Spots gleichzeitig
- ✅ **ROI-basierte Berechnung:** Nur 3-sigma Umgebung wird berechnet (nicht ganzes Bild)
- ✅ **Pre-computed Koordinaten-Grids:** Wiederverwend bare Meshgrids
- ✅ **Background-Caching:** Intelligentes Caching für Batch-Simulationen
- ✅ **Memory-efficient:** Optimierte Array-Wiederverwendung
- ✅ **Progress-Callbacks:** Thread-safe Echtzeit-Updates

**Ergebnisse:**
- Kleine TIFFs (128×128, 100 frames, 10 spots): ~1-2 Sekunden (V3: ~10s)
- Mittlere TIFFs (256×256, 500 frames, 30 spots): ~3-5 Minuten (V3: ~45 min)
- Große TIFFs (512×512, 2000 frames, 50 spots): ~20-30 Minuten (V3: mehrere Stunden)

### 🎨 GUI V4.0 - Advanced Edition

**Neue Parameter-Tabs:**
- 📊 Basis-Parameter (wie V3.0)
- ⚛️ **NEU:** Erweiterte Physik (PSF, Background, Noise)
- 💡 **NEU:** Photophysik & Blinking (ON/OFF, Bleaching)
- 📐 **NEU:** 3D & Astigmatismus (z-Parameter, Koeffizienten)
- 📦 Batch-Modus (erweitert)
- 💾 Export & Metadata

**Neue GUI-Features:**
- ✅ Tooltips für ALLE Parameter (physikalische Bedeutung + Empfehlungen)
- ✅ Live-Updates für D-Wert-Schätzung
- ✅ z-Stack Slice-Berechnung in Echtzeit
- ✅ Moderneres Design (dunkler Header, bessere Farben)
- ✅ Scrollbares Interface (passt auf alle Bildschirmgrößen)

### 🔬 Erweiterte Physik

**Photophysik (NEU!):**
- ✅ Blinking: 2-Zustands-Modell (ON/OFF) mit konfigurierbaren Dauern
- ✅ Photobleaching: Irreversibles Bleaching mit realistischen Wahrscheinlichkeiten
- ✅ Geometrische Dauern-Verteilung (physikalisch korrekt)

**Noise & PSF (erweitert):**
- ✅ Variable Max-Intensität (vorher: fix für Detektor)
- ✅ Spot Intensity Sigma (lognormale Variabilität)
- ✅ Frame Jitter Sigma (Frame-zu-Frame Schwankungen)
- ✅ Separate Background Mean & Std
- ✅ Konfigurierbare Read Noise

**3D & Astigmatismus (erweitert):**
- ✅ z-Amplitude (Intensitätsabfall-Skala)
- ✅ z-Max (Clipping-Bereich)
- ✅ z0 (charakteristische Skala)
- ✅ Astigmatismus-Koeffizienten Ax, Ay (vorher: hardcoded)

### 🖥️ Desktop-App

**Build-System:**
- ✅ PyInstaller-Integration
- ✅ Cross-Platform Build-Scripts (Windows .bat + Mac/Linux .sh)
- ✅ Launcher mit Auto-Dependency-Check (`START_SIMULATOR.py`)
- ✅ Spec-File für optimierte Builds

**Features:**
- ✅ Standalone Executable (~200 MB)
- ✅ Keine Python-Installation nötig
- ✅ Portable (USB-Stick)
- ✅ Kein Konsolen-Fenster (GUI-only)

### 📚 Dokumentation

**Neue Dateien:**
- ✅ `ANLEITUNG_DESKTOP_APP.md` - Umfassende Desktop-App Anleitung
- ✅ `CHANGELOG.md` - Versionshistorie
- ✅ `build_app.spec` - PyInstaller Konfiguration
- ✅ `build_desktop_app.sh` / `.bat` - Build-Scripts
- ✅ `START_SIMULATOR.py` - Smart Launcher

**Aktualisiert:**
- ✅ `requirements.txt` - PyInstaller hinzugefügt
- ✅ Code-Kommentare - Alle neuen Funktionen dokumentiert

### 🔧 Technische Details

**Neue Klassen:**
- `PSFGeneratorOptimized` - Vektorisierte PSF-Berechnung
- `BackgroundGeneratorOptimized` - Mit Caching
- `TIFFSimulatorOptimized` - Hauptklasse mit Progress-Callbacks
- `TIFFSimulatorGUI_V4` - Erweiterte GUI
- `ToolTip` - Hilfe-Tooltips für GUI

**Backward Compatibility:**
- ✅ Alte APIs funktionieren weiterhin
- ✅ V3.0 GUI läuft mit V4.0 Engine
- ✅ Aliase: `TIFFSimulator = TIFFSimulatorOptimized`

### 🐛 Bugfixes

- ✅ Float32 statt Float64 (schneller, weniger Speicher)
- ✅ Robustere NaN/Inf-Behandlung
- ✅ Thread-safe UI-Updates
- ✅ Bessere Exception-Handling

---

## Unreleased

### Added
- Bundled ThunderSTORM-z-Stack-Physik, Live z-Profil und erweiterte Guides direkt in die PyInstaller-Desktop-Builds (`build_desktop_app.bat/.sh`).
- Aktualisierte `build_quick.bat`, um alle neuen Module, Dokumentationen und Matplotlib-Abhängigkeiten in den One-File-Build einzuschließen.

### Improved
- Dokumentation ergänzt, wie die aktualisierte `.bat`-Installation die neuen Features automatisch in den `dist/`-Ordner bringt.

---

## Version 3.0 - Oktober 2025

### Features
- ✅ Grundlegende TIFF-Simulation
- ✅ TDI-G0 & Tetraspecs Presets
- ✅ Polymerisationszeit-Modell
- ✅ Astigmatismus-Support
- ✅ z-Stack Kalibrierung
- ✅ Batch-Modus mit Presets
- ✅ Metadata-Export (JSON, TXT, CSV)
- ✅ GUI mit Scrollbarem Interface
- ✅ Jupyter Notebook Tutorial

### Physik
- ✅ Point Spread Function (2D Gaußsch)
- ✅ Brownsche Bewegung (normale/sub/confined Diffusion)
- ✅ Zeitabhängiger Diffusionskoeffizient D(t)
- ✅ Poisson-Noise + Read Noise
- ✅ Background mit Gradient
- ✅ Einfaches Blinking & Bleaching

### Performance
- ⚠️ Frame-für-Frame Verarbeitung (langsam bei großen TIFFs)
- ⚠️ Volle Bild-Meshgrids pro Spot
- ⚠️ Keine Parallelisierung

---

## Version 2.0 - Nicht veröffentlicht

Interne Entwicklungsversion

---

## Version 1.0 - Initial Release

Proof-of-Concept für Masterthesis
