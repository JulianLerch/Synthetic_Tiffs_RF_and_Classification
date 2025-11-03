# 🚀 SCHNELLSTART-ANLEITUNG V4.1

**TIFF Simulator V4.1 - Mit Track Analysis - In 3 Minuten starten!**

---

## ⚡ Installation

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. GUI starten
python tiff_simulator_gui_v4.py
```

**✅ Das war's!** Alle 7 Tabs sind sofort verfügbar.

---

## 📋 GUI-Tabs Übersicht

### Tab 1: 🔬 Simulator
- Synthetische TIFF-Mikroskopie-Daten generieren
- 4 Diffusionstypen: Normal, Subdiffusion, Confined, Superdiffusion
- Polymerisations-Zeitreihen (D(t) = D₀ · exp(-t/τ))

### Tab 2: 📦 Batch Mode
- Automatische Multi-TIFF-Generierung
- 3 Presets: Simple, Complex, Thesis
- Integriertes RF-Training

### Tab 3: 🤖 RF Training
- Random Forest Classifier trainieren
- **V4.1 Optimierungen:**
  - `step_size: 32` frames (weniger Overlap → weniger Data Leakage)
  - `n_estimators: 2048` (mehr Bäume → bessere Generalisierung)
  - `max_depth: 20` (Regularisierung gegen Overfitting)
  - `base_switch_prob: 0.002` (realistischere Switching-Raten)

### Tab 4: 💾 Export
- CSV-Export von Metadaten
- Track-Daten, Parameter, Timestamps

### Tab 5: 🎛️ Detector Config
- Kamera- und Optik-Parameter
- PSF, Noise Models
- Custom Presets

### Tab 6: 🎯 Tracking
- TrackMate-Integration
- Parameter-Empfehlungen

### Tab 7: 🔬 Track Analysis ⭐ **NEU!**
- **Experimentelle TrackMate XML-Daten auswerten**
- Single/Batch-Modus
- Multi-Scale Sliding Window Analyse
- RF-basierte Diffusionsklassifikation
- **Outputs:**
  - **Excel**: Ein Sheet pro Track (frame-by-frame Labels)
  - **CSV**: Statistiken pro Track
  - **PDF**: Pie Charts, Boxplots, Segment-Tabellen

---

## 🔬 WORKFLOW: Track Analysis

### Schritt 1: RF-Modell trainieren (falls noch nicht vorhanden)

```bash
# GUI-Methode (empfohlen):
1. Tab 2: Batch Mode öffnen
2. "Thesis Preset" auswählen
3. "Start Batch" klicken
4. Warten (~30 Min)
5. RF-Modell wird automatisch gespeichert: output/rf_model_*.joblib
```

### Schritt 2: Experimentelle Daten analysieren

```bash
# GUI-Methode:
1. Tab 7: Track Analysis öffnen
2. Modus wählen:
   - "Single": Eine XML-Datei
   - "Batch": Ganzer Ordner (rekursiv)
3. XML-Datei(en) auswählen (Browse-Button)
4. Preview prüfen (Track-Statistiken)
5. RF-Modell wird auto-detected (oder manuell wählen)
6. Frame Rate einstellen (z.B. 20 Hz)
7. "Start Analysis" klicken
```

### Schritt 3: Ergebnisse ansehen

**Excel** (`output/FILENAME_classification.xlsx`):
- Ein Sheet pro Track
- Spalten: Frame | X | Y | Z | Time | Label | Segment_ID

**CSV** (`output/FILENAME_statistics.csv`):
- Pro Track: Länge, Diffusionstypen, D-Werte, Alpha-Exponenten

**PDF** (`output/FILENAME_report.pdf`):
- Pie Chart: Diffusionstyp-Verteilung
- Boxplots: D und Alpha pro Diffusionstyp
- Tabelle: Segment-Statistiken

---

## 🐛 Fehlerbehebung

### ❌ `ModuleNotFoundError: No module named 'X'`

```bash
pip install -r requirements.txt
```

**Wichtige Pakete:**
- matplotlib ≥3.5.0
- scipy ≥1.8.0
- openpyxl ≥3.0.0
- joblib ≥1.2.0
- scikit-learn ≥1.2.0
- numpy ≥1.21.0

### ❌ GUI startet nicht

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora/RHEL
sudo dnf install python3-tkinter
```

**Windows/macOS**: tkinter sollte built-in sein

### ❌ RF-Modell nicht gefunden

1. Trainiere zuerst ein Modell (Tab 2 oder Tab 3)
2. Oder wähle manuell `.joblib`-Datei in Tab 7

### ❌ Track Analysis schlägt fehl

- **XML-Format prüfen**: Muss TrackMate XML sein
- **RF-Modell checken**: Muss 27 Features haben
- **Frame Rate checken**: Muss zu experimentellen Bedingungen passen

---

## 📚 Weitere Dokumentation

- **SETUP_GUIDE.md** - Detaillierte Installation & Konfiguration
- **TRACK_ANALYSIS_GUIDE.md** - Kompletter Track-Analysis-Workflow
- **CHANGELOG_V4.1.md** - Alle V4.1-Änderungen

---

## 🔧 V4.1 Fixes (2025-11-03)

### Bugfix: Import Error Handling
- **Problem**: `NameError: name 'exit' is not defined`
- **Fix**: `exit(1)` → `sys.exit(1)`
- **Commit**: `e3a40bc`

### Alle Dependencies installiert ✅
- matplotlib 3.10.7
- scipy 1.16.3
- openpyxl 3.1.5
- joblib (via scikit-learn)
- scikit-learn
- numpy

### Alle Module getestet ✅
- ✓ tiff_simulator_v3
- ✓ track_analysis
- ✓ rf_trainer
- ✓ metadata_exporter
- ✓ GUI (alle 7 Tabs)

---

## 🎯 Status: Einsatzbereit!

**Nächster Schritt**: GUI starten und loslegen!

```bash
python tiff_simulator_gui_v4.py
```

---

## 💡 Quick Tipps

### Schnelle Test-Simulation
```python
image_size = (64, 64)
num_spots = 5
num_frames = 30
# → ~10 Sekunden
```

### Realistische Daten
```python
image_size = (128, 128)
num_spots = 15
num_frames = 200
# → ~1 Minute
```

### Publication Quality
```python
image_size = (256, 256)
num_spots = 50
num_frames = 500
# → ~5-10 Minuten
```

---

**Los geht's! 🚀**

Starte mit `python tiff_simulator_gui_v4.py` und analysiere deine ersten Tracks! 🔬✨
