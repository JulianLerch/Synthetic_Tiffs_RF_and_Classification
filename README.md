# 🔬 TIFF-SIMULATOR V5.0

**Wissenschaftlich präzise Simulation von Single-Molecule Tracking Daten für hochauflösende Fluoreszenzmikroskopie**

---

## 📋 ÜBERSICHT

Dieses Software-Paket ermöglicht die realistische Simulation von Fluoreszenzmikroskopie-Daten unter Berücksichtigung physikalisch korrekter Parameter für:

- **Point Spread Function (PSF)**: Gaußsche Approximation der optischen Abbildung
- **Brownsche Bewegung**: Diffusion mit zeitabhängigem Koeffizienten D(t)
- **Astigmatismus**: z-abhängige PSF-Deformation für 3D-Lokalisierung
- **Photon Statistics**: Poisson-verteiltes Shot Noise für realistische SNR

**Version:** 5.0 (November 2025)
**Lizenz:** MIT

---

## 🎯 HAUPTFUNKTIONEN

### 📄 Einzelnes TIFF
- Generiere einzelne TIFF-Dateien
- Mit oder ohne Astigmatismus
- Konfigurierbare Parameter (Bildgröße, Spots, Frames, etc.)
- Polymerisationszeit-abhängige Diffusion

### 📚 Z-Stack
- Z-Stack Kalibrierung für 3D-Tracking
- Automatischer Astigmatismus
- Konfigurierbare z-Range und z-Step
- Für PSF-Kalibrierung in TrackMate/ThunderSTORM

### 🔄 Batch-Modus
- Automatische Generierung mehrerer TIFFs
- Vordefinierte Presets:
  - **Quick Test**: 3 TIFFs, ~2 Minuten
  - **Thesis Quality**: ~60 TIFFs, ~45 Minuten
  - **Publication Quality**: ~30 TIFFs, ~2 Stunden
- Custom Polymerisationszeit-Serien

---

## 📦 INSTALLATION

### Voraussetzungen

- **Python**: ≥ 3.8 (empfohlen: 3.9 oder 3.10)
- **Betriebssystem**: Windows, macOS, Linux

### Dependencies installieren

```bash
pip install -r requirements.txt
```

**Enthaltene Pakete:**
- `numpy` (≥1.21.0): Numerische Berechnungen
- `scipy` (≥1.8.0): Wissenschaftliche Funktionen
- `Pillow` (≥9.2.0): TIFF-Export
- `matplotlib` (≥3.5.0): Optional für Visualisierung
- `tqdm` (≥4.64.0): Progress Bars

**Linux-Nutzer:** tkinter muss ggf. separat installiert werden:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora/RHEL
sudo dnf install python3-tkinter
```

---

## 🚀 QUICK START

### Option 1: Einfacher Start (GUI)

```bash
python START_SIMULATOR.py
```

Dieser Launcher prüft automatisch alle Dependencies und startet die GUI.

### Option 2: Direkt GUI starten

```bash
python tiff_simulator_gui.py
```

### Option 3: Batch-Modus (Command Line)

```bash
# Quick Test (3 TIFFs, ~2 Min)
python batch_simulator.py --preset quick --output ./test_output

# Masterthesis (60+ TIFFs, ~45 Min)
python batch_simulator.py --preset thesis --output ./thesis_data

# Publication Quality (30 TIFFs, ~2 Std)
python batch_simulator.py --preset publication --output ./publication_data

# Custom Polymerisationszeiten
python batch_simulator.py --output ./custom --times 30,60,90,120
```

---

## 🎨 GUI ÜBERSICHT

Die moderne GUI besteht aus 3 Tabs:

### 📄 Tab 1: Einzelnes TIFF
- **Detektor**: TDI-G0 oder Tetraspecs
- **Bild-Parameter**: Größe, Spots, Frames, Frame Rate
- **Polymerisation**: Zeitpunkt (0-180 min)
- **Astigmatismus**: Optional aktivierbar für 3D
- **Output**: Wählbares Verzeichnis

### 📚 Tab 2: Z-Stack
- **Detektor**: TDI-G0 oder Tetraspecs
- **Z-Parameter**: Start, Ende, Schrittweite (in µm)
- **Bild-Parameter**: Größe, Anzahl Spots
- **Automatischer Astigmatismus** für PSF-Kalibrierung

### 🔄 Tab 3: Batch-Modus
- **Preset-Auswahl**: Quick, Thesis, Publication
- **Custom Zeiten**: Optional eigene Polymerisationszeit-Serie
- **Automatische Generierung** mehrerer TIFFs mit Fortschrittsanzeige

---

## 🔧 PROGRAMMTISCHE NUTZUNG

### Einzelnes TIFF generieren

```python
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff

# Simulator erstellen
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode='polyzeit',
    t_poly_min=60.0,
    astigmatism=False
)

# TIFF generieren
tiff = sim.generate_tiff(
    image_size=(128, 128),
    num_spots=15,
    num_frames=200,
    frame_rate_hz=20.0
)

# Speichern
save_tiff("output.tif", tiff)
```

### Z-Stack generieren

```python
from tiff_simulator_v3 import TETRASPECS_PRESET, TIFFSimulator, save_tiff

sim = TIFFSimulator(
    detector=TETRASPECS_PRESET,
    mode='z_stack',
    astigmatism=True
)

zstack = sim.generate_z_stack(
    image_size=(128, 128),
    num_spots=20,
    z_range_um=(-1.0, 1.0),
    z_step_um=0.1
)

save_tiff("zstack.tif", zstack)
```

### Batch-Simulation

```python
from batch_simulator import BatchSimulator, PresetBatches
from tiff_simulator_v3 import TDI_PRESET

# Option 1: Preset nutzen
batch = PresetBatches.quick_test("./output")
batch.run()

# Option 2: Custom Batch
batch = BatchSimulator("./custom_output")
batch.add_polyzeit_series(
    times=[30, 60, 90, 120],
    detector=TDI_PRESET,
    repeats=3,
    image_size=(128, 128),
    num_spots=15,
    num_frames=200
)
batch.run()
```

---

## 📊 OUTPUT-DATEIEN

### TIFF-Dateien
- **Format**: Multi-page TIFF
- **Bit-Tiefe**: 16-bit (uint16)
- **Photon Counts**: Realistisch (50-300 je nach Detektor)

### Metadaten
Für jedes TIFF werden automatisch 3 Dateien erstellt:

1. **JSON** (`*_metadata.json`): Vollständige Parameter
2. **TXT** (`*_metadata.txt`): Menschenlesbare Zusammenfassung
3. **CSV** (`*_metadata.csv`): Tabellarische Parameter

### Batch-Statistik
Im Batch-Modus wird zusätzlich erstellt:

- **`batch_statistics.json`**: Zusammenfassung aller generierten TIFFs

---

## 🔬 PHYSIKALISCHE DETAILS

### Detektor-Presets

| Parameter | TDI-G0 | Tetraspecs |
|-----------|--------|------------|
| Pixel-Größe | 0.108 µm | 0.160 µm |
| Max Intensity | 260 counts | 300 counts |
| PSF FWHM | 0.40 µm | 0.40 µm |
| Typ | sCMOS | sCMOS |

### Diffusionsmodell

**Zeitabhängige Diffusion** während Gel-Polymerisation:

```
D(t) = D₀ · exp(-t/τ) · f(t)
```

- **τ = 40 min**: Charakteristische Zeitkonstante
- **D₀**: Initial-Diffusionskoeffizient
- **t = 0 min**: D ≈ 4.0 µm²/s (freie Diffusion)
- **t = 60 min**: D ≈ 0.5 µm²/s (moderate Vernetzung)
- **t = 180 min**: D ≈ 0.04 µm²/s (maximale Vernetzung)

### Astigmatismus (für 3D)

PSF-Breiten in x und y:
```
σₓ(z) = σ₀ · √(1 + (z/d)² + A₃(z/d)³ + A₄(z/d)⁴)
σᵧ(z) = σ₀ · √(1 + (z/d)² - A₃(z/d)³ - A₄(z/d)⁴)
```

- **d = 0.5 µm**: Depth of field
- **A₃ = -0.15**: Astigmatismus Koeffizient
- **A₄ = 0.05**: Höherer-Ordnung Term

---

## 📁 PROJEKTSTRUKTUR

```
tiff_simulator/
├── tiff_simulator_v3.py      # Core Simulator-Engine
├── tiff_simulator_gui.py     # Moderne GUI (3 Tabs)
├── batch_simulator.py         # Batch-Modus
├── metadata_exporter.py       # Metadaten-Export
├── START_SIMULATOR.py         # Einfacher Launcher
├── requirements.txt           # Dependencies
├── README.md                  # Diese Datei
├── QUICKSTART.md             # Schnellanleitung
├── SETUP_GUIDE.md            # Installations-Guide
├── BATCH_MODE_GUIDE.md       # Batch-Modus Details
├── PHYSICS_VALIDATION.md     # Physikalische Validierung
└── CHANGELOG.md              # Versions-Historie
```

---

## 🐛 TROUBLESHOOTING

### Import-Fehler

```python
ImportError: No module named 'numpy'
```

**Lösung:**
```bash
pip install -r requirements.txt
```

### tkinter fehlt (Linux)

```
ImportError: No module named '_tkinter'
```

**Lösung:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora/RHEL
sudo dnf install python3-tkinter
```

### GUI startet nicht

**Prüfen:**
1. Python Version ≥ 3.8?
2. Alle Dependencies installiert?
3. Dateien im gleichen Ordner?

**Debug:**
```bash
python START_SIMULATOR.py
```

---

## 📝 ZITATION

Falls du diesen Simulator in deiner Forschung verwendest:

```bibtex
@software{tiff_simulator_v5,
  title = {TIFF Simulator: Realistic Single-Molecule Tracking Data Generator},
  version = {5.0},
  year = {2025},
  note = {Synthetic microscopy data with time-dependent diffusion modeling}
}
```

---

## 📄 LIZENZ

MIT License - Siehe LICENSE Datei für Details

---

## 🤝 CONTRIBUTING

Contributions sind willkommen! Bitte:
1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Committe deine Änderungen
4. Erstelle einen Pull Request

---

## 📮 KONTAKT & SUPPORT

Bei Fragen oder Problemen:
- GitHub Issues
- README.md und weitere Guides lesen
- Code-Kommentare durchsehen

---

**Viel Erfolg mit deinen Simulationen! 🚀**
