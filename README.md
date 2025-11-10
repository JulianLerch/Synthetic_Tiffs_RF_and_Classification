# 🔬 HYPERREALISTISCHER TIFF-SIMULATOR V3.0

**Wissenschaftlich präzise Simulation von Single-Molecule Tracking Daten für hochauflösende Fluoreszenzmikroskopie**

---

## 📋 ÜBERSICHT

Dieses Software-Paket ermöglicht die realistische Simulation von Fluoreszenzmikroskopie-Daten unter Berücksichtigung physikalisch korrekter Parameter für:

- **Point Spread Function (PSF)**: Gaußsche Approximation der optischen Abbildung
- **Brownsche Bewegung**: Diffusion mit zeitabhängigem Koeffizienten D(t)
- **Astigmatismus**: z-abhängige PSF-Deformation für 3D-Lokalisierung
- **Photon Statistics**: Poisson-verteiltes Shot Noise für realistische SNR

**Version:** 3.0 (Oktober 2025)  
**Autor:** Generiert für Masterthesis  
**Lizenz:** MIT

---

## 🎯 FEATURES

### ✅ Kern-Features

- **Zwei Detektor-Presets**: TDI-G0 (0.108 µm/px) und Tetraspecs (0.160 µm/px)
- **Drei Simulationsmodi**: Polymerisationszeit, Polymerisationszeit + Astigmatismus, z-Stack
- **Batch-Modus**: Automatisierte Generierung mehrerer TIFFs
- **Progress Bar**: Echtzeit-Fortschrittsanzeige
- **Metadata-Export**: JSON, TXT, CSV für vollständige Dokumentation
- **Wissenschaftlich validiert**: Basierend auf experimentellen Daten

### ✅ Neue Features V3.0

- **Threading**: GUI bleibt während Simulation responsiv
- **Erweiterte Validierung**: Prüft alle Parameter vor Start
- **Vordefinierte Batch-Presets**: Quick Test, Masterthesis, Publication Quality
- **Scrollbare GUI**: Alle Elemente immer sichtbar
- **Tab-Interface**: Übersichtliche Parameterorganisation

---

## 📦 INSTALLATION

### Voraussetzungen

- **Python**: ≥ 3.8
- **Betriebssystem**: Windows, macOS, Linux

### 1. Dependencies installieren

```bash
pip install -r requirements.txt
```

**Enthält:**
- numpy (≥1.20.0): Numerische Berechnungen
- Pillow (≥9.0.0): TIFF-Export
- matplotlib (≥3.5.0): Optional für Visualisierung
- tqdm (≥4.60.0): Progress Bars

### 2. Dateien überprüfen

```
tiff_simulator_complete/
├── tiff_simulator_v3.py      # Core Backend
├── metadata_exporter.py       # Metadata-System
├── batch_simulator.py         # Batch-Modus
├── tiff_simulator_gui.py      # GUI
├── requirements.txt           # Dependencies
└── README.md                  # Diese Datei
```

---

## 🚀 QUICK START

### GUI starten

```bash
python tiff_simulator_gui.py
```

### Batch-Modus (Command Line)

```bash
# Quick Test (3 TIFFs, ~2 Min)
python batch_simulator.py --preset quick --output ./test_output

# Masterthesis (60+ TIFFs, ~45 Min)
python batch_simulator.py --preset thesis --output ./thesis_data

# Publication Quality (30 TIFFs, ~2 Std)
python batch_simulator.py --preset publication --output ./publication_data

# Random-Forest für Diffusionsklassifikation parallel mittrainieren
python batch_simulator.py --preset thesis --output ./rf_batch --train-rf \
    --rf-window 48 --rf-step 16 --rf-estimators 1024 --rf-max-depth 28 \
    --rf-min-leaf 3 --rf-min-split 6 --rf-max-samples 0.85 \
    --rf-max-windows-per-class 100000 --rf-max-windows-per-track 600
```

### Programmatische Verwendung (Python)

```python
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff

# Erstelle Simulator
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode="polyzeit",
    t_poly_min=60.0,
    astigmatism=False
)

# Generiere TIFF
tiff_stack = sim.generate_tiff(
    image_size=(128, 128),
    num_spots=10,
    num_frames=100,
    frame_rate_hz=20.0
)

# Speichere
save_tiff("simulation.tif", tiff_stack)

# Exportiere Metadata
from metadata_exporter import MetadataExporter
exporter = MetadataExporter("./output")
exporter.export_all(sim.get_metadata(), "simulation")
```

---

## 🪟 WINDOWS-DESKTOP-BUILD (PYINSTALLER)

Möchtest du eine eigenständige EXE mit allen aktuellen GUI-Verbesserungen (ThunderSTORM Z-Stack Preset, Live z-Profil, neue Guides) erzeugen? So klappt's:

1. **ZIP entpacken** und ins Projektverzeichnis wechseln.
2. **`build_desktop_app.bat` doppelklicken** (oder `build_quick.bat` für den kompakten One-File-Build).
3. Das Skript installiert falls nötig PyInstaller, räumt alte Builds auf und paketiert die App mit allen neuen Modulen & Dokumentationen.
4. Nach erfolgreichem Lauf findest du die EXE unter `dist/`:
   - `dist/TIFF_Simulator_V4.1.exe` *(One-File)* oder
   - `dist/TIFF_Simulator_V4.1/TIFF_Simulator_V4.1.exe` *(Ordnerstruktur)*
5. Starte die EXE – die neue z-Stack-Physik, ThunderSTORM-Preset, Matplotlib-Vorschau und die mitgelieferten Guides sind direkt integriert.

> 💡 Die `.bat`-Dateien können jederzeit erneut ausgeführt werden, um nach Updates eine frische Desktop-Version zu erzeugen.

---

## 🔬 PHYSIKALISCHES MODELL

### 1. Point Spread Function (PSF)

Die PSF wird als 2D Gaußfunktion modelliert:

```
I(x,y) = I₀ · exp(-[(x-x₀)²/2σₓ² + (y-y₀)²/2σᵧ²])
```

**Parameter:**
- `I₀`: Peak-Intensität [counts]
- `σₓ, σᵧ`: Standardabweichungen [px]
- `FWHM = 2√(2ln2) · σ ≈ 2.355 · σ`

**Beziehung zur numerischen Apertur:**

Die theoretische FWHM für diffraction-limited Bildgebung:

```
FWHM ≈ 0.51 · λ / NA
```

Für λ = 580 nm und NA = 1.2:
```
FWHM ≈ 0.51 · 580 nm / 1.2 ≈ 246 nm
```

In der Praxis liegt die FWHM typisch bei ~400 nm aufgrund optischer Aberrationen.

**Referenzen:**
- Born & Wolf (1999). *Principles of Optics*, 7th Ed.
- Pawley (2006). *Handbook of Biological Confocal Microscopy*, 3rd Ed.

### 2. Astigmatismus (z-abhängige PSF)

Für 3D-Lokalisierung wird Astigmatismus durch z-abhängige Deformation der PSF modelliert:

```
σₓ(z) = σ₀ · √(1 + (z/z₀)²)
σᵧ(z) = σ₀ · √(1 - α(z/z₀)²)
```

**Parameter:**
- `σ₀`: Minimale σ bei z = 0
- `z₀`: Charakteristische Länge (~0.5 µm)
- `α`: Asymmetrie-Parameter (typisch 0.5)

Für |z| > z₀ wird die PSF stark elliptisch, was 3D-Lokalisierung ermöglicht.

**Referenzen:**
- Huang et al. (2008). *Science*, 319(5864), 810-813. "Three-Dimensional Super-Resolution Imaging by Stochastic Optical Reconstruction Microscopy"
- Stallinga & Rieger (2010). *Opt. Express*, 18(24), 24461-24476.

### 3. Diffusionskoeffizient D(t)

Der zeitabhängige Diffusionskoeffizient während der Polymerisationsphase:

```
D(t) = D₀ · exp(-t/τ) · f(t)
```

**Basis-Reduktion** (exponentieller Abfall):
```
τ = 40 min  (charakteristische Zeitkonstante)
```

**Zusätzliche Reduktion** für t ≥ 90 min:
```
f(t) = 0.5 · exp(-(t-90)/30)  für t ≥ 90 min
f(t) = 1                       für t < 90 min
```

**Typische D-Werte:**
```
t = 0 min:     D ≈ 4.0 µm²/s    (freie Diffusion)
t = 30 min:    D ≈ 1.5 µm²/s    (leichte Vernetzung)
t = 60 min:    D ≈ 0.5 µm²/s    (moderate Vernetzung)
t = 90 min:    D ≈ 0.15 µm²/s   (starke Vernetzung)
t = 120 min:   D ≈ 0.08 µm²/s   (sehr starke Vernetzung)
t = 180 min:   D ≈ 0.04 µm²/s   (maximale Vernetzung)
```

**Physikalische Interpretation:**

Die starke Reduktion von D reflektiert:
1. Zunehmende Viskosität η des Hydrogels (Stokes-Einstein: D ∝ 1/η)
2. Sterische Hinderung durch Polymernetzwerk
3. Übergang von normaler zu subdiffusiver Bewegung

**Referenzen:**
- Saxton & Jacobson (1997). *Annu. Rev. Biophys. Biomol. Struct.*, 26, 373-399.
- Masuda et al. (2005). *Phys. Rev. Lett.*, 95(18), 188102.
- Banks & Fradin (2005). *Biophys. J.*, 89(5), 2960-2971.

### 4. Diffusionstypen und Fraktionen

Mit zunehmender Polymerisation ändern sich die Fraktionen verschiedener Diffusionstypen:

**Normale Diffusion** (α = 1):
```
⟨r²(t)⟩ = 4Dt
```

**Subdiffusion** (α < 1):
```
⟨r²(t)⟩ = 4Dt^α    mit α ≈ 0.7
```

**Confined Diffusion**:
```
⟨r²(t)⟩ = R²(1 - exp(-4Dt/R²))
```

**Zeitabhängige Fraktionen:**

| Zeit | Normal | Subdiffusion | Confined |
|------|--------|--------------|----------|
| 0-10 min | 95% | 4% | 1% |
| 60 min | 65% | 24% | 10% |
| 120 min | 40% | 34% | 25% |
| >180 min | 35% | 35% | 28% |

**Referenzen:**
- Höfling & Franosch (2013). *Rep. Prog. Phys.*, 76(4), 046602.
- Metzler et al. (2014). *Phys. Chem. Chem. Phys.*, 16(44), 24128-24164.

### 5. Photon Statistics

**Shot Noise** (Poisson-Statistik):
```
P(n) = (λⁿ/n!) · e^(-λ)
```

wobei λ = erwartete Photonenzahl pro Pixel.

**Signal-to-Noise Ratio (SNR)**:
```
SNR = I_signal / √(I_signal + σ_bg²)
```

Für TDI-G0:
```
SNR ≈ 260 / √(260 + 15²) ≈ 15
```

**Referenzen:**
- Kubitscheck (2017). *Fluorescence Microscopy*, Wiley-VCH.
- Stelzer (2015). *Light Microscopy*, EMBO Practical Course Notes.

---

## 📊 VERWENDUNG

### Workflow 1: Single Simulation

```python
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff
from metadata_exporter import MetadataExporter

# 1. Erstelle Simulator
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode="polyzeit",
    t_poly_min=60.0,  # 60 min Polymerisation
    astigmatism=False
)

# 2. Generiere TIFF
tiff = sim.generate_tiff(
    image_size=(128, 128),
    num_spots=15,
    num_frames=200,
    frame_rate_hz=20.0  # 20 Hz = 50 ms pro Frame
)

# 3. Speichere
save_tiff("tdi_60min.tif", tiff)

# 4. Exportiere Metadata
exporter = MetadataExporter("./output")
exporter.export_all(sim.get_metadata(), "tdi_60min")
```

### Workflow 2: Batch Simulation

```python
from batch_simulator import BatchSimulator
from tiff_simulator_v3 import TDI_PRESET

# Erstelle Batch
batch = BatchSimulator("./batch_output")

# Füge Polymerisationszeit-Serie hinzu
batch.add_polyzeit_series(
    times=[10, 30, 60, 90, 120, 180],  # 6 Zeitpunkte
    detector=TDI_PRESET,
    repeats=3,  # 3 Wiederholungen pro Zeit
    image_size=(128, 128),
    num_spots=15,
    num_frames=200
)

# Führe aus
batch.run()

# Output:
# - 18 TIFF-Dateien (6 Zeiten × 3 Wiederholungen)
# - 18 × 3 Metadata-Dateien (JSON, TXT, CSV)
# - 1 Batch-Statistik (JSON)
```
### Random-Forest Co-Training (Diffusionsklassifikation)

```python
from batch_simulator import BatchSimulator

batch = BatchSimulator(
    "./batch_output",
    enable_rf=True,
    rf_config={
        'window_size': 48,
        'step_size': 16,
        'n_estimators': 1024,
        'max_depth': 28,
        'min_samples_leaf': 3,
        'min_samples_split': 6,
        'max_samples': 0.85,
        'max_windows_per_class': 100_000,
        'max_windows_per_track': 600,
    }
)

batch.add_polyzeit_series(times=[0, 30, 60, 90, 120], detector=TDI_PRESET, repeats=3)
stats = batch.run()

print(stats['rf'])
# → {'trained': True, 'model_path': './batch_output/random_forest_diffusion.joblib',
#    'feature_path': './batch_output/rf_training_features.csv',
#    'summary_path': './batch_output/rf_training_summary.json',
#    'samples': 480000, 'training_accuracy': 0.9642, 'oob_score': 0.9187,
#    'validation_accuracy': 0.9153}
```

**Ergebnis:**

- Parallel zum TIFF-Export wird ein großer, aber gedeckelter Random Forest (1024 Bäume, Tiefe 28, `class_weight="balanced_subsample"`, Out-of-Bag-Validierung mit Konfusionsmatrix) auf Sliding-Window-Features trainiert.
- Pro Fenster werden 27 physikalisch motivierte Merkmale (u. a. MSD-Lags, Log-Log-Slope, Straightness, Radius of Gyration, Richtungs- und Geschwindigkeits-Autokorrelation, Bounding Box, z-Range, Step-Momente) extrahiert.
- Modell (`random_forest_diffusion.joblib`), Feature-Tabelle (`rf_training_features.csv`) und Zusammenfassung (`rf_training_summary.json`) landen im Batch-Ordner.
- Die Summary enthält zusätzlich Feature-Importances, Trainings-Accuracy, echte OOB-Validierungswerte (Accuracy + Konfusionsmatrix), Klassen-/Polyzeit-Verteilungen und die neuen Fenster-Kontingente.
- Funktioniert für alle Polymerisationszeiten in einem Lauf – ideal für spätere Auswertung auf Real-Daten.

> **GUI:** Im Batch-Tab gibt es jetzt die Checkbox „Random-Forest während des Batch-Laufs mittrainieren“ inkl. Fenstergröße, Schrittweite, Baumanzahl, Tiefe, Min-Leaf/Split **sowie** Steuerung für Baum-Subsampling und Limits pro Klasse/Track.


### Workflow 3: z-Stack Kalibrierung

```python
# z-Stack für 3D-Kalibrierung
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode="z_stack",
    astigmatism=True  # WICHTIG für z-Stack!
)

z_stack = sim.generate_z_stack(
    image_size=(128, 128),
    num_spots=20,
    z_range_um=(-1.0, 1.0),  # -1 bis +1 µm
    z_step_um=0.1  # 0.1 µm Steps = 21 Slices
)

save_tiff("z_calibration.tif", z_stack)
```

**Neu ab V4.1:**

- Axiale Profile nutzen jetzt einen physikalisch motivierten Rayleigh-Bereich, eine astigmatismusspezifische Fokusverschiebung
  sowie eine sphärische Aberrationskorrektur basierend auf den Brechungsindizes (Öl/Glas/Probe).
- Die Intensität fällt realistisch gemäß Defokus **und** benutzerdefiniertem Intensitätsboden ab – ideal für ThunderSTORM-
  Kalibrierungen.
- Mit `evaluate_z_profile(detector, z_positions)` lässt sich das berechnete Profil (σx/σy, Intensität, korrigierte z-Positionen)
  ohne TIFF-Rendering inspizieren.
- In der GUI gibt es im Tab „📐 3D & Astigmatismus“ ein neues Info-Panel mit Stage- und Sample-z-Bereichen, Intensitätsskalierung
  sowie dem Verhältnis σx/σy. Der Button „⚙️ ThunderSTORM Preset“ setzt empfohlene Werte (±0.6 µm, 20 nm Schrittweite,
  astigmatische Fokusverschiebung 0.28 µm) für typische Kalibrierungsscans.

### Workflow 4: 3D-Simulation mit Astigmatismus

```python
# 3D-Diffusion mit astigmatischer PSF
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode="polyzeit_astig",
    t_poly_min=60.0,
    astigmatism=True
)

tiff_3d = sim.generate_tiff(
    image_size=(128, 128),
    num_spots=15,
    num_frames=200,
    frame_rate_hz=20.0
)

save_tiff("diffusion_3d.tif", tiff_3d)

# Analyse mit ThunderSTORM, TrackMate, etc.
```

---

## 🔧 PARAMETER-REFERENZ

### Detektor-Presets

#### TDI-G0
```python
{
    'name': 'TDI-G0',
    'max_intensity': 260.0,      # [counts]
    'background_mean': 100.0,    # [counts]
    'background_std': 15.0,      # [counts]
    'pixel_size_um': 0.108,      # [µm]
    'fwhm_um': 0.40,             # [µm]
    'quantum_efficiency': 0.85   # [%]
}
```

#### Tetraspecs
```python
{
    'name': 'Tetraspecs',
    'max_intensity': 300.0,      # [counts]
    'background_mean': 100.0,    # [counts]
    'background_std': 15.0,      # [counts]
    'pixel_size_um': 0.160,      # [µm]
    'fwhm_um': 0.40,             # [µm]
    'quantum_efficiency': 0.90   # [%]
}
```

### Empfohlene Parameter

#### Schnelle Tests
```python
image_size = (64, 64)
num_spots = 3-5
num_frames = 20-50
→ Dauer: ~10 Sekunden
```

#### Realistische Simulationen
```python
image_size = (128, 128)
num_spots = 10-20
num_frames = 100-200
→ Dauer: ~1 Minute
```

#### Publication Quality
```python
image_size = (256, 256)
num_spots = 20-50
num_frames = 500-1000
→ Dauer: ~5-10 Minuten
```

---

## 📋 METADATA-FORMAT

### JSON (Vollständig, maschinenlesbar)

```json
{
  "timestamp": "2025-10-28T10:30:00",
  "detector": "TDI-G0",
  "mode": "polyzeit",
  "t_poly_min": 60.0,
  "image_size": [128, 128],
  "num_spots": 15,
  "num_frames": 200,
  "frame_rate_hz": 20.0,
  "diffusion": {
    "D_initial": 4.0,
    "D_values": {
      "normal": 0.503,
      "subdiffusion": 0.302,
      "confined": 0.151
    },
    "diffusion_fractions": {
      "normal": 0.65,
      "subdiffusion": 0.24,
      "confined": 0.10
    }
  },
  "trajectories": [...]
}
```

### TXT (Menschenlesbar, Zusammenfassung)

```
======================================================================
TIFF SIMULATION - METADATA
======================================================================

Generiert: 2025-10-28T10:30:00
Dateiname: tdi_60min

DETEKTOR
----------------------------------------------------------------------
Name: TDI-G0
FWHM: 0.400 µm
Pixel Size: 0.108 µm
Astigmatismus: Nein

SIMULATIONSPARAMETER
----------------------------------------------------------------------
Modus: polyzeit
Bildgröße: 128 × 128 px
Anzahl Spots: 15
Anzahl Frames: 200
Frame Rate: 20.0 Hz
Gesamt-Dauer: 10.00 s

DIFFUSIONSPARAMETER
----------------------------------------------------------------------
Polymerisationszeit: 60.0 min
D_initial: 4.000 µm²/s
Frame Rate: 20.0 Hz

Diffusionskoeffizienten:
  D_normal: 0.5030 µm²/s
  D_subdiffusion: 0.3018 µm²/s
  D_confined: 0.1509 µm²/s

Diffusionsfraktionen:
  normal: 65.0%
  subdiffusion: 24.0%
  confined: 10.0%

Realisierte Diffusionsfraktionen:
  normal: 63.8% (115200 Frames)
  subdiffusion: 25.4% (45800 Frames)
  confined: 10.8% (19500 Frames)
  Gesamt ausgewertete Frames: 180500
> Hinweis: Ab Version 4.1 werden theoretische und tatsächlich realisierte Fraktionen gleichzeitig dokumentiert.
```

### CSV (Tabellarisch, für Batch-Analysen)

```csv
filename,timestamp,detector,mode,image_width,image_height,num_spots,num_frames,frame_rate_hz,t_poly_min,D_initial,D_normal,D_subdiffusion,realized_total_frames,frac_normal_theory_pct,frac_normal_real_pct,frames_normal,frac_subdiffusion_theory_pct,frac_subdiffusion_real_pct,frames_subdiffusion,frac_confined_theory_pct,frac_confined_real_pct,frames_confined
tdi_60min,2025-10-28T10:30:00,TDI-G0,polyzeit,128,128,15,200,20.0,60.0,4.0,0.503,0.302,180500,65.0,63.8,115200,24.0,25.4,45800,10.0,10.8,19500
```

---

## 🎓 FÜR MASTERTHESIS

### Empfohlene Studien

#### 1. Zeitabhängigkeit von D

```python
batch = BatchSimulator("./thesis_data")
batch.add_polyzeit_series(
    times=[0, 10, 30, 60, 90, 120, 180],
    detector=TDI_PRESET,
    repeats=5,  # Statistik!
    image_size=(128, 128),
    num_spots=20,
    num_frames=200
)
batch.run()

# Analyse:
# - Plot D(t) mit Fehlerbalken
# - MSD-Analyse für jeden Zeitpunkt
# - Vergleich mit experimentellen Daten
```

#### 2. Detektor-Vergleich

```python
batch.add_detector_comparison(
    polyzeit=60.0,
    repeats=5,
    image_size=(128, 128),
    num_spots=20,
    num_frames=200
)

# Analyse:
# - SNR-Vergleich
# - Lokalisierungsgenauigkeit
# - MSD-Unterschiede
```

#### 3. 3D-Lokalisierung

```python
# 1. z-Kalibrierung
batch.add_z_stack(detector=TDI_PRESET)

# 2. 3D-Simulationen
batch.add_3d_series(
    times=[60, 90, 120],
    repeats=3
)

# Analyse mit ThunderSTORM:
# - z-Lokalisierungsgenauigkeit
# - 3D-MSD-Analyse
# - Subdiffusions-Parameter α
```

---

## 📊 DATENANALYSE

### Empfohlene Software

**Tracking & Lokalisierung:**
- **TrackMate** (Fiji/ImageJ): 2D/3D Tracking, MSD-Analyse
- **ThunderSTORM** (Fiji): Sub-pixel Lokalisierung, 3D
- **u-track** (MATLAB): Fortgeschrittenes Tracking

**Datenanalyse:**
- **Python**: pandas, scipy, matplotlib, seaborn
- **R**: ggplot2, dplyr, zoo
- **MATLAB**: Statistics Toolbox, Curve Fitting

### Beispiel: MSD-Analyse (Python)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Lade Metadata
metadata = pd.read_csv("batch_output/batch_metadata.csv")

# Gruppiere nach Polyzeit
grouped = metadata.groupby('t_poly_min')

# Plot D vs. Zeit
fig, ax = plt.subplots(figsize=(10, 6))

for name, group in grouped:
    ax.scatter(group['t_poly_min'], group['D_normal'], 
              label=f'{name} min', alpha=0.6)

ax.set_xlabel('Polymerisationszeit [min]', fontsize=12)
ax.set_ylabel('D [µm²/s]', fontsize=12)
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('D_vs_time.pdf', dpi=300)
```

---

## 🐛 TROUBLESHOOTING

### Problem: Import Error

```
❌ ModuleNotFoundError: No module named 'numpy'
```

**Lösung:**
```bash
pip install -r requirements.txt
```

### Problem: GUI startet nicht

```
❌ TclError: no display name
```

**Lösung (Linux/SSH):**
```bash
export DISPLAY=:0
# oder X11 Forwarding aktivieren
```

### Problem: TIFF zu dunkel

Die Intensitäten sind realistisch! (TDI: ~260, Tetraspecs: ~300)

**Lösung in ImageJ:**
```
Image → Adjust → Brightness/Contrast
→ Auto oder Min=0, Max=500
```

### Problem: Simulation zu langsam

**Lösung:**
- Kleinere Bildgröße (64×64 statt 256×256)
- Weniger Frames (50 statt 500)
- Weniger Spots (5 statt 50)

---

## 📚 REFERENZEN

### Fluoreszenzmikroskopie
1. **Pawley, J. (2006).** *Handbook of Biological Confocal Microscopy*, 3rd Ed. Springer.
2. **Kubitscheck, U. (2017).** *Fluorescence Microscopy: From Principles to Biological Applications*, 2nd Ed. Wiley-VCH.

### Single-Molecule Tracking
3. **Manzo & Garcia-Parajo (2015).** "A review of progress in single particle tracking." *Rep. Prog. Phys.*, 78(12), 124601.
4. **Chenouard et al. (2014).** "Objective comparison of particle tracking methods." *Nature Methods*, 11, 281-289.

### Brownsche Bewegung & Diffusion
5. **Saxton & Jacobson (1997).** "Single-particle tracking: applications to membrane dynamics." *Annu. Rev. Biophys. Biomol. Struct.*, 26, 373-399.
6. **Höfling & Franosch (2013).** "Anomalous transport in the crowded world of biological cells." *Rep. Prog. Phys.*, 76(4), 046602.

### 3D-Lokalisierung
7. **Huang et al. (2008).** "Three-dimensional super-resolution imaging by stochastic optical reconstruction microscopy." *Science*, 319(5864), 810-813.
8. **Stallinga & Rieger (2010).** "Accuracy of the Gaussian point spread function model in 2D localization microscopy." *Opt. Express*, 18(24), 24461-24476.

---

## 📝 LIZENZ

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 🙏 ACKNOWLEDGEMENTS

Entwickelt für Masterthesis im Bereich Single-Molecule Tracking und Hydrogel-Polymerisation.

Bei Fragen oder Problemen: Siehe Troubleshooting-Sektion oder öffne ein Issue.

**Viel Erfolg mit deiner Forschung! 🔬✨**
