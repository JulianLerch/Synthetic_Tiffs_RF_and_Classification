# 🔬 TRACK ANALYSIS GUIDE - Von TrackMate zu Diffusionsklassifikation

**Version 4.1 - Automatische Klassifikation experimenteller Tracking-Daten**

---

## 📋 Übersicht

Das Track Analysis Modul analysiert experimentelle Single-Particle-Tracking Daten aus TrackMate XML-Dateien und klassifiziert automatisch Trajektorien-Segmente in 4 Diffusionsarten:

- **Normal Diffusion** (Brownian Motion)
- **Subdiffusion** (Anomalous, α < 1)
- **Confined Diffusion** (In Poren gefangen)
- **Superdiffusion** (Konvektion, α > 1)

---

## 🎯 Workflow

```
TrackMate XML → Parse Tracks → Multi-Scale Sliding Windows →
RF Classification → Smoothing → MSD/D Calculation →
Excel + CSV + Plots
```

---

## 🚀 Quick Start

### Voraussetzungen

1. **Trainiertes RF-Modell** (aus Batch-Simulation)
   ```bash
   # Generiere Trainings-Daten und trainiere RF
   python batch_simulator.py --preset thesis --train-rf --output ./rf_training

   # Modell ist hier:
   ./rf_training/random_forest_diffusion.joblib
   ```

2. **TrackMate XML-Dateien**
   - Exportiert aus Fiji/ImageJ mit TrackMate Plugin
   - Format: `.xml` mit Spot/Track Struktur

### Single XML Analysis

```bash
python track_analysis.py merged_tracks.xml \
  --model ./rf_training/random_forest_diffusion.joblib \
  --frame-rate 20.0 \
  --output ./analysis_output
```

**Output:**
```
analysis_output/
├── merged_tracks_analysis.xlsx    # Excel mit allen Tracks
├── merged_tracks_statistics.csv   # Aggregierte Statistiken
├── merged_tracks_distribution.pdf # Pie Chart
├── merged_tracks_D_boxplot.pdf    # D-Werte pro Typ
└── merged_tracks_alpha_boxplot.pdf # Alpha-Werte pro Typ
```

### Batch Analysis (ganzer Ordner)

```bash
python track_analysis.py ./experiment_data/ \
  --model ./rf_model.joblib \
  --batch \
  --recursive \
  --output ./all_analyses
```

**Analysiert alle `.xml` Dateien rekursiv und erstellt für jede:**
- Separate Analyse-Ordner
- Alle Output-Dateien

---

## 📊 Output-Dateien im Detail

### 1. Excel-Datei (`{name}_analysis.xlsx`)

**Sheet-Struktur:**

#### `Summary` Sheet:
```
XML File: merged_tracks.xml
Total Tracks: 45
Mean Track Length: 127.3

Diffusion Type | Count | Percentage
normal         | 3542  | 62.5%
subdiffusion   | 1687  | 29.8%
confined       | 389   | 6.9%
superdiffusion | 45    | 0.8%
```

#### `Track_1`, `Track_2`, ... Sheets (ein Sheet pro Track):
```
Frame | X        | Y        | Z      | Time   | Diffusion_Type
0     | 12.4567  | 8.2341   | 0.123  | 0.000  | normal
1     | 12.5123  | 8.2987   | 0.134  | 0.050  | normal
...
75    | 14.2341  | 9.8765   | 0.221  | 3.750  | subdiffusion
```

**Plus Segment Summary:**
```
=== SEGMENT SUMMARY ===
Start_Frame | End_Frame | Type         | Length | D [µm²/s] | Alpha
0           | 67        | normal       | 68     | 0.342     | 0.98
68          | 112       | subdiffusion | 45     | 0.087     | 0.64
113         | 145       | normal       | 33     | 0.298     | 1.02
```

### 2. CSV-Datei (`{name}_statistics.csv`)

```csv
Metric,Value
XML File,merged_tracks.xml
Num Tracks,45
Mean Track Length,127.3

Diffusion Type,Count,Percentage
normal,3542,62.50
subdiffusion,1687,29.80
confined,389,6.90
superdiffusion,45,0.80

Diffusion Type,Mean D [µm²/s],Std D,N
normal,0.3421,0.1234,42
subdiffusion,0.0876,0.0432,38
confined,0.0234,0.0123,28
superdiffusion,1.2341,0.5432,8

Diffusion Type,Mean Alpha,Std Alpha,N
normal,0.98,0.12,42
subdiffusion,0.64,0.09,38
confined,0.23,0.08,28
superdiffusion,1.34,0.21,8
```

### 3. Visualisierungen (PDF/SVG)

#### `{name}_distribution.pdf`:
- Pie Chart mit prozentualer Verteilung der 4 Diffusionsarten
- Farbcodiert: Normal (grün), Sub (gold), Confined (rot), Super (blau)

#### `{name}_D_boxplot.pdf`:
- Boxplots der D-Werte für jede Diffusionsart
- Y-Achse: Diffusionskoeffizient [µm²/s]
- Zeigt Median, Quartile, Ausreißer

#### `{name}_alpha_boxplot.pdf`:
- Boxplots der Alpha-Exponenten für jede Diffusionsart
- Horizontal Line bei α=1 (Normal Diffusion)
- Y-Achse: MSD-Exponent α

---

## 🔧 Technische Details

### Multi-Scale Sliding Window Analyse

**Warum Multi-Scale?**
- Kleine Windows (30 Frames): Hohe zeitliche Auflösung, Details
- Große Windows (96 Frames): Robuste Statistik, Trends
- Kombiniert: Best of both worlds

**Default Window-Größen:**
```python
window_sizes = [30, 48, 64, 96]  # Frames
overlap = 50%  # Windows überlappen
```

**Für jeden Track:**
1. Berechne Features für alle Windows aller Größen
2. Predict Diffusionstyp pro Window
3. Aggregiere Predictions: Mittlere Wahrscheinlichkeit über alle Windows pro Frame
4. Wähle wahrscheinlichstes Label pro Frame

### Glättung (Smoothing)

**Problem:** Rauschen kann zu kurzen, unrealistischen Segmenten führen

**Lösung:** `min_segment_length = 30 Frames`

**Algorithmus:**
1. Finde alle Segmente (zusammenhängende Regionen mit gleichem Label)
2. Für Segmente < 30 Frames:
   - Ersetze durch Label der Nachbar-Segmente
   - Präferiere häufigeren Nachbarn
3. Merge benachbarte Segmente mit gleichem Label

**Resultat:** Keine kurzen "Flackern", stabilere Klassifikation

### MSD & D Berechnung

**Pro klassifiziertes Segment:**

```python
# 1. Berechne MSD für verschiedene Lags
MSD(lag) = mean((r(t+lag) - r(t))²)

# 2. Log-Log Regression
log(MSD) = α · log(lag) + const
→ α ist MSD-Exponent (slope)

# 3. Berechne D aus MSD
D = MSD(lag=1) / (6 * t^α)  # 3D Diffusion
```

**Interpretation:**
- **α ≈ 1.0:** Normal Diffusion, D = Diffusionskoeffizient
- **α < 1.0:** Subdiffusion, D = "effektiver" Koeffizient
- **α > 1.0:** Superdiffusion, D > normal
- **α ≈ 0:** Confined, D sehr klein

---

## 🎨 GUI-Integration (Coming Soon!)

**Neuer Tab "Track Analysis" in der GUI:**

```
┌──────────────────────────────────────────┐
│ 🔬 TRACK ANALYSIS                        │
├──────────────────────────────────────────┤
│                                          │
│ [Select XML File]  [Select Folder]      │
│                                          │
│ ┌────────────────────────────────────┐  │
│ │ PREVIEW                            │  │
│ │                                    │  │
│ │ Tracks: 45                         │  │
│ │ Mean Length: 127.3 frames          │  │
│ │ Min/Max: 42 / 234 frames           │  │
│ │                                    │  │
│ └────────────────────────────────────┘  │
│                                          │
│ RF Model: [Browse...] ✅ Loaded         │
│                                          │
│ Frame Rate [Hz]: [20.0]                 │
│                                          │
│ Output Dir: [Browse...] ✅ Set          │
│                                          │
│ [🚀 START ANALYSIS]                     │
│                                          │
│ Progress: [████████░░] 80%              │
│                                          │
│ ✅ 36/45 tracks analyzed                │
│                                          │
└──────────────────────────────────────────┘
```

**Features:**
- Drag & Drop XML-Dateien
- Live-Preview der XML-Struktur
- Auto-Detect RF-Modell im Output-Ordner
- Real-time Progress Bar
- Automatisches Öffnen der Ergebnisse

---

## 📈 Best Practices

### 1. Frame Rate richtig setzen

**Wichtig:** Die Frame-Rate beeinflusst D-Wert Berechnungen!

```bash
# Wenn deine Aufnahmen 20 Hz (50 ms/frame) sind:
--frame-rate 20.0

# Wenn 10 Hz (100 ms/frame):
--frame-rate 10.0
```

**Falsche Frame-Rate → Falsche D-Werte!**

### 2. Track-Länge beachten

**Minimum:** 30 Frames
- Kürzere Tracks werden automatisch gefiltert
- Grund: Mindestens 30 Frames für robuste Feature-Berechnung

**Optimal:** 100-300 Frames
- Genug für Multi-Scale Windows
- Mehrere Segmente möglich
- Robuste Statistik

**Sehr lange (>500 Frames):** Auch OK
- Mehr Segmente
- Bessere zeitliche Auflösung
- Aber: Länger zur Berechnung

### 3. Trainings-Daten sollten repräsentativ sein

**Wichtig:** RF-Modell nur so gut wie Trainings-Daten!

**Check:**
- Wurden alle 4 Diffusionstypen trainiert?
- Decken Trainings-Daten deine experimentellen Bedingungen ab?
- Gleicher Detektor? Gleiche Frame-Rate?

**Wenn experimentelle Daten sehr anders:**
- Trainiere neues Modell mit synthetischen Daten die näher an Experiment sind
- Oder: Nutze experimentelle Daten für Fine-Tuning (erfordert manuelle Labels)

### 4. Interpretiere Ergebnisse kritisch

**Sanity Checks:**

✅ **Normal Diffusion dominant?** (>50%)
- Für die meisten biologischen Systeme erwartet

⚠️ **Zu viel Superdiffusion?** (>10%)
- Könnte auf Drift, Konvektion oder Fehler hinweisen
- Check: Ist Sample wirklich ruhig?

⚠️ **Sehr kurze Segmente trotz Glättung?**
- Könnte auf schnelles Switching hinweisen
- Oder: Niedrige Klassifikations-Konfidenz

✅ **D-Werte im erwarteten Bereich?**
- Kleine Moleküle in Wasser: ~100-1000 µm²/s
- Proteine in Wasser: ~1-10 µm²/s
- Proteine in Gel: ~0.01-1 µm²/s

---

## 🐛 Troubleshooting

### Problem: "No tracks found in XML"

**Ursachen:**
- XML ist nicht im TrackMate-Format
- XML ist korrupt
- Alle Tracks <30 Frames

**Lösung:**
```bash
# Check XML-Struktur:
head -50 merged_tracks.xml

# Sollte enthalten:
# <AllSpots>
# <AllTracks>
```

### Problem: "RF Model not found"

**Lösung:**
```bash
# Check Modell-Pfad:
ls -lh ./rf_training/random_forest_diffusion.joblib

# Wenn nicht vorhanden, trainiere zuerst:
python batch_simulator.py --preset quick --train-rf --output ./rf_training
```

### Problem: "Excel export failed"

**Ursache:** `openpyxl` nicht installiert

**Lösung:**
```bash
pip install openpyxl>=3.0.0
```

### Problem: "Alle Tracks als 'normal' klassifiziert"

**Mögliche Ursachen:**
1. **Frame-Rate falsch gesetzt**
   - Zu hoch → Alles sieht wie Normal aus
   - Lösung: Korrekte Frame-Rate angeben

2. **RF-Modell schlecht trainiert**
   - Nur Normal-Samples im Training?
   - Lösung: Trainiere mit t=0 bis t=180 min (alle Typen)

3. **Experimentelle Daten tatsächlich nur Normal**
   - Check: Sind Bedingungen unterschiedlich zu Simulation?

### Problem: "Sehr lange Rechenzeit"

**Ursachen:**
- Viele lange Tracks (>500 Frames)
- Viele Windows

**Lösungen:**
```bash
# 1. Reduziere Window-Größen (in Code):
window_sizes = [48, 64]  # Statt [30, 48, 64, 96]

# 2. Filtere kürzere Tracks (in Code):
if len(track) >= 50:  # Statt >= 30
```

**Typische Zeiten:**
- 10 Tracks @ 100 Frames: ~10 Sekunden
- 50 Tracks @ 150 Frames: ~1 Minute
- 100 Tracks @ 200 Frames: ~5 Minuten

---

## 🔬 Wissenschaftliche Validierung

### Vergleich mit manuellen Labels

**Workflow:**
1. Wähle 10-20 representative Tracks
2. Klassifiziere manuell (z.B. per MSD-Plot)
3. Vergleiche mit automatischer Klassifikation
4. Berechne Accuracy

**Beispiel:**
```python
# Manual labels
manual = ["normal", "subdiffusion", "normal", ...]

# Automatic labels (aus Excel)
automatic = ["normal", "subdiffusion", "confined", ...]

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(manual, automatic)
print(f"Accuracy: {accuracy:.2%}")
```

**Erwartete Accuracy:** 80-90%
- Höher: Sehr gut!
- Niedriger: Check Trainings-Daten

### Feature Importance Check

**Welche Features sind wichtig?**
```python
from joblib import load

model_data = load("rf_model.joblib")
importances = model_data["model"].feature_importances_
feature_names = model_data["feature_names"]

# Top 5
top_indices = np.argsort(importances)[::-1][:5]
for i in top_indices:
    print(f"{feature_names[i]}: {importances[i]:.3f}")
```

**Erwartete Top Features:**
1. `msd_loglog_slope` (α-Exponent) - **Wichtigste!**
2. `straightness`
3. `confinement_radius`
4. `mean_step_xy`
5. `directional_persistence`

**Wenn anders:** Könnte auf Probleme hinweisen

---

## 📚 Literatur-Referenzen

**Diffusionstypen:**
- Metzler & Klafter (2000): "The random walk's guide to anomalous diffusion"
- Höfling & Franosch (2013): "Anomalous transport in crowded environments"

**Single-Particle Tracking:**
- Manzo & Garcia-Parajo (2015): "A review of progress in single particle tracking"
- Shen et al. (2017): "Single particle tracking: from theory to biophysical applications"

**Machine Learning für SPT:**
- Granik & Weiss (2019): "Machine learning classification of trajectories"
- Muñoz-Gil et al. (2021): "Objective comparison of methods to decode anomalous diffusion"

---

## 🎯 Zusammenfassung

**Was das Modul macht:**
✅ Automatische Klassifikation von Tracking-Daten
✅ Multi-Scale Analyse für Robustheit
✅ Glättung für stabile Segmente
✅ MSD/D Berechnung pro Segment
✅ Umfassender Output (Excel, CSV, Plots)
✅ Batch-fähig für große Datensätze

**Was es NICHT macht:**
❌ Tracking selbst (nutze TrackMate/ImageJ)
❌ Bildverarbeitung (nur bereits getrackte Daten)
❌ Training (nutze batch_simulator.py mit --train-rf)

**Für wen:**
- Masterthesis: Hydrogel-Polymerisation Analyse
- Biophysik: Protein-Diffusion in Zellen
- Material Science: Nanopartikel-Tracking
- Jeder mit TrackMate XML-Daten

**Next Steps:**
1. Trainiere RF-Modell mit synthetischen Daten
2. Analysiere deine TrackMate XMLs
3. Vergleiche mit manuellen Labels (Validierung)
4. Nutze für Publikation!

---

**Version:** 4.1
**Status:** Production-Ready ✅
**GUI-Integration:** Coming Soon
**Autor:** Julian Lerch / Claude AI
