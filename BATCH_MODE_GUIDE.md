# 🚀 BATCH-MODUS ANLEITUNG - TIFF Simulator V4.0

## ✅ Batch-Modus in der Desktop-GUI (V4.0 Advanced)

Die Desktop-App (Build via `Build Desktop App` → `dist/TIFF_Simulator_V4`) kann den **kompletten Batch-Lauf inklusive Random-Forest-Training** direkt aus dem Batch-Tab starten.

### Schritt-für-Schritt
1. Öffne den Batch-Tab und aktiviere oben die Checkbox **„Batch-Modus aktivieren“**.
2. Trage deine Polymerisationszeiten ein oder nutze die Presets.
3. Lege Wiederholungen, Spot-Range und optional Astigmatismus fest.
4. Wähle dein Ausgabeverzeichnis.
5. Aktiviere **„🌲 Parallel ein Premium-Random-Forest trainieren“**, falls ein Klassifikator mitlaufen soll, und passe Fenstergröße, Schrittweite, Baumanzahl/Tiefe sowie Subsampling- und Fenster-Limits an.
6. Starte den Lauf mit **„SIMULATION STARTEN“** – TIFF-Stacks, Metadata und (falls aktiviert) Random-Forest landen im Output-Ordner.

### Random-Forest mittrainieren
- Während des Batch-Laufs werden nach jedem TIFF automatisch Sliding-Window-Features extrahiert und das Modell incrementell weitertrainiert – **auch über alle Wiederholungen hinweg**.
- Der finale Klassifikator (`random_forest_diffusion.joblib`), die Feature-Tabelle sowie eine Trainingszusammenfassung liegen direkt im Batch-Output.
- Statusmeldungen erscheinen live in der GUI; nach Abschluss zeigt ein Dialog den Speicherpfad und die Anzahl verwendeter Fenster.

### Hinweise
- Der Batch-Tab nutzt exakt dieselben Simulationsparameter wie der Single-Modus – es gibt **keine Änderungen an der TIFF-Physik**.
- Wird der Batch-Modus mehrfach hintereinander gestartet, beginnt jedes Mal ein neues Modelltraining; innerhalb eines Batch-Laufs sammelt der Random Forest jedoch alle Fenster aus allen Wiederholungen.
- Exportoptionen (JSON/TXT/CSV) greifen auch im Batch und werden gemeinsam mit dem Modell gespeichert.

---

## ✅ Alternative: Batch-Simulator direkt nutzen (EMPFOHLEN!)

### Quick Start:

```bash
python batch_simulator.py --preset quick --output ./batch_output

# Random-Forest-Training gleich mitstarten
python batch_simulator.py --preset thesis --output ./rf_output --train-rf \
    --rf-window 48 --rf-step 16 --rf-estimators 1024 --rf-max-depth 28 \
    --rf-min-leaf 3 --rf-min-split 6 --rf-max-samples 0.85 \
    --rf-max-windows-per-class 100000 --rf-max-windows-per-track 600
```

### Alle Optionen:

```python
from batch_simulator import BatchSimulator, PresetBatches

# Methode 1: Preset verwenden
batch = PresetBatches.quick_test("./output")
batch.run()

# Methode 2: Custom Batch
from tiff_simulator_v3 import TDI_PRESET

batch = BatchSimulator("./output")

# Füge Simulationen hinzu
for t_poly in [0, 30, 60, 90, 120]:
    batch.add_task({
        'detector': TDI_PRESET,
        'mode': 'polyzeit',
        't_poly_min': t_poly,
        'astigmatism': False,
        'filename': f"tdi_t{t_poly}min.tif",
        'image_size': (256, 256),
        'num_spots': 30,
        'num_frames': 200,
        'frame_rate_hz': 20.0,
        'd_initial': 0.24  # ← KORRIGIERT!
    })

# Führe aus mit Progress
batch.run(progress_callback=lambda c, t, s: print(f"{c}/{t}: {s}"))
```

---

## 📋 VERFÜGBARE BATCH-PRESETS:

### 1. Quick Test
```python
batch = PresetBatches.quick_test("./output")
```
- **3 TIFFs** in ~2-5 Minuten
- Polyzeiten: 30, 60, 90 min
- 64×64 px, 50 frames
- Perfekt zum Testen!

### 2. Masterthesis
```python
batch = PresetBatches.masterthesis_full("./output")
```
- **60+ TIFFs** in ~1 Stunde
- Vollständige Parameterstudie
- TDI-G0 + Tetraspecs
- 2D + 3D (Astigmatismus)
- z-Stack Kalibrierung
- 3 Wiederholungen pro Bedingung

### 3. Publication Quality
```python
batch = PresetBatches.publication_quality("./output")
```
- **30 TIFFs** in ~2-3 Stunden
- Hohe Auflösung: 256×256 px
- Viele Spots: 50
- Viele Frames: 500
- 5 Wiederholungen für Statistik

---

## 🌲 Random-Forest Co-Training

- **Was passiert?** Jede simulierte Trajektorie wird per Sliding Window (Standard: 48 Frames, Schritt 16) in Trainingsfenster zerlegt. Daraus werden 27 physikalisch motivierte Features extrahiert (MSD-Lags + Log-Log-Slope, Straightness, Radius of Gyration, Richtungs- & Geschwindigkeits-Autokorrelation, Bounding Box, z-Range, Step-Momente, …).
- **Modellgröße:** 1024 Trees mit Tiefe 28, `class_weight="balanced_subsample"`, Out-of-Bag-Validierung mit Konfusionsmatrix – robust für alle vier Diffusionsarten über viele Polymerisationszeiten ohne explodierende Modellgrößen.
- **Outputs:**
  - `random_forest_diffusion.joblib` – fertig trainiertes Modell (inkl. Feature-Namen & Config)
  - `rf_training_features.csv` – alle Fenster + Label + Polyzeit-Metadaten
  - `rf_training_summary.json` – Samples, Klassenverteilung, Feature-Importances, Trainings- und OOB-Accuracy, Validierungs-Konfusionsmatrix & Klassifikationsreport, Fensterlimits pro Klasse/Polyzeit
- **Aktivierung:**
  - **CLI:** `--train-rf` Flag + optionale Feintuning-Parameter (`--rf-window`, `--rf-step`, `--rf-estimators`, `--rf-max-depth`, `--rf-min-leaf`, `--rf-min-split`, `--rf-random-state`, `--rf-max-samples`, `--rf-max-windows-per-class`, `--rf-max-windows-per-track`).
  - **Python:** `BatchSimulator(..., enable_rf=True, rf_config={...})`
  - **GUI:** Batch-Tab → Checkbox „Random-Forest während des Batch-Laufs mittrainieren“.
- **Best Practice:** Lass mehrere Polymerisationszeiten im selben Lauf laufen, damit der Forest sämtliche Übergänge in den synthetischen Tracks sieht – genau dafür ist das Sliding-Window-Feature-Set optimiert.

---

## 🛠️ CUSTOM BATCH FÜR IHRE MASTERTHESIS

**Mit den KORRIGIERTEN D-Werten:**

```python
from batch_simulator import BatchSimulator
from tiff_simulator_v3 import TDI_PRESET, TETRASPECS_PRESET

# Erstelle Batch
batch = BatchSimulator("./masterthesis_data")

# Parameter
d_initial = 0.24  # µm²/s - REALISTISCH!
poly_times = [0, 10, 30, 60, 90, 120, 180]  # min
repeats = 3

for detector in [TDI_PRESET, TETRASPECS_PRESET]:
    for t_poly in poly_times:
        for repeat in range(repeats):
            # 2D Simulation
            batch.add_task({
                'detector': detector,
                'mode': 'polyzeit',
                't_poly_min': t_poly,
                'astigmatism': False,
                'filename': f"{detector.name}_2d_t{int(t_poly)}min_r{repeat+1}.tif",
                'image_size': (256, 256),
                'num_spots': 30,
                'num_frames': 300,
                'frame_rate_hz': 20.0,
                'd_initial': d_initial
            })

            # 3D Simulation (mit Astigmatismus)
            batch.add_task({
                'detector': detector,
                'mode': 'polyzeit_astig',
                't_poly_min': t_poly,
                'astigmatism': True,
                'filename': f"{detector.name}_3d_t{int(t_poly)}min_r{repeat+1}.tif",
                'image_size': (256, 256),
                'num_spots': 30,
                'num_frames': 300,
                'frame_rate_hz': 20.0,
                'd_initial': d_initial
            })

# WICHTIG: Mit Progress-Callback ausführen!
def progress(current, total, status):
    print(f"[{current}/{total}] {status}")
    # Optional: In Datei loggen
    with open("batch_progress.log", "a") as f:
        f.write(f"{datetime.now()}: [{current}/{total}] {status}\n")

batch.run(progress_callback=progress)

print(f"\n✅ Batch fertig! Alle TIFFs in: {batch.output_dir}")
print(f"📊 Metadata CSV: {batch.output_dir}/batch_summary.csv")
```

**Das erstellt:**
- 2 Detektoren × 7 Zeiten × 2 Modi (2D/3D) × 3 Repeats = **84 TIFFs**
- Dauer: ~2-3 Stunden (mit V4.0 Performance!)
- Vollständige Metadata (JSON, TXT, CSV)

---

## 📊 BATCH-MODUS FÜR SPEZIFISCHE ANALYSEN

### A) Nur D-Wert Variation (feste Zeit)

```python
batch = BatchSimulator("./d_variation")

d_values = [0.15, 0.20, 0.24, 0.28, 0.32]  # µm²/s
t_poly = 60  # min (feste Zeit)

for d in d_values:
    for repeat in range(5):
        batch.add_task({
            'detector': TDI_PRESET,
            'mode': 'polyzeit',
            't_poly_min': t_poly,
            'd_initial': d,
            'filename': f"d{d:.2f}_r{repeat+1}.tif",
            'image_size': (256, 256),
            'num_spots': 30,
            'num_frames': 200,
            'frame_rate_hz': 20.0
        })

batch.run()
```

### B) Zeit-Serie (feste D₀)

```python
batch = BatchSimulator("./time_series")

times = [0, 5, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180]  # min
d_initial = 0.24  # µm²/s

for t in times:
    for repeat in range(3):
        batch.add_task({
            'detector': TDI_PRESET,
            'mode': 'polyzeit',
            't_poly_min': t,
            'd_initial': d_initial,
            'filename': f"t{t:03d}min_r{repeat+1}.tif",
            'image_size': (256, 256),
            'num_spots': 30,
            'num_frames': 200,
            'frame_rate_hz': 20.0
        })

batch.run()
```

### C) Nur z-Stack Kalibrierung

```python
batch = BatchSimulator("./z_calibration")

for detector in [TDI_PRESET, TETRASPECS_PRESET]:
    batch.add_task({
        'detector': detector,
        'mode': 'z_stack',
        't_poly_min': 0,  # Keine Polymerisation
        'astigmatism': True,
        'filename': f"zstack_{detector.name}.tif",
        'image_size': (256, 256),
        'num_spots': 50,
        'z_range_um': (-1.0, 1.0),
        'z_step_um': 0.05
    })

batch.run()
```

---

## 🎯 EMPFEHLUNG FÜR IHRE MASTERTHESIS:

**Workflow:**

1. **Testen** (5-10 Minuten):
   ```bash
   python -c "from batch_simulator import PresetBatches; PresetBatches.quick_test('./test').run()"
   ```

2. **Kleine Studie** (~30 Minuten):
   ```python
   # Nur 3 Zeiten, 2 Repeats
   times = [0, 60, 120]
   repeats = 2
   # → 12 TIFFs (2×3×2)
   ```

3. **Vollständige Thesis-Daten** (~2-3 Stunden):
   ```python
   # Custom Batch wie oben (84 TIFFs)
   ```

4. **Analyse**:
   - Alle TIFFs mit TrackMate/ThunderSTORM analysieren
   - Ground Truth aus Metadata CSV
   - D-Wert Rekonstruktion
   - Plots für Thesis

---

## 💡 TIPPS & TRICKS

### Parallele Batches

Wenn Sie mehrere CPU-Kerne haben:

```bash
# Terminal 1
python batch_1.py &

# Terminal 2
python batch_2.py &

# Etc.
```

### Resume bei Absturz

Batch-Simulator erstellt nach jedem TIFF eine CSV.
Bei Absturz: Checken Sie welche TIFFs fehlen und erstellen Sie neuen Batch nur für diese.

### Speicherplatz

Jedes TIFF (256×256, 300 frames, 16-bit):
- ~40 MB pro Datei
- 84 TIFFs = ~3.4 GB
- + Metadata = ~3.5 GB gesamt

**Planen Sie genug Speicher ein!**

---

## ❓ FAQ

**Q: Läuft der Batch auch in der GUI V4.0?**
A: Ja! Der Batch-Tab führt die Simulationen direkt aus (inklusive Random-Forest-Training). Alle Parameter stammen 1:1 aus den GUI-Settings.

**Q: Kann ich die GUI-Parameter für die CLI übernehmen?**
A: Klar – die Werte aus dem Single-Tab entsprechen den Argumenten für `BatchSimulator`. Für automatisierte Runs kannst du sie in ein Python-Script oder die CLI übertragen.

**Q: Progress-Tracking?**
A: Nutzen Sie `progress_callback` - siehe Beispiele oben!

**Q: Kann ich Batch abbrechen?**
A: Ja, mit Ctrl+C. Bereits erstellte TIFFs bleiben erhalten.

---

## 📝 ZUSAMMENFASSUNG

**Batch-Modus nutzen:**
1. ✅ GUI V4.0 Batch-Tab (inkl. optionalem Random-Forest)
2. ✅ `batch_simulator.py` direkt ausführen
3. ✅ Presets: `quick`, `thesis`, `publication`
4. ✅ Custom: Python-Script schreiben
5. ✅ **WICHTIG:** `d_initial = 0.24` verwenden!

**Für Ihre Thesis empfohlen:**
```python
# Custom Batch mit ~84 TIFFs, ~2-3 Stunden
# Alle Zeiten, Detektoren, 2D+3D, Repeats
```

Viel Erfolg! 🎓
