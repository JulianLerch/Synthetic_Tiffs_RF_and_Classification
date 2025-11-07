# 🌲 Random Forest Klassifizierer - Nutzungsanleitung

## Übersicht

Dieser Random Forest (RF) Klassifizierer wurde trainiert, um **4 verschiedene Diffusionsarten** in Single-Particle Tracking Daten zu unterscheiden:

1. **Normal** - Klassische Brownsche Bewegung (α ≈ 1.0)
2. **Subdiffusion** - Verlangsamte Diffusion durch Hindernis-Netzwerk (α ≈ 0.7)
3. **Superdiffusion** - Beschleunigte Diffusion durch Konvektion/aktive Prozesse (α ≈ 1.3)
4. **Confined** - Eingeschränkte Diffusion in begrenztem Raum

---

## 📊 Feature-Set: 27 Features

Der Random Forest verwendet ein umfassendes Set von **27 quantitativen Features**, die aus Trajektorien-Segmenten (typisch 32-64 Frames) extrahiert werden.

### 1. Schrittweiten-Statistiken (6 Features)

Diese Features beschreiben die Verteilung der Schrittlängen in der xy-Ebene:

| Feature | Berechnung | Interpretation |
|---------|------------|----------------|
| `mean_step_xy` | `np.mean(step_xy)` | Durchschnittliche Schrittweite - höher bei schneller Diffusion |
| `std_step_xy` | `np.std(step_xy)` | Streuung der Schrittweiten - höher bei heterogener Bewegung |
| `median_step_xy` | `np.median(step_xy)` | Robuster Mittelwert gegen Ausreißer |
| `mad_step_xy` | `median(abs(step_xy - median))` | Median Absolute Deviation - robuste Streuung |
| `max_step_xy` | `np.max(step_xy)` | Maximale Schrittweite - indikativ für Superdiffusion |
| `min_step_xy` | `np.min(step_xy)` | Minimale Schrittweite |

**Code:**
```python
steps = np.diff(window_pos, axis=0)  # Shape: (N-1, 3)
step_xy = np.linalg.norm(steps[:, :2], axis=1)  # 2D Schrittlängen
mean_step_xy = np.mean(step_xy)
```

---

### 2. Z-Achsen Bewegung (2 Features)

| Feature | Berechnung | Interpretation |
|---------|------------|----------------|
| `mean_step_z` | `np.mean(abs(steps[:, 2]))` | Durchschnittliche z-Bewegung |
| `std_step_z` | `np.std(abs(steps[:, 2]))` | Variabilität in z-Richtung |

---

### 3. Mean Squared Displacement (MSD) - 5 Features

Das MSD ist der Goldstandard zur Charakterisierung von Diffusion. Der Exponent α in MSD(τ) ∝ τ^α unterscheidet die Diffusionstypen:

| Feature | Berechnung | Interpretation |
|---------|------------|----------------|
| `msd_lag1` | `MSD(Δt = 1 Frame)` | Kurzzeitdiffusion |
| `msd_lag2` | `MSD(Δt = 2 Frames)` | |
| `msd_lag4` | `MSD(Δt = 4 Frames)` | |
| `msd_lag8` | `MSD(Δt = 8 Frames)` | Langzeitdiffusion |
| `msd_loglog_slope` | Steigung in log(MSD) vs log(τ) | **α-Exponent**: <1 = Sub, ≈1 = Normal, >1 = Super |

**Code:**
```python
def compute_msd(positions, lag):
    """MSD bei gegebenem Zeitlag"""
    diff = positions[lag:] - positions[:-lag]
    return np.mean(np.sum(diff**2, axis=1))

# Berechne MSD für verschiedene Lags
msd_lag1 = compute_msd(window_pos, 1)
msd_lag2 = compute_msd(window_pos, 2)
msd_lag4 = compute_msd(window_pos, 4)
msd_lag8 = compute_msd(window_pos, 8)

# Berechne α-Exponent aus log-log Steigung
lags = np.array([1, 2, 4, 8])
msds = np.array([msd_lag1, msd_lag2, msd_lag4, msd_lag8])
log_lags = np.log(lags)
log_msds = np.log(msds)

# Lineare Regression in log-log Raum
alpha = np.polyfit(log_lags, log_msds, deg=1)[0]
```

---

### 4. Trajektorien-Form (8 Features)

Diese Features beschreiben die geometrische Form der Trajektorie:

| Feature | Berechnung | Interpretation |
|---------|------------|----------------|
| `straightness` | `end_to_end_distance / total_path_length` | 1.0 = gerade Linie, <0.1 = stark gewunden |
| `confinement_radius` | `sqrt(mean(r²))` vom Schwerpunkt | Größe des erkundeten Bereichs |
| `radius_of_gyration` | `sqrt(sum(eigenvalues(Cov)))` | Isotrope Größe der Trajektorie |
| `gyration_asymmetry` | `(λ_max - λ_min) / sum(λ)` | Anisotropie: 0 = isotrop, 1 = stark anisotrop |
| `bounding_box_area` | `(x_max - x_min) * (y_max - y_min)` | Rechteckige Begrenzung |
| `axial_range` | `z_max - z_min` | Vertikale Ausdehnung |
| `step_p90` | `90. Perzentil(step_xy)` | Große Schritte |
| `step_p10` | `10. Perzentil(step_xy)` | Kleine Schritte |

**Code:**
```python
# Straightness
straight_dist = np.linalg.norm(window_pos[-1] - window_pos[0])
total_path = np.sum(step_xy)
straightness = straight_dist / total_path

# Confinement Radius
xy_centered = window_pos[:, :2] - np.mean(window_pos[:, :2], axis=0)
confinement_radius = np.sqrt(np.mean(np.sum(xy_centered**2, axis=1)))

# Radius of Gyration & Asymmetry
centered = window_pos - np.mean(window_pos, axis=0)
cov = np.dot(centered.T, centered) / centered.shape[0]
eigvals = np.linalg.eigvalsh(cov)  # Eigenwerte der Kovarianzmatrix
radius_of_gyration = np.sqrt(np.sum(eigvals))
gyration_asymmetry = (eigvals[-1] - eigvals[0]) / np.sum(eigvals)
```

---

### 5. Bewegungsdynamik (6 Features)

Diese Features charakterisieren die zeitliche Dynamik der Bewegung:

| Feature | Berechnung | Interpretation |
|---------|------------|----------------|
| `turning_angle_mean` | `mean(arccos(v_i · v_i+1))` | Durchschnittlicher Richtungswechsel |
| `turning_angle_std` | `std(turning_angles)` | Variabilität der Richtungswechsel |
| `directional_persistence` | `mean(cos(turning_angles))` | Persistenz: 1 = geradeaus, -1 = Umkehr |
| `velocity_autocorr` | `corr(v_t, v_t+1)` | Autokorrelation der Geschwindigkeiten |
| `step_skewness` | Schiefe der Schrittweiten-Verteilung | Asymmetrie: >0 = rechtsschief |
| `step_kurtosis` | Kurtosis der Schrittweiten | Schwere Flanken: >0 = Ausreißer-reich |

**Code:**
```python
# Turning Angles
v1 = steps[:-1, :2]  # Schritte i
v2 = steps[1:, :2]   # Schritte i+1
norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
cos_angles = np.sum(v1 * v2, axis=1) / (norms + 1e-12)
turning_angles = np.arccos(np.clip(cos_angles, -1, 1))
turning_angle_mean = np.mean(turning_angles)

# Directional Persistence
directional_persistence = np.mean(cos_angles)

# Velocity Autocorrelation
velocity_autocorr = np.corrcoef(step_xy[:-1], step_xy[1:])[0, 1]

# Moments
step_skewness = scipy.stats.skew(step_xy)
step_kurtosis = scipy.stats.kurtosis(step_xy)
```

---

## 🔧 Verwendung des Random Forest

### 1. Modell Laden

Der trainierte Random Forest wird als `.pkl` Datei mit `joblib` gespeichert:

```python
from joblib import load
import numpy as np

# Lade das trainierte Modell
model_path = "output/batch_2025-XX-XX_XXXXXX/rf_model_balanced.pkl"
rf_model = load(model_path)

# Das Modell-Artefakt enthält:
# - rf_model['model']: Der sklearn RandomForestClassifier
# - rf_model['feature_names']: Liste der 27 Feature-Namen
# - rf_model['label_mapping']: Dict {0: 'normal', 1: 'subdiffusion', ...}
# - rf_model['config']: RFTrainingConfig mit allen Hyperparametern
```

### 2. Features aus Trajektorie Extrahieren

```python
def extract_features(trajectory_positions):
    """
    Extrahiere 27 Features aus einer Trajektorie.

    Parameters:
    -----------
    trajectory_positions : np.ndarray
        Shape (N, 3) - xyz-Positionen über N Frames [µm]

    Returns:
    --------
    features : np.ndarray
        Shape (27,) - Feature-Vektor
    """

    # Berechne Schritte
    steps = np.diff(trajectory_positions, axis=0)
    step_xy = np.linalg.norm(steps[:, :2], axis=1)
    step_z = np.abs(steps[:, 2])

    # Feature 1-6: Schrittweiten-Statistiken
    mean_step_xy = float(np.mean(step_xy))
    std_step_xy = float(np.std(step_xy))
    median_step_xy = float(np.median(step_xy))
    mad_step_xy = float(np.median(np.abs(step_xy - median_step_xy)))
    max_step_xy = float(np.max(step_xy))
    min_step_xy = float(np.min(step_xy))

    # Feature 7-8: Z-Bewegung
    mean_step_z = float(np.mean(step_z))
    std_step_z = float(np.std(step_z))

    # Feature 9-13: MSD
    def compute_msd(lag):
        if lag >= len(trajectory_positions):
            return 0.0
        diff = trajectory_positions[lag:] - trajectory_positions[:-lag]
        return float(np.mean(np.sum(diff**2, axis=1)))

    msd_lag1 = compute_msd(1)
    msd_lag2 = compute_msd(2)
    msd_lag4 = compute_msd(4)
    msd_lag8 = compute_msd(8)

    # MSD Slope (α-Exponent)
    lags = np.array([1, 2, 4, 8])
    msds = np.array([msd_lag1, msd_lag2, msd_lag4, msd_lag8])
    mask = msds > 0
    if np.count_nonzero(mask) >= 2:
        log_lags = np.log(lags[mask])
        log_msds = np.log(msds[mask])
        msd_slope = np.polyfit(log_lags, log_msds, 1)[0]
    else:
        msd_slope = 0.0

    # Feature 14-21: Geometrie
    straight_dist = np.linalg.norm(trajectory_positions[-1, :2] - trajectory_positions[0, :2])
    total_path = np.sum(step_xy) + 1e-9
    straightness = straight_dist / total_path

    xy_centered = trajectory_positions[:, :2] - np.mean(trajectory_positions[:, :2], axis=0)
    confinement_radius = float(np.sqrt(np.mean(np.sum(xy_centered**2, axis=1))))

    centered = trajectory_positions - np.mean(trajectory_positions, axis=0)
    cov = np.dot(centered.T, centered) / centered.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    radius_of_gyration = float(np.sqrt(np.sum(np.clip(eigvals, 0, None))))
    gyration_asymmetry = float((eigvals[-1] - eigvals[0]) / (np.sum(eigvals) + 1e-12))

    step_p90 = float(np.percentile(step_xy, 90))
    step_p10 = float(np.percentile(step_xy, 10))

    bbox_area = float((trajectory_positions[:, 0].max() - trajectory_positions[:, 0].min()) *
                     (trajectory_positions[:, 1].max() - trajectory_positions[:, 1].min()))
    axial_range = float(trajectory_positions[:, 2].max() - trajectory_positions[:, 2].min())

    # Feature 22-27: Dynamik
    if len(steps) >= 2:
        v1 = steps[:-1, :2]
        v2 = steps[1:, :2]
        norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
        cos_angles = np.sum(v1 * v2, axis=1) / norms
        cos_angles = np.clip(cos_angles, -1, 1)
        turning_angles = np.arccos(cos_angles)
        turning_angle_mean = float(np.mean(turning_angles))
        turning_angle_std = float(np.std(turning_angles))
        directional_persistence = float(np.mean(cos_angles))
    else:
        turning_angle_mean = 0.0
        turning_angle_std = 0.0
        directional_persistence = 0.0

    if len(step_xy) >= 2:
        velocity_autocorr = float(np.corrcoef(step_xy[:-1], step_xy[1:])[0, 1])
        if not np.isfinite(velocity_autocorr):
            velocity_autocorr = 0.0
    else:
        velocity_autocorr = 0.0

    if std_step_xy > 1e-12:
        centered_steps = (step_xy - mean_step_xy) / std_step_xy
        step_skewness = float(np.mean(centered_steps**3))
        step_kurtosis = float(np.mean(centered_steps**4) - 3.0)
    else:
        step_skewness = 0.0
        step_kurtosis = 0.0

    # Zusammenstellung aller 27 Features
    features = np.array([
        mean_step_xy, std_step_xy, median_step_xy, mad_step_xy, max_step_xy, min_step_xy,
        mean_step_z, std_step_z,
        msd_lag1, msd_lag2, msd_lag4, msd_lag8, msd_slope,
        straightness, confinement_radius, radius_of_gyration, gyration_asymmetry,
        turning_angle_mean, turning_angle_std,
        step_p90, step_p10,
        bbox_area, axial_range,
        directional_persistence, velocity_autocorr,
        step_skewness, step_kurtosis
    ], dtype=np.float32)

    return features
```

### 3. Klassifizierung Durchführen

```python
# Beispiel-Trajektorie (32-64 Frames empfohlen)
trajectory = np.random.randn(48, 3) * 0.5  # Shape (48, 3) [µm]

# Features extrahieren
features = extract_features(trajectory)

# Klassifizierung
prediction = rf_model['model'].predict([features])[0]
probabilities = rf_model['model'].predict_proba([features])[0]

# Label zuordnen
label_mapping = rf_model['label_mapping']
predicted_label = label_mapping[prediction]

print(f"Vorhergesagte Klasse: {predicted_label}")
print(f"Wahrscheinlichkeiten:")
for idx, prob in enumerate(probabilities):
    print(f"  {label_mapping[idx]}: {prob:.3f}")
```

**Beispiel-Output:**
```
Vorhergesagte Klasse: normal
Wahrscheinlichkeiten:
  normal: 0.872
  subdiffusion: 0.089
  superdiffusion: 0.031
  confined: 0.008
```

---

## 📈 Batch-Klassifizierung mit Sliding Window

Für lange Trajektorien verwendet der RF einen **Sliding Window Ansatz**:

```python
def classify_trajectory_windowed(trajectory, window_size=48, step_size=24):
    """
    Klassifiziere Trajektorie mit Sliding Window.

    Parameters:
    -----------
    trajectory : np.ndarray
        Shape (N, 3) - Komplette Trajektorie
    window_size : int
        Anzahl Frames pro Window (z.B. 48)
    step_size : int
        Schrittweite zwischen Windows (z.B. 24 = 50% Overlap)

    Returns:
    --------
    classifications : list of dict
        Pro Window: {'start': int, 'end': int, 'label': str, 'probs': dict}
    """

    results = []

    for start in range(0, len(trajectory) - window_size + 1, step_size):
        end = start + window_size
        window = trajectory[start:end]

        # Features extrahieren
        features = extract_features(window)

        # Klassifizieren
        pred = rf_model['model'].predict([features])[0]
        probs = rf_model['model'].predict_proba([features])[0]

        label = label_mapping[pred]
        prob_dict = {label_mapping[i]: float(p) for i, p in enumerate(probs)}

        results.append({
            'start_frame': start,
            'end_frame': end,
            'label': label,
            'probabilities': prob_dict,
            'confidence': float(probs.max())
        })

    return results

# Beispiel
trajectory = np.random.randn(500, 3) * 0.5  # 500 Frames
classifications = classify_trajectory_windowed(trajectory, window_size=48, step_size=24)

for c in classifications[:3]:  # Erste 3 Windows
    print(f"Frames {c['start_frame']}-{c['end_frame']}: "
          f"{c['label']} (Konfidenz: {c['confidence']:.2f})")
```

---

## 🎯 Performance-Metriken

Der Random Forest wird mit folgenden Hyperparametern trainiert (optimiert für beste Leistung):

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `n_estimators` | 2048 | Anzahl Decision Trees im Ensemble |
| `max_depth` | 20 | Maximale Tiefe pro Baum (Regularisierung) |
| `min_samples_leaf` | 5 | Minimale Samples pro Blatt |
| `min_samples_split` | 10 | Minimale Samples für Split |
| `max_features` | "sqrt" | √27 ≈ 5 Features pro Split |
| `class_weight` | "balanced_subsample" | Automatisches Class Balancing |
| `oob_score` | True | Out-of-Bag Score für Validierung |

**Erwartete Performance** (auf Simulations-Daten):
- **Accuracy**: 90-95% (abhängig von Polymerisationsgrad)
- **OOB Score**: 88-93%
- **Per-Class Precision/Recall**: >85% für alle 4 Klassen

---

## 🔍 Feature Importance

Die wichtigsten Features (typisch):

1. **msd_loglog_slope** (α-Exponent) - **WICHTIGSTES FEATURE**
2. **mean_step_xy** - Unterscheidet schnelle von langsamen Diffusion
3. **straightness** - Unterscheidet confined von freier Diffusion
4. **confinement_radius** - Räumliche Einschränkung
5. **directional_persistence** - Richtungs-Persistenz
6. **velocity_autocorr** - Zeitliche Korrelation
7. **msd_lag1** bis **msd_lag8** - Multi-Scale MSD

**Feature Importance ausgeben:**
```python
importances = rf_model['model'].feature_importances_
feature_names = rf_model['feature_names']

# Sortiere nach Wichtigkeit
sorted_idx = np.argsort(importances)[::-1]

print("Top 10 wichtigste Features:")
for i in sorted_idx[:10]:
    print(f"{feature_names[i]:25s}: {importances[i]:.4f}")
```

---

## 💡 Best Practices

### 1. Window-Größe Wählen
- **Zu klein (<32 Frames)**: Features werden instabil, schlechte MSD-Schätzung
- **Optimal (48-64 Frames)**: Gute Balance zwischen Auflösung und Stabilität
- **Zu groß (>100 Frames)**: Verlust von zeitlicher Auflösung, mögliche Label-Wechsel im Window

### 2. Konfidenz-Schwellenwert
```python
confidence_threshold = 0.7

if max(probabilities) < confidence_threshold:
    print("WARNUNG: Niedrige Konfidenz - Klassifizierung unsicher!")
```

### 3. Multi-Scale Klassifizierung
Verwende mehrere Window-Größen für robustere Ergebnisse:
```python
window_sizes = [32, 48, 64]
predictions = []

for ws in window_sizes:
    results = classify_trajectory_windowed(trajectory, window_size=ws)
    predictions.append(results)

# Majority Voting über alle Window-Größen
```

### 4. Umgang mit Realdaten
Der RF ist auf **synthetischen Daten** trainiert. Für Realdaten:
- Nutze die gleichen Tracking-Parameter (Pixel-Größe, Frame-Rate)
- Kalibriere Lokalisierungs-Genauigkeit
- Verwende z-Kalibrierungs-Stacks für Astigmatismus
- Eventuell: Fine-Tuning mit gelabelten Realdaten

---

## 📝 Zusammenfassung

**Minimales Working Example:**

```python
from joblib import load
import numpy as np

# 1. Modell laden
rf_artifact = load("output/your_batch/rf_model_balanced.pkl")
rf_model = rf_artifact['model']
label_mapping = rf_artifact['label_mapping']

# 2. Trajektorie laden (z.B. aus TIFF-Analyse)
trajectory = np.loadtxt("trajectory.csv", delimiter=",")  # Shape (N, 3)

# 3. Features extrahieren (verwende extract_features() von oben)
features = extract_features(trajectory)

# 4. Klassifizieren
prediction = rf_model.predict([features])[0]
predicted_label = label_mapping[prediction]

print(f"Diffusionstyp: {predicted_label}")
```

---

## 🆘 Support

Bei Fragen oder Problemen:
1. Überprüfe, ob alle 27 Features korrekt berechnet werden
2. Stelle sicher, dass die Einheiten übereinstimmen (µm für Positionen)
3. Checke Feature-Ranges auf NaN/Inf Werte
4. Vergleiche mit Feature-Werten aus dem Training (siehe `rf_training_summary.json`)

**Viel Erfolg mit der Klassifizierung! 🎉**
