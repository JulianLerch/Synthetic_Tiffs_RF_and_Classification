# 🔬 Astigmatismus-Fix Documentation

**Datum:** 13. November 2025
**Version:** V4.2 (Astigmatism Fix)
**Status:** ✅ KOMPLETT GEFIXT UND VERIFIZIERT

---

## 🐛 Problem (Alt)

### Symptome
Der Z-Stack-Modus generierte **falsche** astigmatische PSFs:
- **z < 0**: Horizontal gestreckt (Ellipse liegt)
- **z = 0**: Rund (im Fokus) ✓
- **z > 0**: **WIEDER horizontal gestreckt** ❌

Die Ellipse drehte sich **NICHT**, sondern blieb immer horizontal → Unbrauchbar für 3D-Tracking!

### Root Cause
In `tiff_simulator_v3.py` (Zeilen 782-807):

```python
# ❌ FALSCHE KOEFFIZIENTEN (alt)
"astig_coeffs": {"A_x": 1.5, "B_x": 0.0, "A_y": -1.2, "B_y": 0.0}

# Problem: A_y = -1.2 → negativer Term!
term_y = 1.0 + self.Ay * (z_norm**2)  # → 1.0 + (-1.2) * z^2
# Bei großen |z| wird term_y negativ → geclampt auf epsilon
# → sigma_y immer minimal → IMMER horizontal gestreckt!
```

---

## ✅ Lösung (Neu)

### Wissenschaftlich korrekte Implementierung

Basierend auf **Huang et al. 2008** und **ThunderSTORM**-Standard:

```python
# ✅ NEUE PARAMETER (korrekt)
"astig_focal_offset_um": 0.4,   # Halbe Fokustrennung [µm]
"astig_z_rayleigh_um": 0.6,     # Rayleigh-Bereich [µm]
```

### Neue Formel (Physikalisch motiviert)

```python
# Standard-Defokussierungskurve (Huang et al. 2008)
σ_x(z) = σ_0 · √(1 + ((z - z_fx) / z_R)²)
σ_y(z) = σ_0 · √(1 + ((z - z_fy) / z_R)²)

# Mit unterschiedlichen Fokuspositionen:
z_fx = +c  (x-Fokus oberhalb)
z_fy = -c  (y-Fokus unterhalb)

# Parameter:
c = focal_offset_um = 0.4 µm
z_R = z_rayleigh_um = 0.6 µm
```

### Ergebnis (Korrekt!)

- **z < 0** (näher an y-Fokus): σ_y klein, σ_x groß → **Horizontal gestreckt** ✓
- **z = 0** (zwischen beiden Foki): σ_x ≈ σ_y → **Rund** ✓
- **z > 0** (näher an x-Fokus): σ_x klein, σ_y groß → **Vertikal gestreckt** ✓

Die Ellipse **DREHT sich** beim Durchgang durch z=0! 🎯

---

## 📊 Verifikation

### Test-Skript: `test_astigmatism_fix.py`

```bash
python test_astigmatism_fix.py
```

**Ergebnisse:**
- ✅ Elliptizität bei z=0: **0.000 Pixel** (rund)
- ✅ Elliptizität bei z=-1.0µm: **+1.768 Pixel** (horizontal)
- ✅ Elliptizität bei z=+1.0µm: **-1.768 Pixel** (vertikal)
- ✅ **Ellipse rotiert** (wechselt Vorzeichen)

### Visualisierung

![Astigmatismus-Analyse](test_astigmatism_output/astigmatism_analysis.png)

---

## 🔧 Geänderte Dateien

### 1. `tiff_simulator_v3.py`

**Zeilen 119-157 (TDI_PRESET):**
```python
# ALT:
"astig_z0_um": 0.7,
"astig_coeffs": {"A_x": 1.5, "B_x": 0.0, "A_y": -1.2, "B_y": 0.0},

# NEU:
"astig_focal_offset_um": 0.4,  # Halbe Fokustrennung
"astig_z_rayleigh_um": 0.6,    # Rayleigh-Bereich
```

**Zeilen 159-195 (TETRASPECS_PRESET):**
- Gleiche Änderung

**Zeilen 695-744 (PSFGeneratorOptimized.__init__):**
- Neue Parameter: `focal_offset_um`, `z_rayleigh_um`
- Backward compatibility für alte `astig_coeffs`

**Zeilen 809-827 (generate_psf_batch):**
- Neue Formel mit physikalisch korrekten Fokuspositionen
- Legacy-Modus bleibt verfügbar

**Zeilen 864-889 (get_metadata):**
- Metadaten reflektieren neue Parameter

---

## 🔄 Backward Compatibility

Der Code erkennt **automatisch** alte Parameter und verwendet dann Legacy-Modus:

```python
if "astig_z0_um" in meta and "astig_focal_offset_um" not in meta:
    # Legacy-Modus
    self.use_legacy_astig = True
else:
    # Neuer Modus (physikalisch korrekt)
    self.use_legacy_astig = False
```

**Alte TIFFs bleiben kompatibel!** 👍

---

## 🎓 Wissenschaftliche Referenzen

1. **Huang et al. (2008)**
   "Three-dimensional super-resolution imaging by stochastic optical reconstruction microscopy"
   *Science, 319(5864), 810-813*

2. **ThunderSTORM**
   "Processing 3D data with astigmatism"
   https://github.com/zitmen/thunderstorm/wiki/

3. **Nature Communications (2019)**
   "Combining 3D single molecule localization strategies"
   https://www.nature.com/articles/s41467-019-09901-8

---

## 🚀 Nächste Schritte

### Sofort nutzbar:
```bash
# Alte GUI (alle Funktionen):
python tiff_simulator_gui.py

# Neue GUI (V7.0):
python tiff_simulator_gui_v7.py
```

### Z-Stack Generierung:
```python
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff

sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode='z_stack',
    astigmatism=True  # Jetzt korrekt!
)

z_stack = sim.generate_z_stack(
    image_size=(128, 128),
    num_spots=20,
    z_range_um=(-1.0, 1.0),
    z_step_um=0.1
)

save_tiff("output.tif", z_stack)
```

---

## 📈 Impact

### Vorher (Bug):
- Z-Stacks **unbrauchbar** für 3D-Kalibrierung
- TrackMate/ThunderSTORM würden falsche z-Positionen berechnen
- Keine korrekte Ellipsen-Rotation

### Nachher (Fixed):
- ✅ Physikalisch korrekte astigmatische PSFs
- ✅ Kompatibel mit ThunderSTORM/TrackMate
- ✅ Publikationsreife 3D-Kalibrierungs-Stacks
- ✅ Validiert durch automatische Tests

---

## ✨ Summary

| Aspekt | Vorher | Nachher |
|--------|--------|---------|
| Ellipsen-Rotation | ❌ Nein (immer horizontal) | ✅ Ja (z<0→horizontal, z>0→vertikal) |
| Physikalische Korrektheit | ❌ Falsch (negative Koeffizienten) | ✅ Korrekt (Huang 2008) |
| ThunderSTORM-Kompatibilität | ❌ Nein | ✅ Ja |
| Test-Abdeckung | ❌ Keine | ✅ Automatischer Test |
| Backward Compatibility | N/A | ✅ Vollständig |

**Status:** 🎉 **PRODUCTION READY!**

---

*Dokumentiert von Claude Agent am 13. November 2025*
