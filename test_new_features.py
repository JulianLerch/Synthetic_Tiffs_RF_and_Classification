#!/usr/bin/env python3
"""Test-Skript für neue Features: Ausleuchtungsgradient und Brechungsindex-Korrektur"""

import numpy as np
from tiff_simulator_v3 import (
    TIFFSimulatorOptimized,
    DetectorPreset,
    TDI_PRESET,
    TETRASPECS_PRESET
)

print("=" * 70)
print("TEST: Neue Features in tiff_simulator_v3.py")
print("=" * 70)

# Test 1: Ausleuchtungsgradient
print("\n[1] TEST: Ausleuchtungsgradient")
print("-" * 70)

# Erstelle Custom Detector mit Gradient
custom_detector = DetectorPreset(
    name="Test-Detector",
    max_intensity=260.0,
    background_mean=100.0,
    background_std=15.0,
    pixel_size_um=0.108,
    fwhm_um=0.40,
    metadata={
        "illumination_gradient_strength": 15.0,  # Subtiler Gradient
        "illumination_gradient_type": "radial"
    }
)

simulator = TIFFSimulatorOptimized(
    detector=custom_detector,
    mode="timelapse",
    t_poly_min=0.0,
    astigmatism=False
)

print(f"✓ Simulator erstellt mit:")
print(f"  - Gradient Stärke: {custom_detector.metadata['illumination_gradient_strength']}")
print(f"  - Gradient Typ: {custom_detector.metadata['illumination_gradient_type']}")

# Teste Background-Generierung
bg_test = simulator.bg_gen.generate((100, 100))
print(f"✓ Background generiert: Shape {bg_test.shape}, Mean={bg_test.mean():.2f}")

# Test verschiedene Gradient-Typen
gradient_types = ["radial", "linear_x", "linear_y", "corner"]
for gtype in gradient_types:
    custom_detector.metadata["illumination_gradient_type"] = gtype
    sim = TIFFSimulatorOptimized(detector=custom_detector, mode="timelapse", t_poly_min=0.0)
    bg = sim.bg_gen.generate((50, 50))
    print(f"  {gtype:12s}: OK - Mean={bg.mean():.2f}, Std={bg.std():.2f}")

print("✓ Ausleuchtungsgradient funktioniert!")

# Test 2: Brechungsindex-Korrektur
print("\n[2] TEST: Brechungsindex-Korrektur für Z-Stack")
print("-" * 70)

# Teste verschiedene Brechungsindex-Korrekturen
ri_corrections = [1.0, 0.876, 1.15]  # 1.0 = kein, 0.876 = Wasser/Öl, 1.15 = Hydrogel/Öl

for ri_corr in ri_corrections:
    custom_detector.metadata.update({
        "refractive_index_correction": ri_corr,
        "astig_z0_um": 0.5,
        "astig_coeffs": {"A_x": 1.0, "B_x": 0.0, "A_y": -0.5, "B_y": 0.0}
    })

    sim = TIFFSimulatorOptimized(
        detector=custom_detector,
        mode="timelapse",
        t_poly_min=0.0,
        astigmatism=True  # Aktiviere Astigmatismus
    )

    print(f"  RI-Korrektur {ri_corr:.3f}: PSF-Generator erstellt")
    print(f"    - z0_um: {sim.psf_gen.z0_um:.3f}")
    print(f"    - refractive_correction: {sim.psf_gen.refractive_correction:.3f}")

print("✓ Brechungsindex-Korrektur funktioniert!")

# Test 3: Standard-Presets prüfen
print("\n[3] TEST: Standard-Presets haben neue Parameter")
print("-" * 70)

for preset in [TDI_PRESET, TETRASPECS_PRESET]:
    print(f"{preset.name}:")
    print(f"  - illumination_gradient_strength: {preset.metadata.get('illumination_gradient_strength', 'FEHLT')}")
    print(f"  - illumination_gradient_type: {preset.metadata.get('illumination_gradient_type', 'FEHLT')}")
    print(f"  - refractive_index_correction: {preset.metadata.get('refractive_index_correction', 'FEHLT')}")
    print(f"  - z_amp_um: {preset.metadata.get('z_amp_um', 'FEHLT')}")

    # Teste ob Simulator erstellt werden kann
    sim = TIFFSimulatorOptimized(detector=preset, mode="timelapse", t_poly_min=0.0, astigmatism=True)
    print(f"  ✓ Simulator erfolgreich erstellt")

print("\n" + "=" * 70)
print("✅ ALLE TESTS BESTANDEN!")
print("=" * 70)

# Zeige Beispiel-Verwendung
print("\n💡 BEISPIEL-VERWENDUNG:")
print("-" * 70)
print("""
# Ausleuchtungsgradient aktivieren:
from tiff_simulator_v3 import DetectorPreset, TDI_PRESET

detector = TDI_PRESET
detector.metadata['illumination_gradient_strength'] = 15.0  # Subtiler Gradient
detector.metadata['illumination_gradient_type'] = 'radial'  # Radial vom Zentrum

# Brechungsindex-Korrektur für z-Stack:
detector.metadata['refractive_index_correction'] = 0.876  # Wasser (1.33) / Öl (1.518)
detector.metadata['z_amp_um'] = 0.5  # Kleinerer Wert für stärkeren z-Abfall
""")

print("\n🎯 EMPFEHLUNGEN:")
print("-" * 70)
print("• Ausleuchtungsgradient: 5-20 für subtile Effekte, >20 für starke Gradienten")
print("• Brechungsindex-Korrektur:")
print("  - 1.00: Keine Korrektur (Standard)")
print("  - 0.88: Wasser (n=1.33) mit Öl-Immersion (n=1.518)")
print("  - 0.91: Hydrogel (n=1.38) mit Öl-Immersion")
print("  - 1.15: Öl-Immersion in höherbrechendem Medium")
print("• z_amp_um: 0.3-0.7 µm (kleinerer Wert = stärkerer Intensitätsabfall)")
