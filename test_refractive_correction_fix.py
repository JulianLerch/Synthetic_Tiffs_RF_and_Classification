"""
Test: Brechungsindex-Korrektur Bug-Fix
=======================================

Verifiziert dass die 1/z Explosion bei z>0.1µm behoben ist.
"""

import numpy as np
from tiff_simulator_v3 import calculate_advanced_refractive_correction

print("=" * 80)
print("🔍 BRECHUNGSINDEX-KORREKTUR BUG-FIX TEST")
print("=" * 80)

# User's z-stack parameters
z_positions = np.arange(-0.5, 0.51, 0.01)

# Refractive index parameters (from preset)
n_oil = 1.518
n_glass = 1.523
n_polymer = 1.54
NA = 1.5
d_glass_um = 170.0

print(f"\nParameter:")
print(f"  n_oil: {n_oil}")
print(f"  n_glass: {n_glass}")
print(f"  n_polymer: {n_polymer}")
print(f"  NA: {NA}")
print(f"  d_glass: {d_glass_um}µm")

# Apply correction
z_corrected = calculate_advanced_refractive_correction(
    z_positions, n_oil, n_glass, n_polymer, NA, d_glass_um
)

# Calculate correction factor
correction_factor = z_corrected / np.where(np.abs(z_positions) < 1e-6, 1e-6, z_positions)

print(f"\n{'='*80}")
print("📊 ANALYSE: FRAMES RUND UM FRAME 61 (User's Problem)")
print("=" * 80)

for frame_idx in range(58, 65):
    if frame_idx < len(z_positions):
        z_app = z_positions[frame_idx]
        z_corr = z_corrected[frame_idx]
        factor = correction_factor[frame_idx]

        warning = ""
        if abs(factor) > 5:
            warning = " ⚠️ EXPLOSION!"
        elif abs(factor) > 2:
            warning = " ⚠️ SEHR GROSS"
        elif frame_idx == 61:
            warning = " ← Frame 61 (User Problem)"

        print(f"Frame {frame_idx}: z_app={z_app:+.2f}µm → z_corr={z_corr:+.2f}µm (Faktor: {factor:.2f}x){warning}")

# Check maximum correction factor in z-stack range
max_factor = np.max(np.abs(correction_factor))
print(f"\n{'='*80}")
print("📈 STATISTIK")
print("=" * 80)
print(f"Maximaler Korrekturfaktor: {max_factor:.2f}x")

if max_factor < 2.0:
    print(f"✅ GUT! Korrekturfaktor bleibt moderat (<2x)")
elif max_factor < 5.0:
    print(f"⚠️ GRENZWERTIG: Korrekturfaktor bis {max_factor:.2f}x")
else:
    print(f"❌ PROBLEM: Korrekturfaktor explodiert ({max_factor:.2f}x)!")

# Expected behavior with NEW threshold (2.0µm)
print(f"\n{'='*80}")
print("🎯 ERWARTETES VERHALTEN (mit z_threshold=2.0µm)")
print("=" * 80)
print("Für z in [-0.5, +0.5]µm:")
print("  → f_depth sollte immer 1.0 sein (keine Tiefenkorrektur)")
print("  → z_corrected = z_apparent * f_base * f_na")
print(f"  → Erwarteter Faktor: {n_polymer/n_oil:.3f} * {np.sqrt(n_oil**2 - NA**2) / np.sqrt(n_polymer**2 - NA**2):.3f} = {(n_polymer/n_oil) * (np.sqrt(n_oil**2 - NA**2) / np.sqrt(n_polymer**2 - NA**2)):.3f}")

actual_factor = z_corrected[50] / z_positions[50] if z_positions[50] != 0 else 0  # Frame 50 ist bei z≈0
expected_factor = (n_polymer/n_oil) * (np.sqrt(n_oil**2 - NA**2) / np.sqrt(n_polymer**2 - NA**2))

print(f"\nTatsächlicher Faktor bei z≈0: {actual_factor:.3f}")
print(f"Erwarteter Faktor: {expected_factor:.3f}")

if abs(actual_factor - expected_factor) < 0.01:
    print(f"✅ PERFEKT! Korrekturfaktor stimmt überein!")
else:
    print(f"⚠️ Abweichung: {abs(actual_factor - expected_factor):.3f}")

print(f"\n{'='*80}")
if max_factor < 2.0:
    print("🎉 SUCCESS! Bug ist behoben!")
    print("   Z-Stack sollte jetzt keine PSF-Explosion mehr haben!")
else:
    print("❌ PROBLEM BLEIBT! Weitere Anpassungen nötig.")
print("=" * 80)
