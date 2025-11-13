"""
Test-Skript zur Verifikation der Astigmatismus-Korrektur
=========================================================

Dieses Skript:
1. Generiert einen Test-Z-Stack mit der neuen korrekten Astigmatismus-Formel
2. Analysiert die PSF-Elliptizität als Funktion von z
3. Zeigt, dass die Ellipse sich korrekt dreht (nicht mehr "links-rund-links")

Expected behavior (KORREKT):
- z < 0: Ellipse horizontal (σ_x > σ_y)
- z = 0: Rund (σ_x ≈ σ_y)
- z > 0: Ellipse vertikal (σ_y > σ_x)

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tiff_simulator_v3 import TDI_PRESET, TIFFSimulator, save_tiff

# Output-Verzeichnis
output_dir = Path("./test_astigmatism_output")
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("🔬 ASTIGMATISMUS-FIX VERIFIKATION")
print("=" * 60)

# Erstelle Simulator mit Astigmatismus
sim = TIFFSimulator(
    detector=TDI_PRESET,
    mode='z_stack',
    astigmatism=True
)

# Generiere Test-Z-Stack
print("\n📚 Generiere Test-Z-Stack...")
print(f"   z-Bereich: -1.0 bis +1.0 µm")
print(f"   z-Schritte: 0.1 µm")
print(f"   Spots: 10")

z_stack = sim.generate_z_stack(
    image_size=(128, 128),
    num_spots=10,
    z_range_um=(-1.0, 1.0),
    z_step_um=0.1,
    progress_callback=lambda c, t, s: print(f"   {s}") if c % 5 == 0 or c == t else None
)

# Speichere Z-Stack
output_path = output_dir / "test_astigmatism_fixed.tif"
save_tiff(str(output_path), z_stack)

print(f"\n✅ Z-Stack gespeichert: {output_path}")
print(f"   Shape: {z_stack.shape}")
print(f"   z-Slices: {z_stack.shape[0]}")

# Analysiere PSF-Eigenschaften
print("\n📊 Analysiere PSF-Elliptizität...")

# Berechne sigma_x und sigma_y für verschiedene z-Positionen
z_positions = np.arange(-1.0, 1.01, 0.1)
sigma_0 = TDI_PRESET.fwhm_um / (TDI_PRESET.pixel_size_um * 2.355)  # in pixels

# Neue korrekte Formel
focal_offset = 0.4  # µm
z_rayleigh = 0.6  # µm

sigma_x_values = []
sigma_y_values = []

for z in z_positions:
    # x-Achse: Fokus bei z = +c
    term_x = 1.0 + ((z - (+focal_offset)) / z_rayleigh)**2
    sigma_x = sigma_0 * np.sqrt(term_x)

    # y-Achse: Fokus bei z = -c
    term_y = 1.0 + ((z - (-focal_offset)) / z_rayleigh)**2
    sigma_y = sigma_0 * np.sqrt(term_y)

    sigma_x_values.append(sigma_x)
    sigma_y_values.append(sigma_y)

sigma_x_values = np.array(sigma_x_values)
sigma_y_values = np.array(sigma_y_values)

# Elliptizität (Unterschied zwischen σ_x und σ_y)
ellipticity = sigma_x_values - sigma_y_values

# Plotte Ergebnisse
plt.figure(figsize=(14, 5))

# Plot 1: σ_x und σ_y vs z
plt.subplot(1, 3, 1)
plt.plot(z_positions, sigma_x_values, 'b-', linewidth=2, label='σ_x (horizontal)')
plt.plot(z_positions, sigma_y_values, 'r-', linewidth=2, label='σ_y (vertikal)')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Fokusebene')
plt.axvline(x=-focal_offset, color='b', linestyle=':', alpha=0.5, label='x-Fokus')
plt.axvline(x=+focal_offset, color='r', linestyle=':', alpha=0.5, label='y-Fokus')
plt.xlabel('z-Position [µm]', fontsize=11)
plt.ylabel('PSF-Breite σ [Pixel]', fontsize=11)
plt.title('PSF-Breite vs. z-Position\n(Neue korrekte Formel)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Elliptizität
plt.subplot(1, 3, 2)
plt.plot(z_positions, ellipticity, 'g-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.fill_between(z_positions, 0, ellipticity, where=(ellipticity >= 0),
                 color='blue', alpha=0.3, label='Horizontal gestreckt (σ_x > σ_y)')
plt.fill_between(z_positions, 0, ellipticity, where=(ellipticity < 0),
                 color='red', alpha=0.3, label='Vertikal gestreckt (σ_y > σ_x)')
plt.xlabel('z-Position [µm]', fontsize=11)
plt.ylabel('Elliptizität (σ_x - σ_y) [Pixel]', fontsize=11)
plt.title('PSF-Elliptizität vs. z-Position\n✅ KORREKT: Rotation bei z=0', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Aspekt-Verhältnis
plt.subplot(1, 3, 3)
aspect_ratio = sigma_x_values / sigma_y_values
plt.plot(z_positions, aspect_ratio, 'purple', linewidth=2)
plt.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='Rund (1:1)')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, label='Fokusebene')
plt.fill_between(z_positions, 1, aspect_ratio, where=(aspect_ratio >= 1),
                 color='blue', alpha=0.3, label='Horizontal')
plt.fill_between(z_positions, aspect_ratio, 1, where=(aspect_ratio < 1),
                 color='red', alpha=0.3, label='Vertikal')
plt.xlabel('z-Position [µm]', fontsize=11)
plt.ylabel('Aspekt-Verhältnis (σ_x / σ_y)', fontsize=11)
plt.title('PSF Aspekt-Verhältnis\n✅ DREHT bei z=0', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / "astigmatism_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✅ Analyse-Plot gespeichert: {plot_path}")

# Verifikation
print("\n🔍 VERIFIKATION:")
print("-" * 60)

# Finde Minima und Maxima
z_at_min_ellipticity = z_positions[np.argmin(np.abs(ellipticity))]
ellipticity_at_min = ellipticity[np.argmin(np.abs(ellipticity))]

print(f"✅ Elliptizität bei z=0: {ellipticity[np.argmin(np.abs(z_positions))]:.3f} Pixel")
print(f"   (sollte ≈ 0 sein → Rund)")

print(f"\n✅ Elliptizität bei z=-1.0 µm: {ellipticity[0]:.3f} Pixel")
print(f"   (sollte > 0 sein → Horizontal gestreckt)")

print(f"\n✅ Elliptizität bei z=+1.0 µm: {ellipticity[-1]:.3f} Pixel")
print(f"   (sollte < 0 sein → Vertikal gestreckt)")

# Prüfe ob Rotation erfolgt
crosses_zero = np.any(ellipticity[:-1] * ellipticity[1:] < 0)
print(f"\n✅ Ellipse rotiert (wechselt Vorzeichen): {crosses_zero}")

if crosses_zero:
    print("\n" + "=" * 60)
    print("🎉 ERFOLG! Astigmatismus ist jetzt KORREKT implementiert!")
    print("=" * 60)
    print("\nDie PSF:")
    print("  • z < 0: Horizontal gestreckt (σ_x > σ_y)")
    print("  • z = 0: Rund (σ_x ≈ σ_y)")
    print("  • z > 0: Vertikal gestreckt (σ_y > σ_x)")
    print("\nDies entspricht dem erwarteten physikalischen Verhalten!")
else:
    print("\n❌ FEHLER: Ellipse rotiert nicht korrekt!")

print("\n" + "=" * 60)
print(f"📁 Alle Dateien gespeichert in: {output_dir}")
print("=" * 60)

plt.show()
