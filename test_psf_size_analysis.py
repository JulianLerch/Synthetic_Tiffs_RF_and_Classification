"""
Test: PSF-Größe über z-Range für verschiedene Astigmatismus-Parameter
======================================================================

Testet ob die PSF-Größe vernünftig bleibt oder explodiert.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("./test_zstack_calibration_output")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("🔬 PSF-GRÖßEN ANALYSE (Sprung-Problem)")
print("=" * 80)

# Parameter-Sets zum Testen
param_sets = [
    {"name": "AKTUELL (zu stark?)", "c": 0.7, "z_R": 0.3, "color": "red"},
    {"name": "MODERAT", "c": 0.5, "z_R": 0.4, "color": "green"},
    {"name": "SANFT", "c": 0.4, "z_R": 0.5, "color": "blue"},
]

z_positions = np.arange(-0.6, 0.61, 0.02)
sigma_0 = 1.5  # px (baseline PSF width bei z=0, in Fokus)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: σ_x und σ_y über z (alle Parameter-Sets)
ax = axes[0, 0]
for params in param_sets:
    c = params['c']
    z_R = params['z_R']

    term_x = 1.0 + ((z_positions - c) / z_R) ** 2
    sigma_x = sigma_0 * np.sqrt(term_x)

    term_y = 1.0 + ((z_positions + c) / z_R) ** 2
    sigma_y = sigma_0 * np.sqrt(term_y)

    ax.plot(z_positions, sigma_x, '--', linewidth=2, color=params['color'],
            alpha=0.7, label=f"{params['name']} σ_x")
    ax.plot(z_positions, sigma_y, '-', linewidth=2, color=params['color'],
            label=f"{params['name']} σ_y")

ax.axhline(y=sigma_0, linestyle=':', color='gray', alpha=0.5, label='σ_0')
ax.axvline(x=0, linestyle='--', color='k', alpha=0.3)
ax.set_xlabel('z-Position [µm]', fontsize=12)
ax.set_ylabel('PSF Width σ [px]', fontsize=12)
ax.set_title('PSF-Breite über z\n(Je flacher, desto besser!)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 8])

# Plot 2: Maximale PSF-Breite bei ±0.6µm
ax = axes[0, 1]
max_widths = []
labels = []
colors_plot = []

for params in param_sets:
    c = params['c']
    z_R = params['z_R']

    # Bei z = -0.6µm (worst case für σ_y)
    term_y_neg = 1.0 + ((-0.6 + c) / z_R) ** 2
    sigma_y_neg = sigma_0 * np.sqrt(term_y_neg)

    # Bei z = +0.6µm (worst case für σ_y)
    term_y_pos = 1.0 + ((0.6 + c) / z_R) ** 2
    sigma_y_pos = sigma_0 * np.sqrt(term_y_pos)

    max_sigma = max(sigma_y_neg, sigma_y_pos)
    max_widths.append(max_sigma)
    labels.append(params['name'])
    colors_plot.append(params['color'])

    print(f"\n{params['name']} (c={c}, z_R={z_R}):")
    print(f"  Max σ bei ±0.6µm: {max_sigma:.2f}px ({max_sigma/sigma_0:.2f}x größer)")
    if max_sigma > 5:
        print(f"  ⚠️ ZU GROSS! PSF wird {max_sigma/sigma_0:.1f}x größer als in Fokus!")
    elif max_sigma > 3.5:
        print(f"  ⚠️ GRENZWERTIG: PSF wird {max_sigma/sigma_0:.1f}x größer")
    else:
        print(f"  ✅ OK: PSF bleibt moderat ({max_sigma/sigma_0:.1f}x größer)")

bars = ax.bar(labels, max_widths, color=colors_plot, alpha=0.7, edgecolor='black')
ax.axhline(y=3.5, linestyle='--', color='orange', label='Grenzwert (3.5px)')
ax.axhline(y=5.0, linestyle='--', color='red', label='Zu groß (5px)')
ax.set_ylabel('Maximale PSF-Breite [px]', fontsize=12)
ax.set_title('Maximale PSF-Größe bei z=±0.6µm\n(soll < 3.5px sein!)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Elliptizität über z
ax = axes[1, 0]
for params in param_sets:
    c = params['c']
    z_R = params['z_R']

    term_x = 1.0 + ((z_positions - c) / z_R) ** 2
    sigma_x = sigma_0 * np.sqrt(term_x)

    term_y = 1.0 + ((z_positions + c) / z_R) ** 2
    sigma_y = sigma_0 * np.sqrt(term_y)

    ellipticity = sigma_x - sigma_y

    ax.plot(z_positions, ellipticity, linewidth=2, color=params['color'],
            label=params['name'])

ax.axhline(y=0, linestyle='--', color='k', alpha=0.3)
ax.axvline(x=0, linestyle='--', color='k', alpha=0.3)
ax.set_xlabel('z-Position [µm]', fontsize=12)
ax.set_ylabel('Elliptizität (σ_x - σ_y) [px]', fontsize=12)
ax.set_title('Elliptizität über z\n(Steigung wichtig für z-Präzision)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Kalibrierungskurven-Steigung
ax = axes[1, 1]
slopes = []
colors_plot2 = []

for params in param_sets:
    c = params['c']
    z_R = params['z_R']

    term_x = 1.0 + ((z_positions - c) / z_R) ** 2
    sigma_x = sigma_0 * np.sqrt(term_x)

    term_y = 1.0 + ((z_positions + c) / z_R) ** 2
    sigma_y = sigma_0 * np.sqrt(term_y)

    ellipticity = sigma_x - sigma_y

    # Steigung im Messbereich ±0.5µm
    mask = np.abs(z_positions) <= 0.5
    slope, _ = np.polyfit(z_positions[mask], ellipticity[mask], 1)
    slopes.append(abs(slope))
    colors_plot2.append(params['color'])

    print(f"\n{params['name']}:")
    print(f"  Steigung: {abs(slope):.2f} px/µm")
    if abs(slope) < 3.0:
        print(f"  ⚠️ ZU FLACH! Schlechte z-Präzision")
    elif abs(slope) < 5.0:
        print(f"  ⚠️ GRENZWERTIG")
    else:
        print(f"  ✅ GUT! Steile Kalibrierungskurve")

bars = ax.bar(labels, slopes, color=colors_plot2, alpha=0.7, edgecolor='black')
ax.axhline(y=5.0, linestyle='--', color='green', label='Minimum (5 px/µm)')
ax.axhline(y=3.0, linestyle='--', color='orange', label='Zu flach (3 px/µm)')
ax.set_ylabel('Steigung [px/µm]', fontsize=12)
ax.set_title('Kalibrierungskurven-Steigung\n(höher = bessere z-Präzision)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = output_dir / "psf_size_analysis.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot gespeichert: {plot_path}")

print("\n" + "=" * 80)
print("📊 ZUSAMMENFASSUNG & EMPFEHLUNG")
print("=" * 80)

# Finde optimalen Parameter-Set
best_idx = None
for idx, params in enumerate(param_sets):
    c = params['c']
    z_R = params['z_R']

    # Berechne Max-Sigma
    term_y_pos = 1.0 + ((0.6 + c) / z_R) ** 2
    max_sigma = sigma_0 * np.sqrt(term_y_pos)

    # Berechne Steigung
    term_x = 1.0 + ((z_positions - c) / z_R) ** 2
    sigma_x = sigma_0 * np.sqrt(term_x)
    term_y = 1.0 + ((z_positions + c) / z_R) ** 2
    sigma_y = sigma_0 * np.sqrt(term_y)
    ellipticity = sigma_x - sigma_y
    mask = np.abs(z_positions) <= 0.5
    slope, _ = np.polyfit(z_positions[mask], ellipticity[mask], 1)

    # Optimum: Steigung > 5 UND max_sigma < 3.5
    if abs(slope) >= 5.0 and max_sigma <= 3.5:
        best_idx = idx

if best_idx is not None:
    best = param_sets[best_idx]
    print(f"\n✅ EMPFEHLUNG: {best['name']}")
    print(f"   • focal_offset: {best['c']} µm")
    print(f"   • z_rayleigh:   {best['z_R']} µm")
    print(f"   • Gute Balance: Steile Kurve + moderate PSF-Größe")
else:
    # Fallback: Wähle den mit bester Balance
    print("\n⚠️ Kein perfekter Parameter-Set gefunden!")
    print("   Wähle besten Kompromiss...")
    best = param_sets[1]  # MODERAT ist meist gut
    print(f"\n✅ EMPFEHLUNG: {best['name']}")
    print(f"   • focal_offset: {best['c']} µm")
    print(f"   • z_rayleigh:   {best['z_R']} µm")

print("\n" + "=" * 80)

plt.show()
