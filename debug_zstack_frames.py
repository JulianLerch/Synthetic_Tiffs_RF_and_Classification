"""
Debug: Z-Stack Frame-by-Frame Analyse
=====================================

Simuliert genau das, was der User gemacht hat:
-0.5 bis +0.5 in 0.01 Schritten

Zeigt für jeden Frame:
- z-Position
- σ_x und σ_y
- PSF-Größe

Findet den "Sprung" bei Frame 61→62
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("./test_zstack_calibration_output")
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("🔍 Z-STACK FRAME-BY-FRAME DEBUG (User's Problem)")
print("=" * 80)

# User's parameters
z_start = -0.5
z_end = 0.5
z_step = 0.01

# Current astigmatism parameters
c = 0.5  # focal_offset_um
z_R = 0.4  # z_rayleigh_um
sigma_0 = 1.5  # px (baseline PSF width)

# Generate z-positions (like in Z-stack)
z_positions = np.arange(z_start, z_end + z_step/2, z_step)
n_frames = len(z_positions)

print(f"\nZ-Stack Info:")
print(f"  Range: {z_start} to {z_end} µm")
print(f"  Step: {z_step} µm")
print(f"  Total frames: {n_frames}")
print(f"\nAstigmatism parameters:")
print(f"  focal_offset (c): {c} µm")
print(f"  z_rayleigh (z_R): {z_R} µm")
print(f"  sigma_0: {sigma_0} px")

# Calculate sigma_x and sigma_y for each frame
sigma_x_values = []
sigma_y_values = []
psf_area_values = []

for z in z_positions:
    # EXACT formula from code
    term_x = 1.0 + ((z - c) / z_R) ** 2
    term_y = 1.0 + ((z + c) / z_R) ** 2

    sigma_x = sigma_0 * np.sqrt(term_x)
    sigma_y = sigma_0 * np.sqrt(term_y)

    # PSF area (proportional to sigma_x * sigma_y)
    psf_area = sigma_x * sigma_y

    sigma_x_values.append(sigma_x)
    sigma_y_values.append(sigma_y)
    psf_area_values.append(psf_area)

sigma_x_values = np.array(sigma_x_values)
sigma_y_values = np.array(sigma_y_values)
psf_area_values = np.array(psf_area_values)

# Find the problematic frames
print(f"\n{'='*80}")
print("📊 FRAME-BY-FRAME ANALYSIS AROUND FRAME 60-62")
print("=" * 80)

for frame_idx in range(max(0, 58), min(n_frames, 65)):
    z = z_positions[frame_idx]
    sx = sigma_x_values[frame_idx]
    sy = sigma_y_values[frame_idx]
    area = psf_area_values[frame_idx]

    # Check for explosion
    explosion = ""
    if sx > 3*sigma_0 or sy > 3*sigma_0:
        explosion = " ⚠️ EXPLOSION!"
    elif sx > 2.5*sigma_0 or sy > 2.5*sigma_0:
        explosion = " ⚠️ SEHR GROSS"
    elif frame_idx in [60, 61, 62]:
        explosion = " ← USER'S PROBLEM AREA"

    print(f"Frame {frame_idx:3d}: z={z:+.2f}µm  σ_x={sx:.2f}px  σ_y={sy:.2f}px  Area={area:.1f}{explosion}")

# Find maximum values
max_sigma_x = np.max(sigma_x_values)
max_sigma_y = np.max(sigma_y_values)
max_area = np.max(psf_area_values)

print(f"\n{'='*80}")
print("📈 STATISTIK ÜBER GESAMTEN Z-STACK")
print("=" * 80)
print(f"Minimum σ_x: {np.min(sigma_x_values):.2f} px")
print(f"Maximum σ_x: {max_sigma_x:.2f} px ({max_sigma_x/sigma_0:.2f}x größer als σ_0)")
print(f"Minimum σ_y: {np.min(sigma_y_values):.2f} px")
print(f"Maximum σ_y: {max_sigma_y:.2f} px ({max_sigma_y/sigma_0:.2f}x größer als σ_0)")
print(f"Maximum Area: {max_area:.1f} px² ({max_area/(sigma_0**2):.2f}x größer als σ_0²)")

# Find frame with maximum PSF size
max_frame = np.argmax(psf_area_values)
print(f"\nGRÖßTE PSF bei Frame {max_frame} (z={z_positions[max_frame]:.2f}µm)")
print(f"  σ_x={sigma_x_values[max_frame]:.2f}px, σ_y={sigma_y_values[max_frame]:.2f}px")

# Check at z=0
idx_z0 = np.argmin(np.abs(z_positions))
print(f"\nBei z≈0 (Frame {idx_z0}, z={z_positions[idx_z0]:.2f}µm):")
print(f"  σ_x={sigma_x_values[idx_z0]:.2f}px, σ_y={sigma_y_values[idx_z0]:.2f}px")
print(f"  Erwartung: σ_x ≈ σ_y (rund)")
print(f"  Tatsächlich: Δσ = {abs(sigma_x_values[idx_z0]-sigma_y_values[idx_z0]):.3f}px ✓" if abs(sigma_x_values[idx_z0]-sigma_y_values[idx_z0]) < 0.1 else "  Tatsächlich: NICHT rund! ✗")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: σ_x and σ_y over z
ax = axes[0, 0]
ax.plot(z_positions, sigma_x_values, 'b-', linewidth=2, label='σ_x')
ax.plot(z_positions, sigma_y_values, 'r-', linewidth=2, label='σ_y')
ax.axhline(y=sigma_0, linestyle='--', color='gray', alpha=0.5, label='σ_0')
ax.axvline(x=0, linestyle='--', color='k', alpha=0.3, label='z=0')
# Mark frames 60-62
for frame_idx in [60, 61, 62]:
    if frame_idx < len(z_positions):
        z = z_positions[frame_idx]
        sx = sigma_x_values[frame_idx]
        sy = sigma_y_values[frame_idx]
        ax.plot(z, sx, 'bo', markersize=10)
        ax.plot(z, sy, 'ro', markersize=10)
        ax.text(z, max(sx, sy) + 0.3, f'F{frame_idx}', ha='center', fontsize=9)

ax.set_xlabel('z-Position [µm]', fontsize=12)
ax.set_ylabel('PSF Width [px]', fontsize=12)
ax.set_title('PSF-Breite über Z-Stack\n(Frames 60-62 markiert)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Frame index vs PSF size
ax = axes[0, 1]
frame_indices = np.arange(len(z_positions))
ax.plot(frame_indices, sigma_x_values, 'b-', linewidth=2, label='σ_x')
ax.plot(frame_indices, sigma_y_values, 'r-', linewidth=2, label='σ_y')
# Mark problem area
ax.axvline(x=61, linestyle='--', color='red', alpha=0.5, linewidth=2, label='Frame 61 (User Problem)')
ax.axvspan(60, 62, alpha=0.2, color='red')
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('PSF Width [px]', fontsize=12)
ax.set_title('PSF-Breite vs. Frame Index\n(Problem bei Frame 61?)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: PSF Area
ax = axes[1, 0]
ax.plot(frame_indices, psf_area_values, 'g-', linewidth=2)
ax.axvline(x=61, linestyle='--', color='red', alpha=0.5, linewidth=2, label='Frame 61')
ax.axvspan(60, 62, alpha=0.2, color='red')
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('PSF Area (σ_x × σ_y) [px²]', fontsize=12)
ax.set_title('PSF-Fläche vs. Frame Index\n(Zeigt "Sprünge")', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Difference sigma_y - sigma_x (to show jumps)
ax = axes[1, 1]
diff = sigma_y_values - sigma_x_values
ax.plot(frame_indices, diff, 'purple', linewidth=2)
ax.axhline(y=0, linestyle='--', color='k', alpha=0.3)
ax.axvline(x=61, linestyle='--', color='red', alpha=0.5, linewidth=2, label='Frame 61')
ax.axvspan(60, 62, alpha=0.2, color='red')
ax.set_xlabel('Frame Index', fontsize=12)
ax.set_ylabel('σ_y - σ_x [px]', fontsize=12)
ax.set_title('Elliptizität vs. Frame Index\n(Positiv=vertikal, Negativ=horizontal)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = output_dir / "frame_by_frame_debug.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot gespeichert: {plot_path}")

# Check for sudden jumps
print(f"\n{'='*80}")
print("🔍 SUCHE NACH PLÖTZLICHEN SPRÜNGEN")
print("=" * 80)

# Calculate frame-to-frame changes
delta_area = np.diff(psf_area_values)
jump_threshold = 0.5  # px² (adjust if needed)

jumps_found = False
for i, delta in enumerate(delta_area):
    if abs(delta) > jump_threshold:
        print(f"Frame {i} → {i+1}: PSF Area Änderung = {delta:+.2f} px² (z={z_positions[i]:.2f} → {z_positions[i+1]:.2f}µm)")
        jumps_found = True

if not jumps_found:
    print("Keine plötzlichen Sprünge gefunden!")
    print("\n⚠️ ABER: PSF kann trotzdem zu groß sein (kontinuierlich)!")

print("\n" + "=" * 80)

plt.show()
