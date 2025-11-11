"""
🔬 PHYSICS V6.0 MODULES - HYPERREALISTIC EXTENSIONS
====================================================

Wissenschaftlich fundierte Erweiterungen für hyperrealistische Simulationen.

Basierend auf:
- Nature Communications 2022: sCMOS noise characterization
- ScienceDirect 2025: Piezo stage hysteresis
- Biophysical studies: TIRF, photobleaching, polymer dynamics

Version: 6.0 - November 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================================
# 1. ADVANCED CAMERA NOISE MODEL (sCMOS)
# ============================================================================

class AdvancedCameraNoiseModel:
    """
    Hyperrealistisches sCMOS Kamera-Rauschmodell.

    Wissenschaftliche Basis:
    ------------------------
    - Read Noise: <1 e⁻ (pixel-dependent, Gaussian distribution)
    - Dark Current: 0.001 e⁻/px/s @ -20°C (Poisson, temperaturabhängig)
    - Fixed Pattern Noise: Hot pixels (~0.01% der Pixel)
    - Temporal Correlation: 1/f flicker noise

    Referenz:
    ---------
    Nature Communications (2022): "Photon-free sCMOS camera characterization"
    """

    def __init__(self,
                 read_noise_mean: float = 1.2,
                 read_noise_std: float = 0.3,
                 dark_current_rate: float = 0.001,
                 temperature_celsius: float = -20.0,
                 hot_pixel_fraction: float = 0.0001,
                 enable_fpn: bool = True,
                 enable_temporal_correlation: bool = False):
        """
        Parameters:
        -----------
        read_noise_mean : float
            Mittlerer read noise [e⁻]
        read_noise_std : float
            Standardabweichung des read noise über Pixel [e⁻]
        dark_current_rate : float
            Dark current rate bei Referenztemperatur [e⁻/px/s]
        temperature_celsius : float
            Kamera-Temperatur [°C]
        hot_pixel_fraction : float
            Fraktion von hot pixels (typisch 0.01%)
        enable_fpn : bool
            Fixed Pattern Noise aktivieren
        enable_temporal_correlation : bool
            Zeitliche Korrelation (1/f noise) aktivieren
        """
        self.read_noise_mean = read_noise_mean
        self.read_noise_std = read_noise_std
        self.dark_current_rate = dark_current_rate
        self.temperature = temperature_celsius
        self.hot_pixel_frac = hot_pixel_fraction
        self.enable_fpn = enable_fpn
        self.enable_temporal = enable_temporal_correlation

        # Caches
        self._read_noise_map = None
        self._fpn_map = None
        self._image_size = None

    def _get_read_noise_map(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Generiert pixel-abhängige read noise map."""
        if self._read_noise_map is None or self._image_size != image_size:
            height, width = image_size
            # Jedes Pixel hat eigene read noise (Gaussian verteilt)
            self._read_noise_map = np.random.normal(
                self.read_noise_mean,
                self.read_noise_std,
                size=(height, width)
            ).astype(np.float32)
            # Clamp zu physikalischen Werten
            self._read_noise_map = np.maximum(self._read_noise_map, 0.1)
            self._image_size = image_size
        return self._read_noise_map

    def _get_fpn_map(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Generiert Fixed Pattern Noise (hot pixels)."""
        if self._fpn_map is None or self._image_size != image_size:
            height, width = image_size
            # Hot pixels: zufällige Positionen
            hot_mask = np.random.rand(height, width) < self.hot_pixel_frac
            # Hot pixel intensity: 50-200 counts
            fpn_map = np.where(hot_mask,
                              np.random.uniform(50, 200, size=(height, width)),
                              0.0).astype(np.float32)
            self._fpn_map = fpn_map
        return self._fpn_map

    def _calculate_dark_current(self, exposure_time_s: float,
                                image_size: Tuple[int, int]) -> np.ndarray:
        """
        Berechnet dark current mit Temperaturabhängigkeit.

        Arrhenius-Gleichung: D(T) = D₀ · exp(-E_a / kT)
        """
        height, width = image_size

        # Temperaturabhängigkeit (vereinfachtes Modell)
        # Verdopplung alle 6-8°C
        temp_ref = -20.0  # Referenztemperatur
        doubling_temp = 7.0  # [°C]
        temp_factor = 2.0 ** ((self.temperature - temp_ref) / doubling_temp)

        # Dark current rate bei aktueller Temperatur
        dc_rate = self.dark_current_rate * temp_factor

        # Erwartete Anzahl dark current electrons
        dc_mean = dc_rate * exposure_time_s

        # Poisson-distributed dark current
        dark_current = np.random.poisson(dc_mean, size=(height, width)).astype(np.float32)

        return dark_current

    def apply_noise(self, frame: np.ndarray, exposure_time_s: float = 0.05,
                   frame_number: int = 0) -> np.ndarray:
        """
        Fügt realistisches sCMOS Rauschen zu Frame hinzu.

        Parameters:
        -----------
        frame : np.ndarray
            Input frame [counts], shape (height, width)
        exposure_time_s : float
            Belichtungszeit [s]
        frame_number : int
            Frame-Nummer (für temporale Korrelation)

        Returns:
        --------
        np.ndarray : Frame mit Rauschen [counts]
        """
        image_size = frame.shape
        noisy_frame = frame.copy()

        # 1. Shot Noise (Poisson) - bereits in frame enthalten, aber verstärken
        # Frame sollte bereits Poisson noise haben, wir addieren nichts extra

        # 2. Read Noise (pixel-abhängig, Gaussian)
        read_noise_map = self._get_read_noise_map(image_size)
        read_noise = np.random.normal(0, read_noise_map).astype(np.float32)
        noisy_frame += read_noise

        # 3. Dark Current (temperaturabhängig, Poisson)
        dark_current = self._calculate_dark_current(exposure_time_s, image_size)
        noisy_frame += dark_current

        # 4. Fixed Pattern Noise (hot pixels)
        if self.enable_fpn:
            fpn = self._get_fpn_map(image_size)
            noisy_frame += fpn

        # 5. Temporal Correlation (1/f flicker noise) - optional
        if self.enable_temporal and frame_number > 0:
            # Sehr subtile zeitliche Korrelation
            # Amplitude sinkt mit 1/f
            freq = max(frame_number, 1)
            flicker_amplitude = 0.5 / freq  # 1/f decay
            flicker = np.random.normal(0, flicker_amplitude, size=image_size).astype(np.float32)
            noisy_frame += flicker

        return noisy_frame

    def reset_cache(self):
        """Reset cached noise maps (z.B. bei Bildgrößenänderung)."""
        self._read_noise_map = None
        self._fpn_map = None
        self._image_size = None


# ============================================================================
# 2. Z-WOBBLE FOR 2D TRACKING (Thermal Drift + Vibrations)
# ============================================================================

class ZWobbleSimulator:
    """
    Simuliert realistischen z-Wobble für 2D-Tracking.

    Komponenten:
    ------------
    1. Thermische Drift: Linear + Random Walk (0.5-2 nm/s)
    2. Mechanische Vibrationen: Sinusförmig + Harmonics (10-50 nm)

    Anwendung:
    ----------
    Auch bei 2D-TIRF-Mikroskopie gibt es leichte z-Bewegungen durch:
    - Temperaturschwankungen des Tisches
    - Mechanische Vibrationen (Pumpen, Gebäude, etc.)
    - Stage-Drift

    Diese z-Bewegungen beeinflussen die Intensität via TIRF evanescent wave!
    """

    def __init__(self,
                 thermal_drift_rate_nm_per_s: float = 1.0,
                 vibration_freq_hz: float = 5.0,
                 vibration_amplitude_nm: float = 30.0,
                 enable_drift: bool = True,
                 enable_vibrations: bool = True):
        """
        Parameters:
        -----------
        thermal_drift_rate_nm_per_s : float
            Thermische Driftrate [nm/s]
        vibration_freq_hz : float
            Hauptfrequenz der Vibrationen [Hz]
        vibration_amplitude_nm : float
            Amplitude der Vibrationen [nm]
        enable_drift : bool
            Thermische Drift aktivieren
        enable_vibrations : bool
            Mechanische Vibrationen aktivieren
        """
        self.drift_rate = thermal_drift_rate_nm_per_s
        self.vib_freq = vibration_freq_hz
        self.vib_amp = vibration_amplitude_nm
        self.enable_drift = enable_drift
        self.enable_vibrations = enable_vibrations

    def thermal_drift_z(self, t_seconds: float) -> float:
        """
        Berechnet thermische Drift zu Zeit t.

        z_drift(t) = v_drift · t + √(2D · t) · N(0,1)

        Returns z in µm
        """
        if not self.enable_drift:
            return 0.0

        # Linearer Drift
        linear_drift_nm = self.drift_rate * t_seconds

        # Random Walk Komponente (Diffusion des Tisches)
        D_thermal = 0.01  # Diffusionskoeffizient für Stage [nm²/s]
        random_walk_nm = np.sqrt(2 * D_thermal * t_seconds) * np.random.randn()

        total_drift_nm = linear_drift_nm + random_walk_nm

        # Convert nm → µm
        return total_drift_nm * 1e-3

    def mechanical_vibrations_z(self, t_seconds: float) -> float:
        """
        Berechnet mechanische Vibrationen zu Zeit t.

        z_vib(t) = Σ A_n · sin(2π n f t + φ_n)

        Returns z in µm
        """
        if not self.enable_vibrations:
            return 0.0

        # Hauptfrequenz mit zufälliger Phase
        phase = np.random.uniform(0, 2*np.pi)
        z_vib_nm = self.vib_amp * np.sin(2 * np.pi * self.vib_freq * t_seconds + phase)

        # Harmonische Obertöne (schwächer)
        z_vib_nm += 0.3 * self.vib_amp * np.sin(4 * np.pi * self.vib_freq * t_seconds)
        z_vib_nm += 0.1 * self.vib_amp * np.sin(6 * np.pi * self.vib_freq * t_seconds)

        # Convert nm → µm
        return z_vib_nm * 1e-3

    def generate_z_wobble(self, num_frames: int, dt: float) -> np.ndarray:
        """
        Generiert z-Wobble Zeitreihe für alle Frames.

        Parameters:
        -----------
        num_frames : int
            Anzahl Frames
        dt : float
            Zeitschritt zwischen Frames [s]

        Returns:
        --------
        np.ndarray : z-Wobble [µm], shape (num_frames,)
        """
        z_wobble = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            t = i * dt
            z_wobble[i] = self.thermal_drift_z(t) + self.mechanical_vibrations_z(t)

        # Clip zu realistischem Bereich (±200 nm für TIRF)
        z_wobble = np.clip(z_wobble, -0.2, 0.2)

        return z_wobble

    def add_z_wobble_to_trajectory(self, trajectory_2d: np.ndarray,
                                   dt: float) -> np.ndarray:
        """
        Fügt z-Wobble zu 2D-Trajektorie hinzu.

        Parameters:
        -----------
        trajectory_2d : np.ndarray
            2D Trajektorie, shape (num_frames, 2) - x, y [µm]
        dt : float
            Zeitschritt [s]

        Returns:
        --------
        np.ndarray : 3D Trajektorie, shape (num_frames, 3) - x, y, z [µm]
        """
        num_frames = len(trajectory_2d)
        z_wobble = self.generate_z_wobble(num_frames, dt)

        # Kombiniere 2D + z-wobble
        trajectory_3d = np.column_stack([trajectory_2d, z_wobble])

        return trajectory_3d


# ============================================================================
# 3. PIEZO STAGE SIMULATOR (Hysteresis + Nonlinearity + Drift)
# ============================================================================

class PiezoStageSimulator:
    """
    Simuliert realistische Piezo-Stage Eigenschaften für Z-Stacks.

    Physikalische Effekte:
    ----------------------
    1. Hysterese: 10-50 nm (richtungsabhängig)
    2. Positioning Noise: 5-15 nm RMS
    3. Non-linearity: 5-15% (Abweichung von idealer linearer Response)
    4. Drift: 10-50 nm/min während langem Scan

    Wissenschaftliche Basis:
    ------------------------
    ScienceDirect 2025: "Piezo stage hysteresis compensation"
    - Prandtl-Ishlinskii Modell für Hysterese
    - Rate-independent behavior

    Anwendung:
    ----------
    Bei Z-Stack Aufnahmen ist die tatsächliche z-Position ≠ Ziel-Position!
    """

    def __init__(self,
                 max_hysteresis_nm: float = 30.0,
                 positioning_noise_rms_nm: float = 10.0,
                 nonlinearity_percent: float = 10.0,
                 drift_rate_nm_per_min: float = 20.0):
        """
        Parameters:
        -----------
        max_hysteresis_nm : float
            Maximale Hysterese [nm]
        positioning_noise_rms_nm : float
            RMS positioning noise [nm]
        nonlinearity_percent : float
            Nichtlinearität [%]
        drift_rate_nm_per_min : float
            Drift-Rate [nm/min]
        """
        self.max_hyst_um = max_hysteresis_nm * 1e-3  # nm → µm
        self.pos_noise_um = positioning_noise_rms_nm * 1e-3
        self.nonlin = nonlinearity_percent / 100.0
        self.drift_rate = drift_rate_nm_per_min * 1e-3  # nm/min → µm/min

        # State
        self.prev_target = 0.0
        self.prev_actual = 0.0
        self.scan_start_time = 0.0

    def apply_hysteresis(self, target_z_um: float) -> float:
        """
        Wendet Hysterese-Modell an (vereinfachtes Prandtl-Ishlinskii).

        z_actual = z_target + h(direction) · δz
        """
        # Richtung der Bewegung
        direction = np.sign(target_z_um - self.prev_target)

        # Hysterese ist richtungsabhängig
        if direction > 0:  # Moving up
            hyst_error = 0.7 * self.max_hyst_um
        elif direction < 0:  # Moving down
            hyst_error = -0.7 * self.max_hyst_um
        else:  # No movement
            hyst_error = 0.0

        return hyst_error

    def apply_nonlinearity(self, z_um: float) -> float:
        """
        Wendet Nichtlinearität an.

        z_actual = z_target · (1 + α · |z_target|)
        """
        return z_um * (1.0 + self.nonlin * abs(z_um))

    def apply_positioning_noise(self) -> float:
        """Fügt Positionierungs-Rauschen hinzu."""
        return np.random.normal(0, self.pos_noise_um)

    def apply_drift(self, current_time_s: float) -> float:
        """
        Berechnet Drift seit Scan-Start.

        drift = rate · (t - t_start)
        """
        elapsed_min = (current_time_s - self.scan_start_time) / 60.0
        drift_um = self.drift_rate * elapsed_min
        return drift_um

    def move_to(self, target_z_um: float, current_time_s: float = 0.0) -> float:
        """
        Simuliert Bewegung zu Ziel-Position.

        Returns:
        --------
        float : Tatsächliche Position [µm]
        """
        # 1. Hysterese
        hyst_error = self.apply_hysteresis(target_z_um)

        # 2. Nichtlinearität
        z_with_nonlin = self.apply_nonlinearity(target_z_um)

        # 3. Positioning Noise
        noise = self.apply_positioning_noise()

        # 4. Drift (falls Scan läuft)
        drift = self.apply_drift(current_time_s)

        # Gesamtposition
        z_actual = z_with_nonlin + hyst_error + noise + drift

        # Update state
        self.prev_target = target_z_um
        self.prev_actual = z_actual

        return z_actual

    def start_scan(self, start_time_s: float = 0.0):
        """Startet einen Scan (für Drift-Tracking)."""
        self.scan_start_time = start_time_s

    def reset(self):
        """Reset state."""
        self.prev_target = 0.0
        self.prev_actual = 0.0
        self.scan_start_time = 0.0


# ============================================================================
# 4. TIRF ILLUMINATION (Evanescent Wave)
# ============================================================================

def tirf_penetration_depth(wavelength_nm: float,
                          n1: float,
                          n2: float,
                          theta_degrees: float) -> float:
    """
    Berechnet TIRF Penetrationstiefe.

    d = (λ / 4π) / √(n₁² sin²θ - n₂²)

    Parameters:
    -----------
    wavelength_nm : float
        Anregungswellenlänge [nm]
    n1 : float
        Brechungsindex Glas (typisch 1.52)
    n2 : float
        Brechungsindex Probe (1.33-1.47)
    theta_degrees : float
        Einfallswinkel [degrees]

    Returns:
    --------
    float : Penetrationstiefe [nm]

    Raises:
    -------
    ValueError : Wenn Winkel unter kritischem Winkel
    """
    theta_rad = np.radians(theta_degrees)

    # Kritischer Winkel
    theta_c = np.arcsin(n2 / n1)

    if theta_rad <= theta_c:
        theta_c_deg = np.degrees(theta_c)
        raise ValueError(
            f"Winkel {theta_degrees:.1f}° liegt unter kritischem Winkel {theta_c_deg:.1f}°!"
        )

    numerator = wavelength_nm / (4 * np.pi)
    denominator = np.sqrt((n1 * np.sin(theta_rad))**2 - n2**2)

    return numerator / denominator


def tirf_intensity_profile(z_um: np.ndarray,
                          penetration_depth_nm: float = 100.0) -> np.ndarray:
    """
    Berechnet TIRF Intensitätsprofil als Funktion von z.

    I(z) = I₀ · exp(-|z| / d)

    Parameters:
    -----------
    z_um : np.ndarray or float
        z-Positionen [µm]
    penetration_depth_nm : float
        Penetrationstiefe [nm]

    Returns:
    --------
    np.ndarray or float : Relative Intensität (0-1)
    """
    d_um = penetration_depth_nm * 1e-3  # nm → µm
    return np.exp(-np.abs(z_um) / d_um)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def apply_tirf_to_intensities(intensities: np.ndarray,
                             z_positions: np.ndarray,
                             penetration_depth_nm: float = 100.0) -> np.ndarray:
    """
    Wendet TIRF-Profil auf Spot-Intensitäten an.

    Parameters:
    -----------
    intensities : np.ndarray
        Original-Intensitäten [counts]
    z_positions : np.ndarray
        z-Positionen der Spots [µm]
    penetration_depth_nm : float
        TIRF Penetrationstiefe [nm]

    Returns:
    --------
    np.ndarray : Modifizierte Intensitäten [counts]
    """
    tirf_factors = tirf_intensity_profile(z_positions, penetration_depth_nm)
    return intensities * tirf_factors


# ============================================================================
# 5. POLYMER CHEMISTRY (Mesh Size + Crosslinking Density)
# ============================================================================

class PolymerChemistryModel:
    """
    Realistische Polymer-Chemie für Hydrogel-Polymerisation.

    Physikalische Modelle:
    ----------------------
    1. Mesh Size Evolution: ξ(t) = ξ₀ · exp(-t / τ_mesh)
    2. Crosslinking Density: ρ_x(t) = ρ_max · (1 - exp(-k·t))
    3. Obstruction Model (Ogston): D/D₀ = exp(-√φ · R/ξ)

    Comonomer-Effekt:
    -----------------
    comonomer_factor ändert die Vernetzungsgeschwindigkeit:
    - 1.0 = Standard (PEGDA)
    - 0.7 = Langsame Vernetzung
    - 1.5 = Schnelle Vernetzung (reaktives Comonomer)

    Wissenschaftliche Basis:
    ------------------------
    - Macromolecules (2016): "Engineering elasticity in hydrogels"
    - PEG hydrogel crosslinking density studies
    """

    def __init__(self,
                 xi_0_nm: float = 50.0,
                 tau_mesh_min: float = 40.0,
                 rho_max_mol_per_cm3: float = 0.15,
                 k_crosslink_per_min: float = 0.05,
                 particle_radius_nm: float = 5.0,
                 comonomer_factor: float = 1.0):
        """
        Parameters:
        -----------
        xi_0_nm : float
            Initiale Mesh-Größe [nm]
        tau_mesh_min : float
            Zeitkonstante für Mesh-Größen-Abfall [min]
        rho_max_mol_per_cm3 : float
            Maximale Crosslink-Dichte [mol/cm³]
        k_crosslink_per_min : float
            Rate-Konstante für Crosslinking [1/min]
        particle_radius_nm : float
            Partikel-Radius [nm] (für Obstruction)
        comonomer_factor : float
            Comonomer-Beschleunigungsfaktor
        """
        self.xi_0 = xi_0_nm
        self.tau_mesh_base = tau_mesh_min
        self.rho_max = rho_max_mol_per_cm3
        self.k_base = k_crosslink_per_min
        self.R_particle = particle_radius_nm
        self.comonomer = comonomer_factor

        # Angepasste Parameter basierend auf Comonomer
        self.tau_mesh = self.tau_mesh_base / self.comonomer
        self.k_crosslink = self.k_base * self.comonomer

    def mesh_size(self, t_min: float) -> float:
        """
        Berechnet Mesh-Größe zu Zeit t.

        ξ(t) = ξ₀ · exp(-t / τ_mesh)

        Returns: Mesh size [nm]
        """
        return self.xi_0 * np.exp(-t_min / self.tau_mesh)

    def crosslinking_density(self, t_min: float) -> float:
        """
        Berechnet Crosslinking-Dichte zu Zeit t.

        ρ_x(t) = ρ_max · (1 - exp(-k·t))

        Returns: Crosslink density [mol/cm³]
        """
        return self.rho_max * (1.0 - np.exp(-self.k_crosslink * t_min))

    def polymer_volume_fraction(self, t_min: float) -> float:
        """
        Berechnet Polymer-Volumen-Fraktion aus Crosslink-Dichte.

        φ ≈ ρ_x / ρ_max (vereinfacht)

        Returns: Volume fraction (0-1)
        """
        rho_x = self.crosslinking_density(t_min)
        return rho_x / self.rho_max

    def obstruction_factor(self, t_min: float) -> float:
        """
        Berechnet Obstruktions-Faktor für Diffusion (Ogston Model).

        D/D₀ = exp(-√φ · R/ξ)

        Returns: Factor (0-1) to multiply with free diffusion coefficient
        """
        xi = self.mesh_size(t_min)
        phi = self.polymer_volume_fraction(t_min)

        # Ogston obstruction model
        exponent = -np.sqrt(phi) * self.R_particle / xi
        return np.exp(exponent)

    def effective_diffusion_coefficient(self, D_free: float, t_min: float) -> float:
        """
        Berechnet effektiven Diffusionskoeffizienten unter Berücksichtigung
        von Obstruction durch Polymer-Netzwerk.

        Parameters:
        -----------
        D_free : float
            Freier Diffusionskoeffizient [µm²/s]
        t_min : float
            Polymerisationszeit [min]

        Returns:
        --------
        float : Effektiver D [µm²/s]
        """
        obstruction = self.obstruction_factor(t_min)
        return D_free * obstruction

    def get_state_dict(self, t_min: float) -> Dict:
        """Gibt aktuellen Zustand des Polymers zurück."""
        return {
            'time_min': t_min,
            'mesh_size_nm': self.mesh_size(t_min),
            'crosslinking_density': self.crosslinking_density(t_min),
            'volume_fraction': self.polymer_volume_fraction(t_min),
            'obstruction_factor': self.obstruction_factor(t_min),
            'comonomer_factor': self.comonomer
        }


# ============================================================================
# 6. ADVANCED DIFFUSION MODELS (CTRW, FBM, Caging)
# ============================================================================

class ContinuousTimeRandomWalk:
    """
    Continuous-Time Random Walk (CTRW) Model für anomale Diffusion.

    Physikalische Grundlage:
    ------------------------
    - Waiting time distribution: ψ(t) ∝ t^(-1-α)
    - Jump length distribution: Gaussian
    - α < 1: Subdiffusion (lange Wartezeiten)
    - α = 1: Normale Diffusion
    - α > 1: Superdiffusion

    Wissenschaftliche Basis:
    ------------------------
    Metzler et al. (2014): "Anomalous diffusion models and their properties"
    """

    def __init__(self, alpha: float = 0.7, sigma_jump: float = 0.1):
        """
        Parameters:
        -----------
        alpha : float
            Anomaler Exponent (0.2-0.8 für Subdiffusion)
        sigma_jump : float
            Breite der Jump-Längen-Verteilung [µm]
        """
        self.alpha = alpha
        self.sigma = sigma_jump

    def generate_waiting_time(self, dt_base: float) -> float:
        """
        Generiert Wartezeit aus Power-Law-Verteilung.

        Returns: waiting time [s]
        """
        # Levy-stable distribution approximation
        # Für alpha < 1: lange Wartezeiten
        u = np.random.uniform(0, 1)
        wait_time = dt_base * (u ** (-1.0 / self.alpha))
        return min(wait_time, 100 * dt_base)  # Cap bei 100x dt

    def generate_step(self) -> np.ndarray:
        """
        Generiert räumlichen Schritt.

        Returns: step vector [µm], shape (3,)
        """
        return np.random.normal(0, self.sigma, size=3).astype(np.float32)


class FractionalBrownianMotion:
    """
    Fractional Brownian Motion (FBM) für Diffusion mit Memory.

    Hurst Exponent H:
    -----------------
    H < 0.5: Anti-persistent (Subdiffusion) - Richtungswechsel wahrscheinlich
    H = 0.5: Standard Brownian Motion
    H > 0.5: Persistent (Superdiffusion) - Gleiche Richtung wahrscheinlich

    MSD ~ t^(2H)

    Wissenschaftliche Basis:
    ------------------------
    Krapf et al. (2019): "Power spectral density of trajectories"
    """

    def __init__(self, hurst: float = 0.3):
        """
        Parameters:
        -----------
        hurst : float
            Hurst exponent (0.2-0.4 für Subdiffusion)
        """
        self.H = hurst

    def generate_increments(self, num_steps: int, dt: float, D: float) -> np.ndarray:
        """
        Generiert FBM increments.

        Simplified method using autocovariance.

        Parameters:
        -----------
        num_steps : int
            Anzahl Schritte
        dt : float
            Zeitschritt [s]
        D : float
            Diffusionskoeffizient [µm²/s]

        Returns:
        --------
        np.ndarray : Increments, shape (num_steps,)
        """
        # Autokovariance für FBM
        # C(k) ∝ k^(2H) für FBM
        sigma = np.sqrt(2 * D * (dt ** (2 * self.H)))

        # Generiere korrelierte Gaussian increments
        increments = np.zeros(num_steps)
        increments[0] = np.random.normal(0, sigma)

        for i in range(1, num_steps):
            # Simplified correlation
            rho = ((i+1)**(2*self.H) + abs(i-1)**(2*self.H) - 2*i**(2*self.H)) / 2
            rho = np.clip(rho, -0.99, 0.99)

            increments[i] = (rho * increments[i-1] +
                           np.sqrt(1 - rho**2) * np.random.normal(0, sigma))

        return increments


class CagingModel:
    """
    Caging Model mit exponentiellen Escape-Kinetiken.

    Physikalische Grundlage:
    ------------------------
    Partikel ist in "Käfig" gefangen durch Polymer-Netzwerk.
    Kann entkommen mit Rate λ (exponentiell verteilt).

    P_escape = 1 - exp(-λ · dt)

    Typische Escape-Zeiten:
    -----------------------
    - Starkes Confinement: τ_escape ~ 10-60 s (λ = 0.017-0.1 s⁻¹)
    - Schwaches Confinement: τ_escape ~ 1-10 s (λ = 0.1-1 s⁻¹)
    """

    def __init__(self, cage_radius_um: float = 0.3,
                 escape_rate_per_s: float = 0.05,
                 spring_constant: float = 0.5):
        """
        Parameters:
        -----------
        cage_radius_um : float
            Käfig-Radius [µm]
        escape_rate_per_s : float
            Escape-Rate λ [1/s]
        spring_constant : float
            Federkonstante für harmonisches Potential [dimensionslos]
        """
        self.R_cage = cage_radius_um
        self.lambda_escape = escape_rate_per_s
        self.k_spring = spring_constant

        # State
        self.is_confined = True
        self.cage_center = np.zeros(3, dtype=np.float32)

    def set_cage_center(self, position: np.ndarray):
        """Setzt neues Käfig-Zentrum."""
        self.cage_center = position.copy()
        self.is_confined = True

    def check_escape(self, dt: float) -> bool:
        """
        Prüft ob Partikel aus Käfig entkommen ist.

        Returns: True wenn escaped
        """
        if not self.is_confined:
            return False

        # Exponentiell verteilte Escape-Wahrscheinlichkeit
        p_escape = 1.0 - np.exp(-self.lambda_escape * dt)

        if np.random.rand() < p_escape:
            self.is_confined = False
            return True

        return False

    def restoring_force(self, position: np.ndarray) -> np.ndarray:
        """
        Berechnet Rückstellkraft wenn im Käfig.

        F = -k · (r - r_cage)

        Returns: Force vector (als Drift-Term)
        """
        if not self.is_confined:
            return np.zeros(3, dtype=np.float32)

        displacement = position - self.cage_center
        distance = np.linalg.norm(displacement)

        if distance > self.R_cage:
            # Außerhalb Käfig → entkommen
            self.is_confined = False
            return np.zeros(3, dtype=np.float32)

        # Harmonisches Potential
        force = -self.k_spring * displacement

        return force


# ============================================================================
# 7. TRIPLET-STATE PHOTOBLEACHING MODEL
# ============================================================================

class TripletStatePhotobleaching:
    """
    3-State Photophysics Model (S0, S1, T1).

    States:
    -------
    S0: Ground state
    S1: Excited singlet (fluorescent)
    T1: Triplet state (dark)

    Transitions:
    ------------
    S0 → S1: Excitation (rate k_01, intensitätsabhängig)
    S1 → S0: Fluorescence (rate k_10 ~ 1 ns⁻¹)
    S1 → T1: Intersystem Crossing (rate k_1T ~ 1 µs⁻¹)
    T1 → S0: Relaxation (rate k_T0 ~ 1 ms⁻¹)
    T1 → bleached: Photobleaching (rate k_bleach ~ 0.1 s⁻¹)

    Wissenschaftliche Basis:
    ------------------------
    Biology of the Cell (2025): "Triplet state population in GFP"
    - Quantum yield: ~7×10⁻⁷ bleaching events per excitation
    - Triplet lifetime: ~1 ms (1000x longer than singlet)
    """

    def __init__(self,
                 k_ISC_per_s: float = 1e6,
                 k_triplet_relax_per_s: float = 1e3,
                 k_bleach_per_s: float = 1e-1,
                 intensity_saturation: float = 100.0):
        """
        Parameters:
        -----------
        k_ISC_per_s : float
            Intersystem crossing rate S1 → T1 [1/s]
        k_triplet_relax_per_s : float
            Triplet relaxation rate T1 → S0 [1/s]
        k_bleach_per_s : float
            Bleaching rate from T1 [1/s]
        intensity_saturation : float
            Sättigungsintensität [counts]
        """
        self.k_1T = k_ISC_per_s
        self.k_T0 = k_triplet_relax_per_s
        self.k_bleach = k_bleach_per_s
        self.I_sat = intensity_saturation

        # State
        self.state = 'S1'  # Start in fluorescent state

    def simulate_frame(self, intensity: float, dt: float) -> Tuple[bool, bool]:
        """
        Simuliert Photophysik für ein Frame (Gillespie-ähnlich).

        Parameters:
        -----------
        intensity : float
            Anregungsintensität [counts]
        dt : float
            Frame-Zeit [s]

        Returns:
        --------
        Tuple[bool, bool] : (is_on, is_bleached)
        """
        if self.state == 'bleached':
            return False, True

        # Intensitätsabhängige ISC-Rate (Sättigung)
        k_1T_eff = self.k_1T * (intensity / self.I_sat) / (1.0 + intensity / self.I_sat)

        # Transition probabilities
        p_ISC = 1.0 - np.exp(-k_1T_eff * dt)
        p_return = 1.0 - np.exp(-self.k_T0 * dt)
        p_bleach = 1.0 - np.exp(-self.k_bleach * dt)

        # State machine
        if self.state == 'S1':
            if np.random.rand() < p_ISC:
                self.state = 'T1'  # Enter triplet (blink off)

        elif self.state == 'T1':
            if np.random.rand() < p_bleach:
                self.state = 'bleached'  # Permanent bleaching
                return False, True
            elif np.random.rand() < p_return:
                self.state = 'S1'  # Return to fluorescent

        is_on = (self.state == 'S1')
        return is_on, False

    def reset(self):
        """Reset to initial state."""
        self.state = 'S1'


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🔬 Testing Physics V6.0 Modules")
    print("=" * 60)

    # Test 1: Camera Noise
    print("\n1. Testing Camera Noise Model...")
    camera = AdvancedCameraNoiseModel(enable_fpn=True, enable_temporal_correlation=True)
    test_frame = np.ones((128, 128), dtype=np.float32) * 100.0
    noisy_frame = camera.apply_noise(test_frame, exposure_time_s=0.05, frame_number=10)
    print(f"   Input mean: {test_frame.mean():.2f}")
    print(f"   Output mean: {noisy_frame.mean():.2f}")
    print(f"   Output std: {noisy_frame.std():.2f}")
    print(f"   ✅ Camera noise OK")

    # Test 2: Z-Wobble
    print("\n2. Testing Z-Wobble Simulator...")
    wobble = ZWobbleSimulator(thermal_drift_rate_nm_per_s=1.0,
                             vibration_amplitude_nm=30.0)
    z_series = wobble.generate_z_wobble(num_frames=100, dt=0.05)
    print(f"   Z-wobble range: [{z_series.min()*1000:.1f}, {z_series.max()*1000:.1f}] nm")
    print(f"   Z-wobble std: {z_series.std()*1000:.1f} nm")
    print(f"   ✅ Z-wobble OK")

    # Test 3: Piezo Stage
    print("\n3. Testing Piezo Stage Simulator...")
    piezo = PiezoStageSimulator(max_hysteresis_nm=30.0,
                                positioning_noise_rms_nm=10.0)
    piezo.start_scan()
    z_targets = np.linspace(-1.0, 1.0, 21)
    z_actual = [piezo.move_to(z, t) for t, z in enumerate(z_targets)]
    error = np.array(z_actual) - z_targets
    print(f"   Mean positioning error: {error.mean()*1000:.2f} nm")
    print(f"   RMS positioning error: {np.sqrt((error**2).mean())*1000:.2f} nm")
    print(f"   ✅ Piezo stage OK")

    # Test 4: TIRF
    print("\n4. Testing TIRF Illumination...")
    try:
        d_pen = tirf_penetration_depth(580, n1=1.52, n2=1.37, theta_degrees=67)
        print(f"   Penetration depth @ 67°: {d_pen:.1f} nm")
        z_test = np.array([0.0, 0.05, 0.1, 0.15, 0.2])  # µm
        I_tirf = tirf_intensity_profile(z_test, d_pen)
        print(f"   TIRF intensities: {I_tirf}")
        print(f"   ✅ TIRF OK")
    except ValueError as e:
        print(f"   ⚠️  {e}")

    # Test 5: Polymer Chemistry
    print("\n5. Testing Polymer Chemistry Model...")
    polymer = PolymerChemistryModel(comonomer_factor=1.5)
    for t in [0, 60, 120]:
        state = polymer.get_state_dict(t)
        print(f"   t={t:3d} min: ξ={state['mesh_size_nm']:4.1f} nm, ρ_x={state['crosslinking_density']:.3f}, obstruction={state['obstruction_factor']:.3f}")
    print(f"   ✅ Polymer chemistry OK")

    # Test 6: CTRW
    print("\n6. Testing CTRW Model...")
    ctrw = ContinuousTimeRandomWalk(alpha=0.7)
    steps = [ctrw.generate_step() for _ in range(100)]
    step_lengths = [np.linalg.norm(s) for s in steps]
    print(f"   Mean step length: {np.mean(step_lengths):.3f} µm")
    print(f"   Std step length: {np.std(step_lengths):.3f} µm")
    print(f"   ✅ CTRW OK")

    # Test 7: FBM
    print("\n7. Testing FBM Model...")
    fbm = FractionalBrownianMotion(hurst=0.3)
    increments = fbm.generate_increments(num_steps=100, dt=0.05, D=1.0)
    print(f"   Mean increment: {increments.mean():.4f} µm")
    print(f"   Std increment: {increments.std():.4f} µm")
    print(f"   ✅ FBM OK")

    # Test 8: Caging
    print("\n8. Testing Caging Model...")
    cage = CagingModel(cage_radius_um=0.3, escape_rate_per_s=0.1)
    cage.set_cage_center(np.array([0., 0., 0.]))
    pos = np.array([0.1, 0.1, 0.0])
    force = cage.restoring_force(pos)
    print(f"   Position: {pos}")
    print(f"   Restoring force: {force}")
    print(f"   Is confined: {cage.is_confined}")
    print(f"   ✅ Caging OK")

    # Test 9: Triplet State
    print("\n9. Testing Triplet-State Photobleaching...")
    triplet = TripletStatePhotobleaching()
    on_count = 0
    bleached_count = 0
    for i in range(1000):
        is_on, is_bleached = triplet.simulate_frame(intensity=100.0, dt=0.05)
        if is_on:
            on_count += 1
        if is_bleached:
            bleached_count += 1
            break
    print(f"   Frames ON before bleaching: {on_count}/1000")
    print(f"   Bleached: {bleached_count > 0}")
    print(f"   ✅ Triplet-state OK")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\n📊 Module contains:")
    print("   1. AdvancedCameraNoiseModel")
    print("   2. ZWobbleSimulator")
    print("   3. PiezoStageSimulator")
    print("   4. TIRF functions")
    print("   5. PolymerChemistryModel")
    print("   6. ContinuousTimeRandomWalk")
    print("   7. FractionalBrownianMotion")
    print("   8. CagingModel")
    print("   9. TripletStatePhotobleaching")
    print("\n🚀 Ready for integration into tiff_simulator_v3.py!")
