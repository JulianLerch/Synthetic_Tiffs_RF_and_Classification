"""
ðŸ”¬ HYPERREALISTISCHES TIFF-SIMULATIONSSYSTEM V3.0
=================================================

Wissenschaftlich prÃ¤zise Simulation von Single-Molecule Tracking Daten
fÃ¼r hochauflÃ¶sende Fluoreszenzmikroskopie.

Physikalische Grundlagen:
-------------------------
- Point Spread Function (PSF): 2D GauÃŸsche Approximation
- Diffusion: Brownsche Bewegung mit zeitabhÃ¤ngigem D(t)
- Astigmatismus: Elliptische PSF-Deformation als Funktion von z
- Photon Noise: Poisson-Statistik fÃ¼r realistische Bildgebung

Autor: Generiert fÃ¼r Masterthesis
Version: 3.0 - Oktober 2025
Lizenz: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


@dataclass
class DetectorPreset:
    """
    Detektor-Konfiguration mit physikalisch realistischen Parametern.
    
    Attributes:
    -----------
    name : str
        Detektorbezeichnung
    max_intensity : float
        Maximale Photonenzahl pro Spot [counts]
    background_mean : float
        Mittlerer Background [counts]
    background_std : float
        Standardabweichung des Backgrounds [counts]
    pixel_size_um : float
        Physikalische PixelgrÃ¶ÃŸe [Âµm]
    fwhm_um : float
        Full Width at Half Maximum der PSF [Âµm]
    """
    name: str
    max_intensity: float
    background_mean: float
    background_std: float
    pixel_size_um: float
    fwhm_um: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# DETEKTOR PRESETS - Experimentell validiert
# ============================================================================

TDI_PRESET = DetectorPreset(
    name="TDI-G0",
    max_intensity=260.0,
    background_mean=100.0,
    background_std=15.0,
    pixel_size_um=0.108,
    fwhm_um=0.40,
    metadata={
        "detector_type": "TDI Line Scan Camera",
        "numerical_aperture": 1.2,
        "wavelength_nm": 580,
        "quantum_efficiency": 0.85,
        # ZusÃ¤tzliche Simulationsparameter
        "read_noise_std": 1.2,                 # [counts]
        "spot_intensity_sigma": 0.25,          # Lognormal-Jitter pro Spot (Multiplikativ)
        "frame_jitter_sigma": 0.10,            # Lognormal-Jitter pro Frame
        "on_mean_frames": 4.0,                  # mittlere ON-Dauer [frames]
        "off_mean_frames": 6.0,                 # mittlere OFF-Dauer [frames]
        "bleach_prob_per_frame": 0.002,         # Bleach-Wahrscheinlichkeit je Frame
        "z_amp_um": 0.7,                        # IntensitÃ¤tsabfall-Skala in z [Âµm]
        "z_max_um": 0.6,                        # Begrenzung des z-Bereichs [µm]
        "astig_z0_um": 0.5,
        "astig_coeffs": {"A_x": 1.0, "B_x": 0.0, "A_y": -0.5, "B_y": 0.0}
    }
)

TETRASPECS_PRESET = DetectorPreset(
    name="Tetraspecs",
    max_intensity=300.0,
    background_mean=100.0,
    background_std=15.0,
    pixel_size_um=0.160,
    fwhm_um=0.40,
    metadata={
        "detector_type": "sCMOS Camera",
        "numerical_aperture": 1.2,
        "wavelength_nm": 580,
        "quantum_efficiency": 0.90,
        # ZusÃ¤tzliche Simulationsparameter
        "read_noise_std": 1.8,
        "spot_intensity_sigma": 0.25,
        "frame_jitter_sigma": 0.12,
        "on_mean_frames": 5.0,
        "off_mean_frames": 7.0,
        "bleach_prob_per_frame": 0.0015,
        "z_amp_um": 0.7,
        "z_max_um": 0.6,
        "astig_z0_um": 0.5,
        "astig_coeffs": {"A_x": 1.0, "B_x": 0.0, "A_y": -0.5, "B_y": 0.0}
    }
)


# ============================================================================
# ZEITABHÃ„NGIGE DIFFUSIONSKOEFFIZIENTEN
# ============================================================================

def get_time_dependent_D(t_poly_min: float, D_initial: float, 
                         diffusion_type: str = "normal") -> float:
    """
    Berechnet zeitabhÃ¤ngigen Diffusionskoeffizienten wÃ¤hrend der 
    Polymerisationsphase basierend auf experimentellen Daten.
    
    Physikalisches Modell:
    ----------------------
    D(t) = Dâ‚€ Â· exp(-t/Ï„) Â· f(t)
    
    wobei:
    - Dâ‚€: Initialer Diffusionskoeffizient [ÂµmÂ²/s]
    - Ï„: Charakteristische Zeitkonstante (40 min)
    - f(t): ZusÃ¤tzliche Reduktionsfunktion fÃ¼r t > 90 min
    
    Die starke Reduktion von D bei langen Polymerisationszeiten reflektiert
    die zunehmende Netzwerkdichte und ViskositÃ¤t des Hydrogels.
    
    Parameters:
    -----------
    t_poly_min : float
        Polymerisationszeit [min]
    D_initial : float
        Initialer D-Wert bei t=0 [ÂµmÂ²/s], typisch 3-5 ÂµmÂ²/s
    diffusion_type : str
        "normal", "subdiffusion", "superdiffusion", "confined"
    
    Returns:
    --------
    float : ZeitabhÃ¤ngiger D-Wert [ÂµmÂ²/s]
    
    Referenzen:
    -----------
    - Exponentieller Abfall: Saxton & Jacobson, Annu Rev Biophys (1997)
    - Subdiffusion in Gelen: Masuda et al., Phys Rev Lett (2005)
    """
    
    # Basis-Reduktion: Exponentieller Abfall
    tau = 40.0  # Charakteristische Zeitkonstante [min]
    reduction_factor = np.exp(-t_poly_min / tau)
    
    # ZusÃ¤tzliche Reduktion ab 90 min (starke Vernetzung)
    if t_poly_min >= 90:
        extra_reduction = 0.5 * np.exp(-(t_poly_min - 90) / 30.0)
        reduction_factor *= extra_reduction
    
    D_base = D_initial * reduction_factor
    
    # Diffusionstyp-spezifische Modifikationen
    if diffusion_type == "subdiffusion":
        # Subdiffusion: ZusÃ¤tzliche Verlangsamung
        D_base *= 0.6
    elif diffusion_type == "superdiffusion":
        # Superdiffusion: Leichte ErhÃ¶hung (selten)
        D_base *= 1.3
    elif diffusion_type == "confined":
        # Confined: Stark reduziert
        D_base *= 0.3
    
    return max(D_base, 0.001)  # Minimum: 0.001 ÂµmÂ²/s


def get_diffusion_fractions(t_poly_min: float) -> Dict[str, float]:
    """
    Berechnet Fraktionen verschiedener Diffusionstypen als Funktion der Zeit.
    
    Mit zunehmender Polymerisation steigt der Anteil von Sub- und Confined
    Diffusion, wÃ¤hrend normale Brownsche Bewegung abnimmt.
    
    Parameters:
    -----------
    t_poly_min : float
        Polymerisationszeit [min]
    
    Returns:
    --------
    Dict[str, float] : Fraktionen fÃ¼r jeden Diffusionstyp
    """
    
    # ZeitabhÃ¤ngige Fraktionen (Summe = 1.0)
    if t_poly_min < 10:
        fractions = {
            "normal": 0.95,
            "subdiffusion": 0.04,
            "confined": 0.01,
            "superdiffusion": 0.0
        }
    elif t_poly_min < 60:
        # Lineare Interpolation
        progress = t_poly_min / 60.0
        fractions = {
            "normal": 0.95 - 0.30 * progress,
            "subdiffusion": 0.04 + 0.20 * progress,
            "confined": 0.01 + 0.09 * progress,
            "superdiffusion": 0.0
        }
    elif t_poly_min < 120:
        progress = (t_poly_min - 60.0) / 60.0
        fractions = {
            "normal": 0.65 - 0.25 * progress,
            "subdiffusion": 0.24 + 0.10 * progress,
            "confined": 0.10 + 0.15 * progress,
            "superdiffusion": 0.01
        }
    else:
        # > 120 min: Stark vernetzt
        progress = min((t_poly_min - 120.0) / 60.0, 1.0)
        fractions = {
            "normal": 0.40 - 0.05 * progress,
            "subdiffusion": 0.34 + 0.01 * progress,
            "confined": 0.25 + 0.03 * progress,
            "superdiffusion": 0.01
        }
    
    # Normalisierung
    total = sum(fractions.values())
    return {k: v/total for k, v in fractions.items()}


# ============================================================================
# PSF GENERATOR - Physikalisch korrekte Point Spread Function
# ============================================================================

class PSFGenerator:
    """
    Generiert physikalisch realistische Point Spread Functions (PSF).
    
    Die PSF wird als 2D GauÃŸfunktion modelliert:
    
    I(x,y) = Iâ‚€ Â· exp(-[(x-xâ‚€)Â²/Ïƒâ‚“Â² + (y-yâ‚€)Â²/Ïƒáµ§Â²]/2)
    
    wobei:
    - Iâ‚€: Peak-IntensitÃ¤t [counts]
    - Ïƒâ‚“, Ïƒáµ§: Standardabweichungen in x,y [px]
    - Beziehung: FWHM = 2âˆš(2ln2) Â· Ïƒ â‰ˆ 2.355 Â· Ïƒ
    
    FÃ¼r Astigmatismus (z-abhÃ¤ngig):
    Ïƒâ‚“(z) = Ïƒâ‚€ Â· âˆš(1 + (z/zâ‚€)Â²)
    Ïƒáµ§(z) = Ïƒâ‚€ Â· âˆš(1 - (z/zâ‚€)Â²)
    """
    
    def __init__(self, detector: DetectorPreset, astigmatism: bool = False):
        self.detector = detector
        self.astigmatism = astigmatism
        
        # Konvertiere FWHM zu Ïƒ (Standardabweichung)
        # FWHM = 2.355 * Ïƒ
        fwhm_px = detector.fwhm_um / detector.pixel_size_um
        self.sigma_px = fwhm_px / 2.355
        # Numerischer Mindestwert fÃ¼r Sigma, um InstabilitÃ¤ten zu vermeiden
        self._sigma_eps = 1e-6
        
        # Astigmatismus-Parameter
        if astigmatism:
            # Load calibration from detector metadata
            meta = getattr(detector, 'metadata', {}) or {}
            self.z0_um = float(meta.get("astig_z0_um", 0.5))
            coeffs = meta.get("astig_coeffs", {}) or {}
            self.Ax = float(coeffs.get("A_x", 1.0))
            self.Bx = float(coeffs.get("B_x", 0.0))
            self.Ay = float(coeffs.get("A_y", -0.5))
            self.By = float(coeffs.get("B_y", 0.0))
    
    def generate_psf(self, center_x: float, center_y: float, 
                     intensity: float, z_um: float = 0.0,
                     image_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        Generiert eine einzelne PSF.
        
        Parameters:
        -----------
        center_x, center_y : float
            Spot-Position [px]
        intensity : float
            Peak-IntensitÃ¤t [counts]
        z_um : float
            z-Position fÃ¼r Astigmatismus [Âµm]
        image_size : Tuple[int, int]
            BildgrÃ¶ÃŸe (height, width) [px]
        
        Returns:
        --------
        np.ndarray : PSF-Bild [counts]
        """
        
        height, width = image_size
        
        # Berechne sigma_x, sigma_y basierend auf z-Position
        if self.astigmatism and z_um != 0.0:
            # Astigmatische PSF mit kalibrierten Koeffizienten
            z_norm = z_um / self.z0_um
            term_x = max(1.0 + getattr(self, 'Ax', 1.0) * (z_norm**2) + getattr(self, 'Bx', 0.0) * (z_norm**4), self._sigma_eps)
            term_y = max(1.0 + getattr(self, 'Ay', -0.5) * (z_norm**2) + getattr(self, 'By', 0.0) * (z_norm**4), self._sigma_eps)
            sigma_x = self.sigma_px * np.sqrt(term_x)
            sigma_y = self.sigma_px * np.sqrt(term_y)
        else:
            # Symmetrische PSF
            sigma_x = sigma_y = self.sigma_px

        # Untere Schranke fÃ¼r numerische StabilitÃ¤t
        sigma_x = max(sigma_x, self._sigma_eps)
        sigma_y = max(sigma_y, self._sigma_eps)
        
        # Erstelle Koordinaten-Meshgrid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 2D GauÃŸfunktion
        psf = intensity * np.exp(
            -(((x - center_x)**2 / (2 * sigma_x**2)) +
              ((y - center_y)**2 / (2 * sigma_y**2)))
        )
        
        return psf
    
    def get_metadata(self) -> Dict:
        """Gibt PSF-Metadata zurÃ¼ck"""
        return {
            "fwhm_um": self.detector.fwhm_um,
            "sigma_px": self.sigma_px,
            "pixel_size_um": self.detector.pixel_size_um,
            "astigmatism": self.astigmatism,
            "z0_um": self.z0_um if self.astigmatism else None
        }


# ============================================================================
# TRAJEKTORIEN-GENERATOR - Brownsche Bewegung
# ============================================================================

class TrajectoryGenerator:
    """
    Generiert realistische Trajektorien basierend auf verschiedenen
    Diffusionsmodellen.
    
    Brownsche Bewegung (normale Diffusion):
    ---------------------------------------
    Î”rÂ² = 2Â·dÂ·DÂ·Î”t
    
    wobei:
    - d: DimensionalitÃ¤t (2 fÃ¼r 2D, 3 fÃ¼r 3D)
    - D: Diffusionskoeffizient [ÂµmÂ²/s]
    - Î”t: Zeitintervall [s]
    
    Subdiffusion (Anomale Diffusion):
    ----------------------------------
    Î”rÂ² = 2Â·dÂ·DÂ·Î”t^Î±
    mit Î± < 1 (typisch 0.6-0.8)
    """
    
    def __init__(self, D_initial: float, t_poly_min: float, 
                 frame_rate_hz: float, pixel_size_um: float):
        self.D_initial = D_initial
        self.t_poly_min = t_poly_min
        self.dt = 1.0 / frame_rate_hz
        self.pixel_size_um = pixel_size_um
        
        # Hole Diffusionsfraktionen
        self.fractions = get_diffusion_fractions(t_poly_min)
        
        # Berechne D-Werte fÃ¼r jeden Typ
        self.D_values = {
            dtype: get_time_dependent_D(t_poly_min, D_initial, dtype)
            for dtype in self.fractions.keys()
        }
    
    def generate_trajectory(self, start_pos: Tuple[float, float, float],
                           num_frames: int, 
                           diffusion_type: str = "normal") -> np.ndarray:
        """
        Generiert eine 3D-Trajektorie.
        
        Parameters:
        -----------
        start_pos : Tuple[float, float, float]
            Startposition (x, y, z) [Âµm]
        num_frames : int
            Anzahl Frames
        diffusion_type : str
            Typ der Diffusion
        
        Returns:
        --------
        np.ndarray : Trajektorie (num_frames, 3) [Âµm]
        """
        
        D = self.D_values[diffusion_type]
        trajectory = np.zeros((num_frames, 3))
        trajectory[0] = start_pos
        
        # Anomaler Exponent fÃ¼r Subdiffusion
        alpha = 0.7 if diffusion_type == "subdiffusion" else 1.0
        
        for i in range(1, num_frames):
            # Mean Square Displacement
            if diffusion_type == "confined":
                # Confined: Begrenzter Raum
                msd = 2 * D * self.dt
                # RÃ¼ckstellkraft zum Zentrum
                drift = -0.1 * (trajectory[i-1] - start_pos)
            else:
                # Normale/Sub/Super Diffusion
                msd = 2 * D * (self.dt ** alpha)
                drift = np.zeros(3)
            
            # Brownsche Schritte
            step = np.random.normal(0, np.sqrt(msd), size=3)
            
            trajectory[i] = trajectory[i-1] + step + drift
        
        return trajectory
    
    def generate_multi_trajectory(self, num_spots: int, num_frames: int,
                                  image_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        Generiert mehrere Trajektorien mit verschiedenen Diffusionstypen.
        
        Returns:
        --------
        List[np.ndarray] : Liste von Trajektorien
        """
        
        height, width = image_size
        trajectories = []
        
        for _ in range(num_spots):
            # WÃ¤hle Diffusionstyp basierend auf Fraktionen
            dtype = np.random.choice(
                list(self.fractions.keys()),
                p=list(self.fractions.values())
            )
            
            # ZufÃ¤llige Startposition (in Âµm, innerhalb des Bildes)
            start_x = np.random.uniform(0.2 * width, 0.8 * width) * self.pixel_size_um
            start_y = np.random.uniform(0.2 * height, 0.8 * height) * self.pixel_size_um
            start_z = np.random.uniform(-0.5, 0.5)  # z in [-0.5, 0.5] Âµm
            
            trajectory = self.generate_trajectory(
                (start_x, start_y, start_z),
                num_frames,
                dtype
            )
            
            trajectories.append({
                "positions": trajectory,
                "diffusion_type": dtype,
                "D_value": self.D_values[dtype]
            })
        
        return trajectories
    
    def get_metadata(self) -> Dict:
        """Gibt Trajektorien-Metadata zurÃ¼ck"""
        return {
            "D_initial": self.D_initial,
            "t_poly_min": self.t_poly_min,
            "frame_rate_hz": 1.0 / self.dt,
            "diffusion_fractions": self.fractions,
            "D_values": self.D_values
        }


# ============================================================================
# BACKGROUND GENERATOR - Realistischer Hintergrund
# ============================================================================

class BackgroundGenerator:
    """Generiert realistischen Background mit Gradiente und Rauschen."""
    
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def generate(self, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Generiert Background-Bild.
        
        Returns:
        --------
        np.ndarray : Background [counts]
        """
        
        height, width = image_size
        
        # Basis-Background (Poisson-Rauschen)
        background = np.random.poisson(self.mean, size=(height, width)).astype(float)
        
        # GauÃŸsches Rauschen
        noise = np.random.normal(0, self.std, size=(height, width))
        background += noise
        
        # Leichter Gradient (simuliert ungleichmÃ¤ÃŸige Beleuchtung)
        y, x = np.meshgrid(np.linspace(-1, 1, height), 
                          np.linspace(-1, 1, width), 
                          indexing='ij')
        gradient = 5 * (x**2 + y**2)
        background += gradient
        
        return np.maximum(background, 0)  # Keine negativen Werte


# ============================================================================
# PHOTOPHYSICS - Blinking & Bleaching
# ============================================================================

class PhotoPhysics:
    """
    Einfache 2-Zustands-Photophysik (ON/OFF) mit Bleaching.

    - ON/OFF-Dauern ~ geometrisch (entspricht exponential in kontinuierlicher Zeit)
    - Bleaching: pro ON-Frame mit kleiner Wahrscheinlichkeit permanent OFF
    """

    def __init__(self, on_mean_frames: float = 4.0, off_mean_frames: float = 6.0,
                 bleach_prob_per_frame: float = 0.002):
        self.on_mean = max(on_mean_frames, 1e-3)
        self.off_mean = max(off_mean_frames, 1e-3)
        self.bleach_prob = max(bleach_prob_per_frame, 0.0)

    def _sample_duration(self, mean_frames: float) -> int:
        # Geometric(p) with mean = 1/p -> p = 1/mean
        p = 1.0 / max(mean_frames, 1.0)
        # numpy geometric returns k >= 1
        return int(np.random.geometric(p))

    def generate_on_mask(self, num_spots: int, num_frames: int) -> np.ndarray:
        mask = np.zeros((num_spots, num_frames), dtype=bool)
        p0_on = self.on_mean / (self.on_mean + self.off_mean)
        for s in range(num_spots):
            t = 0
            state_on = np.random.rand() < p0_on
            bleached = False
            while t < num_frames:
                if bleached:
                    break
                duration = self._sample_duration(self.on_mean if state_on else self.off_mean)
                end = min(num_frames, t + duration)
                if state_on:
                    mask[s, t:end] = True
                    # Bleaching mit Wahrscheinlichkeit Ã¼ber die Dauer
                    if np.random.rand() < (1.0 - (1.0 - self.bleach_prob) ** (end - t)):
                        bleached = True
                        break
                # Toggle Zustand und weiter
                t = end
                state_on = not state_on
        return mask


# ============================================================================
# HAUPTSIMULATOR
# ============================================================================

class TIFFSimulator:
    """
    Hauptklasse fÃ¼r TIFF-Simulation.
    
    Workflow:
    ---------
    1. Initialisierung mit Detektor-Preset
    2. Generierung von Trajektorien
    3. Rendering der PSFs
    4. Addition von Background & Noise
    5. Export als TIFF
    """
    
    def __init__(self, detector: DetectorPreset, mode: str = "polyzeit",
                 t_poly_min: float = 60.0, astigmatism: bool = False):
        
        self.detector = detector
        self.mode = mode
        self.t_poly_min = t_poly_min
        self.astigmatism = astigmatism
        
        # Initialisiere Generatoren
        self.psf_gen = PSFGenerator(detector, astigmatism)
        self.bg_gen = BackgroundGenerator(
            detector.background_mean,
            detector.background_std
        )
        
        # Metadata
        self.metadata = {
            "detector": detector.name,
            "mode": mode,
            "t_poly_min": t_poly_min,
            "astigmatism": astigmatism,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_tiff(self, image_size: Tuple[int, int], num_spots: int,
                     num_frames: int, frame_rate_hz: float,
                     d_initial: float = 0.5,
                     exposure_substeps: int = 1,
                     enable_photophysics: bool = False) -> np.ndarray:
        """
        Generiert TIFF-Stack.
        
        Parameters:
        -----------
        image_size : Tuple[int, int]
            (height, width) [px]
        num_spots : int
            Anzahl Fluoreszenz-Spots
        num_frames : int
            Anzahl Frames
        frame_rate_hz : float
            Frame Rate [Hz]
        
        Returns:
        --------
        np.ndarray : TIFF-Stack (num_frames, height, width) [counts]
        """
        
        height, width = image_size
        
        # Initialisiere Trajektorien-Generator
        traj_gen = TrajectoryGenerator(
            D_initial=float(d_initial),
            t_poly_min=self.t_poly_min,
            frame_rate_hz=frame_rate_hz,
            pixel_size_um=self.detector.pixel_size_um
        )
        
        # Generiere Trajektorien
        trajectories = traj_gen.generate_multi_trajectory(
            num_spots, num_frames, image_size
        )

        # Detector-Parameter (Photophysik & z-Begrenzung)
        meta = self.detector.metadata or {}
        read_noise_std = float(meta.get("read_noise_std", 1.5))
        spot_sigma = float(meta.get("spot_intensity_sigma", 0.25))
        frame_sigma = float(meta.get("frame_jitter_sigma", 0.10))
        on_mean = float(meta.get("on_mean_frames", 4.0))
        off_mean = float(meta.get("off_mean_frames", 6.0))
        bleach_p = float(meta.get("bleach_prob_per_frame", 0.002))
        z_amp_um = float(meta.get("z_amp_um", 0.7))
        z_max_um = float(meta.get("z_max_um", 0.6))

        # Photophysik (Blinking & Bleaching)
        if enable_photophysics:
            phot = PhotoPhysics(on_mean, off_mean, bleach_p)
            on_mask = phot.generate_on_mask(num_spots, num_frames)
        else:
            on_mask = np.ones((num_spots, num_frames), dtype=bool)

        # Spot-IntensitÃ¤ten (Lognormal-Verteilung)
        base_intensities = self.detector.max_intensity * np.exp(
            np.random.normal(0.0, spot_sigma, size=num_spots)
        )

        # Initialisiere TIFF-Stack
        tiff_stack = np.zeros((num_frames, height, width), dtype=np.uint16)
        
        # Generiere jeden Frame
        for frame_idx in range(num_frames):
            # Background
            frame = self.bg_gen.generate(image_size)
            
            # FÃ¼ge jeden Spot hinzu
            for si, traj_data in enumerate(trajectories):
                pos = traj_data["positions"][frame_idx]
                
                # Konvertiere Âµm zu Pixel
                x_px = pos[0] / self.detector.pixel_size_um
                y_px = pos[1] / self.detector.pixel_size_um
                z_um = float(pos[2]) if self.astigmatism else 0.0
                if self.astigmatism:
                    if z_um > z_max_um:
                        z_um = z_max_um
                    elif z_um < -z_max_um:
                        z_um = -z_max_um
                
                # Check ob im Bild
                if 0 <= x_px < width and 0 <= y_px < height:
                    # Photophysik: nur wenn Spot an ist
                    if on_mask[si, frame_idx]:
                        frame_jitter = float(np.exp(np.random.normal(0.0, frame_sigma)))
                        amp = np.exp(- (z_um / z_amp_um) ** 2) if self.astigmatism else 1.0
                        intensity = base_intensities[si] * frame_jitter * amp
                        # Motion-Blur: mehrere Substeps pro Belichtung
                        substeps = max(int(exposure_substeps), 1)
                        if frame_idx > 0 and substeps > 1:
                            prev = traj_data["positions"][frame_idx-1]
                        else:
                            prev = traj_data["positions"][frame_idx]
                        for ss in range(substeps):
                            if frame_idx > 0 and substeps > 1:
                                frac = (ss + 0.5) / substeps
                                px = prev[0] + frac * (pos[0] - prev[0])
                                py = prev[1] + frac * (pos[1] - prev[1])
                                pz = prev[2] + frac * (pos[2] - prev[2]) if self.astigmatism else 0.0
                            else:
                                px, py, pz = pos[0], pos[1], z_um
                            psf = self.psf_gen.generate_psf(
                                px / self.detector.pixel_size_um,
                                py / self.detector.pixel_size_um,
                                intensity / substeps,
                                float(pz) if self.astigmatism else 0.0,
                                image_size
                            )
                            frame += psf
            
            # Poisson-Noise (shot noise) â€“ robuste Vorverarbeitung
            frame = np.nan_to_num(frame, nan=0.0, posinf=1e6, neginf=0.0)
            frame = np.clip(frame, 0, 1e6)
            frame = np.random.poisson(frame)
            # Kamera-Read-Noise
            if read_noise_std > 0:
                frame = frame.astype(float)
                frame += np.random.normal(0.0, read_noise_std, size=frame.shape)
            
            # Clip & Convert
            tiff_stack[frame_idx] = np.clip(frame, 0, 65535).astype(np.uint16)
        
        # Update Metadata
        self.metadata.update({
            "image_size": image_size,
            "num_spots": num_spots,
            "num_frames": num_frames,
            "frame_rate_hz": frame_rate_hz,
            "d_initial": float(d_initial),
            "exposure_substeps": int(exposure_substeps),
            "photophysics": bool(enable_photophysics),
            "trajectories": trajectories,
            "psf": self.psf_gen.get_metadata(),
            "diffusion": traj_gen.get_metadata()
        })
        
        return tiff_stack
    
    def generate_z_stack(self, image_size: Tuple[int, int], num_spots: int,
                        z_range_um: Tuple[float, float], 
                        z_step_um: float) -> np.ndarray:
        """
        Generiert z-Stack fÃ¼r Kalibrierung (statische Spots).
        
        Returns:
        --------
        np.ndarray : z-Stack (n_slices, height, width)
        """
        
        z_min, z_max = z_range_um
        z_positions = np.arange(z_min, z_max + z_step_um, z_step_um)
        n_slices = len(z_positions)
        
        height, width = image_size
        
        # Generiere zufÃ¤llige, aber STATISCHE Spot-Positionen
        spot_positions = []
        for _ in range(num_spots):
            x_px = np.random.uniform(0.2 * width, 0.8 * width)
            y_px = np.random.uniform(0.2 * height, 0.8 * height)
            spot_positions.append((x_px, y_px))
        
        # Erstelle z-Stack
        z_stack = np.zeros((n_slices, height, width), dtype=np.uint16)
        
        for z_idx, z_um in enumerate(z_positions):
            # Background
            frame = self.bg_gen.generate(image_size)
            
            # FÃ¼ge Spots hinzu (immer an gleicher Position!)
            # z-abhÃ¤ngige IntensitÃ¤t (Defokus)
            meta = self.detector.metadata or {}
            z_amp_um = float(meta.get("z_amp_um", 0.7))
            amp = np.exp(- (z_um / z_amp_um) ** 2)
            for x_px, y_px in spot_positions:
                psf = self.psf_gen.generate_psf(
                    x_px, y_px,
                    self.detector.max_intensity * amp,
                    z_um,
                    image_size
                )
                frame += psf
            
            # Poisson-Noise â€“ robuste Vorverarbeitung
            frame = np.nan_to_num(frame, nan=0.0, posinf=1e6, neginf=0.0)
            frame = np.clip(frame, 0, 1e6)
            frame = np.random.poisson(frame)
            # Kamera-Read-Noise
            read_noise_std = float((self.detector.metadata or {}).get("read_noise_std", 1.5))
            if read_noise_std > 0:
                frame = frame.astype(float)
                frame += np.random.normal(0.0, read_noise_std, size=frame.shape)
            z_stack[z_idx] = np.clip(frame, 0, 65535).astype(np.uint16)
        
        # Update Metadata
        self.metadata.update({
            "image_size": image_size,
            "num_spots": num_spots,
            "z_range_um": z_range_um,
            "z_step_um": z_step_um,
            "n_slices": n_slices,
            "spot_positions": spot_positions
        })
        
        return z_stack
    
    def get_metadata(self) -> Dict:
        """Gibt alle Metadata zurÃ¼ck"""
        return self.metadata.copy()


# ============================================================================
# TIFF EXPORT
# ============================================================================

def save_tiff(filepath: str, tiff_stack: np.ndarray, 
              metadata: Optional[Dict] = None) -> None:
    """
    Speichert TIFF-Stack mit Metadata.
    
    Parameters:
    -----------
    filepath : str
        Pfad zur TIFF-Datei
    tiff_stack : np.ndarray
        TIFF-Stack (frames, height, width)
    metadata : Dict, optional
        Metadata-Dictionary
    """
    
    from PIL import Image
    
    # Erstelle TIFF
    images = [Image.fromarray(frame) for frame in tiff_stack]
    
    # Speichere als Multi-Page TIFF
    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        compression='tiff_deflate'
    )
    
    print(f"âœ… TIFF gespeichert: {filepath}")
    print(f"   Shape: {tiff_stack.shape}")
    print(f"   Dtype: {tiff_stack.dtype}")
    print(f"   Range: [{tiff_stack.min()}, {tiff_stack.max()}]")


# ============================================================================
# QUICK TESTING
# ============================================================================

if __name__ == "__main__":
    print("ðŸ”¬ TIFF Simulator V3.0 - Backend Test")
    print("=" * 50)
    
    # Test: Einfache Simulation
    sim = TIFFSimulator(
        detector=TDI_PRESET,
        mode="polyzeit",
        t_poly_min=60.0,
        astigmatism=False
    )
    
    tiff = sim.generate_tiff(
        image_size=(64, 64),
        num_spots=3,
        num_frames=10,
        frame_rate_hz=20.0
    )
    
    print(f"\nâœ… Test erfolgreich!")
    print(f"   TIFF Shape: {tiff.shape}")
    print(f"   Mean Intensity: {tiff.mean():.1f}")
    print(f"   Max Intensity: {tiff.max()}")



