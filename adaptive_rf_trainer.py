"""
🤖 ADAPTIVE RF TRAINER - Quick Fine-Tuning for Experimental Data
==================================================================

Schätzt den Polymerisationsgrad aus experimentellen Daten und trainiert
einen Random Forest Classifier speziell auf diese Bedingungen.

Workflow:
1. Parse experimentelle TrackMate XML
2. Schätze durchschnittlichen Diffusionskoeffizienten
3. Berechne daraus den Polymerisationsgrad t_poly
4. Generiere ~100-200 Tracks bei diesem t_poly (mit BatchSimulator)
5. Quick-Train RF (weniger Bäume, aber spezialisiert)
6. Nutze diesen RF für die Analyse

Vorteil: RF ist optimal auf die experimentellen Bedingungen abgestimmt!

Version: 1.0 - November 2025
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from scipy import stats
import warnings
import tempfile
import shutil
import json

from tiff_simulator_v3 import DetectorPreset
from batch_simulator import BatchSimulator
from rf_trainer import RandomForestTrainer, RFTrainingConfig

warnings.filterwarnings('ignore')


@dataclass
class PolygradEstimate:
    """Geschätzter Polymerisationsgrad aus experimentellen Daten."""

    t_poly_min: float  # Geschätzte Polymerisationszeit in Minuten
    mean_D: float      # Mittlerer Diffusionskoeffizient (µm²/s)
    std_D: float       # Standardabweichung von D
    D0_reference: float = 1.0  # Referenz-D0 für t=0 (µm²/s)
    tau_min: float = 32.0      # Zeitkonstante (Minuten)
    confidence: str = "medium" # low, medium, high
    n_tracks_analyzed: int = 0

    def __post_init__(self):
        """Berechne Konfidenz basierend auf Anzahl Tracks."""
        if self.n_tracks_analyzed >= 100:
            self.confidence = "high"
        elif self.n_tracks_analyzed >= 30:
            self.confidence = "medium"
        else:
            self.confidence = "low"


class PolygradEstimator:
    """Schätzt Polymerisationsgrad aus experimentellen Tracks."""

    def __init__(self, D0_reference: float = 1.0, tau_min: float = 32.0):
        """
        Args:
            D0_reference: Referenz-Diffusionskoeffizient bei t=0 (µm²/s)
            tau_min: Zeitkonstante für Polymerisation (Minuten)
        """
        self.D0_reference = D0_reference
        self.tau_min = tau_min

    def estimate_from_xml(
        self,
        xml_path: Path,
        frame_rate_hz: float = 20.0,
        min_track_length: int = 48
    ) -> PolygradEstimate:
        """
        Schätzt Polymerisationsgrad aus TrackMate XML.

        Methode:
        1. Parse alle Tracks
        2. Berechne MSD für jeden Track (log-log fit)
        3. Berechne D aus MSD (D = MSD/(4*Δt) für 2D)
        4. Mittlere über alle Tracks
        5. Invertiere D(t) = D0 * exp(-t/tau) → t = -tau * ln(D/D0)

        Args:
            xml_path: Pfad zur TrackMate XML
            frame_rate_hz: Frame rate in Hz
            min_track_length: Minimale Track-Länge für Analyse

        Returns:
            PolygradEstimate mit geschätztem t_poly
        """
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        frame_interval_sec = float(root.attrib.get('frameInterval', 1.0))
        dt = frame_interval_sec  # Time per frame in seconds

        D_values = []

        # Iteriere über alle Tracks
        for particle in root.findall('particle'):
            n_spots = int(particle.attrib['nSpots'])

            if n_spots < min_track_length:
                continue

            # Extrahiere Positionen
            positions = []
            for detection in particle.findall('detection'):
                x = float(detection.attrib['x'])
                y = float(detection.attrib['y'])
                positions.append([x, y])

            positions = np.array(positions)

            # Berechne MSD
            D = self._estimate_D_from_trajectory(positions, dt)

            if D > 0 and not np.isnan(D) and not np.isinf(D):
                D_values.append(D)

        D_values = np.array(D_values)

        if len(D_values) == 0:
            print("⚠️  Keine Tracks für D-Schätzung gefunden!")
            return PolygradEstimate(
                t_poly_min=0.0,
                mean_D=self.D0_reference,
                std_D=0.0,
                D0_reference=self.D0_reference,
                tau_min=self.tau_min,
                n_tracks_analyzed=0
            )

        mean_D = np.mean(D_values)
        std_D = np.std(D_values)

        # Berechne t_poly: D(t) = D0 * exp(-t/tau) → t = -tau * ln(D/D0)
        # Clipping um negative/unrealistische Werte zu vermeiden
        if mean_D >= self.D0_reference:
            # D >= D0 → t = 0 (keine Polymerisation oder sehr früh)
            t_poly_min = 0.0
        else:
            # D < D0 → Polymerisation hat stattgefunden
            ratio = mean_D / self.D0_reference
            if ratio <= 0:
                ratio = 1e-6  # Safety
            t_poly_min = -self.tau_min * np.log(ratio)

            # Clipping: max 180 min (realistischer Bereich)
            t_poly_min = min(t_poly_min, 180.0)

        estimate = PolygradEstimate(
            t_poly_min=t_poly_min,
            mean_D=mean_D,
            std_D=std_D,
            D0_reference=self.D0_reference,
            tau_min=self.tau_min,
            n_tracks_analyzed=len(D_values)
        )

        return estimate

    def _estimate_D_from_trajectory(
        self,
        positions: np.ndarray,
        dt: float,
        max_tau: int = 10
    ) -> float:
        """
        Schätzt D aus einer einzelnen Trajektorie via MSD.

        Args:
            positions: (N, 2) array mit [x, y] Positionen
            dt: Zeit pro Frame (Sekunden)
            max_tau: Maximales τ für MSD-Berechnung

        Returns:
            Diffusionskoeffizient D in µm²/s
        """
        N = len(positions)

        if N < 4:
            return np.nan

        # MSD für verschiedene τ
        max_tau = min(max_tau, N // 4)

        msds = []
        taus = []

        for tau in range(1, max_tau + 1):
            displacements = positions[tau:] - positions[:-tau]
            squared_displacements = np.sum(displacements**2, axis=1)
            msd = np.mean(squared_displacements)

            msds.append(msd)
            taus.append(tau * dt)

        msds = np.array(msds)
        taus = np.array(taus)

        # Linear fit: MSD = 4*D*τ (für 2D Diffusion)
        # D = slope / 4
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(taus, msds)
            D = slope / 4.0  # 2D Diffusion

            # Validierung
            if D <= 0 or r_value**2 < 0.5:  # Schlechter Fit
                return np.nan

            return D

        except:
            return np.nan


def quick_train_adaptive_rf(
    xml_path: Path,
    detector: DetectorPreset,
    frame_rate_hz: float = 20.0,
    n_tracks_total: int = 200,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    cleanup_temp: bool = True
) -> Tuple[RandomForestTrainer, PolygradEstimate]:
    """
    Adaptive RF Quick-Training: Schätzt Polygrad und trainiert RF darauf.

    Args:
        xml_path: Pfad zur experimentellen TrackMate XML
        detector: Detector Preset für Simulation
        frame_rate_hz: Frame rate
        n_tracks_total: Totale Anzahl Tracks für Training (wird auf 2-3 TIFFs verteilt)
        output_dir: Optional output directory für RF-Modell
        verbose: Print progress
        cleanup_temp: Temporäre TIFFs nach Training löschen

    Returns:
        (RandomForestTrainer, PolygradEstimate)
    """
    if verbose:
        print("=" * 70)
        print("🤖 ADAPTIVE RF QUICK-TRAINING")
        print("=" * 70)

    # Schritt 1: Schätze Polygrad
    if verbose:
        print(f"\n📊 Schritt 1: Analysiere experimentelle Daten...")
        print(f"   XML: {xml_path.name}")

    estimator = PolygradEstimator()
    estimate = estimator.estimate_from_xml(xml_path, frame_rate_hz)

    if verbose:
        print(f"\n✅ Polygrad-Schätzung:")
        print(f"   t_poly = {estimate.t_poly_min:.1f} min")
        print(f"   mean D = {estimate.mean_D:.4f} µm²/s")
        print(f"   std D  = {estimate.std_D:.4f} µm²/s")
        print(f"   Konfidenz: {estimate.confidence}")
        print(f"   Tracks analysiert: {estimate.n_tracks_analyzed}")

    # Schritt 2: Generiere Trainings-TIFFs
    if verbose:
        print(f"\n🔬 Schritt 2: Generiere Trainings-TIFFs...")
        print(f"   Polygrad: t = {estimate.t_poly_min:.1f} min")
        print(f"   Total Tracks: {n_tracks_total}")

    # Temporäres Verzeichnis für TIFFs
    temp_dir = Path(tempfile.mkdtemp(prefix="adaptive_rf_"))

    try:
        # Batch-Simulation mit 2 TIFFs (je ~n_tracks_total/2 Spots)
        batch = BatchSimulator(
            output_dir=str(temp_dir),
            enable_rf=False  # RF trainieren wir selbst
        )

        # 2 TIFFs für Variation (gleicher Polygrad)
        num_spots = n_tracks_total // 2

        for i in range(2):
            batch.add_task({
                'detector': detector,
                'mode': 'polyzeit',
                't_poly_min': estimate.t_poly_min,
                'filename': f'train_t{estimate.t_poly_min:.0f}_rep{i+1}.tif',
                'image_size': (128, 128),
                'num_spots': num_spots,
                'num_frames': 200,
                'frame_rate_hz': frame_rate_hz
            })

        if verbose:
            print(f"   → Generiere 2 TIFFs mit je {num_spots} Tracks...")

        batch.run()

        if verbose:
            print(f"   ✓ TIFFs generiert in {temp_dir}")

        # Schritt 3: RF Training
        if verbose:
            print(f"\n🌲 Schritt 3: RF Quick-Training...")

        # Quick Config
        config = RFTrainingConfig(
            window_size=48,
            step_size=32,
            n_estimators=512,  # Weniger Bäume (4x schneller als 2048)
            max_depth=15,      # Kleiner (schneller)
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=42
        )

        trainer = RandomForestTrainer(output_dir=temp_dir, config=config)

        # Lade Metadaten und trainiere
        metadata_files = list(temp_dir.glob("*_metadata.json"))

        if verbose:
            print(f"   → Lade {len(metadata_files)} Metadaten-Dateien...")

        for meta_file in metadata_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                trainer.update_with_metadata(metadata)

        if verbose:
            print(f"   → Finalisiere RF-Training...")

        # Finalize & Train
        result = trainer.finalize()

        if verbose:
            print(f"\n✅ RF Training abgeschlossen!")
            print(f"   Modell: {config.n_estimators} Bäume, max_depth={config.max_depth}")
            print(f"   Features: 27")
            if result.get('model_path'):
                print(f"   Modell gespeichert: {Path(result['model_path']).name}")

        # Kopiere Modell zu output_dir
        if output_dir and result.get('model_path'):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Ziel-Pfad
            model_path = output_dir / f"rf_adaptive_t{estimate.t_poly_min:.0f}min.joblib"

            # Kopiere von temp_dir zu output_dir
            shutil.copy(result['model_path'], model_path)

            if verbose:
                print(f"   Kopiert nach: {model_path.name}")

    finally:
        # Cleanup
        if cleanup_temp:
            if verbose:
                print(f"\n🧹 Cleanup: Lösche temporäre Dateien...")
            shutil.rmtree(temp_dir, ignore_errors=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✅ Adaptive RF Quick-Training abgeschlossen!")
        print(f"{'=' * 70}\n")

    return trainer, estimate


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    """Quick test mit merged_tracks.xml"""

    from tiff_simulator_v3 import TDI_PRESET

    xml_path = Path("merged_tracks.xml")

    if not xml_path.exists():
        print(f"❌ {xml_path} nicht gefunden!")
        exit(1)

    print("\n" + "=" * 70)
    print("🧪 TEST: Adaptive RF Quick-Training")
    print("=" * 70)

    # Quick-Train
    trainer, estimate = quick_train_adaptive_rf(
        xml_path=xml_path,
        detector=TDI_PRESET,
        frame_rate_hz=20.0,
        n_tracks_total=200,  # 2 TIFFs mit je 100 Tracks
        output_dir=Path("output"),
        verbose=True,
        cleanup_temp=True
    )

    print("\n✅ Test erfolgreich!")
    print(f"   RF Modell: {trainer.rf.n_estimators} Bäume")
    print(f"   Polygrad: t = {estimate.t_poly_min:.1f} min")
    print(f"   Mean D: {estimate.mean_D:.4f} µm²/s\n")
