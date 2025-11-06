"""
🔄 BATCH SIMULATOR
==================

Automatisierte Generierung mehrerer TIFF-Simulationen mit systematischer
Parametervariatio für umfassende Parameterstudien.

Anwendungsfälle:
- Polymerisationszeit-Serien (t = 0, 30, 60, 90, 120 min)
- Detektor-Vergleiche (TDI-G0 vs. Tetraspecs)
- D-Wert-Studien (verschiedene D_initial)
- Reproduzierbarkeit (n Wiederholungen gleicher Parameter)

Version: 3.0 - Oktober 2025
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Tuple, Sequence
from datetime import datetime
import json
from tqdm import tqdm

try:
    from tiff_simulator_v3 import (
        TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    )
    from metadata_exporter import MetadataExporter
    from rf_trainer import RandomForestTrainer, RFTrainingConfig
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Stelle sicher, dass tiff_simulator_v3.py und metadata_exporter.py")
    print("   im gleichen Ordner sind!")
    exit(1)


class BatchSimulator:
    """
    Automatisierte Batch-Simulation mit Fortschrittsanzeige.
    
    Workflow:
    ---------
    1. Definition der Parametervariationen
    2. Systematische Generierung aller Kombinationen
    3. Parallel-Export von TIFF + Metadata
    4. Zusammenfassung in Master-CSV
    
    Beispiel:
    ---------
    >>> batch = BatchSimulator(output_dir="my_simulations")
    >>> batch.add_polyzeit_series(times=[30, 60, 90], detector=TDI_PRESET)
    >>> batch.run()
    """
    
    def __init__(
        self,
        output_dir: str = "./batch_output",
        *,
        enable_rf: bool = False,
        rf_config: Dict = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.tasks = []
        self.metadata_exporter = MetadataExporter(self.output_dir)

        self.rf_enabled = bool(enable_rf)
        self.rf_trainer = None
        self.rf_auto_balance = True
        self._balanced_training_added = False
        if self.rf_enabled:
            rf_cfg = dict(rf_config or {})
            self.rf_auto_balance = bool(rf_cfg.pop('auto_balance_training', True))
            config = self._create_rf_config(rf_cfg)
            self.rf_trainer = RandomForestTrainer(self.output_dir, config)

        # Statistik
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
            'rf': {
                'trained': False,
                'model_path': None,
                'feature_path': None,
                'summary_path': None,
                'training_accuracy': None,
                'oob_score': None,
                'validation_accuracy': None,
                'samples': 0
            }
        }
    
    def add_task(self, task_config: Dict) -> None:
        """
        Fügt eine Simulations-Aufgabe hinzu.
        
        Parameters:
        -----------
        task_config : Dict
            Konfiguration mit allen Parametern:
            - detector: DetectorPreset
            - mode: str
            - t_poly_min: float (optional)
            - astigmatism: bool
            - image_size: Tuple[int, int]
            - num_spots: int
            - num_frames: int
            - frame_rate_hz: float
            - filename: str
            - z_range_um: Tuple[float, float] (für z-Stack)
            - z_step_um: float (für z-Stack)
        """
        
        # Validierung
        required = ['detector', 'mode', 'filename']
        for key in required:
            if key not in task_config:
                raise ValueError(f"Fehlendes Required Field: {key}")
        
        # Defaults setzen
        defaults = {
            'astigmatism': False,
            'image_size': (128, 128),
            'num_spots': 10,
            'num_frames': 100,
            'frame_rate_hz': 20.0,
            't_poly_min': 60.0
        }
        
        for key, value in defaults.items():
            if key not in task_config:
                task_config[key] = value
        
        self.tasks.append(task_config)
        self.stats['total_tasks'] += 1
    
    def add_polyzeit_series(self, times: List[float], 
                           detector=TDI_PRESET,
                           repeats: int = 1,
                           **kwargs) -> None:
        """
        Fügt eine Polymerisationszeit-Serie hinzu.
        
        Parameters:
        -----------
        times : List[float]
            Liste von Polymerisationszeiten [min]
        detector : DetectorPreset
            Detektor (TDI_PRESET oder TETRASPECS_PRESET)
        repeats : int
            Anzahl Wiederholungen pro Zeit
        **kwargs : dict
            Weitere Parameter (image_size, num_spots, etc.)
        
        Beispiel:
        ---------
        >>> batch.add_polyzeit_series(
        ...     times=[30, 60, 90, 120],
        ...     detector=TDI_PRESET,
        ...     repeats=3,
        ...     num_spots=20,
        ...     num_frames=200
        ... )
        """
        
        for t in times:
            for rep in range(repeats):
                # Dateiname mit Zeit und Repeat
                filename = f"{detector.name.lower()}_t{int(t)}min"
                if repeats > 1:
                    filename += f"_rep{rep+1}"
                filename += ".tif"
                
                task = {
                    'detector': detector,
                    'mode': 'polyzeit',
                    't_poly_min': t,
                    'filename': filename,
                    **kwargs
                }
                
                self.add_task(task)
    
    def add_detector_comparison(self, polyzeit: float = 60.0,
                               repeats: int = 1,
                               **kwargs) -> None:
        """
        Fügt Detektor-Vergleich hinzu (TDI-G0 vs. Tetraspecs).
        
        Parameters:
        -----------
        polyzeit : float
            Polymerisationszeit [min]
        repeats : int
            Anzahl Wiederholungen
        **kwargs : dict
            Weitere Parameter
        """
        
        for detector in [TDI_PRESET, TETRASPECS_PRESET]:
            for rep in range(repeats):
                filename = f"{detector.name.lower()}_comparison_t{int(polyzeit)}min"
                if repeats > 1:
                    filename += f"_rep{rep+1}"
                filename += ".tif"
                
                task = {
                    'detector': detector,
                    'mode': 'polyzeit',
                    't_poly_min': polyzeit,
                    'filename': filename,
                    **kwargs
                }
                
                self.add_task(task)
    
    def add_3d_series(self, times: List[float], 
                     detector=TDI_PRESET,
                     repeats: int = 1,
                     **kwargs) -> None:
        """
        Fügt 3D-Simulationen mit Astigmatismus hinzu.
        
        Parameters:
        -----------
        times : List[float]
            Polymerisationszeiten [min]
        detector : DetectorPreset
            Detektor
        repeats : int
            Wiederholungen
        **kwargs : dict
            Weitere Parameter
        """
        
        for t in times:
            for rep in range(repeats):
                filename = f"{detector.name.lower()}_3d_t{int(t)}min"
                if repeats > 1:
                    filename += f"_rep{rep+1}"
                filename += ".tif"
                
                task = {
                    'detector': detector,
                    'mode': 'polyzeit_astig',
                    't_poly_min': t,
                    'astigmatism': True,
                    'filename': filename,
                    **kwargs
                }
                
                self.add_task(task)
    
    def add_z_stack(self, detector=TDI_PRESET,
                   z_range: Tuple[float, float] = (-1.0, 1.0),
                   z_step: float = 0.1,
                   **kwargs) -> None:
        """
        Fügt z-Stack Kalibrierung hinzu.
        
        Parameters:
        -----------
        detector : DetectorPreset
            Detektor
        z_range : Tuple[float, float]
            z-Range [µm]
        z_step : float
            z-Step [µm]
        **kwargs : dict
            Weitere Parameter
        """
        
        filename = f"{detector.name.lower()}_zstack.tif"
        
        task = {
            'detector': detector,
            'mode': 'z_stack',
            'astigmatism': True,
            'z_range_um': z_range,
            'z_step_um': z_step,
            'filename': filename,
            **kwargs
        }
        
        self.add_task(task)

    def add_balanced_rf_training_set(
        self,
        detector=TDI_PRESET,
        t_poly_series: Sequence[float] = (0.0, 45.0, 90.0),
        num_frames: int = 192,
        spots_per_type: int = 48,
        frame_rate_hz: float = 20.0,
        image_size: Tuple[int, int] = (128, 128),
        enable_photophysics: bool = False,
    ) -> None:
        """Plan balanced Training TIFFs (eine pro Diffusionstyp & Polymerisationszeit).

        Jede Kombination aus Zeit und Diffusionstyp erzeugt einen eigenen TIFF,
        bei dem Switching deaktiviert wird. Dadurch entstehen homogene Segmente
        für das RF-Training.
        """

        diffusion_types = ["normal", "subdiffusion", "confined", "superdiffusion"]

        for t_poly in t_poly_series:
            for dtype in diffusion_types:
                filename = (
                    f"rf_balanced_{detector.name.lower()}_{dtype}_t{int(round(t_poly))}min.tif"
                )

                task = {
                    'detector': detector,
                    'mode': 'rf_balanced',
                    't_poly_min': float(t_poly),
                    'filename': filename,
                    'image_size': image_size,
                    'num_spots': spots_per_type,
                    'num_frames': num_frames,
                    'frame_rate_hz': frame_rate_hz,
                    'enable_photophysics': enable_photophysics,
                    'trajectory_options': {
                        'force_diffusion_type': dtype,
                        'enable_switching': False,
                        'max_switches': 0,
                    },
                }

                self.add_task(task)
    
    def run(self, progress_callback: Callable = None) -> Dict:
        """
        Führt alle Batch-Simulationen aus.
        
        Parameters:
        -----------
        progress_callback : Callable, optional
            Callback-Funktion für Fortschritt: callback(current, total, status)
        
        Returns:
        --------
        Dict : Statistik-Dictionary
        """
        
        if len(self.tasks) == 0:
            print("⚠️  Keine Tasks definiert!")
            return self.stats
        
        print(f"\n🔄 BATCH SIMULATION START")
        print(f"=" * 70)
        print(f"Anzahl Tasks: {len(self.tasks)}")
        print(f"Output Dir: {self.output_dir}")
        print(f"=" * 70)

        self.stats['start_time'] = datetime.now()

        if self.rf_enabled and self.rf_auto_balance:
            self._ensure_balanced_rf_tasks()

        # Fortschrittsbalken mit tqdm
        for idx, task in enumerate(tqdm(self.tasks, desc="Simulationen")):
            try:
                # Status-Update
                if progress_callback:
                    progress_callback(idx + 1, len(self.tasks),
                                    f"Generiere {task['filename']}")

                # Simulation ausführen
                self._run_single_task(task)

                self.stats['completed'] += 1

            except Exception as e:
                print(f"\n❌ FEHLER bei {task['filename']}: {e}")
                self.stats['failed'] += 1

        self.stats['end_time'] = datetime.now()

        if self.rf_trainer:
            rf_info = self.rf_trainer.finalize()
            self.stats['rf'].update({
                'trained': bool(rf_info.get('model_path')),
                'model_path': rf_info.get('model_path'),
                'feature_path': rf_info.get('feature_path'),
                'summary_path': rf_info.get('summary_path'),
                'samples': rf_info.get('samples', 0),
                'training_accuracy': rf_info.get('training_accuracy'),
                'oob_score': rf_info.get('oob_score'),
                'validation_accuracy': rf_info.get('validation_accuracy'),
            })

        # Zusammenfassung
        self._print_summary()

        # Speichere Batch-Statistik
        self._save_batch_stats()

        return self.stats

    def _run_single_task(self, task: Dict) -> None:
        """Führt eine einzelne Simulation aus."""
        
        # Erstelle Simulator
        sim = TIFFSimulator(
            detector=task['detector'],
            mode=task['mode'],
            t_poly_min=task.get('t_poly_min', 60.0),
            astigmatism=task['astigmatism']
        )
        
        # Generiere TIFF
        if task['mode'] == 'z_stack':
            tiff_stack = sim.generate_z_stack(
                image_size=task['image_size'],
                num_spots=task['num_spots'],
                z_range_um=task['z_range_um'],
                z_step_um=task['z_step_um']
            )
        else:
            tiff_stack = sim.generate_tiff(
                image_size=task['image_size'],
                num_spots=task['num_spots'],
                num_frames=task['num_frames'],
                frame_rate_hz=task['frame_rate_hz'],
                d_initial=task.get('d_initial', 0.5),
                exposure_substeps=task.get('exposure_substeps', 1),
                enable_photophysics=task.get('enable_photophysics', False),
                trajectory_options=task.get('trajectory_options')
            )
        
        # Speichere TIFF
        tiff_path = self.output_dir / task['filename']
        save_tiff(str(tiff_path), tiff_stack)
        
        # Exportiere Metadata
        metadata = sim.get_metadata()
        base_filename = Path(task['filename']).stem
        self.metadata_exporter.export_all(metadata, base_filename)

        if self.rf_trainer:
            self.rf_trainer.update_with_metadata(metadata)

    def _ensure_balanced_rf_tasks(self) -> None:
        if self._balanced_training_added:
            return

        if any(task.get('mode') == 'rf_balanced' for task in self.tasks):
            self._balanced_training_added = True
            return

        base_task = self.tasks[0] if self.tasks else {}
        detector = base_task.get('detector', TDI_PRESET)
        image_size = base_task.get('image_size', (128, 128))
        num_frames = max(192, int(base_task.get('num_frames', 192)))
        frame_rate = float(base_task.get('frame_rate_hz', 20.0))
        spots = int(base_task.get('num_spots', 48))
        enable_photophysics = bool(base_task.get('enable_photophysics', False))

        times = sorted({float(task.get('t_poly_min', 60.0)) for task in self.tasks if 't_poly_min' in task})
        if not times:
            times = [0.0, 45.0, 90.0]
        elif len(times) > 3:
            mid_idx = len(times) // 2
            times = [times[0], times[mid_idx], times[-1]]

        spots_per_type = max(24, min(96, spots))

        self.add_balanced_rf_training_set(
            detector=detector,
            t_poly_series=times,
            num_frames=num_frames,
            spots_per_type=spots_per_type,
            frame_rate_hz=frame_rate,
            image_size=image_size,
            enable_photophysics=enable_photophysics,
        )

        self._balanced_training_added = True

    def _print_summary(self) -> None:
        """Druckt Zusammenfassung."""

        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print(f"\n" + "=" * 70)
        print(f"🎉 BATCH SIMULATION ABGESCHLOSSEN")
        print(f"=" * 70)
        print(f"Gesamt: {self.stats['total_tasks']}")
        print(f"✅ Erfolgreich: {self.stats['completed']}")
        print(f"❌ Fehlgeschlagen: {self.stats['failed']}")
        print(f"⏱️  Dauer: {duration:.1f} s ({duration/60:.1f} min)")
        print(f"📁 Output: {self.output_dir}")
        if self.rf_trainer:
            if self.stats['rf']['trained']:
                print(f"🌲 RF Modell: {self.stats['rf']['samples']} Fenster → {self.stats['rf']['model_path']}")
                if self.stats['rf'].get('training_accuracy') is not None:
                    print(f"   📈 Training Accuracy: {self.stats['rf']['training_accuracy']:.4f}")
                if self.stats['rf'].get('oob_score') is not None:
                    print(f"   🧪 OOB-Score: {self.stats['rf']['oob_score']:.4f}")
                if self.stats['rf'].get('validation_accuracy') is not None:
                    print(f"   ✅ OOB-Validierung: {self.stats['rf']['validation_accuracy']:.4f}")
                if self.stats['rf'].get('summary_path'):
                    print(f"   📄 Zusammenfassung: {self.stats['rf']['summary_path']}")
            else:
                print("🌲 RF Modell: Keine gültigen Trajektorienfenster extrahiert.")
        print(f"=" * 70)
    
    def _save_batch_stats(self) -> None:
        """Speichert Batch-Statistik als JSON."""
        
        stats_file = self.output_dir / "batch_statistics.json"
        
        stats_export = {
            'total_tasks': self.stats['total_tasks'],
            'completed': self.stats['completed'],
            'failed': self.stats['failed'],
            'start_time': self.stats['start_time'].isoformat(),
            'end_time': self.stats['end_time'].isoformat(),
            'duration_seconds': (self.stats['end_time'] - self.stats['start_time']).total_seconds(),
            'output_directory': str(self.output_dir),
            'tasks': [
                {
                    'filename': task['filename'],
                    'detector': task['detector'].name,
                    'mode': task['mode'],
                    't_poly_min': task.get('t_poly_min', None)
                }
                for task in self.tasks
            ],
            'random_forest': self.stats.get('rf', {})
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_export, f, indent=2)

        print(f"📊 Statistik gespeichert: {stats_file}")

    def _create_rf_config(self, config_dict: Dict) -> RFTrainingConfig:
        if not config_dict:
            return RFTrainingConfig()

        config = RFTrainingConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# ============================================================================
# VORDEFINIERTE BATCH-KONFIGURATIONEN
# ============================================================================

class PresetBatches:
    """Vordefinierte Batch-Konfigurationen für typische Use-Cases."""

    @staticmethod
    def masterthesis_full(output_dir: str = "./masterthesis_data", **batch_kwargs) -> BatchSimulator:
        """
        Vollständige Parameterstudie für Masterthesis.
        
        Enthält:
        - Polymerisationszeit-Serie (6 Zeitpunkte, 3 Wiederholungen)
        - TDI vs. Tetraspecs Vergleich
        - 3D-Simulationen
        - z-Stack Kalibrierung
        
        Gesamt: ~60 TIFFs
        Dauer: ~30-60 Minuten (abhängig von Hardware)
        """
        
        batch = BatchSimulator(output_dir, **batch_kwargs)
        
        # 1. Polymerisationszeit-Serie (TDI-G0)
        # OPTIMIERT: Inkludiert t=0 für Superdiffusion-Training
        batch.add_polyzeit_series(
            times=[0, 10, 30, 60, 90, 120, 180],
            detector=TDI_PRESET,
            repeats=3,
            image_size=(128, 128),
            num_spots=15,
            num_frames=200,
            frame_rate_hz=20.0
        )
        
        # 2. Detektor-Vergleich bei 60 min
        batch.add_detector_comparison(
            polyzeit=60.0,
            repeats=3,
            image_size=(128, 128),
            num_spots=15,
            num_frames=200
        )
        
        # 3. 3D-Simulationen mit Astigmatismus
        batch.add_3d_series(
            times=[60, 90, 120],
            detector=TDI_PRESET,
            repeats=2,
            image_size=(128, 128),
            num_spots=15,
            num_frames=200
        )
        
        # 4. z-Stack Kalibrierung (beide Detektoren)
        batch.add_z_stack(
            detector=TDI_PRESET,
            z_range=(-1.0, 1.0),
            z_step=0.1,
            image_size=(128, 128),
            num_spots=20
        )
        
        batch.add_z_stack(
            detector=TETRASPECS_PRESET,
            z_range=(-1.0, 1.0),
            z_step=0.1,
            image_size=(128, 128),
            num_spots=20
        )
        
        return batch
    
    @staticmethod
    def quick_test(output_dir: str = "./quick_test", **batch_kwargs) -> BatchSimulator:
        """
        Schneller Test mit wenigen Parametern.
        
        Enthält:
        - 3 Polymerisationszeiten
        - Kleine Bilder (64x64)
        - Wenige Frames (50)
        
        Gesamt: ~3 TIFFs
        Dauer: ~1-2 Minuten
        """
        
        batch = BatchSimulator(output_dir, **batch_kwargs)
        
        batch.add_polyzeit_series(
            times=[30, 60, 90],
            detector=TDI_PRESET,
            repeats=1,
            image_size=(64, 64),
            num_spots=5,
            num_frames=50,
            frame_rate_hz=20.0
        )
        
        return batch
    
    @staticmethod
    def publication_quality(output_dir: str = "./publication_data", **batch_kwargs) -> BatchSimulator:
        """
        Publication-Quality Daten.
        
        Enthält:
        - Hohe Auflösung (256x256)
        - Viele Spots (50)
        - Lange Zeitreihen (500 Frames)
        - Mehrere Wiederholungen (5)
        
        Gesamt: ~30 TIFFs
        Dauer: ~2-3 Stunden
        """
        
        batch = BatchSimulator(output_dir, **batch_kwargs)
        
        batch.add_polyzeit_series(
            times=[30, 60, 90, 120, 180],
            detector=TDI_PRESET,
            repeats=5,
            image_size=(256, 256),
            num_spots=50,
            num_frames=500,
            frame_rate_hz=20.0
        )
        
        return batch


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """CLI für Batch-Simulator."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🔄 TIFF Batch Simulator V3.0'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['quick', 'thesis', 'publication'],
        default='quick',
        help='Vordefinierte Batch-Konfiguration'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./batch_output',
        help='Output-Verzeichnis'
    )

    parser.add_argument(
        '--train-rf',
        action='store_true',
        help='Aktiviert begleitendes Random-Forest-Training'
    )
    parser.add_argument(
        '--rf-window',
        type=int,
        default=48,
        help='Fenstergröße für Sliding-Window-Features'
    )
    parser.add_argument(
        '--rf-window-sizes',
        type=str,
        default='32,48,64',
        help='Komma-separierte Liste zusätzlicher Fenstergrößen (Frames)'
    )
    parser.add_argument(
        '--rf-step',
        type=int,
        default=16,
        help='Schrittweite zwischen Fenstern'
    )
    parser.add_argument(
        '--rf-step-fraction',
        type=float,
        default=0.5,
        help='Schrittweite relativ zur Fenstergröße (0.0-1.0, 0 deaktiviert)'
    )
    parser.add_argument(
        '--rf-estimators',
        type=int,
        default=1024,
        help='Anzahl Bäume des Random Forest'
    )
    parser.add_argument(
        '--rf-max-depth',
        type=int,
        default=28,
        help='Maximale Tiefe der Bäume (<=0 für unbegrenzt)'
    )
    parser.add_argument(
        '--rf-min-leaf',
        type=int,
        default=3,
        help='Minimale Samples pro Blatt'
    )
    parser.add_argument(
        '--rf-min-split',
        type=int,
        default=6,
        help='Minimale Samples pro Split'
    )
    parser.add_argument(
        '--rf-random-state',
        type=int,
        default=42,
        help='Random-State für Reproduzierbarkeit'
    )
    parser.add_argument(
        '--rf-max-samples',
        type=float,
        default=0.85,
        help='Anteil der Fenster pro Baum (0<value<=1, <=0 deaktiviert Unterabtastung)'
    )
    parser.add_argument(
        '--rf-max-windows-per-class',
        type=int,
        default=100_000,
        help='Maximale Anzahl gespeicherter Sliding-Window-Samples pro Diffusionsklasse'
    )
    parser.add_argument(
        '--rf-max-windows-per-track',
        type=int,
        default=600,
        help='Maximale Anzahl gespeicherter Sliding-Window-Samples pro Trajektorie'
    )
    parser.add_argument(
        '--rf-min-majority',
        type=float,
        default=0.7,
        help='Minimale Mehrheitsfraktion eines Labels pro Fenster (0-1)'
    )
    parser.add_argument(
        '--rf-max-switches',
        type=int,
        default=1,
        help='Maximale Anzahl Labelwechsel innerhalb eines Fensters (<=0 deaktiviert Filter)'
    )
    parser.add_argument(
        '--rf-no-auto-balance',
        action='store_true',
        help='Deaktiviert automatische Balanced-RF-Trainingssimulationen'
    )

    args, unknown = parser.parse_known_args()

    batch_kwargs = {}
    if args.train_rf:
        max_depth = None if args.rf_max_depth is not None and args.rf_max_depth <= 0 else args.rf_max_depth
        max_samples = None if args.rf_max_samples is None or args.rf_max_samples <= 0 else min(1.0, float(args.rf_max_samples))
        window_sizes = []
        for part in args.rf_window_sizes.split(','):
            part = part.strip()
            if not part:
                continue
            try:
                window_sizes.append(int(part))
            except ValueError:
                continue
        if not window_sizes:
            window_sizes = [args.rf_window]

        window_sizes = sorted({max(3, int(w)) for w in window_sizes})
        median_index = len(window_sizes) // 2
        primary_window = window_sizes[median_index]

        batch_kwargs = {
            'enable_rf': True,
            'rf_config': {
                'window_size': primary_window,
                'window_sizes': tuple(window_sizes),
                'step_size': max(0, args.rf_step),
                'step_size_fraction': max(0.0, min(1.0, args.rf_step_fraction)),
                'n_estimators': max(10, args.rf_estimators),
                'max_depth': max_depth,
                'min_samples_leaf': max(1, args.rf_min_leaf),
                'min_samples_split': max(2, args.rf_min_split),
                'random_state': args.rf_random_state,
                'max_samples': max_samples,
                'max_windows_per_class': max(0, args.rf_max_windows_per_class),
                'max_windows_per_track': max(0, args.rf_max_windows_per_track),
                'min_majority_fraction': max(0.5, min(0.95, args.rf_min_majority)),
                'max_label_switches': max(0, args.rf_max_switches),
                'auto_balance_training': not args.rf_no_auto_balance,
            }
        }

    # Wähle Preset
    # Optional: Custom Zeiten via --times (unknown args)
    custom_times = None
    custom_detector = 'tdi'
    custom_repeats = 1
    try:
        for i, tok in enumerate(unknown):
            if tok == '--times' and i + 1 < len(unknown):
                import re as _re
                parts = _re.split(r'[,:;\\s]+', unknown[i+1].strip())
                custom_times = [float(p) for p in parts if p]
            elif tok.startswith('--times='):
                import re as _re
                parts = _re.split(r'[,:;\\s]+', tok.split('=',1)[1].strip())
                custom_times = [float(p) for p in parts if p]
            elif tok == '--detector' and i + 1 < len(unknown):
                if unknown[i+1].lower() in ('tdi', 'tetraspecs'):
                    custom_detector = unknown[i+1].lower()
            elif tok.startswith('--detector='):
                val = tok.split('=',1)[1].lower()
                if val in ('tdi', 'tetraspecs'):
                    custom_detector = val
            elif tok == '--repeats' and i + 1 < len(unknown):
                custom_repeats = int(unknown[i+1])
            elif tok.startswith('--repeats='):
                custom_repeats = int(tok.split('=',1)[1])
    except Exception:
        custom_times = custom_times  # ignore parse errors silently

    if custom_times is not None:
        det = TDI_PRESET if custom_detector == 'tdi' else TETRASPECS_PRESET
        batch = BatchSimulator(args.output, **batch_kwargs)
        batch.add_polyzeit_series(times=custom_times, detector=det, repeats=custom_repeats)
        print('Custom Times: ' + str(custom_times) + ' | Detector: ' + det.name + ' | Repeats: ' + str(custom_repeats))
        batch.run()
        return

    if args.preset == 'quick':
        batch = PresetBatches.quick_test(args.output, **batch_kwargs)
        print("🚀 Quick Test Batch ausgewählt")
    elif args.preset == 'thesis':
        batch = PresetBatches.masterthesis_full(args.output, **batch_kwargs)
        print("🎓 Masterthesis Batch ausgewählt")
    else:
        batch = PresetBatches.publication_quality(args.output, **batch_kwargs)
        print("📄 Publication Quality Batch ausgewählt")

    if args.train_rf:
        print("🌲 Random-Forest-Training aktiviert: Modell wird parallel zu den Simulationen aufgebaut.")

    # Run
    batch.run()


if __name__ == "__main__":
    main()
