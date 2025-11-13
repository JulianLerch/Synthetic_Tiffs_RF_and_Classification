"""
🔄 BATCH SIMULATOR
==================

Automatisierte Generierung mehrerer TIFF-Simulationen mit systematischer
Parametervariation für umfassende Parameterstudien.

Anwendungsfälle:
- Polymerisationszeit-Serien (t = 0, 30, 60, 90, 120 min)
- Detektor-Vergleiche (TDI-G0 vs. Tetraspecs)
- D-Wert-Studien (verschiedene D_initial)
- Reproduzierbarkeit (n Wiederholungen gleicher Parameter)

Version: 4.0 - November 2025
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Callable, Tuple
from datetime import datetime
import json
from tqdm import tqdm

try:
    from tiff_simulator_v3 import (
        TDI_PRESET, TETRASPECS_PRESET, TIFFSimulator, save_tiff
    )
    from metadata_exporter import MetadataExporter
    from trackmate_exporter import TrackMateExporter
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Stelle sicher, dass tiff_simulator_v3.py, metadata_exporter.py")
    print("   und trackmate_exporter.py im gleichen Ordner sind!")
    exit(1)


class BatchSimulator:
    """
    Automatisierte Batch-Simulation mit Fortschrittsanzeige.

    Workflow:
    ---------
    1. Definition der Parametervariationen
    2. Systematische Generierung aller Kombinationen
    3. Parallel-Export von TIFF + Metadata
    4. Zusammenfassung in Master-JSON

    Beispiel:
    ---------
    >>> batch = BatchSimulator(output_dir="my_simulations")
    >>> batch.add_polyzeit_series(times=[30, 60, 90], detector=TDI_PRESET)
    >>> batch.run()
    """

    def __init__(self, output_dir: str = "./batch_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.tasks = []
        self.metadata_exporter = MetadataExporter(self.output_dir)
        self.trackmate_exporter = TrackMateExporter(pixel_size_um=0.11)

        # Track completed tasks by polyzeit for XML export
        self.completed_tasks_by_polyzeit = {}  # {polyzeit: [task_info, ...]}

        # Statistik
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
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

        # Generiere TrackMate XML-Dateien für jede Polyzeit
        self._generate_trackmate_xmls()

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
                enable_photophysics=task.get('enable_photophysics', False)
            )

        # Determine output subfolder based on polyzeit
        # Group by polyzeit (not for z_stack mode)
        if task['mode'] != 'z_stack' and 't_poly_min' in task:
            polyzeit = task['t_poly_min']
            polyzeit_folder = self.output_dir / f"polyzeit_{int(polyzeit)}min"
            polyzeit_folder.mkdir(exist_ok=True, parents=True)
            output_folder = polyzeit_folder
        else:
            # z_stack and other modes go to root
            output_folder = self.output_dir

        # Speichere TIFF
        tiff_path = output_folder / task['filename']
        save_tiff(str(tiff_path), tiff_stack)

        # Exportiere Metadata
        metadata = sim.get_metadata()
        base_filename = Path(task['filename']).stem

        # Temporarily change metadata_exporter output_dir for this file
        original_export_dir = self.metadata_exporter.output_dir
        self.metadata_exporter.output_dir = output_folder
        self.metadata_exporter.export_all(metadata, base_filename)
        self.metadata_exporter.output_dir = original_export_dir

        # Track completed task for XML export
        if task['mode'] != 'z_stack' and 't_poly_min' in task:
            polyzeit = task['t_poly_min']
            if polyzeit not in self.completed_tasks_by_polyzeit:
                self.completed_tasks_by_polyzeit[polyzeit] = []

            self.completed_tasks_by_polyzeit[polyzeit].append({
                'metadata_path': output_folder / f"{base_filename}_metadata.json",
                'tiff_path': tiff_path,
                'output_folder': output_folder,
                'frame_rate_hz': task.get('frame_rate_hz', 20.0),
                'has_astigmatism': task['astigmatism']
            })

    def _generate_trackmate_xmls(self) -> None:
        """
        Generiert TrackMate XML-Dateien für jede Polyzeit.

        Aggregiert alle Tracks aller Wiederholungen einer Polyzeit in eine XML.
        """

        if len(self.completed_tasks_by_polyzeit) == 0:
            return

        print(f"\n📊 GENERIERE TRACKMATE XML-DATEIEN")
        print(f"=" * 70)

        for polyzeit, tasks in sorted(self.completed_tasks_by_polyzeit.items()):
            try:
                # Collect all metadata paths for this polyzeit
                metadata_paths = [task['metadata_path'] for task in tasks]

                # Determine output path (in polyzeit folder)
                output_folder = tasks[0]['output_folder']
                xml_filename = f"trackmate_polyzeit_{int(polyzeit)}min_all_tracks.xml"
                xml_path = output_folder / xml_filename

                # Get frame rate (should be same for all tasks in this polyzeit)
                frame_rate_hz = tasks[0]['frame_rate_hz']

                # Check if any task has astigmatism
                has_astigmatism = any(task['has_astigmatism'] for task in tasks)

                # Generate XML
                print(f"  Polyzeit {int(polyzeit)} min: {len(tasks)} Datei(en) → {xml_filename}")

                self.trackmate_exporter.export_from_metadata_files(
                    metadata_paths=metadata_paths,
                    output_path=xml_path,
                    frame_rate_hz=frame_rate_hz
                )

            except Exception as e:
                print(f"  ❌ Fehler bei Polyzeit {int(polyzeit)} min: {e}")

        print(f"=" * 70)

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
            ]
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_export, f, indent=2)

        print(f"📊 Statistik gespeichert: {stats_file}")


# ============================================================================
# VORDEFINIERTE BATCH-KONFIGURATIONEN
# ============================================================================

class PresetBatches:
    """Vordefinierte Batch-Konfigurationen für typische Use-Cases."""

    @staticmethod
    def masterthesis_full(output_dir: str = "./masterthesis_data") -> BatchSimulator:
        """
        Vollständige Parameterstudie für Masterthesis.

        Enthält:
        - Polymerisationszeit-Serie (7 Zeitpunkte, 3 Wiederholungen)
        - TDI vs. Tetraspecs Vergleich
        - 3D-Simulationen
        - z-Stack Kalibrierung

        Gesamt: ~60 TIFFs
        Dauer: ~30-60 Minuten (abhängig von Hardware)
        """

        batch = BatchSimulator(output_dir)

        # 1. Polymerisationszeit-Serie (TDI-G0)
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
    def quick_test(output_dir: str = "./quick_test") -> BatchSimulator:
        """
        Schneller Test mit wenigen Parametern.

        Enthält:
        - 3 Polymerisationszeiten
        - Kleine Bilder (64x64)
        - Wenige Frames (50)

        Gesamt: ~3 TIFFs
        Dauer: ~1-2 Minuten
        """

        batch = BatchSimulator(output_dir)

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
    def publication_quality(output_dir: str = "./publication_data") -> BatchSimulator:
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

        batch = BatchSimulator(output_dir)

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
        description='🔄 TIFF Batch Simulator V4.0'
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

    args, unknown = parser.parse_known_args()

    # Optional: Custom Zeiten via --times
    custom_times = None
    custom_detector = 'tdi'
    custom_repeats = 1
    try:
        for i, tok in enumerate(unknown):
            if tok == '--times' and i + 1 < len(unknown):
                import re as _re
                parts = _re.split(r'[,:;\s]+', unknown[i+1].strip())
                custom_times = [float(p) for p in parts if p]
            elif tok.startswith('--times='):
                import re as _re
                parts = _re.split(r'[,:;\s]+', tok.split('=',1)[1].strip())
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
        pass

    if custom_times is not None:
        det = TDI_PRESET if custom_detector == 'tdi' else TETRASPECS_PRESET
        batch = BatchSimulator(args.output)
        batch.add_polyzeit_series(times=custom_times, detector=det, repeats=custom_repeats)
        print('Custom Times: ' + str(custom_times) + ' | Detector: ' + det.name + ' | Repeats: ' + str(custom_repeats))
        batch.run()
        return

    # Wähle Preset
    if args.preset == 'quick':
        batch = PresetBatches.quick_test(args.output)
        print("🚀 Quick Test Batch ausgewählt")
    elif args.preset == 'thesis':
        batch = PresetBatches.masterthesis_full(args.output)
        print("🎓 Masterthesis Batch ausgewählt")
    else:
        batch = PresetBatches.publication_quality(args.output)
        print("📄 Publication Quality Batch ausgewählt")

    # Run
    batch.run()


if __name__ == "__main__":
    main()
