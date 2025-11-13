"""
TrackMate XML Exporter
======================

Exportiert simulierte Trajektorien im TrackMate XML Format.

Features:
- Pixel → µm Konvertierung
- z-Positionen aus Astigmatismus-Metadata
- Kompatibel mit TrackMate v7.x
- Aggregate über multiple TIFFs

Version: 1.0 - Januar 2025
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np


class TrackMateExporter:
    """
    Exportiert Trajektorien im TrackMate XML Format.

    TrackMate Format:
    -----------------
    <Tracks spaceUnits="µm" timeUnits="sec" frameInterval="1.0">
        <particle nSpots="123">
            <detection t="0" x="1.23" y="4.56" z="0.0"/>
            <detection t="1" x="1.25" y="4.58" z="0.0"/>
            ...
        </particle>
    </Tracks>
    """

    def __init__(self, pixel_size_um: float = 0.11):
        """
        Parameters:
        -----------
        pixel_size_um : float
            Pixelgröße in µm (default: 0.11 µm/pixel)
        """
        self.pixel_size_um = pixel_size_um

    def export_tracks(
        self,
        tracks_data: List[Dict],
        output_path: Path,
        frame_interval_sec: float = 0.05,
        has_astigmatism: bool = False
    ) -> None:
        """
        Exportiert Tracks als TrackMate XML.

        Parameters:
        -----------
        tracks_data : List[Dict]
            Liste von Trajektorien. Jeder Track ist ein Dict mit:
            - 'positions_px': np.array shape (n_frames, 2) [x, y] in Pixel
            - 'z_positions_um': np.array shape (n_frames,) [z] in µm (optional)
            - 'frames': np.array shape (n_frames,) Frame-Indices
        output_path : Path
            Pfad zur XML-Datei
        frame_interval_sec : float
            Zeit zwischen Frames [sec]
        has_astigmatism : bool
            Ob z-Positionen enthalten sind
        """

        # Root Element
        root = ET.Element('Tracks')
        root.set('spaceUnits', 'µm')
        root.set('timeUnits', 'sec')
        root.set('frameInterval', f'{frame_interval_sec:.2f}')
        root.set('from', 'TrackMate v7.14.0')
        root.set('nTracks', str(len(tracks_data)))

        # Tracks durchgehen
        for track_idx, track in enumerate(tracks_data):
            positions_px = track['positions_px']  # Shape: (n_spots, 2)
            frames = track['frames']  # Shape: (n_spots,)
            z_positions_um = track.get('z_positions_um', None)  # Shape: (n_spots,)

            n_spots = len(frames)

            # Particle Element
            particle = ET.SubElement(root, 'particle')
            particle.set('nSpots', str(n_spots))

            # Detections
            for i, frame_idx in enumerate(frames):
                # Pixel → µm Konvertierung
                x_um = positions_px[i, 0] * self.pixel_size_um
                y_um = positions_px[i, 1] * self.pixel_size_um

                # z-Position
                if has_astigmatism and z_positions_um is not None:
                    z_um = z_positions_um[i]
                else:
                    z_um = 0.0

                # Detection Element
                detection = ET.SubElement(particle, 'detection')
                detection.set('t', str(int(frame_idx)))
                detection.set('x', f'{x_um:.15f}')  # Hohe Präzision
                detection.set('y', f'{y_um:.15f}')
                detection.set('z', f'{z_um:.15f}')

        # Pretty Print XML
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')

        # Schreibe Datei
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        print(f"✅ TrackMate XML exportiert: {output_path}")
        print(f"   Tracks: {len(tracks_data)}")
        print(f"   Format: TrackMate v7.14.0")
        print(f"   Astigmatismus: {'Ja (mit z)' if has_astigmatism else 'Nein (z=0)'}")

    def load_tracks_from_metadata(
        self,
        metadata_paths: List[Path]
    ) -> Tuple[List[Dict], bool]:
        """
        Lädt Tracks aus Metadata JSON-Files.

        Parameters:
        -----------
        metadata_paths : List[Path]
            Liste von metadata.json Dateien

        Returns:
        --------
        tracks_data : List[Dict]
            Liste von Tracks
        has_astigmatism : bool
            Ob Astigmatismus aktiv war
        """

        all_tracks = []
        has_astigmatism = False
        frame_offset = 0  # Frame offset für mehrere TIFFs

        for meta_path in metadata_paths:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # Check Astigmatismus
            if metadata.get('astigmatism', False):
                has_astigmatism = True

            # Extrahiere Trajektorien
            trajectories = metadata.get('trajectories', [])

            for traj in trajectories:
                # Positions können 2D [x, y] oder 3D [x, y, z] sein
                positions = np.array(traj['positions'])  # Shape: (n_spots, 2 oder 3)

                # Determine dimensionality
                if positions.ndim == 2 and positions.shape[1] == 3:
                    # 3D positions [x, y, z]
                    positions_px = positions[:, :2]  # Extrahiere x, y
                    z_positions_um = positions[:, 2]  # Extrahiere z
                elif positions.ndim == 2 and positions.shape[1] == 2:
                    # 2D positions [x, y]
                    positions_px = positions
                    z_positions_um = None
                else:
                    # Fallback: assume 2D
                    positions_px = positions
                    z_positions_um = None

                # Generate frame indices (0, 1, 2, ..., n_spots-1) + offset
                n_spots = len(positions_px)
                frames = np.arange(n_spots) + frame_offset

                track = {
                    'positions_px': positions_px,
                    'frames': frames,
                    'z_positions_um': z_positions_um
                }

                all_tracks.append(track)

            # Frame offset erhöhen für nächstes TIFF
            num_frames = metadata.get('num_frames', 0)
            frame_offset += num_frames

        return all_tracks, has_astigmatism

    def export_from_metadata_files(
        self,
        metadata_paths: List[Path],
        output_path: Path,
        frame_rate_hz: float = 20.0
    ) -> None:
        """
        Convenience Funktion: Lädt Tracks aus Metadata und exportiert XML.

        Parameters:
        -----------
        metadata_paths : List[Path]
            Liste von metadata.json Dateien
        output_path : Path
            Pfad zur XML-Datei
        frame_rate_hz : float
            Framerate [Hz]
        """

        # Lade Tracks
        tracks, has_astig = self.load_tracks_from_metadata(metadata_paths)

        # Frame interval
        frame_interval_sec = 1.0 / frame_rate_hz

        # Exportiere
        self.export_tracks(
            tracks_data=tracks,
            output_path=output_path,
            frame_interval_sec=frame_interval_sec,
            has_astigmatism=has_astig
        )


if __name__ == "__main__":
    # Test
    print("🧪 TrackMate Exporter Test")
    print("=" * 60)

    # Erstelle Test-Daten
    test_track_1 = {
        'positions_px': np.array([[10.0, 20.0], [10.5, 20.3], [11.0, 20.1]]),
        'frames': np.array([0, 1, 2]),
        'z_positions_um': np.array([0.0, 0.05, -0.02])
    }

    test_track_2 = {
        'positions_px': np.array([[50.0, 60.0], [50.2, 59.8]]),
        'frames': np.array([5, 6]),
        'z_positions_um': np.array([0.1, 0.15])
    }

    exporter = TrackMateExporter(pixel_size_um=0.11)

    test_path = Path("./test_tracks.xml")
    exporter.export_tracks(
        tracks_data=[test_track_1, test_track_2],
        output_path=test_path,
        frame_interval_sec=0.05,
        has_astigmatism=True
    )

    print(f"\n📄 Test XML erstellt: {test_path}")
    print("Öffne die Datei zum Überprüfen!")
