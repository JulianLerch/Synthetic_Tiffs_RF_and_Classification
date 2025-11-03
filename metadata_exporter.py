"""
📋 METADATA EXPORTER
===================

Exportiert vollständige Simulations-Metadata in verschiedenen Formaten:
- JSON (maschinenlesbar)
- TXT (menschenlesbar)
- CSV (für Batch-Analysen)

Version: 3.0 - Oktober 2025
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np


class MetadataExporter:
    """
    Exportiert Metadata in verschiedenen Formaten.
    
    Unterstützte Formate:
    - JSON: Vollständige Metadata, verschachtelt
    - TXT: Lesbare Zusammenfassung
    - CSV: Tabellarische Daten für Batch-Analysen
    """
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def export_json(self, metadata: Dict, filename: str) -> str:
        """
        Exportiert Metadata als JSON.
        
        Parameters:
        -----------
        metadata : Dict
            Metadata-Dictionary
        filename : str
            Dateiname (ohne .json)
        
        Returns:
        --------
        str : Pfad zur JSON-Datei
        """
        
        filepath = self.output_dir / f"{filename}_metadata.json"
        
        # Konvertiere numpy types zu Python types
        clean_metadata = self._clean_for_json(metadata)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_metadata, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_txt(self, metadata: Dict, filename: str) -> str:
        """
        Exportiert Metadata als lesbares TXT.
        
        Returns:
        --------
        str : Pfad zur TXT-Datei
        """
        
        filepath = self.output_dir / f"{filename}_metadata.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("TIFF SIMULATION - METADATA\n")
            f.write("=" * 70 + "\n\n")
            
            # Zeitstempel
            f.write(f"Generiert: {metadata.get('timestamp', 'N/A')}\n")
            f.write(f"Dateiname: {filename}\n\n")
            
            # Detektor
            f.write("DETEKTOR\n")
            f.write("-" * 70 + "\n")
            f.write(f"Name: {metadata.get('detector', 'N/A')}\n")
            if 'psf' in metadata:
                psf = metadata['psf']
                f.write(f"FWHM: {psf.get('fwhm_um', 'N/A'):.3f} µm\n")
                f.write(f"Pixel Size: {psf.get('pixel_size_um', 'N/A'):.3f} µm\n")
                f.write(f"Astigmatismus: {'Ja' if psf.get('astigmatism') else 'Nein'}\n")
            f.write("\n")
            
            # Simulationsparameter
            f.write("SIMULATIONSPARAMETER\n")
            f.write("-" * 70 + "\n")
            f.write(f"Modus: {metadata.get('mode', 'N/A')}\n")
            
            if 'image_size' in metadata:
                h, w = metadata['image_size']
                f.write(f"Bildgröße: {w} × {h} px\n")
            
            f.write(f"Anzahl Spots: {metadata.get('num_spots', 'N/A')}\n")
            
            if 'num_frames' in metadata:
                f.write(f"Anzahl Frames: {metadata.get('num_frames', 'N/A')}\n")
                f.write(f"Frame Rate: {metadata.get('frame_rate_hz', 'N/A'):.1f} Hz\n")
                duration = metadata['num_frames'] / metadata['frame_rate_hz']
                f.write(f"Gesamt-Dauer: {duration:.2f} s\n")
            
            if 'z_range_um' in metadata:
                z_min, z_max = metadata['z_range_um']
                f.write(f"z-Range: [{z_min:.2f}, {z_max:.2f}] µm\n")
                f.write(f"z-Step: {metadata.get('z_step_um', 'N/A'):.3f} µm\n")
                f.write(f"Anzahl z-Slices: {metadata.get('n_slices', 'N/A')}\n")
            
            f.write("\n")
            
            # Diffusion
            if 'diffusion' in metadata:
                diff = metadata['diffusion']
                f.write("DIFFUSIONSPARAMETER\n")
                f.write("-" * 70 + "\n")
                f.write(f"Polymerisationszeit: {diff.get('t_poly_min', 'N/A'):.1f} min\n")
                f.write(f"D_initial: {diff.get('D_initial', 'N/A'):.3f} µm²/s\n")
                f.write(f"Frame Rate: {diff.get('frame_rate_hz', 'N/A'):.1f} Hz\n\n")
                
                # D-Werte
                f.write("Diffusionskoeffizienten:\n")
                if 'D_values' in diff:
                    for dtype, D in diff['D_values'].items():
                        f.write(f"  D_{dtype}: {D:.4f} µm²/s\n")
                f.write("\n")
                
                # Fraktionen
                f.write("Diffusionsfraktionen:\n")
                if 'diffusion_fractions' in diff:
                    for dtype, frac in diff['diffusion_fractions'].items():
                        f.write(f"  {dtype}: {frac*100:.1f}%\n")
                f.write("\n")

                realized = diff.get('realized_fractions')
                if realized:
                    frame_counts = diff.get('realized_frame_counts', {})
                    total_frames = diff.get('realized_total_frames')
                    f.write("Realisierte Diffusionsfraktionen:\n")
                    for dtype, frac in realized.items():
                        frames = frame_counts.get(dtype, 0)
                        f.write(f"  {dtype}: {frac*100:.1f}% ({frames} Frames)\n")
                    if total_frames is not None:
                        f.write(f"  Gesamt ausgewertete Frames: {total_frames}\n")
                    f.write("\n")
            
            # Trajektorien-Statistik
            if 'trajectories' in metadata:
                trajs = metadata['trajectories']
                f.write("TRAJEKTORIEN-STATISTIK\n")
                f.write("-" * 70 + "\n")
                f.write(f"Anzahl Trajektorien: {len(trajs)}\n\n")
                
                # Pro Diffusionstyp
                types_count = {}
                for traj in trajs:
                    dtype = traj.get('diffusion_type', 'unknown')
                    types_count[dtype] = types_count.get(dtype, 0) + 1
                
                f.write("Verteilung nach Typ:\n")
                for dtype, count in sorted(types_count.items()):
                    percentage = (count / len(trajs)) * 100
                    f.write(f"  {dtype}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n")
            f.write("=" * 70 + "\n")
        
        return str(filepath)
    
    def export_csv_row(self, metadata: Dict, filename: str, 
                      csv_filepath: str = None) -> str:
        """
        Fügt Metadata-Zeile zu CSV hinzu (für Batch-Analysen).
        
        Returns:
        --------
        str : Pfad zur CSV-Datei
        """
        
        if csv_filepath is None:
            csv_filepath = self.output_dir / "batch_metadata.csv"
        else:
            csv_filepath = Path(csv_filepath)
        
        # Extrahiere wichtige Werte
        row = {
            'filename': filename,
            'timestamp': metadata.get('timestamp', ''),
            'detector': metadata.get('detector', ''),
            'mode': metadata.get('mode', ''),
            'image_width': metadata.get('image_size', [0, 0])[1],
            'image_height': metadata.get('image_size', [0, 0])[0],
            'num_spots': metadata.get('num_spots', 0),
            'num_frames': metadata.get('num_frames', 0),
            'frame_rate_hz': metadata.get('frame_rate_hz', 0),
            't_poly_min': metadata.get('diffusion', {}).get('t_poly_min', 0),
            'D_initial': metadata.get('diffusion', {}).get('D_initial', 0),
            'astigmatism': metadata.get('astigmatism', False)
        }
        
        # D-Werte hinzufügen
        if 'diffusion' in metadata and 'D_values' in metadata['diffusion']:
            for dtype, D in metadata['diffusion']['D_values'].items():
                row[f'D_{dtype}'] = D
        
        # Fraktionen hinzufügen
        if 'diffusion' in metadata:
            diff = metadata['diffusion']
            theory = diff.get('diffusion_fractions', {})
            realized = diff.get('realized_fractions', {})
            frame_counts = diff.get('realized_frame_counts', {})
            total_frames = diff.get('realized_total_frames')
            if total_frames is not None:
                row['realized_total_frames'] = int(total_frames)

            all_labels = sorted(set().union(theory.keys(), realized.keys(), frame_counts.keys()))
            for dtype in all_labels:
                row[f'frac_{dtype}_theory_pct'] = theory.get(dtype, 0.0) * 100.0
                row[f'frac_{dtype}_real_pct'] = realized.get(dtype, 0.0) * 100.0
                row[f'frames_{dtype}'] = int(frame_counts.get(dtype, 0))
        
        # CSV erstellen oder erweitern
        file_exists = csv_filepath.exists()
        
        with open(csv_filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        return str(csv_filepath)
    
    def export_all(self, metadata: Dict, filename: str) -> Dict[str, str]:
        """
        Exportiert Metadata in allen Formaten.
        
        Returns:
        --------
        Dict[str, str] : Dictionary mit Pfaden zu allen Dateien
        """
        
        paths = {
            'json': self.export_json(metadata, filename),
            'txt': self.export_txt(metadata, filename),
            'csv': self.export_csv_row(metadata, filename)
        }
        
        return paths
    
    def _clean_for_json(self, obj):
        """Konvertiert numpy types zu Python types für JSON."""
        
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._clean_for_json(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            return self._clean_for_json(obj.tolist())
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("📋 Metadata Exporter Test")
    
    # Test-Metadata
    test_metadata = {
        'timestamp': datetime.now().isoformat(),
        'detector': 'TDI-G0',
        'mode': 'polyzeit',
        'image_size': (128, 128),
        'num_spots': 10,
        'num_frames': 100,
        'frame_rate_hz': 20.0,
        'astigmatism': False,
        'diffusion': {
            't_poly_min': 60.0,
            'D_initial': 4.0,
            'D_values': {
                'normal': 0.503,
                'subdiffusion': 0.302
            },
            'diffusion_fractions': {
                'normal': 0.65,
                'subdiffusion': 0.24
            }
        }
    }
    
    exporter = MetadataExporter("./test_output")
    paths = exporter.export_all(test_metadata, "test_simulation")
    
    print("\n✅ Export erfolgreich!")
    for format_name, path in paths.items():
        print(f"   {format_name.upper()}: {path}")
