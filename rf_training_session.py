"""High-level orchestration for dedicated RF training sessions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import json

from tiff_simulator_v3 import TIFFSimulator, TDI_PRESET, save_tiff
from metadata_exporter import MetadataExporter
from rf_trainer import RandomForestTrainer, RFTrainingConfig


DIFFUSION_TYPES_DEFAULT = ("normal", "subdiffusion", "confined", "superdiffusion")


@dataclass
class RFTrainingSessionConfig:
    """Configuration for a standalone RF training session."""

    output_dir: str = "./rf_training_session"
    detector: object = TDI_PRESET
    polymerization_times: Sequence[float] = (0.0, 45.0, 90.0)
    diffusion_types: Sequence[str] = field(default_factory=lambda: list(DIFFUSION_TYPES_DEFAULT))
    frames_per_track: Sequence[int] = (96,)
    tracks_per_diffusion: int = 64
    frame_rate_hz: float = 20.0
    astigmatism: bool = False
    enable_photophysics: bool = False
    training_mode: str = "window"
    polygrade_strategy: str = "per_grade"
    rf_config_overrides: Dict[str, object] = field(default_factory=dict)
    random_seed: Optional[int] = 42
    save_tiffs: bool = False
    export_metadata: bool = True
    metadata_formats: Sequence[str] = ("json",)


class RFTrainingSession:
    """Generate balanced simulations and train RF models without batch mode."""

    def __init__(self, config: RFTrainingSessionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rf_config = RFTrainingConfig()
        for key, value in config.rf_config_overrides.items():
            if hasattr(rf_config, key):
                setattr(rf_config, key, value)

        rf_config.training_mode = config.training_mode
        rf_config.polygrade_strategy = config.polygrade_strategy

        self.trainer = RandomForestTrainer(self.output_dir, rf_config)

        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True, parents=True)
        self.metadata_exporter = MetadataExporter(metadata_dir)

        self.simulation_log: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        progress_callback: Optional[Callable[[int, int, Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        """Generate simulations for all configured combos and train the RF."""

        combos = self._iter_combinations()
        total = len(combos)
        for idx, (t_poly, diffusion, frames) in enumerate(combos, start=1):
            metadata, tiff_path = self._simulate_single(t_poly, diffusion, frames)
            self.trainer.update_with_metadata(metadata)

            self.simulation_log.append({
                "poly_time_min": float(t_poly),
                "diffusion_type": diffusion,
                "frames": int(frames),
                "tracks": int(self.config.tracks_per_diffusion),
                "metadata_path": metadata.get("__exported_metadata__"),
                "tiff_path": tiff_path,
            })

            if progress_callback:
                progress_callback(
                    idx,
                    total,
                    {
                        "poly_time_min": float(t_poly),
                        "diffusion_type": diffusion,
                        "frames": int(frames),
                    },
                )

        trainer_summary = self.trainer.finalize()

        report = {
            "config": asdict(self.config),
            "trainer_summary": trainer_summary,
            "simulations": self.simulation_log,
        }

        session_report_path = self.output_dir / "rf_training_session_report.json"
        with open(session_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        report["session_report_path"] = str(session_report_path)
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _iter_combinations(self) -> List[Tuple[float, str, int]]:
        times = [float(t) for t in self.config.polymerization_times]
        types = list(self.config.diffusion_types) or list(DIFFUSION_TYPES_DEFAULT)
        frames = [int(max(8, f)) for f in self.config.frames_per_track]

        combinations: List[Tuple[float, str, int]] = []
        for t_poly in times:
            for diffusion in types:
                for frame_count in frames:
                    combinations.append((t_poly, diffusion, frame_count))
        return combinations

    def _simulate_single(
        self,
        t_poly: float,
        diffusion_type: str,
        num_frames: int,
    ) -> Tuple[Dict[str, object], Optional[str]]:
        simulator = TIFFSimulator(
            detector=self.config.detector,
            mode="rf_training",
            t_poly_min=float(t_poly),
            astigmatism=self.config.astigmatism,
        )

        trajectory_options = {
            "force_diffusion_type": diffusion_type,
            "enable_switching": False,
            "max_switches": 0,
        }

        if self.config.training_mode == "track":
            trajectory_options["max_switches"] = 0

        stack = simulator.generate_tiff(
            image_size=(128, 128),
            num_spots=self.config.tracks_per_diffusion,
            num_frames=int(num_frames),
            frame_rate_hz=float(self.config.frame_rate_hz),
            enable_photophysics=self.config.enable_photophysics,
            trajectory_options=trajectory_options,
        )

        metadata = simulator.get_metadata()
        metadata.setdefault("session", {})
        metadata["session"].update({
            "poly_time_min": float(t_poly),
            "diffusion_type": diffusion_type,
            "num_frames": int(num_frames),
            "tracks_per_diffusion": int(self.config.tracks_per_diffusion),
        })

        exported_metadata_path: Optional[str] = None
        if self.config.export_metadata:
            base_name = f"rftrain_{diffusion_type}_t{int(round(t_poly))}m_{num_frames}f"
            exported_metadata_path = self.metadata_exporter.export_json(metadata, base_name)
        metadata["__exported_metadata__"] = exported_metadata_path

        tiff_path: Optional[str] = None
        if self.config.save_tiffs:
            tiff_filename = f"rftrain_{diffusion_type}_t{int(round(t_poly))}m_{num_frames}f.tif"
            tiff_full_path = self.output_dir / tiff_filename
            save_tiff(str(tiff_full_path), stack)
            tiff_path = str(tiff_full_path)

        return metadata, tiff_path
