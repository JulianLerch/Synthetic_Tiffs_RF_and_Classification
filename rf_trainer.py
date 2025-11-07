"""Random Forest Trainer for Diffusion Classification.

Dieses Modul extrahiert ein reichhaltiges Set von Sliding-Window-Features
aus den simulierten Trajektorien und trainiert darauf einen extrem
leistungsstarken Random-Forest-Klassifikator, der zwischen vier
Diffusionsarten unterscheidet. Das Training ist so ausgelegt, dass es über
mehrere Polymerisationszeiten hinweg robuste Modelle erzeugt, umfangreiche
Leistungsmetriken (Accuracy, OOB-Score, Confusion Matrix) sammelt und das
finale Modell inklusive Feature-Daten in das Batch-Output-Verzeichnis
schreibt.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, NamedTuple
import random

import json
import csv

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from diffusion_label_utils import expand_labels


DiffusionLabel = str


@dataclass
class RFTrainingConfig:
    """Konfigurationsparameter für das Random-Forest-Training.

    OPTIMIZED DEFAULTS (V4.1):
    - Reduziertes Overlapping (step_size = 32 statt 16) für weniger Data Leakage
    - Höhere n_estimators (2048 statt 1024) für robustere Ensemble-Performance
    - Moderate Regularisierung (max_depth=20, min_samples_leaf=5, min_samples_split=10)
    """

    window_size: int = 48
    window_sizes: Tuple[int, ...] = (32, 48, 64)
    step_size: Optional[int] = None
    step_size_fraction: float = 0.5
    n_estimators: int = 2048  # OPTIMIERT: Mehr Bäume (war 1024)
    max_depth: Optional[int] = 20      # OPTIMIERT: Konservativer (war 28)
    min_samples_leaf: int = 5          # OPTIMIERT: Stärker regularisiert (war 3)
    min_samples_split: int = 10        # OPTIMIERT: Stärker regularisiert (war 6)
    max_features: str = "sqrt"
    bootstrap: bool = True
    oob_score: bool = True
    random_state: int = 42
    class_weight: str = "balanced_subsample"
    max_samples: Optional[float] = 0.85
    max_windows_per_class: Optional[int] = 100_000
    max_windows_per_track: Optional[int] = 600
    min_majority_fraction: float = 0.7
    max_label_switches: int = 1
    min_segment_length_factor: float = 0.75
    training_mode: str = "window"  # "window" oder "track"
    polygrade_strategy: str = "combined"  # "combined", "per_grade", "auto"

    def __post_init__(self) -> None:
        sizes = self.window_sizes or (self.window_size,)
        normalized = sorted({int(max(3, s)) for s in sizes})
        if not normalized:
            normalized = [max(3, int(self.window_size))]
        if self.window_size not in normalized:
            normalized.append(int(self.window_size))
            normalized = sorted({int(max(3, s)) for s in normalized})
        self.window_sizes = tuple(normalized)

        mode = (self.training_mode or "window").strip().lower()
        if mode not in {"window", "track"}:
            mode = "window"
        self.training_mode = mode

        strategy = (self.polygrade_strategy or "combined").strip().lower()
        if strategy not in {"combined", "per_grade", "auto"}:
            strategy = "combined"
        self.polygrade_strategy = strategy

    @property
    def smallest_window(self) -> int:
        return int(self.window_sizes[0])

    def iter_windows(self) -> Iterable[int]:
        for size in self.window_sizes:
            yield int(size)

    def step_for_window(self, window: int) -> int:
        if self.step_size is not None and self.step_size > 0:
            return max(1, min(int(window), int(self.step_size)))
        frac = max(0.05, float(self.step_size_fraction or 0.5))
        return max(1, int(round(window * frac)))


class SampleMeta(NamedTuple):
    start_frame: int
    end_frame: int
    poly_time_min: Optional[float]
    frame_rate_hz: Optional[float]
    label: DiffusionLabel
    track_index: int
    track_length: int
    window_size: int
    majority_fraction: float
    label_switches: int
    sample_type: str


class RandomForestTrainer:
    """Extrahiert Features aus Trajektorien und trainiert einen Random Forest."""

    FEATURE_NAMES: Sequence[str] = (
        "mean_step_xy",
        "std_step_xy",
        "median_step_xy",
        "mad_step_xy",
        "max_step_xy",
        "min_step_xy",
        "mean_step_z",
        "std_step_z",
        "msd_lag1",
        "msd_lag2",
        "msd_lag4",
        "msd_lag8",
        "msd_loglog_slope",
        "straightness",
        "confinement_radius",
        "radius_of_gyration",
        "gyration_asymmetry",
        "turning_angle_mean",
        "turning_angle_std",
        "step_p90",
        "step_p10",
        "bounding_box_area",
        "axial_range",
        "directional_persistence",
        "velocity_autocorr",
        "step_skewness",
        "step_kurtosis",
    )

    def __init__(self, output_dir: Path, config: Optional[RFTrainingConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.config = config or RFTrainingConfig()

        self.features: List[List[float]] = []
        self.labels: List[DiffusionLabel] = []
        self.sample_meta: List[SampleMeta] = []
        self.model: Optional[RandomForestClassifier] = None
        self.models_per_grade: Dict[str, RandomForestClassifier] = {}
        self._feature_matrix: Optional[np.ndarray] = None
        self._needs_fit: bool = False
        self._class_seen_counts: Counter = Counter()
        self._class_kept_counts: Counter = Counter()
        self._class_indices: Dict[DiffusionLabel, List[int]] = defaultdict(list)
        self._frame_label_counts: Counter = Counter()
        self._polygrade_indices: Dict[str, List[int]] = defaultdict(list)
        self._polygrade_values: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_with_metadata(self, metadata: Dict) -> None:
        """Extrahiert Features aus Metadata und aktualisiert das Modell."""

        trajectories = metadata.get("trajectories", [])
        if not trajectories:
            return

        poly_time = metadata.get("diffusion", {}).get("t_poly_min", None)
        frame_rate = metadata.get("diffusion", {}).get("frame_rate_hz", None)

        initial_count = len(self.features)

        min_required = self.config.smallest_window if self.config.training_mode == "window" else 3

        for track_idx, traj in enumerate(trajectories):
            positions = np.asarray(traj.get("positions"))
            if positions.ndim != 2 or positions.shape[0] < min_required:
                continue

            labels = expand_labels(
                traj.get("diffusion_type", "unknown"),
                traj.get("switch_log", []),
                positions.shape[0],
            )

            self._frame_label_counts.update(labels)

            self._collect_samples(
                positions,
                labels,
                poly_time=poly_time,
                frame_rate=frame_rate,
                track_index=track_idx,
            )

        if len(self.features) > initial_count:
            self._needs_fit = True
            self._feature_matrix = None

    def finalize(self) -> Dict[str, Optional[str]]:
        """Speichert Modell, Feature-Daten und Konfiguration."""

        if not self.features:
            return {
                "model_path": None,
                "feature_path": None,
                "summary_path": None,
                "samples": 0,
                "training_accuracy": None,
                "oob_score": None,
            }

        self._ensure_model_fitted()

        if self.model is None:
            return {
                "model_path": None,
                "feature_path": None,
                "summary_path": None,
                "samples": len(self.features),
                "training_accuracy": None,
                "oob_score": None,
            }

        model_path = self.output_dir / "random_forest_diffusion.joblib"
        models_dict: Dict[str, RandomForestClassifier] = {"combined": self.model}
        models_dict.update(self.models_per_grade)

        model_index = {"combined": {"type": "combined", "poly_time_min": None}}
        for key, model in self.models_per_grade.items():
            model_index[key] = {
                "type": "polygrade",
                "poly_time_min": self._polygrade_values.get(key),
                "classes": list(getattr(model, "classes_", [])),
            }

        artifact = {
            "model": self.model,
            "models": models_dict,
            "default_model_key": "combined",
            "model_index": model_index,
            "feature_names": list(self.FEATURE_NAMES),
            "config": asdict(self.config),
            "polygrade_values": dict(self._polygrade_values),
        }

        dump(artifact, model_path)

        # Kopiere RF Usage Guide
        self._copy_rf_usage_guide()

        feature_path = self._export_feature_table()
        metrics = self._collect_training_metrics()
        summary_path = self._export_training_summary(model_path, feature_path, metrics)

        return {
            "model_path": str(model_path),
            "feature_path": feature_path,
            "summary_path": summary_path,
            "samples": len(self.features),
            "training_accuracy": metrics.get("training_accuracy"),
            "oob_score": metrics.get("oob_score"),
            "validation_accuracy": metrics.get("validation_accuracy"),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _collect_samples(
        self,
        positions: np.ndarray,
        labels: Sequence[DiffusionLabel],
        *,
        poly_time: Optional[float],
        frame_rate: Optional[float],
        track_index: int,
    ) -> None:
        if self.config.training_mode == "track":
            self._collect_track_sample(
                positions,
                labels,
                poly_time=poly_time,
                frame_rate=frame_rate,
                track_index=track_index,
            )
        else:
            self._collect_windows(
                positions,
                labels,
                poly_time=poly_time,
                frame_rate=frame_rate,
                track_index=track_index,
            )

    def _collect_windows(
        self,
        positions: np.ndarray,
        labels: Sequence[DiffusionLabel],
        *,
        poly_time: Optional[float],
        frame_rate: Optional[float],
        track_index: int,
    ) -> None:
        max_per_track = self.config.max_windows_per_track or 0
        windows_taken = 0

        for window in self.config.iter_windows():
            if positions.shape[0] < window:
                continue

            step = self.config.step_for_window(window)

            for start in range(0, positions.shape[0] - window + 1, step):
                end = start + window
                window_pos = positions[start:end]
                window_labels = labels[start:end]

                feature_vec = self._compute_features(window_pos)
                if feature_vec is None:
                    continue

                label_summary = self._summarize_window_labels(window_labels)
                if label_summary is None:
                    continue

                target_label, majority_fraction, switch_count = label_summary

                if majority_fraction < self.config.min_majority_fraction:
                    continue
                if (
                    self.config.max_label_switches is not None
                    and switch_count > self.config.max_label_switches
                ):
                    continue

                meta = SampleMeta(
                    start_frame=start,
                    end_frame=end,
                    poly_time_min=poly_time,
                    frame_rate_hz=frame_rate,
                    label=target_label,
                    track_index=track_index,
                    track_length=positions.shape[0],
                    window_size=window,
                    majority_fraction=majority_fraction,
                    label_switches=switch_count,
                    sample_type="window",
                )

                if self._maybe_store_sample(feature_vec.tolist(), target_label, meta):
                    windows_taken += 1

                if max_per_track and windows_taken >= max_per_track:
                    break
            if max_per_track and windows_taken >= max_per_track:
                break

    def _compute_features(self, window_pos: np.ndarray) -> Optional[np.ndarray]:
        steps = np.diff(window_pos, axis=0)
        if steps.size == 0:
            return None

        step_xy = np.linalg.norm(steps[:, :2], axis=1)
        step_z = np.abs(steps[:, 2])

        mean_step_xy = float(np.mean(step_xy))
        std_step_xy = float(np.std(step_xy))
        median_step_xy = float(np.median(step_xy))
        mad_step_xy = float(np.median(np.abs(step_xy - median_step_xy)))
        max_step_xy = float(np.max(step_xy))
        min_step_xy = float(np.min(step_xy))

        def _msd(lag: int) -> float:
            if lag >= window_pos.shape[0]:
                return 0.0
            diff = window_pos[lag:] - window_pos[:-lag]
            return float(np.mean(np.sum(diff**2, axis=1)))

        straight_dist = np.linalg.norm(window_pos[-1, :2] - window_pos[0, :2])
        total_path = np.sum(step_xy) + 1e-9
        straightness = straight_dist / total_path

        xy_centered = window_pos[:, :2] - np.mean(window_pos[:, :2], axis=0)
        confinement = float(np.sqrt(np.mean(np.sum(xy_centered**2, axis=1))))

        radius_of_gyration, gyration_asymmetry = self._gyration_statistics(window_pos)

        turning_angles = self._compute_turning_angles(steps[:, :2])
        turning_angle_mean = float(np.mean(turning_angles) if turning_angles.size else 0.0)
        directional_persistence = self._directional_persistence(steps[:, :2])

        velocity_autocorr = self._velocity_autocorr(step_xy)

        msd_lag1 = _msd(1)
        msd_lag2 = _msd(2)
        msd_lag4 = _msd(4)
        msd_lag8 = _msd(8)
        msd_slope = self._loglog_msd_slope([1, 2, 4, 8], [msd_lag1, msd_lag2, msd_lag4, msd_lag8])

        step_skewness, step_kurtosis = self._moment_statistics(step_xy, mean_step_xy, std_step_xy)

        bbox = window_pos[:, :2]
        bbox_area = float(
            (bbox[:, 0].max() - bbox[:, 0].min()) *
            (bbox[:, 1].max() - bbox[:, 1].min())
        )

        axial_range = float(window_pos[:, 2].max() - window_pos[:, 2].min())

        feature_values = np.array([
            mean_step_xy,
            std_step_xy,
            median_step_xy,
            mad_step_xy,
            max_step_xy,
            min_step_xy,
            float(np.mean(step_z)),
            float(np.std(step_z)),
            msd_lag1,
            msd_lag2,
            msd_lag4,
            msd_lag8,
            msd_slope,
            float(straightness),
            confinement,
            radius_of_gyration,
            gyration_asymmetry,
            turning_angle_mean,
            float(np.std(turning_angles) if turning_angles.size else 0.0),
            float(np.percentile(step_xy, 90)),
            float(np.percentile(step_xy, 10)),
            bbox_area,
            axial_range,
            directional_persistence,
            velocity_autocorr,
            step_skewness,
            step_kurtosis,
        ], dtype=np.float32)

        if np.any(~np.isfinite(feature_values)):
            return None

        return feature_values

    def _collect_track_sample(
        self,
        positions: np.ndarray,
        labels: Sequence[DiffusionLabel],
        *,
        poly_time: Optional[float],
        frame_rate: Optional[float],
        track_index: int,
    ) -> None:
        feature_vec = self._compute_features(positions)
        if feature_vec is None:
            return

        label_summary = self._summarize_window_labels(labels)
        if label_summary is None:
            return

        target_label, majority_fraction, switch_count = label_summary

        if majority_fraction < self.config.min_majority_fraction:
            return
        if (
            self.config.max_label_switches is not None
            and switch_count > self.config.max_label_switches
        ):
            return

        meta = SampleMeta(
            start_frame=0,
            end_frame=positions.shape[0],
            poly_time_min=poly_time,
            frame_rate_hz=frame_rate,
            label=target_label,
            track_index=track_index,
            track_length=positions.shape[0],
            window_size=positions.shape[0],
            majority_fraction=majority_fraction,
            label_switches=switch_count,
            sample_type="track",
        )

        self._maybe_store_sample(feature_vec.tolist(), target_label, meta)

    def _compute_turning_angles(self, step_vectors: np.ndarray) -> np.ndarray:
        if step_vectors.shape[0] < 2:
            return np.array([], dtype=np.float32)

        v1 = step_vectors[:-1]
        v2 = step_vectors[1:]

        norms = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-12
        dots = np.sum(v1 * v2, axis=1)
        cos_angles = np.clip(dots / norms, -1.0, 1.0)
        return np.arccos(cos_angles).astype(np.float32)

    def _directional_persistence(self, step_vectors: np.ndarray) -> float:
        if step_vectors.shape[0] < 2:
            return 0.0

        v1 = step_vectors[:-1]
        v2 = step_vectors[1:]
        norms = (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)) + 1e-12
        cos_vals = np.sum(v1 * v2, axis=1) / norms
        return float(np.mean(cos_vals))

    def _velocity_autocorr(self, step_magnitudes: np.ndarray) -> float:
        if step_magnitudes.size < 2:
            return 0.0

        a = step_magnitudes[:-1]
        b = step_magnitudes[1:]
        if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
            return 0.0

        corr = np.corrcoef(a, b)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(corr)

    def _gyration_statistics(self, positions: np.ndarray) -> Tuple[float, float]:
        centered = positions - np.mean(positions, axis=0)
        cov = np.dot(centered.T, centered) / max(1, centered.shape[0])
        eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))
        radius = float(np.sqrt(np.sum(np.clip(eigvals, 0.0, None))))
        asymmetry = 0.0
        denom = float(np.sum(np.abs(eigvals))) + 1e-12
        if denom > 0:
            asymmetry = float((eigvals[-1] - eigvals[0]) / denom)
        return radius, asymmetry

    def _loglog_msd_slope(self, lags: Sequence[int], msd_values: Sequence[float]) -> float:
        lags_arr = np.asarray(lags, dtype=np.float32)
        msd_arr = np.asarray(msd_values, dtype=np.float32)
        mask = msd_arr > 0
        if np.count_nonzero(mask) < 2:
            return 0.0

        log_lags = np.log(lags_arr[mask])
        log_msds = np.log(msd_arr[mask])
        x_mean = float(np.mean(log_lags))
        y_mean = float(np.mean(log_msds))
        denom = float(np.sum((log_lags - x_mean) ** 2))
        if denom < 1e-12:
            return 0.0
        slope = float(np.sum((log_lags - x_mean) * (log_msds - y_mean)) / denom)
        return slope

    def _moment_statistics(self, values: np.ndarray, mean: float, std: float) -> Tuple[float, float]:
        if std < 1e-12:
            return 0.0, 0.0

        centered = (values - mean) / std
        skewness = float(np.mean(centered**3))
        kurtosis = float(np.mean(centered**4) - 3.0)
        return skewness, kurtosis

    def _summarize_window_labels(
        self, labels: Sequence[DiffusionLabel]
    ) -> Optional[Tuple[DiffusionLabel, float, int]]:
        if not labels:
            return None

        label_counts = Counter(labels)
        most_common = label_counts.most_common()
        if not most_common:
            return None

        top_count = most_common[0][1]
        tied = [lab for lab, cnt in most_common if cnt == top_count]

        if len(tied) == 1:
            winner = tied[0]
        else:
            center_label = labels[len(labels) // 2]
            winner = center_label if center_label in tied else tied[0]

        majority_fraction = top_count / float(len(labels))
        label_switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])

        return winner, majority_fraction, label_switches

    def _ensure_model_fitted(self) -> None:
        if not self.features:
            return

        if self.model is None or self._needs_fit:
            self._fit_models()
            self._needs_fit = False

    def _fit_models(self) -> None:
        X = np.asarray(self.features, dtype=np.float32)
        y = np.array(self.labels)

        self.model = self._train_single_model(X, y)
        self._feature_matrix = X

        self.models_per_grade = {}

        strategy = self.config.polygrade_strategy
        polygrade_keys = list(self._polygrade_indices.keys())
        if not polygrade_keys:
            return

        if strategy == "combined":
            return

        if strategy == "auto":
            if len(polygrade_keys) < 2:
                return

        for key in polygrade_keys:
            indices = self._polygrade_indices.get(key) or []
            if len(indices) < max(20, self.config.min_samples_leaf * 10):
                continue
            idx_arr = np.array(indices, dtype=int)
            X_subset = X[idx_arr]
            y_subset = y[idx_arr]
            if len(np.unique(y_subset)) < 2:
                continue
            self.models_per_grade[key] = self._train_single_model(X_subset, y_subset)

    def _train_single_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        rf_kwargs = dict(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            min_samples_split=self.config.min_samples_split,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            n_jobs=-1,
            random_state=self.config.random_state,
            bootstrap=self.config.bootstrap,
            oob_score=self.config.oob_score,
            max_samples=self.config.max_samples,
        )

        model = RandomForestClassifier(**rf_kwargs)
        model.fit(X, y)
        return model

    def _maybe_store_sample(
        self,
        feature_vec: List[float],
        label: DiffusionLabel,
        meta: SampleMeta,
    ) -> bool:
        limit = self.config.max_windows_per_class or 0
        self._class_seen_counts[label] += 1

        if limit <= 0 or self._class_kept_counts[label] < limit:
            idx = len(self.features)
            self.features.append(feature_vec)
            self.labels.append(label)
            self.sample_meta.append(meta)
            self._class_indices[label].append(idx)
            self._class_kept_counts[label] += 1
            self._assign_polygrade_index(idx, meta.poly_time_min)
            self._needs_fit = True
            self._feature_matrix = None
            return True

        # Reservoir sampling to keep class balance when limit is exceeded
        j = random.randint(0, self._class_seen_counts[label] - 1)
        if j < limit:
            replace_idx = self._class_indices[label][j]
            self.features[replace_idx] = feature_vec
            self.labels[replace_idx] = label
            self.sample_meta[replace_idx] = meta
            self._assign_polygrade_index(replace_idx, meta.poly_time_min)
            self._needs_fit = True
            self._feature_matrix = None
            return True

        return False

    def _assign_polygrade_index(self, idx: int, poly_time: Optional[float]) -> None:
        for key, indices in list(self._polygrade_indices.items()):
            if idx in indices:
                indices.remove(idx)
            if not indices:
                self._polygrade_indices.pop(key, None)

        key = self._polygrade_key(poly_time)
        if key:
            self._polygrade_indices[key].append(idx)
            if poly_time is not None:
                self._polygrade_values[key] = float(poly_time)

    def _polygrade_key(self, poly_time: Optional[float]) -> Optional[str]:
        if poly_time is None:
            return None
        rounded = round(float(poly_time), 3)
        return f"poly_{rounded:.3f}"

    def _export_feature_table(self) -> str:
        feature_path = self.output_dir / "rf_training_features.csv"
        header = [
            "label",
            *self.FEATURE_NAMES,
            "poly_time_min",
            "frame_rate_hz",
            "track_index",
            "track_length",
            "window_size",
            "sample_type",
            "majority_fraction",
            "label_switches",
        ]

        with open(feature_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for label, feats, meta in zip(self.labels, self.features, self.sample_meta):
                feature_array = np.asarray(feats, dtype=np.float64)
                writer.writerow([
                    label,
                    *np.round(feature_array, 6),
                    meta.poly_time_min,
                    meta.frame_rate_hz,
                    meta.track_index,
                    meta.track_length,
                    meta.window_size,
                    meta.sample_type,
                    round(meta.majority_fraction, 4),
                    meta.label_switches,
                ])

        return str(feature_path)

    def _collect_training_metrics(self) -> Dict:
        self._ensure_model_fitted()

        if self.model is None:
            return {}

        X = (
            self._feature_matrix
            if self._feature_matrix is not None
            else np.asarray(self.features, dtype=np.float32)
        )
        y = np.array(self.labels)

        predictions = self.model.predict(X)
        accuracy = float(np.mean(predictions == y)) if y.size else None

        labels_sorted = list(self.model.classes_) if y.size else []
        confusion = None
        report = None
        validation_confusion = None
        validation_report = None
        validation_accuracy = None

        if y.size and labels_sorted:
            confusion = confusion_matrix(y, predictions, labels=labels_sorted).astype(int).tolist()
            report = classification_report(
                y,
                predictions,
                labels=labels_sorted,
                target_names=labels_sorted,
                output_dict=True,
                zero_division=0,
            )

        oob_score = None
        if getattr(self.model, "oob_score_", None) is not None and self.config.oob_score:
            oob_score = float(self.model.oob_score_)
            oob_decision = getattr(self.model, "oob_decision_function_", None)
            if oob_decision is not None:
                oob_decision = np.asarray(oob_decision)
                if oob_decision.ndim == 2 and oob_decision.size:
                    valid_mask = ~np.all(oob_decision <= 0, axis=1)
                    if np.any(valid_mask):
                        oob_probs = oob_decision[valid_mask]
                        y_valid = y[valid_mask]
                        oob_preds = self.model.classes_[np.argmax(oob_probs, axis=1)]
                        validation_accuracy = float(np.mean(oob_preds == y_valid))
                        validation_confusion = confusion_matrix(
                            y_valid,
                            oob_preds,
                            labels=labels_sorted,
                        ).astype(int).tolist()
                        validation_report = classification_report(
                            y_valid,
                            oob_preds,
                            labels=labels_sorted,
                            target_names=labels_sorted,
                            output_dict=True,
                            zero_division=0,
                        )

        window_counts_by_poly = Counter()
        window_counts_by_size = Counter()
        label_switch_hist = Counter()
        for meta in self.sample_meta:
            key = "unknown" if meta.poly_time_min is None else str(meta.poly_time_min)
            window_counts_by_poly[key] += 1
            window_counts_by_size[meta.window_size] += 1
            label_switch_hist[meta.label_switches] += 1

        majority_fractions = [meta.majority_fraction for meta in self.sample_meta]
        avg_majority_fraction = float(np.mean(majority_fractions)) if majority_fractions else None

        polygrade_summary = {
            key: {
                "poly_time_min": self._polygrade_values.get(key),
                "samples": len(indices),
                "model_trained": key in self.models_per_grade,
            }
            for key, indices in self._polygrade_indices.items()
        }

        return {
            "samples": len(self.features),
            "labels": dict(Counter(self.labels)),
            "feature_names": list(self.FEATURE_NAMES),
            "feature_importances": self.model.feature_importances_.tolist(),
            "training_accuracy": accuracy,
            "oob_score": oob_score,
            "validation_accuracy": validation_accuracy,
            "confusion_matrix": confusion,
            "confusion_matrix_labels": labels_sorted,
            "classification_report": report,
            "validation_confusion_matrix": validation_confusion,
            "validation_classification_report": validation_report,
            "windows_seen_per_class": dict(self._class_seen_counts),
            "windows_kept_per_class": dict(self._class_kept_counts),
            "window_counts_by_poly": dict(window_counts_by_poly),
            "window_counts_by_size": dict(window_counts_by_size),
            "label_switch_histogram": dict(label_switch_hist),
            "mean_majority_fraction": avg_majority_fraction,
            "frame_label_counts": dict(self._frame_label_counts),
            "polygrade_models": polygrade_summary,
        }

    def _copy_rf_usage_guide(self) -> None:
        """Kopiert die RF Usage Guide in das Output-Verzeichnis."""
        try:
            import shutil
            source = Path(__file__).parent / "RF_USAGE_GUIDE.md"
            if source.exists():
                dest = self.output_dir / "RF_USAGE_GUIDE.md"
                shutil.copy2(source, dest)
                print(f"✓ RF Usage Guide kopiert nach: {dest}")
        except Exception as e:
            print(f"⚠ Konnte RF Usage Guide nicht kopieren: {e}")

    def _export_training_summary(self, model_path: Path, feature_path: str, metrics: Dict) -> str:
        summary = {
            **metrics,
            "model_path": str(model_path),
            "feature_path": feature_path,
            "config": asdict(self.config),
        }

        summary_path = self.output_dir / "rf_training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return str(summary_path)

