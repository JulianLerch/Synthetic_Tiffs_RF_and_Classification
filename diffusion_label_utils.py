"""Utility helpers for handling diffusion label timelines."""

from __future__ import annotations

from typing import Dict, List, Sequence

DiffusionLabel = str


def expand_labels(
    initial_label: DiffusionLabel,
    switch_log: Sequence[Dict],
    num_frames: int,
) -> List[DiffusionLabel]:
    """Expand a trajectory's switch log to frame-level labels.

    Parameters
    ----------
    initial_label:
        Label of the trajectory before any switches.
    switch_log:
        Iterable of switch entries with ``frame`` and ``to`` keys.
    num_frames:
        Total number of frames for which labels should be produced.

    Returns
    -------
    list[str]
        A list of labels with length ``num_frames`` where each entry
        corresponds to the diffusion state of the respective frame.
    """

    expanded = [initial_label] * max(0, num_frames)

    if not switch_log:
        return expanded

    current_label = initial_label
    for entry in sorted(switch_log, key=lambda e: e.get("frame", 0)):
        frame_idx = int(entry.get("frame", 0))
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        new_label = entry.get("to", current_label) or current_label
        if new_label == current_label:
            continue
        current_label = new_label
        expanded[frame_idx:] = [current_label] * (num_frames - frame_idx)

    return expanded


__all__ = ["expand_labels", "DiffusionLabel"]
