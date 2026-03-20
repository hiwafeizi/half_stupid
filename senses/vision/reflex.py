"""Reflex vision processing.

Takes the 3×3 `ObservationFromGrid` slice positioned under the agent and converts
raw block identifiers into lightweight semantic features suitable for the reflex
controller.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grid_utils import GridSpec, reshape_floor_slice, slice_to_feature_grid


def get_reflex_vision(
    block_values: Sequence[str],
    *,
    spec: Optional[GridSpec] = None,
    sound_map: Optional[np.ndarray] = None,
):
    """Return a dictionary of reflex-friendly features.

    Parameters
    ----------
    block_values:
        Flattened list returned by Malmo for the reflex-sized observation (3×3).
    spec:
        Optional `GridSpec` describing the observation window. If omitted we
        infer a square 3×3 slice.
    sound_map:
        Optional 2D array of sound proximity scores (same footprint as the
        reshaped grid). Use this when an audio sensor is active.
    """

    block_matrix = reshape_floor_slice(block_values, spec=spec)
    feature_grid = slice_to_feature_grid(block_matrix)

    if sound_map is not None and sound_map.shape != block_matrix.shape:
        raise ValueError("sound_map shape must match block grid shape")

    cells = []
    for z in range(block_matrix.shape[0]):
        row = []
        for x in range(block_matrix.shape[1]):
            features = feature_grid[z, x]
            row.append(
                {
                    "block": block_matrix[z, x],
                    "walkable": float(features[0]),
                    "danger": float(features[1]),
                    "reward": float(features[2]),
                    "unknown": float(features[6]),
                    "sound": float(sound_map[z, x]) if sound_map is not None else 0.0,
                }
            )
        cells.append(row)

    return {
        "block_ids": block_matrix,
        "feature_grid": feature_grid,
        "cells": cells,
    }
