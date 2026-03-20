"""Fast PFC vision processing.

Converts the 5×5 `ObservationFromGrid` slice into a compact feature embedding
for the fast, near-reactive prefrontal system.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grid_utils import GridSpec, reshape_floor_slice, slice_to_feature_grid


def get_fast_pfc_vision(
    block_values: Sequence[str],
    *,
    spec: Optional[GridSpec] = None,
) -> dict:
    """Return block tensor, feature tensor and flattened embedding."""

    block_matrix = reshape_floor_slice(block_values, spec=spec)
    feature_grid = slice_to_feature_grid(block_matrix)

    embedding = feature_grid.reshape(-1).astype(np.float32)
    summary = {
        "mean_danger": float(feature_grid[..., 1].mean()),
        "mean_walkable": float(feature_grid[..., 0].mean()),
        "max_reward": float(feature_grid[..., 2].max()),
    }

    return {
        "block_ids": block_matrix,
        "feature_grid": feature_grid,
        "embedding": embedding,
        "summary": summary,
    }
