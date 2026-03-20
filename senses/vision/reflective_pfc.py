"""Reflective PFC vision processing.

Transforms the 7×7 `ObservationFromGrid` slice into tensors that feed the
slower, learned behaviour controller (typically a small CNN/MLP hybrid).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grid_utils import GridSpec, reshape_floor_slice, slice_to_feature_grid


def get_reflective_pfc_vision(
    block_values: Sequence[str],
    *,
    spec: Optional[GridSpec] = None,
) -> dict:
    """Return tensors suited for reflective processing."""

    block_matrix = reshape_floor_slice(block_values, spec=spec)
    feature_grid = slice_to_feature_grid(block_matrix)

    # Channel-first tensor for CNNs
    spatial_tensor = np.transpose(feature_grid, (2, 0, 1)).astype(np.float32)
    embedding = feature_grid.reshape(-1).astype(np.float32)

    stats = {
        "danger_mean": float(feature_grid[..., 1].mean()),
        "danger_max": float(feature_grid[..., 1].max()),
        "walkable_ratio": float((feature_grid[..., 0] > 0.5).mean()),
        "unknown_ratio": float((feature_grid[..., 6] > 0.5).mean()),
    }

    return {
        "block_ids": block_matrix,
        "feature_grid": feature_grid,
        "spatial_tensor": spatial_tensor,
        "embedding": embedding,
        "stats": stats,
    }
