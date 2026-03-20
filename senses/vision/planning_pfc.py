"""Planning PFC vision processing.

Builds a coarse, memory-aware map from the large 15×15 grid observation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grid_utils import (
    GridSpec,
    attach_overlays,
    downsample_square_feature_grid,
    reshape_floor_slice,
    slice_to_feature_grid,
)


def get_planning_pfc_vision(
    block_values: Sequence[str],
    *,
    spec: Optional[GridSpec] = None,
    risk_map: Optional[np.ndarray] = None,
    reward_map: Optional[np.ndarray] = None,
    safe_zone_map: Optional[np.ndarray] = None,
    agent_density_map: Optional[np.ndarray] = None,
    block_size: int = 3,
) -> dict:
    """Return the coarse planning map and supporting tensors."""

    block_matrix = reshape_floor_slice(block_values, spec=spec)
    feature_grid = slice_to_feature_grid(block_matrix)

    overlays = {
        "risk_map": risk_map,
        "reward_map": reward_map,
        "safe_zone_map": safe_zone_map,
        "agent_density_map": agent_density_map,
    }
    for name, overlay in overlays.items():
        if overlay is not None and overlay.shape != block_matrix.shape:
            raise ValueError(f"{name} shape must match block grid shape")

    augmented_grid = attach_overlays(
        feature_grid,
        risk=risk_map,
        reward=reward_map,
        safe=safe_zone_map,
        agents=agent_density_map,
    )

    coarse_map = downsample_square_feature_grid(augmented_grid, block_size=block_size)

    # Prepare stats for planners (global context signals).
    summary = {
        "danger_mean": float(feature_grid[..., 1].mean()),
        "danger_max": float(feature_grid[..., 1].max()),
        "walkable_ratio": float((feature_grid[..., 0] > 0.5).mean()),
        "unknown_ratio": float((feature_grid[..., 6] > 0.5).mean()),
    }

    return {
        "block_ids": block_matrix,
        "feature_grid": feature_grid,
        "augmented_grid": augmented_grid,
        "coarse_map": coarse_map,
        "summary": summary,
    }
