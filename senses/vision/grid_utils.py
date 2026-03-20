"""Helpers for reshaping Malmo grid observations into structured arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .tile_features import batch_tiles_to_vectors


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    depth: int

    @property
    def total(self) -> int:
        return self.width * self.height * self.depth


def infer_2d_spec(grid_values: Sequence[str]) -> GridSpec:
    """Infers a square 2D grid spec (height=1) from the observation length."""
    length = len(grid_values)
    width = int(round(length ** 0.5))
    if width * width != length:
        raise ValueError(f"Cannot infer square grid from {length} entries")
    return GridSpec(width=width, height=1, depth=width)


def reshape_floor_slice(grid_values: Sequence[str], *, spec: Optional[GridSpec] = None) -> np.ndarray:
    """Reshape a single-layer (height=1) observation into [depth, width] array of block ids."""
    if spec is None:
        spec = infer_2d_spec(grid_values)
    if spec.height != 1:
        raise ValueError("reshape_floor_slice expects height == 1")
    if len(grid_values) != spec.total:
        raise ValueError("Observation length does not match grid spec")

    matrix = np.array(grid_values, dtype=object)
    matrix = matrix.reshape((spec.depth, spec.width), order="C")
    return matrix


def slice_to_feature_grid(block_matrix: np.ndarray) -> np.ndarray:
    """Convert a 2D array of block ids into a [depth, width, features] float array."""
    flat_vectors = batch_tiles_to_vectors(block_matrix.flatten())
    depth, width = block_matrix.shape
    return flat_vectors.reshape((depth, width, -1))


def summarise_region(features: np.ndarray) -> np.ndarray:
    """Summarise a region of feature vectors into a single feature vector.

    Currently uses simple mean pooling. Extend with custom statistics as needed.
    """
    return features.mean(axis=(0, 1))


def downsample_square_feature_grid(feature_grid: np.ndarray, *, block_size: int) -> np.ndarray:
    """Downsample a square feature grid by averaging non-overlapping blocks."""
    depth, width, channels = feature_grid.shape
    if depth != width:
        raise ValueError("Expected square grid for downsampling")
    if depth % block_size != 0:
        raise ValueError("Grid size must be divisible by block size")

    new_size = depth // block_size
    downsampled = np.zeros((new_size, new_size, channels), dtype=np.float32)
    for i in range(new_size):
        for j in range(new_size):
            region = feature_grid[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
                :,
            ]
            downsampled[i, j] = summarise_region(region)
    return downsampled


def attach_overlays(base_features: np.ndarray, *, risk: Optional[np.ndarray] = None,
                     reward: Optional[np.ndarray] = None, safe: Optional[np.ndarray] = None,
                     agents: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenate optional overlays to the base feature grid."""
    overlays: List[np.ndarray] = []
    if risk is not None:
        overlays.append(risk[..., None])
    if reward is not None:
        overlays.append(reward[..., None])
    if safe is not None:
        overlays.append(safe[..., None])
    if agents is not None:
        overlays.append(agents[..., None])
    if not overlays:
        return base_features
    overlay_stack = np.concatenate(overlays, axis=-1)
    return np.concatenate((base_features, overlay_stack), axis=-1)
