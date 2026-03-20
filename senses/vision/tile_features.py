"""Utilities for translating Malmo block identifiers into semantic features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass(frozen=True)
class TileFeatures:
    block: str
    walkable: float
    danger: float
    reward: float
    friction: float
    solid: float
    liquid: float
    unknown: float

    def as_vector(self) -> np.ndarray:
        """Returns the feature vector in a consistent order."""
        return np.array(
            [
                self.walkable,
                self.danger,
                self.reward,
                self.friction,
                self.solid,
                self.liquid,
                self.unknown,
            ],
            dtype=np.float32,
        )


# --- Core feature catalogue ------------------------------------------------------------------

# Scores are heuristics in [0, 1]. Adjust as we calibrate behaviour.
_BASE_FEATURES: Dict[str, TileFeatures] = {
    "air": TileFeatures("air", walkable=1.0, danger=0.0, reward=0.0, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "grass": TileFeatures("grass", walkable=1.0, danger=0.05, reward=0.2, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "dirt": TileFeatures("dirt", walkable=1.0, danger=0.05, reward=0.05, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "stone": TileFeatures("stone", walkable=1.0, danger=0.05, reward=0.0, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "sand": TileFeatures("sand", walkable=1.0, danger=0.1, reward=0.0, friction=0.9, solid=1.0, liquid=0.0, unknown=0.0),
    "gravel": TileFeatures("gravel", walkable=1.0, danger=0.1, reward=0.0, friction=0.85, solid=1.0, liquid=0.0, unknown=0.0),
    "water": TileFeatures("water", walkable=0.2, danger=0.4, reward=0.0, friction=0.5, solid=0.0, liquid=1.0, unknown=0.0),
    "flowing_water": TileFeatures("flowing_water", walkable=0.1, danger=0.45, reward=0.0, friction=0.4, solid=0.0, liquid=1.0, unknown=0.0),
    "lava": TileFeatures("lava", walkable=0.0, danger=1.0, reward=0.0, friction=0.3, solid=0.0, liquid=1.0, unknown=0.0),
    "flowing_lava": TileFeatures("flowing_lava", walkable=0.0, danger=1.0, reward=0.0, friction=0.2, solid=0.0, liquid=1.0, unknown=0.0),
    "fire": TileFeatures("fire", walkable=0.0, danger=0.9, reward=0.0, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "ice": TileFeatures("ice", walkable=0.8, danger=0.1, reward=0.0, friction=1.2, solid=1.0, liquid=0.0, unknown=0.0),
    "packed_ice": TileFeatures("packed_ice", walkable=0.8, danger=0.05, reward=0.0, friction=1.2, solid=1.0, liquid=0.0, unknown=0.0),
    "soul_sand": TileFeatures("soul_sand", walkable=0.9, danger=0.2, reward=0.0, friction=0.6, solid=1.0, liquid=0.0, unknown=0.0),
    "vine": TileFeatures("vine", walkable=0.5, danger=0.05, reward=0.0, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "log": TileFeatures("log", walkable=1.0, danger=0.05, reward=0.1, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "leaves": TileFeatures("leaves", walkable=0.6, danger=0.05, reward=0.0, friction=0.8, solid=0.3, liquid=0.0, unknown=0.0),
    "planks": TileFeatures("planks", walkable=1.0, danger=0.05, reward=0.15, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "wheat": TileFeatures("wheat", walkable=0.9, danger=0.05, reward=0.7, friction=1.0, solid=0.1, liquid=0.0, unknown=0.0),
    "carrots": TileFeatures("carrots", walkable=0.9, danger=0.05, reward=0.7, friction=1.0, solid=0.1, liquid=0.0, unknown=0.0),
    "potatoes": TileFeatures("potatoes", walkable=0.9, danger=0.05, reward=0.7, friction=1.0, solid=0.1, liquid=0.0, unknown=0.0),
    "beetroots": TileFeatures("beetroots", walkable=0.9, danger=0.05, reward=0.7, friction=1.0, solid=0.1, liquid=0.0, unknown=0.0),
    "cactus": TileFeatures("cactus", walkable=0.1, danger=0.7, reward=0.0, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
    "web": TileFeatures("web", walkable=0.1, danger=0.4, reward=0.0, friction=0.2, solid=0.1, liquid=0.0, unknown=0.0),
    "tallgrass": TileFeatures("tallgrass", walkable=1.0, danger=0.05, reward=0.1, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "flower": TileFeatures("flower", walkable=1.0, danger=0.0, reward=0.2, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "torch": TileFeatures("torch", walkable=1.0, danger=0.0, reward=0.1, friction=1.0, solid=0.0, liquid=0.0, unknown=0.0),
    "obsidian": TileFeatures("obsidian", walkable=1.0, danger=0.1, reward=0.0, friction=1.0, solid=1.0, liquid=0.0, unknown=0.0),
}

_DEFAULT = TileFeatures("unknown", walkable=0.5, danger=0.3, reward=0.0, friction=1.0, solid=1.0, liquid=0.0, unknown=1.0)


def tile_to_features(block: str) -> TileFeatures:
    """Return semantic features for a block name."""
    block = (block or "").lower()
    if block in _BASE_FEATURES:
        return _BASE_FEATURES[block]
    return TileFeatures(
        block=block if block else "unknown",
        walkable=_DEFAULT.walkable,
        danger=_DEFAULT.danger,
        reward=_DEFAULT.reward,
        friction=_DEFAULT.friction,
        solid=_DEFAULT.solid,
        liquid=_DEFAULT.liquid,
        unknown=1.0,
    )


def batch_tiles_to_vectors(blocks: Iterable[str]) -> np.ndarray:
    """Convert an iterable of block ids to an array of feature vectors."""
    vectors = [tile_to_features(b).as_vector() for b in blocks]
    return np.vstack(vectors) if vectors else np.zeros((0, 7), dtype=np.float32)


def register_block_features(block: str, *, walkable: float, danger: float, reward: float,
                             friction: float, solid: float, liquid: float) -> None:
    """Extend the feature catalogue at runtime."""
    norm_block = block.lower()
    feature = TileFeatures(
        block=norm_block,
        walkable=walkable,
        danger=danger,
        reward=reward,
        friction=friction,
        solid=solid,
        liquid=liquid,
        unknown=0.0,
    )
    # Cast to mutable copy for extension.
    global _BASE_FEATURES
    mutable: Dict[str, TileFeatures] = dict(_BASE_FEATURES)
    mutable[norm_block] = feature
    _BASE_FEATURES = mutable
