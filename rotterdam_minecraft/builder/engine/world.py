"""3D block buffer (sparse voxel grid).

Central data structure: accumulates all placed blocks from building scripts.
Supports querying, merging, material lists, and bounding box calculation.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from collections import Counter

Coord3D = Tuple[int, int, int]


class World:
    """Sparse 3D voxel grid for accumulating Minecraft blocks."""

    def __init__(self):
        self._blocks: Dict[Coord3D, str] = {}

    def set_block(self, x: int, y: int, z: int, block: str) -> None:
        """Place a single block."""
        self._blocks[(x, y, z)] = block

    def set_blocks(self, positions: Set[Coord3D], block: str) -> None:
        """Place many blocks of the same type."""
        for pos in positions:
            self._blocks[pos] = block

    def set_block_dict(self, block_dict: Dict[Coord3D, str]) -> None:
        """Merge a {(x,y,z): block_type} dict into this world."""
        self._blocks.update(block_dict)

    def get_block(self, x: int, y: int, z: int) -> Optional[str]:
        """Get block at position, or None if empty."""
        return self._blocks.get((x, y, z))

    def has_block(self, x: int, y: int, z: int) -> bool:
        """Check if a block exists at position."""
        return (x, y, z) in self._blocks

    def remove_block(self, x: int, y: int, z: int) -> None:
        """Remove a block at position."""
        self._blocks.pop((x, y, z), None)

    def remove_region(
        self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int
    ) -> int:
        """Remove all blocks in a rectangular region. Returns count removed."""
        to_remove = []
        for pos in self._blocks:
            x, y, z = pos
            if x1 <= x <= x2 and y1 <= y <= y2 and z1 <= z <= z2:
                to_remove.append(pos)
        for pos in to_remove:
            del self._blocks[pos]
        return len(to_remove)

    def fill_region(
        self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, block: str
    ) -> None:
        """Fill a rectangular region with a block type."""
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for z in range(min(z1, z2), max(z1, z2) + 1):
                    self._blocks[(x, y, z)] = block

    def merge(self, other: World) -> None:
        """Merge another world's blocks into this one. Other's blocks win on conflict."""
        self._blocks.update(other._blocks)

    def bounds(self) -> Tuple[Coord3D, Coord3D]:
        """Get (min_corner, max_corner) bounding box."""
        if not self._blocks:
            return (0, 0, 0), (0, 0, 0)
        xs = [p[0] for p in self._blocks]
        ys = [p[1] for p in self._blocks]
        zs = [p[2] for p in self._blocks]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def dimensions(self) -> Tuple[int, int, int]:
        """Get (width_x, height_y, depth_z) of the bounding box."""
        (x1, y1, z1), (x2, y2, z2) = self.bounds()
        return (x2 - x1 + 1, y2 - y1 + 1, z2 - z1 + 1)

    def block_count(self) -> int:
        """Total number of placed blocks."""
        return len(self._blocks)

    def material_list(self) -> Dict[str, int]:
        """Count of each block type used."""
        return dict(Counter(self._blocks.values()))

    def blocks(self) -> Dict[Coord3D, str]:
        """Direct access to the block dict."""
        return self._blocks

    def all_positions(self) -> Set[Coord3D]:
        """Get all occupied positions."""
        return set(self._blocks.keys())

    def slice_y(self, y: int) -> Dict[Tuple[int, int], str]:
        """Get a horizontal slice at height Y. Returns {(x,z): block}."""
        return {
            (pos[0], pos[2]): block
            for pos, block in self._blocks.items()
            if pos[1] == y
        }

    def __repr__(self) -> str:
        dims = self.dimensions()
        return (
            f"World({self.block_count()} blocks, "
            f"{dims[0]}x{dims[1]}x{dims[2]})"
        )
