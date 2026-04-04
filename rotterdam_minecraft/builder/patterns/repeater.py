"""Unit duplication and array patterns.

Clone a set of blocks along a line or path. Used for:
- Cube Houses: 38 identical cubes along an S-curve
- Van Nelle: identical factory bays repeated along X axis
- Markthal: arch cross-section repeated along Z axis
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set
from ..core.transforms import translate, translate_dict

Coord3D = Tuple[int, int, int]


def array_linear(
    unit_blocks: Dict[Coord3D, str],
    count: int,
    spacing_x: int = 0,
    spacing_y: int = 0,
    spacing_z: int = 0,
) -> Dict[Coord3D, str]:
    """Repeat a block set in a straight line.

    Args:
        unit_blocks: The template block dict to repeat.
        count: Number of copies.
        spacing_x/y/z: Offset between each copy.

    Returns:
        Combined dict of all copies.
    """
    result: Dict[Coord3D, str] = {}

    for i in range(count):
        dx = i * spacing_x
        dy = i * spacing_y
        dz = i * spacing_z
        translated = translate_dict(unit_blocks, dx, dy, dz)
        result.update(translated)

    return result


def array_along_path(
    unit_blocks: Dict[Coord3D, str],
    positions: List[Tuple[int, int, int]],
    remove_shared_faces: bool = False,
    shared_axis: str = "x",
) -> Dict[Coord3D, str]:
    """Place a block template at each position in a path.

    More flexible than array_linear: positions can follow curves.
    Used for Cube Houses along the zigzag path.

    Args:
        unit_blocks: Template blocks (centered at origin).
        positions: List of (x, y, z) anchor points.
        remove_shared_faces: If True, remove walls between adjacent units.
        shared_axis: Which axis adjacent units share walls on.

    Returns:
        Combined dict of all placed units.
    """
    result: Dict[Coord3D, str] = {}

    for pos in positions:
        translated = translate_dict(unit_blocks, pos[0], pos[1], pos[2])
        result.update(translated)

    if remove_shared_faces and len(positions) > 1:
        # Find and remove blocks that exist in overlapping regions
        # between adjacent units (shared walls)
        for i in range(len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            # Find blocks that would be in the overlap zone
            overlap_blocks = set()
            for pos, block in list(result.items()):
                x, y, z = pos
                if shared_axis == "x":
                    mid_x = (p1[0] + p2[0]) // 2
                    if abs(x - mid_x) <= 1:
                        overlap_blocks.add(pos)
                elif shared_axis == "z":
                    mid_z = (p1[2] + p2[2]) // 2
                    if abs(z - mid_z) <= 1:
                        overlap_blocks.add(pos)

            # Don't actually remove floor/ceiling, just walls
            for pos in overlap_blocks:
                if pos in result:
                    del result[pos]

    return result


def array_with_variations(
    base_blocks: Dict[Coord3D, str],
    count: int,
    spacing_x: int = 0,
    spacing_z: int = 0,
    variations: Dict[int, Dict[Coord3D, str]] = None,
) -> Dict[Coord3D, str]:
    """Repeat a base unit with per-instance variations.

    Args:
        base_blocks: Common template.
        count: Number of copies.
        spacing_x/z: Offset per copy.
        variations: Dict of {copy_index: {extra_blocks}} to overlay.

    Returns:
        Combined dict with base + any per-instance overrides.
    """
    result: Dict[Coord3D, str] = {}

    for i in range(count):
        dx = i * spacing_x
        dz = i * spacing_z
        translated = translate_dict(base_blocks, dx, 0, dz)
        result.update(translated)

        # Apply variations for this copy
        if variations and i in variations:
            var_translated = translate_dict(variations[i], dx, 0, dz)
            result.update(var_translated)

    return result
