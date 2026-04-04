"""Regular window pattern insertion into solid walls.

Used by: Witte Huis, Hotel New York, Markthal exterior, Cube Houses.
Takes a solid wall and punches window holes at regular intervals.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional
from ..core.blocks import Block

Coord3D = Tuple[int, int, int]


def window_grid(
    wall_width: int,
    wall_height: int,
    window_width: int = 1,
    window_height: int = 2,
    h_spacing: int = 3,
    v_spacing: int = 4,
    margin_x: int = 1,
    margin_y: int = 1,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    wall_block: str = Block.WHITE_CONCRETE,
    window_block: str = Block.GLASS_PANE,
) -> Dict[Coord3D, str]:
    """Generate a wall with regularly spaced windows.

    Args:
        wall_width/height: Total wall dimensions.
        window_width/height: Size of each window opening.
        h_spacing: Horizontal distance between window centers.
        v_spacing: Vertical distance between window centers.
        margin_x/y: Edge margin before first window.
        start_x/y/z: Wall origin.
        face: "z" = wall on XY plane, "x" = wall on ZY plane.
        wall_block: Solid wall material.
        window_block: Glass block for windows.

    Returns:
        Dict mapping (x, y, z) -> block_type (wall + windows).
    """
    blocks: Dict[Coord3D, str] = {}

    # Build the window positions
    window_positions = set()
    wx = margin_x
    while wx + window_width <= wall_width - margin_x:
        wy = margin_y
        while wy + window_height <= wall_height - margin_y:
            for dx in range(window_width):
                for dy in range(window_height):
                    window_positions.add((wx + dx, wy + dy))
            wy += v_spacing
        wx += h_spacing

    # Fill wall and windows
    for w in range(wall_width):
        for h in range(wall_height):
            if face == "z":
                pos = (start_x + w, start_y + h, start_z)
            else:
                pos = (start_x, start_y + h, start_z + w)

            if (w, h) in window_positions:
                blocks[pos] = window_block
            else:
                blocks[pos] = wall_block

    return blocks


def add_windows_to_wall(
    wall_blocks: Dict[Coord3D, str],
    window_positions: Set[Coord3D],
    window_block: str = Block.GLASS_PANE,
) -> Dict[Coord3D, str]:
    """Replace specific positions in an existing wall with windows.

    Args:
        wall_blocks: Existing wall block dict.
        window_positions: Set of positions to replace with windows.
        window_block: Block type for windows.

    Returns:
        Updated wall dict with windows inserted.
    """
    result = dict(wall_blocks)
    for pos in window_positions:
        if pos in result:
            result[pos] = window_block
    return result
