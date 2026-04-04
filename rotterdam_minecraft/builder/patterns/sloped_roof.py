"""Roof geometry generators.

Handles mansard, asymmetric peaked, hipped, conical, and flat roofs.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, List
from ..core.blocks import Block
from ..core.shapes import filled_rectangle, rectangle_shell, filled_circle, circle_outline

Coord3D = Tuple[int, int, int]


def asymmetric_peak(
    width: int,
    depth: int,
    peak_height: int,
    peak_x_position: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    block: str = Block.IRON_BLOCK,
) -> Dict[Coord3D, str]:
    """Generate an asymmetric pointed roof (Rotterdam Centraal canopy).

    The peak is off-center; left side is steeper than right side.

    Args:
        width: Total roof width (X axis).
        depth: Roof depth (Z axis).
        peak_height: Height at the peak point.
        peak_x_position: X position of peak from left edge.
        start_x/y/z: Origin.
        block: Roof surface block.
    """
    blocks: Dict[Coord3D, str] = {}

    for z in range(depth):
        for x in range(width):
            # Calculate height at this x position
            if x <= peak_x_position:
                # Left slope (steep)
                if peak_x_position > 0:
                    h = round(peak_height * x / peak_x_position)
                else:
                    h = peak_height
            else:
                # Right slope (gentle)
                remaining = width - peak_x_position - 1
                if remaining > 0:
                    h = round(peak_height * (width - 1 - x) / remaining)
                else:
                    h = 0

            # Place roof surface block at the calculated height
            if h >= 0:
                blocks[(start_x + x, start_y + h, start_z + z)] = block

    return blocks


def mansard_taper(
    base_width: int,
    base_depth: int,
    height: int,
    taper_per_layer: float = 0.5,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    wall_block: str = Block.DEEPSLATE_TILES,
    fill_interior: bool = False,
) -> Dict[Coord3D, str]:
    """Generate a mansard (tapering) roof.

    Used by Witte Huis and Hotel New York. Each layer shrinks inward.

    Args:
        base_width/depth: Starting footprint size.
        height: Total roof height in blocks.
        taper_per_layer: How many blocks to shrink per layer (each side).
        start_x/y/z: Origin.
        wall_block: Roof surface material.
        fill_interior: If True, fill solid. If False, only walls.
    """
    blocks: Dict[Coord3D, str] = {}

    for layer in range(height):
        shrink = round(layer * taper_per_layer)
        w = max(1, base_width - shrink * 2)
        d = max(1, base_depth - shrink * 2)

        if fill_interior:
            profile = filled_rectangle(w, d)
        else:
            profile = rectangle_shell(w, d, thickness=1)

        for x, z in profile:
            blocks[(start_x + x, start_y + layer, start_z + z)] = wall_block

    return blocks


def hipped_roof(
    width: int,
    depth: int,
    height: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    block: str = Block.DEEPSLATE_TILES,
) -> Dict[Coord3D, str]:
    """Generate a hipped roof (slopes on all 4 sides).

    Used by Hotel New York main body.
    """
    blocks: Dict[Coord3D, str] = {}

    for layer in range(height):
        shrink_x = round(layer * width / (2 * height))
        shrink_z = round(layer * depth / (2 * height))
        w = max(1, width - shrink_x * 2)
        d = max(1, depth - shrink_z * 2)

        # Only the outer edge at each layer (roof surface)
        profile = rectangle_shell(w, d, thickness=1)
        for x, z in profile:
            blocks[(start_x + x, start_y + layer, start_z + z)] = block

    return blocks


def conical_cap(
    base_diameter: int,
    height: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    block: str = Block.OXIDIZED_COPPER,
) -> Dict[Coord3D, str]:
    """Generate a conical/dome cap (Hotel NY tower cupolas, Witte Huis turrets).

    Each layer is a circle that shrinks toward the top.
    """
    blocks: Dict[Coord3D, str] = {}

    for layer in range(height):
        # Linear taper from base_diameter to 1
        t = layer / max(1, height - 1)
        d = max(1, round(base_diameter * (1 - t)))
        r = d / 2

        outline = circle_outline(r, thickness=1) if d > 2 else filled_circle(r)
        for x, z in outline:
            blocks[(start_x + x, start_y + layer, start_z + z)] = block

    return blocks


def flat_roof(
    width: int,
    depth: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    block: str = Block.SMOOTH_STONE,
) -> Dict[Coord3D, str]:
    """Simple flat roof slab."""
    blocks: Dict[Coord3D, str] = {}
    for x, z in filled_rectangle(width, depth):
        blocks[(start_x + x, start_y, start_z + z)] = block
    return blocks
