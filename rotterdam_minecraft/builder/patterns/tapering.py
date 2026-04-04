"""Variable-footprint stacking for bowl, diamond, and mansard profiles.

Distinct from sloped_roof: these generate the main building mass, not just a roof cap.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, List, Callable
from ..core.blocks import Block
from ..core.shapes import filled_circle, circle_ring, filled_diamond, diamond_shell, filled_rectangle, rectangle_shell
from ..core.curves import parabolic_diameter

Coord3D = Tuple[int, int, int]


def bowl_profile(
    base_diameter: float,
    top_diameter: float,
    height: int,
    exponent: float = 1.6,
    wall_thickness: int = 2,
    start_y: int = 0,
    wall_block: str = Block.IRON_BLOCK,
    floor_block: str = Block.SMOOTH_STONE,
    floor_interval: int = 6,
    atrium_width: int = 0,
    atrium_depth: int = 0,
) -> Dict[Coord3D, str]:
    """Generate a bowl/vase shaped building (Depot Boijmans).

    Narrow at base, widens toward top. Uses parabolic diameter formula.

    Args:
        base_diameter/top_diameter: Diameter at bottom/top.
        height: Total height in blocks.
        exponent: Parabolic curve exponent (1.6 for Depot).
        wall_thickness: Wall ring thickness.
        start_y: Y origin.
        wall_block: Exterior wall material.
        floor_block: Internal floor material.
        floor_interval: Blocks between floors.
        atrium_width/depth: Central void for atrium.
    """
    blocks: Dict[Coord3D, str] = {}
    diameters = parabolic_diameter(base_diameter, top_diameter, height, exponent)

    # Atrium void
    atrium = set()
    if atrium_width > 0 and atrium_depth > 0:
        atrium = filled_rectangle(atrium_width, atrium_depth)

    for y_layer in range(height):
        d = diameters[y_layer]
        r = d / 2
        inner_r = max(0, r - wall_thickness)

        # Wall ring
        wall = circle_ring(r, inner_r)
        for x, z in wall:
            blocks[(x, start_y + y_layer, z)] = wall_block

        # Floor slabs at intervals
        if y_layer % floor_interval == 0:
            floor = filled_circle(inner_r) - atrium
            for x, z in floor:
                blocks[(x, start_y + y_layer, z)] = floor_block

    return blocks


def diamond_taper(
    layers: List[dict],
    start_y: int = 0,
    wall_block: str = Block.YELLOW_CONCRETE,
    window_block: str = Block.GLASS_PANE,
) -> Dict[Coord3D, str]:
    """Generate a tilted cube shape by diamond expansion/contraction.

    Used by Cube Houses. Each layer specifies the diamond half-width.

    Args:
        layers: List of {"half_width": int, "is_window_layer": bool}.
        start_y: Y origin.
        wall_block: Cube wall material.
        window_block: Block for window positions.
    """
    blocks: Dict[Coord3D, str] = {}

    for i, layer in enumerate(layers):
        hw = layer["half_width"]
        is_window = layer.get("is_window_layer", False)

        if hw <= 0:
            continue

        # Diamond shell (just the edges)
        shell = diamond_shell(hw, thickness=1)
        fill_block = window_block if is_window else wall_block

        if is_window:
            # Alternate glass and wall on the shell
            for x, z in shell:
                # Windows on the flat faces, walls on corners
                if abs(x) + abs(z) == hw:  # Edge points
                    blocks[(x, start_y + i, z)] = wall_block
                else:
                    blocks[(x, start_y + i, z)] = window_block
        else:
            for x, z in shell:
                blocks[(x, start_y + i, z)] = wall_block

    return blocks


def square_taper(
    base_size: int,
    top_size: int,
    height: int,
    start_y: int = 0,
    wall_block: str = Block.DEEPSLATE_TILES,
    shell_only: bool = True,
    thickness: int = 1,
) -> Dict[Coord3D, str]:
    """Generate a tapering square cross-section (Witte Huis mansard zone).

    Linearly interpolates between base_size and top_size over height.
    """
    blocks: Dict[Coord3D, str] = {}

    for layer in range(height):
        t = layer / max(1, height - 1)
        size = round(base_size + (top_size - base_size) * t)
        size = max(1, size)

        if shell_only and size > 2:
            profile = rectangle_shell(size, size, thickness)
        else:
            profile = filled_rectangle(size, size)

        for x, z in profile:
            blocks[(x, start_y + layer, z)] = wall_block

    return blocks
