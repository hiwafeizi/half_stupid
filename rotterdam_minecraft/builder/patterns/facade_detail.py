"""Decorative facade elements: dormers, cornices, balconies, arcades, clocks.

These handle the detailed architectural features that make each building unique.
"""

from __future__ import annotations
from typing import Dict, Tuple
from ..core.blocks import Block

Coord3D = Tuple[int, int, int]


def dormer(
    width: int = 3,
    height: int = 3,
    depth: int = 2,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    wall_block: str = Block.WHITE_CONCRETE,
    roof_block: str = Block.DEEPSLATE_TILES,
    window_block: str = Block.GLASS_PANE,
) -> Dict[Coord3D, str]:
    """Generate a dormer window protruding from a sloped roof.

    Used by Witte Huis mansard and Hotel New York.
    """
    blocks: Dict[Coord3D, str] = {}

    for dx in range(width):
        for dy in range(height):
            for dz in range(depth):
                if face == "z":
                    pos = (start_x + dx, start_y + dy, start_z + dz)
                else:
                    pos = (start_x + dz, start_y + dy, start_z + dx)

                # Front face (dz == 0): window in center, wall around
                if dz == 0:
                    center_x = width // 2
                    if dx == center_x and 0 < dy < height - 1:
                        blocks[pos] = window_block
                    else:
                        blocks[pos] = wall_block
                # Top layer: roof
                elif dy == height - 1:
                    blocks[pos] = roof_block
                # Side walls
                elif dx == 0 or dx == width - 1:
                    blocks[pos] = wall_block

    return blocks


def cornice_band(
    width: int,
    depth: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    block: str = Block.SMOOTH_STONE_SLAB,
    extend_out: int = 0,
) -> Dict[Coord3D, str]:
    """Generate a horizontal decorative band/cornice around a building.

    A single-layer ring at a specific height. Can extend outward by 1 block.
    Used by Witte Huis and Hotel New York for belt courses.
    """
    from ..core.shapes import rectangle_shell

    blocks: Dict[Coord3D, str] = {}
    actual_w = width + extend_out * 2
    actual_d = depth + extend_out * 2

    shell = rectangle_shell(actual_w, actual_d, thickness=1)
    for x, z in shell:
        blocks[(start_x + x, start_y, start_z + z)] = block

    return blocks


def balcony(
    width: int = 4,
    depth: int = 2,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    floor_block: str = Block.SMOOTH_STONE_SLAB,
    railing_block: str = Block.IRON_BARS,
) -> Dict[Coord3D, str]:
    """Generate a balcony protruding from a wall face.

    Used by Hotel New York (ship-railing style).
    """
    blocks: Dict[Coord3D, str] = {}

    for dx in range(width):
        for dz in range(depth):
            if face == "z":
                # Floor
                blocks[(start_x + dx, start_y, start_z - dz)] = floor_block
                # Railing on outer edge and sides
                if dz == depth - 1 or dx == 0 or dx == width - 1:
                    blocks[(start_x + dx, start_y + 1, start_z - dz)] = railing_block
            else:
                blocks[(start_x - dz, start_y, start_z + dx)] = floor_block
                if dz == depth - 1 or dx == 0 or dx == width - 1:
                    blocks[(start_x - dz, start_y + 1, start_z + dx)] = railing_block

    return blocks


def column_arcade(
    width: int,
    height: int,
    num_columns: int,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    column_block: str = Block.QUARTZ_PILLAR,
    arch_block: str = Block.SMOOTH_STONE,
    floor_block: str = Block.POLISHED_ANDESITE,
) -> Dict[Coord3D, str]:
    """Generate a ground-floor column arcade (Witte Huis).

    Columns at regular intervals with arched openings between them.
    """
    blocks: Dict[Coord3D, str] = {}
    spacing = width // max(1, num_columns + 1)

    for i in range(1, num_columns + 1):
        col_x = i * spacing
        for h in range(height):
            if face == "z":
                blocks[(start_x + col_x, start_y + h, start_z)] = column_block
            else:
                blocks[(start_x, start_y + h, start_z + col_x)] = column_block

    # Arch tops between columns
    for i in range(num_columns + 1):
        arch_start = i * spacing
        arch_end = min((i + 1) * spacing, width)
        arch_center = (arch_start + arch_end) // 2
        # Simple pointed arch: highest at center
        for x in range(arch_start, arch_end):
            dist_from_center = abs(x - arch_center)
            arch_y = height - 1 - dist_from_center
            if arch_y > 0:
                if face == "z":
                    blocks[(start_x + x, start_y + arch_y, start_z)] = arch_block
                else:
                    blocks[(start_x, start_y + arch_y, start_z + x)] = arch_block

    return blocks


def clock_face(
    diameter: int = 8,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    face_block: str = Block.WHITE_CONCRETE,
    rim_block: str = Block.BLACK_CONCRETE,
    hand_block: str = Block.BLACK_CONCRETE,
) -> Dict[Coord3D, str]:
    """Generate a clock face on a building facade.

    Used by Rotterdam Centraal and Hotel New York.
    """
    from ..core.shapes import filled_circle, circle_outline

    blocks: Dict[Coord3D, str] = {}
    r = diameter / 2

    # Rim
    rim = circle_outline(r, thickness=1)
    # Face
    face_pts = filled_circle(r - 1)
    # Center dot
    center = (0, 0)

    for x, z in rim:
        if face == "z":
            blocks[(start_x + x, start_y + z, start_z)] = rim_block
        else:
            blocks[(start_x, start_y + z, start_z + x)] = rim_block

    for x, z in face_pts:
        if face == "z":
            blocks[(start_x + x, start_y + z, start_z)] = face_block
        else:
            blocks[(start_x, start_y + z, start_z + x)] = face_block

    # Simple clock hands (12 o'clock and 3 o'clock)
    ri = int(r) - 1
    for i in range(1, ri):
        # Hour hand pointing up
        if face == "z":
            blocks[(start_x, start_y + i, start_z)] = hand_block
        else:
            blocks[(start_x, start_y + i, start_z)] = hand_block
    for i in range(1, ri - 1):
        # Minute hand pointing right
        if face == "z":
            blocks[(start_x + i, start_y, start_z)] = hand_block
        else:
            blocks[(start_x, start_y, start_z + i)] = hand_block

    return blocks
