"""Glass curtain wall generator.

Used by 6 buildings: Van Nelle, De Rotterdam, Rotterdam Centraal,
Markthal (glass ends), Depot (glass railing), Cube Houses.

Creates a flat or curved glass wall with vertical mullions,
horizontal spandrels/floor bands, and optional column setback.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Set
from ..core.blocks import Block

Coord3D = Tuple[int, int, int]


def curtain_wall(
    width: int,
    height: int,
    mullion_spacing: int = 2,
    spandrel_spacing: int = 4,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    glass_block: str = Block.GLASS_PANE,
    mullion_block: str = Block.IRON_BARS,
    spandrel_block: str = Block.LIGHT_GRAY_CONCRETE,
    column_setback: int = 0,
    column_block: Optional[str] = None,
    column_spacing: int = 6,
    solid_panel_chance: float = 0.0,
) -> Dict[Coord3D, str]:
    """Generate a curtain wall with glass, mullions, and spandrels.

    Args:
        width: Wall width in blocks.
        height: Wall height in blocks.
        mullion_spacing: Blocks between vertical mullions.
        spandrel_spacing: Blocks between horizontal floor bands.
        start_x/y/z: Origin position.
        face: "z" = wall on XY plane (facing Z), "x" = wall on ZY plane.
        glass_block: Block for glass panels.
        mullion_block: Block for vertical mullions.
        spandrel_block: Block for horizontal floor bands.
        column_setback: How far structural columns sit behind glass (0 or 1).
        column_block: Block for setback columns (None = no columns).
        column_spacing: Blocks between structural columns.
        solid_panel_chance: Fraction of glass panels replaced with solid (De Rotterdam hotel zone).

    Returns:
        Dict mapping (x, y, z) -> block_type.
    """
    import random
    blocks: Dict[Coord3D, str] = {}

    for w in range(width):
        for h in range(height):
            # Determine position based on face direction
            if face == "z":
                pos = (start_x + w, start_y + h, start_z)
            else:  # face == "x"
                pos = (start_x, start_y + h, start_z + w)

            # Determine block type
            is_mullion = (w % mullion_spacing == 0)
            is_spandrel = (h % spandrel_spacing == 0) and h > 0

            if is_mullion:
                blocks[pos] = mullion_block
            elif is_spandrel:
                blocks[pos] = spandrel_block
            else:
                if solid_panel_chance > 0 and random.random() < solid_panel_chance:
                    blocks[pos] = spandrel_block
                else:
                    blocks[pos] = glass_block

    # Add setback structural columns
    if column_block and column_setback > 0:
        for w in range(0, width, column_spacing):
            for h in range(height):
                if face == "z":
                    col_pos = (start_x + w, start_y + h, start_z - column_setback)
                else:
                    col_pos = (start_x - column_setback, start_y + h, start_z + w)
                blocks[col_pos] = column_block

    return blocks


def cable_net_glass(
    width: int,
    height: int,
    cable_spacing: int = 5,
    start_x: int = 0,
    start_y: int = 0,
    start_z: int = 0,
    face: str = "z",
    glass_block: str = Block.GLASS_PANE,
    cable_block: str = Block.IRON_BARS,
) -> Dict[Coord3D, str]:
    """Generate a cable-net glass facade (Markthal end walls).

    A grid of cables (iron bars) with glass panes between them.
    Unlike curtain_wall, this has cables in BOTH directions.
    """
    blocks: Dict[Coord3D, str] = {}

    for w in range(width):
        for h in range(height):
            if face == "z":
                pos = (start_x + w, start_y + h, start_z)
            else:
                pos = (start_x, start_y + h, start_z + w)

            is_cable = (w % cable_spacing == 0) or (h % cable_spacing == 0)
            blocks[pos] = cable_block if is_cable else glass_block

    return blocks
