"""Cable and diagonal line systems.

Used by Erasmus Bridge (stay cables) and Van Nelle Factory (conveyor bridges).
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from ..core.blocks import Block
from ..core.curves import bresenham_line_3d

Coord3D = Tuple[int, int, int]


def fan_cables(
    anchor_x: int,
    anchor_y: int,
    anchor_z: int,
    deck_y: int,
    deck_z: int,
    deck_start_x: int,
    deck_end_x: int,
    count: int,
    block: str = Block.END_ROD,
) -> Dict[Coord3D, str]:
    """Generate a fan of cables from a single anchor point to deck positions.

    Used for Erasmus Bridge main span cables.
    All cables originate near the pylon top and fan down to evenly
    spaced points along the deck centerline.

    Args:
        anchor_x/y/z: Top attachment point (pylon top).
        deck_y: Y level of deck surface.
        deck_z: Z position along deck centerline.
        deck_start_x: X of first cable attachment on deck.
        deck_end_x: X of last cable attachment on deck.
        count: Number of cables.
        block: Block type for cables.
    """
    blocks: Dict[Coord3D, str] = {}

    if count <= 1:
        spacing = 0
    else:
        spacing = (deck_end_x - deck_start_x) / (count - 1)

    for i in range(count):
        deck_x = round(deck_start_x + i * spacing)
        line = bresenham_line_3d(
            anchor_x, anchor_y, anchor_z,
            deck_x, deck_y, deck_z,
        )
        for point in line:
            blocks[point] = block

    return blocks


def converging_cables(
    anchor_x: int,
    anchor_y: int,
    anchor_z: int,
    targets: List[Tuple[int, int, int]],
    block: str = Block.END_ROD,
) -> Dict[Coord3D, str]:
    """Generate cables from one anchor to multiple target points.

    Used for Erasmus Bridge backstays and any converging cable pattern.
    """
    blocks: Dict[Coord3D, str] = {}

    for tx, ty, tz in targets:
        line = bresenham_line_3d(anchor_x, anchor_y, anchor_z, tx, ty, tz)
        for point in line:
            blocks[point] = block

    return blocks


def sloped_bridge(
    start_x: int, start_y: int, start_z: int,
    end_x: int, end_y: int, end_z: int,
    width: int = 3,
    height: int = 3,
    wall_block: str = Block.GLASS_PANE,
    floor_block: str = Block.SMOOTH_STONE,
    frame_block: str = Block.IRON_BARS,
) -> Dict[Coord3D, str]:
    """Generate a sloped enclosed bridge/walkway (Van Nelle conveyor bridges).

    Creates a rectangular tube that follows a diagonal line between two points.

    Args:
        start_x/y/z: Start position (bottom of one building).
        end_x/y/z: End position (connects to other building).
        width: Bridge width in blocks.
        height: Bridge height in blocks.
        wall_block: Side wall material (glass for Van Nelle).
        floor_block: Floor material.
        frame_block: Edge frame material.
    """
    blocks: Dict[Coord3D, str] = {}

    # Get the centerline path
    centerline = bresenham_line_3d(start_x, start_y, start_z, end_x, end_y, end_z)

    half_w = width // 2

    for cx, cy, cz in centerline:
        for w in range(-half_w, half_w + 1):
            for h in range(height):
                # Position depends on bridge orientation
                # Determine primary direction
                dx = abs(end_x - start_x)
                dz = abs(end_z - start_z)

                if dx >= dz:
                    # Bridge runs along X, width along Z
                    pos = (cx, cy + h, cz + w)
                else:
                    # Bridge runs along Z, width along X
                    pos = (cx + w, cy + h, cz)

                # Determine block type
                is_edge = abs(w) == half_w or h == 0 or h == height - 1
                is_floor = h == 0
                is_frame = is_edge and (h == 0 or h == height - 1)

                if is_floor:
                    blocks[pos] = floor_block
                elif is_frame:
                    blocks[pos] = frame_block
                elif is_edge:
                    blocks[pos] = wall_block
                # Interior is air (not placed)

    return blocks
